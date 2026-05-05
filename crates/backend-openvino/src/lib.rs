use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use libloading::Library;
use loci_protocol::{
    AcceleratorKind, Backend, BackendAssetCapabilities, BackendDescriptor, BackendError,
    BackendExecutionProfile, BackendKernelCatalog, BackendLoweringCapabilities, BackendOutput,
    BackendResult, BackendRuntimeFamily, BackendTelemetry, ChipOperatorClass, DeviceDescriptor,
    ExecutionArtifactKind, ExecutionPlan, HardwareTopology, ImageInput, KernelDescriptor,
    KernelImplementationKind, KernelMaturity, KernelOrigin, LoweringGranularity, ModelAssetLayout,
    ModelDescriptor, ModelFormat, OpenVinoExecutionMode, OpenVinoExecutionProfile, PipelineStage,
    PlacementDecision, PowerState, PreparedModel, PreparedResidency, SessionRequest, ThermalState,
};
use openvino::{CompiledModel, Core, DeviceType, ElementType, Shape, Tensor};
use openvino_genai::{
    DecodedResults, GenerationConfig, LlmPipeline, VlmDecodedResults, VlmPipeline,
};
use std::{
    borrow::Cow,
    collections::HashMap,
    env,
    ffi::{c_char, CString},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
};

pub fn boxed_backend() -> Box<dyn Backend> {
    Box::new(OpenVinoBackend::default())
}

#[derive(Default)]
struct OpenVinoBackend {
    runtime: OpenVinoRuntime,
}

struct OpenVinoRuntime {
    sessions: Mutex<HashMap<String, SessionSlot>>,
}

enum SessionSlot {
    Real(RealSession),
    Fallback(FallbackSession),
}

#[derive(Debug, Clone)]
enum ModelPreparationState {
    ReadyOpenVinoArtifact {
        model_root: PathBuf,
        source_root: Option<PathBuf>,
    },
    MaterializationPlanned {
        prepared_root: PathBuf,
        source_root: PathBuf,
        expected_entrypoint: &'static str,
        metadata_path: PathBuf,
        detail: String,
    },
    MaterializedPlaceholder {
        model_root: PathBuf,
        source_root: PathBuf,
        expected_entrypoint: &'static str,
        metadata_path: PathBuf,
        detail: String,
    },
    RequiresBackendAdaptation {
        source_root: PathBuf,
        expected_entrypoint: &'static str,
        detail: String,
    },
    Unsupported {
        detail: String,
    },
}

#[derive(Debug, Clone)]
struct PreparedArtifactResolver {
    cache_root: PathBuf,
    toolchain: OpenVinoToolchainConfig,
}

#[derive(Debug, Clone)]
struct MaterializedPreparation {
    prepared_root: PathBuf,
    metadata_path: PathBuf,
    expected_entrypoint: &'static str,
    placeholder_entrypoint: PathBuf,
}

#[derive(Debug, Clone, Copy)]
enum PreparationJobKind {
    DiscoverOnly,
    MaterializePlaceholder,
    ValidateTextSourceAsset,
    MaterializeTextExecutable,
}

#[derive(Debug, Clone, Copy)]
enum PreparationJobStatus {
    Ready,
    MaterializedPlaceholder,
    ValidatedSourceAsset,
    MaterializationAttempted,
    RequiresAdaptation,
    Unsupported,
}

#[derive(Debug, Clone)]
struct PreparationJobSpec {
    kind: PreparationJobKind,
    expected_entrypoint: &'static str,
}

#[derive(Debug, Clone)]
struct PreparationJobOutcome {
    status: PreparationJobStatus,
    model_root: Option<PathBuf>,
    source_root: Option<PathBuf>,
    metadata_path: Option<PathBuf>,
    expected_entrypoint: &'static str,
    tool_invocation: Option<String>,
    stdout_path: Option<PathBuf>,
    stderr_path: Option<PathBuf>,
    validation_summary: Option<String>,
    detail: String,
}

#[derive(Debug, Clone)]
struct TextSourceValidation {
    config_path: PathBuf,
    tokenizer_path: Option<PathBuf>,
    index_path: Option<PathBuf>,
    config_excerpt: String,
}

#[derive(Debug, Clone)]
struct OpenVinoToolchainConfig {
    strategy: OpenVinoToolchainStrategy,
    install_root: Option<PathBuf>,
    text_materializer: Option<PathBuf>,
    ffi_library: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
enum OpenVinoToolchainStrategy {
    Deferred,
    ExternalCli,
    FfiBridge,
}

type BridgeVersionFn = unsafe extern "C" fn() -> *const c_char;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BridgeTextMaterializeRequest {
    source_root: *const c_char,
    prepared_root: *const c_char,
    model_name: *const c_char,
    architecture: *const c_char,
    config_json_path: *const c_char,
    tokenizer_json_path: *const c_char,
    safetensors_index_path: *const c_char,
    options_json: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BridgeTextMaterializeResponse {
    status: i32,
    artifact_root: *mut c_char,
    entrypoint_path: *mut c_char,
    metadata_json: *mut c_char,
    error_message: *mut c_char,
}

type BridgeMaterializeFn = unsafe extern "C" fn(
    request: *const BridgeTextMaterializeRequest,
) -> BridgeTextMaterializeResponse;

type BridgeFreeStringFn = unsafe extern "C" fn(ptr: *mut c_char);

struct RealSession {
    pipeline: RealPipeline,
    lowering_diagnostics: Option<String>,
}

enum RealPipeline {
    Llm(LlmPipeline),
    Vlm(VlmPipeline),
    Onnx(OnnxSession),
}

struct OnnxSession {
    compiled_model: CompiledModel,
}

struct FallbackSession {
    reason: String,
    device_name: String,
    model_root: Option<PathBuf>,
    lowering_diagnostics: Option<String>,
}

#[derive(Debug, Clone)]
struct RuntimeBootstrap {
    root_dir: PathBuf,
    lib_paths: Vec<PathBuf>,
    applied_environment: bool,
}

impl Default for OpenVinoRuntime {
    fn default() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }
}

impl PreparedArtifactResolver {
    fn new() -> Self {
        Self {
            cache_root: std::env::temp_dir().join("loci-openvino-artifacts"),
            toolchain: OpenVinoToolchainConfig::from_environment(),
        }
    }

    fn inspect(&self, model: &ModelDescriptor) -> ModelPreparationState {
        let validation_spec = PreparationJobSpec {
            kind: PreparationJobKind::ValidateTextSourceAsset,
            expected_entrypoint: expected_openvino_entrypoint(model),
        };
        let validation_state = self.execute_job(model, validation_spec);
        if matches!(
            validation_state,
            ModelPreparationState::RequiresBackendAdaptation { .. }
        ) {
            if !model.is_multimodal_architecture() {
                let materialize_text_spec = PreparationJobSpec {
                    kind: PreparationJobKind::MaterializeTextExecutable,
                    expected_entrypoint: expected_openvino_entrypoint(model),
                };
                let materialized_state = self.execute_job(model, materialize_text_spec);
                if matches!(
                    materialized_state,
                    ModelPreparationState::MaterializationPlanned { .. }
                        | ModelPreparationState::MaterializedPlaceholder { .. }
                        | ModelPreparationState::ReadyOpenVinoArtifact { .. }
                ) {
                    return materialized_state;
                }
            }

            let materialize_spec = PreparationJobSpec {
                kind: PreparationJobKind::MaterializePlaceholder,
                expected_entrypoint: expected_openvino_entrypoint(model),
            };
            return self.execute_job(model, materialize_spec);
        }
        validation_state
    }

    fn execute_job(
        &self,
        model: &ModelDescriptor,
        spec: PreparationJobSpec,
    ) -> ModelPreparationState {
        let Some(source_root) = resolve_model_root(model) else {
            return ModelPreparationState::Unsupported {
                detail: format!(
                    "model path `{}` does not resolve to a usable local asset root",
                    model.path.display()
                ),
            };
        };

        let expected_entrypoint = spec.expected_entrypoint;

        if source_root.is_file()
            && matches!(
                model.inferred_format(),
                ModelFormat::Gguf
                    | ModelFormat::Onnx
                    | ModelFormat::OpenVinoIr
                    | ModelFormat::OpenVinoBlob
            )
        {
            return self.outcome_to_state(PreparationJobOutcome {
                status: PreparationJobStatus::Ready,
                model_root: Some(source_root),
                source_root: None,
                metadata_path: None,
                expected_entrypoint,
                tool_invocation: None,
                stdout_path: None,
                stderr_path: None,
                validation_summary: Some(
                    "source file is directly executable by the OpenVINO backend".to_string(),
                ),
                detail: "source file is directly executable by the OpenVINO backend".to_string(),
            });
        }

        if source_root.join(expected_entrypoint).is_file() {
            return self.outcome_to_state(PreparationJobOutcome {
                status: PreparationJobStatus::Ready,
                model_root: Some(source_root),
                source_root: None,
                metadata_path: None,
                expected_entrypoint,
                tool_invocation: None,
                stdout_path: None,
                stderr_path: None,
                validation_summary: Some(
                    "source root already contains a directly executable OpenVINO artifact"
                        .to_string(),
                ),
                detail:
                    "source asset root already contains a directly executable OpenVINO artifact"
                        .to_string(),
            });
        }

        if let Some(prepared_root) =
            self.discover_prepared_root(model, &source_root, expected_entrypoint)
        {
            return self.outcome_to_state(PreparationJobOutcome {
                status: PreparationJobStatus::Ready,
                model_root: Some(prepared_root),
                source_root: Some(source_root),
                metadata_path: None,
                expected_entrypoint,
                tool_invocation: None,
                stdout_path: None,
                stderr_path: None,
                validation_summary: Some(
                    "backend-managed prepared root already contains an executable OpenVINO artifact"
                        .to_string(),
                ),
                detail: "backend-managed prepared root already contains an executable OpenVINO artifact"
                    .to_string(),
            });
        }

        if is_raw_transformers_checkpoint(&source_root) {
            if matches!(spec.kind, PreparationJobKind::ValidateTextSourceAsset)
                && !model.is_multimodal_architecture()
            {
                if let Ok(validation) = self.validate_text_source_asset(&source_root) {
                    return self.outcome_to_state(PreparationJobOutcome {
                        status: PreparationJobStatus::ValidatedSourceAsset,
                        model_root: None,
                        source_root: Some(source_root),
                        metadata_path: None,
                        expected_entrypoint,
                        tool_invocation: None,
                        stdout_path: None,
                        stderr_path: None,
                        validation_summary: Some(format!(
                            "config=`{}`, tokenizer_present={}, index_present={}",
                            validation.config_path.display(),
                            validation.tokenizer_path.is_some(),
                            validation.index_path.is_some()
                        )),
                        detail: format!(
                            "text source asset validated for backend-managed preparation: config=`{}`, tokenizer_present={}, index_present={}, architecture_hint={} ",
                            validation.config_path.display(),
                            validation.tokenizer_path.is_some(),
                            validation.index_path.is_some(),
                            validation.config_excerpt
                        ),
                    });
                }
            }

            if matches!(spec.kind, PreparationJobKind::MaterializeTextExecutable)
                && !model.is_multimodal_architecture()
            {
                if let Ok(attempt) =
                    self.attempt_text_materialization(model, &source_root, expected_entrypoint)
                {
                    return self.outcome_to_state(attempt);
                }
            }

            if matches!(spec.kind, PreparationJobKind::MaterializePlaceholder) {
                if let Ok(materialized) =
                    self.materialize_placeholder(model, &source_root, expected_entrypoint)
                {
                    return self.outcome_to_state(PreparationJobOutcome {
                        status: PreparationJobStatus::MaterializedPlaceholder,
                        model_root: Some(materialized.prepared_root),
                        source_root: Some(source_root),
                        metadata_path: Some(materialized.metadata_path),
                        expected_entrypoint: materialized.expected_entrypoint,
                        tool_invocation: None,
                        stdout_path: None,
                        stderr_path: None,
                        validation_summary: Some(format!(
                            "placeholder entrypoint materialized at `{}`",
                            materialized.placeholder_entrypoint.display()
                        )),
                        detail: format!(
                            "backend-managed preparation metadata created at `{}`; executable artifact is still missing",
                            materialized.placeholder_entrypoint.display()
                        ),
                    });
                }
            }

            return self.outcome_to_state(PreparationJobOutcome {
                status: PreparationJobStatus::RequiresAdaptation,
                model_root: None,
                source_root: Some(source_root),
                metadata_path: None,
                expected_entrypoint,
                tool_invocation: None,
                stdout_path: None,
                stderr_path: None,
                validation_summary: None,
                detail: format!(
                    "raw Transformers checkpoint detected ({}); checked backend-managed prepared roots under `{}`",
                    if model.is_multimodal_architecture() {
                        "multimodal"
                    } else {
                        "text"
                    },
                    self.cache_root.display()
                ),
            });
        }

        self.outcome_to_state(PreparationJobOutcome {
            status: PreparationJobStatus::Unsupported,
            model_root: None,
            source_root: Some(source_root.clone()),
            metadata_path: None,
            expected_entrypoint,
            tool_invocation: None,
            stdout_path: None,
            stderr_path: None,
            validation_summary: None,
            detail: format!(
                "model directory `{}` does not contain a prepared OpenVINO artifact or a recognized raw checkpoint layout",
                source_root.display()
            ),
        })
    }

    fn outcome_to_state(&self, outcome: PreparationJobOutcome) -> ModelPreparationState {
        match outcome.status {
            PreparationJobStatus::Ready => ModelPreparationState::ReadyOpenVinoArtifact {
                model_root: outcome
                    .model_root
                    .unwrap_or_else(|| self.cache_root.clone()),
                source_root: outcome.source_root,
            },
            PreparationJobStatus::MaterializationAttempted => {
                ModelPreparationState::MaterializationPlanned {
                    prepared_root: outcome
                        .model_root
                        .unwrap_or_else(|| self.cache_root.clone()),
                    source_root: outcome
                        .source_root
                        .unwrap_or_else(|| self.cache_root.clone()),
                    expected_entrypoint: outcome.expected_entrypoint,
                    metadata_path: outcome
                        .metadata_path
                        .unwrap_or_else(|| self.cache_root.join("loci-prepared-artifact.json")),
                    detail: format!(
                        "{}{}{}{}{}",
                        outcome.detail,
                        outcome
                            .tool_invocation
                            .as_ref()
                            .map(|cmd| format!("; tool_invocation=`{cmd}`"))
                            .unwrap_or_default(),
                        outcome
                            .stdout_path
                            .as_ref()
                            .map(|path| format!("; stdout=`{}`", path.display()))
                            .unwrap_or_default(),
                        outcome
                            .stderr_path
                            .as_ref()
                            .map(|path| format!("; stderr=`{}`", path.display()))
                            .unwrap_or_default(),
                        outcome
                            .validation_summary
                            .as_ref()
                            .map(|summary| format!("; validation_summary=`{summary}`"))
                            .unwrap_or_default(),
                    ),
                }
            }
            PreparationJobStatus::MaterializedPlaceholder => {
                ModelPreparationState::MaterializedPlaceholder {
                    model_root: outcome
                        .model_root
                        .unwrap_or_else(|| self.cache_root.clone()),
                    source_root: outcome
                        .source_root
                        .unwrap_or_else(|| self.cache_root.clone()),
                    expected_entrypoint: outcome.expected_entrypoint,
                    metadata_path: outcome
                        .metadata_path
                        .unwrap_or_else(|| self.cache_root.join("loci-prepared-artifact.json")),
                    detail: outcome.detail,
                }
            }
            PreparationJobStatus::ValidatedSourceAsset => {
                ModelPreparationState::RequiresBackendAdaptation {
                    source_root: outcome
                        .source_root
                        .unwrap_or_else(|| self.cache_root.clone()),
                    expected_entrypoint: outcome.expected_entrypoint,
                    detail: outcome.detail,
                }
            }
            PreparationJobStatus::RequiresAdaptation => {
                ModelPreparationState::RequiresBackendAdaptation {
                    source_root: outcome
                        .source_root
                        .unwrap_or_else(|| self.cache_root.clone()),
                    expected_entrypoint: outcome.expected_entrypoint,
                    detail: outcome.detail,
                }
            }
            PreparationJobStatus::Unsupported => ModelPreparationState::Unsupported {
                detail: outcome.detail,
            },
        }
    }

    fn discover_prepared_root(
        &self,
        model: &ModelDescriptor,
        source_root: &Path,
        expected_entrypoint: &str,
    ) -> Option<PathBuf> {
        let mut candidates = Vec::new();
        candidates.push(source_root.join(".loci/openvino"));
        candidates.push(source_root.join("openvino"));
        candidates.push(self.cache_root.join(sanitize_name(&model.name)));

        candidates.into_iter().find(|candidate| {
            let entrypoint = candidate.join(expected_entrypoint);
            entrypoint.is_file() && !is_placeholder_artifact(&entrypoint)
        })
    }

    fn materialize_placeholder(
        &self,
        model: &ModelDescriptor,
        source_root: &Path,
        expected_entrypoint: &'static str,
    ) -> Result<MaterializedPreparation, std::io::Error> {
        let prepared_root = self.cache_root.join(sanitize_name(&model.name));
        fs::create_dir_all(&prepared_root)?;

        let metadata_path = prepared_root.join("loci-prepared-artifact.json");
        let metadata = format!(
            concat!(
                "{{\n",
                "  \"backend\": \"openvino\",\n",
                "  \"model_name\": {:?},\n",
                "  \"source_root\": {:?},\n",
                "  \"prepared_root\": {:?},\n",
                "  \"expected_entrypoint\": {:?},\n",
                "  \"job_kind\": \"materialize_placeholder\",\n",
                "  \"job_status\": \"placeholder_materialized\",\n",
                "  \"toolchain_kind\": \"rust-native-placeholder\",\n",
                "  \"validated\": false,\n",
                "  \"artifact_files\": [{:?}]\n",
                "}}\n"
            ),
            model.name,
            source_root.display().to_string(),
            prepared_root.display().to_string(),
            expected_entrypoint,
            expected_entrypoint,
        );
        fs::write(&metadata_path, metadata)?;

        let placeholder_entrypoint = prepared_root.join(expected_entrypoint);
        if !placeholder_entrypoint.is_file() {
            let placeholder = format!(
                concat!(
                    "# loci backend-managed placeholder\n",
                    "backend=openvino\n",
                    "model_name={}\n",
                    "source_root={}\n",
                    "status=placeholder_only\n",
                    "note=real Intel executable artifact has not been materialized yet\n"
                ),
                model.name,
                source_root.display()
            );
            fs::write(&placeholder_entrypoint, placeholder)?;
        }

        Ok(MaterializedPreparation {
            prepared_root,
            metadata_path,
            expected_entrypoint,
            placeholder_entrypoint,
        })
    }

    fn validate_text_source_asset(
        &self,
        source_root: &Path,
    ) -> Result<TextSourceValidation, std::io::Error> {
        let config_path = source_root.join("config.json");
        let config_excerpt = fs::read_to_string(&config_path)
            .map(|content| content.chars().take(160).collect::<String>())?;
        let tokenizer_path = source_root.join("tokenizer.json");
        let index_path = source_root.join("model.safetensors.index.json");

        Ok(TextSourceValidation {
            config_path,
            tokenizer_path: tokenizer_path.is_file().then_some(tokenizer_path),
            index_path: index_path.is_file().then_some(index_path),
            config_excerpt: config_excerpt.replace('\n', " ").replace('"', "'"),
        })
    }

    fn attempt_text_materialization(
        &self,
        model: &ModelDescriptor,
        source_root: &Path,
        expected_entrypoint: &'static str,
    ) -> Result<PreparationJobOutcome, std::io::Error> {
        let prepared_root = self.cache_root.join(sanitize_name(&model.name));
        fs::create_dir_all(&prepared_root)?;

        let stdout_path = prepared_root.join("materialize-text.stdout.log");
        let stderr_path = prepared_root.join("materialize-text.stderr.log");
        let metadata_path = prepared_root.join("loci-prepared-artifact.json");
        let tool_invocation = self
            .toolchain
            .describe_text_materializer(source_root, &prepared_root);
        let validation_summary = self.toolchain.validation_summary();

        if matches!(
            self.toolchain.strategy,
            OpenVinoToolchainStrategy::FfiBridge
        ) {
            return self.invoke_text_materializer_bridge(
                model,
                source_root,
                &prepared_root,
                expected_entrypoint,
                metadata_path,
                stdout_path,
                stderr_path,
            );
        }

        if matches!(
            self.toolchain.strategy,
            OpenVinoToolchainStrategy::ExternalCli
        ) {
            return self.invoke_text_materializer_cli(
                model,
                source_root,
                &prepared_root,
                expected_entrypoint,
                metadata_path,
                stdout_path,
                stderr_path,
            );
        }

        fs::write(
            &stdout_path,
            format!(
                "planned text-only materialization for model `{}` from `{}` into `{}` with strategy `{:?}`\n",
                model.name,
                source_root.display(),
                prepared_root.display(),
                self.toolchain.strategy,
            ),
        )?;
        fs::write(
            &stderr_path,
            format!(
                "materializer hook not yet connected to a real Rust/C/C++ OpenVINO conversion tool; strategy={:?}; install_root={}; text_materializer={}\n",
                self.toolchain.strategy,
                self.toolchain
                    .install_root
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<unset>".to_string()),
                self.toolchain
                    .text_materializer
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<unset>".to_string())
            ),
        )?;

        let strategy_label = match self.toolchain.strategy {
            OpenVinoToolchainStrategy::Deferred => "deferred",
            OpenVinoToolchainStrategy::ExternalCli => "external_cli",
            OpenVinoToolchainStrategy::FfiBridge => "ffi_bridge",
        };

        fs::write(
            prepared_root.join("toolchain-config.json"),
            format!(
                concat!(
                    "{{\n",
                    "  \"strategy\": {:?},\n",
                    "  \"install_root\": {:?},\n",
                    "  \"text_materializer\": {:?},\n",
                    "  \"ffi_bridge\": {:?}\n",
                    "}}\n"
                ),
                strategy_label,
                self.toolchain
                    .install_root
                    .as_ref()
                    .map(|path| path.display().to_string()),
                self.toolchain
                    .text_materializer
                    .as_ref()
                    .map(|path| path.display().to_string()),
                self.toolchain
                    .ffi_library
                    .as_ref()
                    .map(|path| path.display().to_string())
            ),
        )?;

        let metadata = format!(
            concat!(
                "{{\n",
                "  \"backend\": \"openvino\",\n",
                "  \"model_name\": {:?},\n",
                "  \"source_root\": {:?},\n",
                "  \"prepared_root\": {:?},\n",
                "  \"expected_entrypoint\": {:?},\n",
                "  \"job_kind\": \"materialize_text_executable\",\n",
                "  \"job_status\": \"materialization_attempted\",\n",
                "  \"toolchain_kind\": {:?},\n",
                "  \"toolchain_install_root\": {:?},\n",
                "  \"toolchain_text_materializer\": {:?},\n",
                "  \"toolchain_ffi_bridge\": {:?},\n",
                "  \"validated\": false,\n",
                "  \"validation_summary\": {:?},\n",
                "  \"stdout_path\": {:?},\n",
                "  \"stderr_path\": {:?},\n",
                "  \"artifact_files\": []\n",
                "}}\n"
            ),
            model.name,
            source_root.display().to_string(),
            prepared_root.display().to_string(),
            expected_entrypoint,
            strategy_label,
            self.toolchain
                .install_root
                .as_ref()
                .map(|path| path.display().to_string()),
            self.toolchain
                .text_materializer
                .as_ref()
                .map(|path| path.display().to_string()),
            self.toolchain
                .ffi_library
                .as_ref()
                .map(|path| path.display().to_string()),
            validation_summary,
            stdout_path.display().to_string(),
            stderr_path.display().to_string(),
        );
        fs::write(&metadata_path, metadata)?;

        Ok(PreparationJobOutcome {
            status: PreparationJobStatus::MaterializationAttempted,
            model_root: Some(prepared_root),
            source_root: Some(source_root.to_path_buf()),
            metadata_path: Some(metadata_path),
            expected_entrypoint,
            tool_invocation: Some(tool_invocation),
            stdout_path: Some(stdout_path),
            stderr_path: Some(stderr_path),
            validation_summary: Some(validation_summary),
            detail: "text-only real materialization hook reached planning stage, but no executable OpenVINO conversion backend is connected yet".to_string(),
        })
    }

    fn invoke_text_materializer_cli(
        &self,
        model: &ModelDescriptor,
        source_root: &Path,
        prepared_root: &Path,
        expected_entrypoint: &'static str,
        metadata_path: PathBuf,
        stdout_path: PathBuf,
        stderr_path: PathBuf,
    ) -> Result<PreparationJobOutcome, std::io::Error> {
        let Some(materializer) = self.toolchain.text_materializer.as_ref() else {
            fs::write(
                &stdout_path,
                "external_cli strategy selected, but LOCI_OPENVINO_TEXT_MATERIALIZER is unset\n",
            )?;
            fs::write(
                &stderr_path,
                "missing external materializer path for OpenVINO text conversion\n",
            )?;
            return Ok(PreparationJobOutcome {
                status: PreparationJobStatus::MaterializationAttempted,
                model_root: Some(prepared_root.to_path_buf()),
                source_root: Some(source_root.to_path_buf()),
                metadata_path: Some(metadata_path),
                expected_entrypoint,
                tool_invocation: Some(
                    "external_cli::<missing LOCI_OPENVINO_TEXT_MATERIALIZER>".to_string(),
                ),
                stdout_path: Some(stdout_path),
                stderr_path: Some(stderr_path),
                validation_summary: Some(self.toolchain.validation_summary()),
                detail: "external_cli strategy selected, but no text materializer executable was configured".to_string(),
            });
        };

        let output = Command::new(materializer)
            .arg("--source")
            .arg(source_root)
            .arg("--output")
            .arg(prepared_root)
            .arg("--model-name")
            .arg(&model.name)
            .arg("--architecture")
            .arg(&model.architecture)
            .output();

        match output {
            Ok(output) => {
                fs::write(&stdout_path, &output.stdout)?;
                fs::write(&stderr_path, &output.stderr)?;

                let produced_entrypoint = prepared_root.join(expected_entrypoint);
                let status = if output.status.success() && produced_entrypoint.is_file() {
                    PreparationJobStatus::Ready
                } else {
                    PreparationJobStatus::MaterializationAttempted
                };

                let metadata = format!(
                    concat!(
                        "{{\n",
                        "  \"backend\": \"openvino\",\n",
                        "  \"model_name\": {:?},\n",
                        "  \"source_root\": {:?},\n",
                        "  \"prepared_root\": {:?},\n",
                        "  \"expected_entrypoint\": {:?},\n",
                        "  \"job_kind\": \"materialize_text_executable\",\n",
                        "  \"job_status\": {:?},\n",
                        "  \"toolchain_kind\": \"external_cli\",\n",
                        "  \"toolchain_text_materializer\": {:?},\n",
                        "  \"exit_code\": {:?},\n",
                        "  \"validated\": true,\n",
                        "  \"stdout_path\": {:?},\n",
                        "  \"stderr_path\": {:?},\n",
                        "  \"artifact_files\": [{:?}]\n",
                        "}}\n"
                    ),
                    model.name,
                    source_root.display().to_string(),
                    prepared_root.display().to_string(),
                    expected_entrypoint,
                    match status {
                        PreparationJobStatus::Ready => "ready",
                        _ => "materialization_attempted",
                    },
                    materializer.display().to_string(),
                    output.status.code(),
                    stdout_path.display().to_string(),
                    stderr_path.display().to_string(),
                    expected_entrypoint,
                );
                fs::write(&metadata_path, metadata)?;

                Ok(PreparationJobOutcome {
                    status,
                    model_root: Some(prepared_root.to_path_buf()),
                    source_root: Some(source_root.to_path_buf()),
                    metadata_path: Some(metadata_path),
                    expected_entrypoint,
                    tool_invocation: Some(format!(
                        "{} --source {} --output {} --model-name {} --architecture {}",
                        materializer.display(),
                        source_root.display(),
                        prepared_root.display(),
                        model.name,
                        model.architecture
                    )),
                    stdout_path: Some(stdout_path),
                    stderr_path: Some(stderr_path),
                    validation_summary: Some(self.toolchain.validation_summary()),
                    detail: if produced_entrypoint.is_file() {
                        "external text materializer produced an executable OpenVINO artifact"
                            .to_string()
                    } else {
                        "external text materializer ran, but did not produce the expected OpenVINO artifact"
                            .to_string()
                    },
                })
            }
            Err(error) => {
                fs::write(
                    &stdout_path,
                    format!(
                        "failed to launch external materializer `{}`\n",
                        materializer.display()
                    ),
                )?;
                fs::write(&stderr_path, format!("{error}\n"))?;

                Ok(PreparationJobOutcome {
                    status: PreparationJobStatus::MaterializationAttempted,
                    model_root: Some(prepared_root.to_path_buf()),
                    source_root: Some(source_root.to_path_buf()),
                    metadata_path: Some(metadata_path),
                    expected_entrypoint,
                    tool_invocation: Some(format!(
                        "{} --source {} --output {}",
                        materializer.display(),
                        source_root.display(),
                        prepared_root.display()
                    )),
                    stdout_path: Some(stdout_path),
                    stderr_path: Some(stderr_path),
                    validation_summary: Some(self.toolchain.validation_summary()),
                    detail: format!("external text materializer could not be launched: {error}"),
                })
            }
        }
    }

    fn invoke_text_materializer_bridge(
        &self,
        model: &ModelDescriptor,
        source_root: &Path,
        prepared_root: &Path,
        expected_entrypoint: &'static str,
        metadata_path: PathBuf,
        stdout_path: PathBuf,
        stderr_path: PathBuf,
    ) -> Result<PreparationJobOutcome, std::io::Error> {
        let bridge_path = self.toolchain.bridge_library_path();
        let Some(bridge_path) = bridge_path else {
            fs::write(
                &stdout_path,
                "ffi bridge strategy selected, but no bridge library path could be resolved\n",
            )?;
            fs::write(&stderr_path, "failed to resolve bridge library path\n")?;
            return Ok(PreparationJobOutcome {
                status: PreparationJobStatus::MaterializationAttempted,
                model_root: Some(prepared_root.to_path_buf()),
                source_root: Some(source_root.to_path_buf()),
                metadata_path: Some(metadata_path),
                expected_entrypoint,
                tool_invocation: Some("ffi_bridge::<unresolved>".to_string()),
                stdout_path: Some(stdout_path),
                stderr_path: Some(stderr_path),
                validation_summary: Some(self.toolchain.validation_summary()),
                detail: "ffi bridge strategy selected, but bridge library path is unresolved"
                    .to_string(),
            });
        };

        let source_root_c = CString::new(source_root.display().to_string()).map_err(io_other)?;
        let prepared_root_c =
            CString::new(prepared_root.display().to_string()).map_err(io_other)?;
        let model_name_c = CString::new(model.name.clone()).map_err(io_other)?;
        let architecture_c = CString::new(model.architecture.clone()).map_err(io_other)?;
        let config_json_c = CString::new(source_root.join("config.json").display().to_string())
            .map_err(io_other)?;
        let tokenizer_json_c =
            CString::new(source_root.join("tokenizer.json").display().to_string())
                .map_err(io_other)?;
        let safetensors_index_c = CString::new(
            source_root
                .join("model.safetensors.index.json")
                .display()
                .to_string(),
        )
        .map_err(io_other)?;
        let options_json_c = CString::new("{}".to_string()).map_err(io_other)?;

        let request = BridgeTextMaterializeRequest {
            source_root: source_root_c.as_ptr(),
            prepared_root: prepared_root_c.as_ptr(),
            model_name: model_name_c.as_ptr(),
            architecture: architecture_c.as_ptr(),
            config_json_path: config_json_c.as_ptr(),
            tokenizer_json_path: tokenizer_json_c.as_ptr(),
            safetensors_index_path: safetensors_index_c.as_ptr(),
            options_json: options_json_c.as_ptr(),
        };

        let mut bridge_stdout = String::new();
        let mut bridge_stderr = String::new();

        let bridge_result: Result<
            (
                i32,
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
            ),
            std::io::Error,
        > = unsafe {
            let library = Library::new(&bridge_path).map_err(io_other)?;
            let version_fn: libloading::Symbol<'_, BridgeVersionFn> =
                library.get(b"loci_ov_bridge_version\0").map_err(io_other)?;
            let materialize_fn: libloading::Symbol<'_, BridgeMaterializeFn> = library
                .get(b"loci_ov_materialize_text_artifact\0")
                .map_err(io_other)?;
            let free_fn: libloading::Symbol<'_, BridgeFreeStringFn> = library
                .get(b"loci_ov_bridge_free_string\0")
                .map_err(io_other)?;

            let version_ptr = version_fn();
            if !version_ptr.is_null() {
                bridge_stdout.push_str("bridge_version=");
                bridge_stdout.push_str(&c_ptr_to_string(version_ptr));
                bridge_stdout.push('\n');
            }

            let response = materialize_fn(&request as *const _);
            let artifact_root = c_ptr_to_owned_string(response.artifact_root, &free_fn);
            let entrypoint_path = c_ptr_to_owned_string(response.entrypoint_path, &free_fn);
            let metadata_json = c_ptr_to_owned_string(response.metadata_json, &free_fn);
            let error_message = c_ptr_to_owned_string(response.error_message, &free_fn);

            Ok((
                response.status,
                artifact_root,
                entrypoint_path,
                metadata_json,
                error_message,
            ))
        };

        match bridge_result {
            Ok((status, artifact_root, entrypoint_path, metadata_json, error_message)) => {
                bridge_stdout.push_str(&format!("bridge_status={status}\n"));
                if let Some(root) = artifact_root {
                    bridge_stdout.push_str(&format!("artifact_root={root}\n"));
                }
                if let Some(entrypoint) = entrypoint_path {
                    bridge_stdout.push_str(&format!("entrypoint_path={entrypoint}\n"));
                }
                if let Some(metadata) = metadata_json {
                    bridge_stdout.push_str(&format!("bridge_metadata={metadata}\n"));
                }
                if let Some(error) = error_message {
                    bridge_stderr.push_str(&error);
                    bridge_stderr.push('\n');
                }
            }
            Err(error) => {
                bridge_stderr.push_str(&format!("bridge_load_or_call_error={error}\n"));
            }
        }

        fs::write(
            &stdout_path,
            format!(
                "attempting ffi bridge text materialization for model `{}` with library `{}`\n{}",
                model.name,
                bridge_path.display(),
                bridge_stdout
            ),
        )?;

        fs::write(
            &stderr_path,
            if bridge_stderr.is_empty() {
                "native ffi bridge invocation completed without bridge stderr output\n".to_string()
            } else {
                bridge_stderr.clone()
            },
        )?;

        let metadata = format!(
            concat!(
                "{{\n",
                "  \"backend\": \"openvino\",\n",
                "  \"model_name\": {:?},\n",
                "  \"source_root\": {:?},\n",
                "  \"prepared_root\": {:?},\n",
                "  \"expected_entrypoint\": {:?},\n",
                "  \"job_kind\": \"materialize_text_executable\",\n",
                "  \"job_status\": \"materialization_attempted\",\n",
                "  \"toolchain_kind\": \"ffi_bridge\",\n",
                "  \"toolchain_ffi_bridge\": {:?},\n",
                "  \"validated\": false,\n",
                "  \"stdout_path\": {:?},\n",
                "  \"stderr_path\": {:?},\n",
                "  \"artifact_files\": []\n",
                "}}\n"
            ),
            model.name,
            source_root.display().to_string(),
            prepared_root.display().to_string(),
            expected_entrypoint,
            bridge_path.display().to_string(),
            stdout_path.display().to_string(),
            stderr_path.display().to_string(),
        );
        fs::write(&metadata_path, metadata)?;

        Ok(PreparationJobOutcome {
            status: PreparationJobStatus::MaterializationAttempted,
            model_root: Some(prepared_root.to_path_buf()),
            source_root: Some(source_root.to_path_buf()),
            metadata_path: Some(metadata_path),
            expected_entrypoint,
            tool_invocation: Some(format!("ffi_bridge::{}", bridge_path.display())),
            stdout_path: Some(stdout_path),
            stderr_path: Some(stderr_path),
            validation_summary: Some(self.toolchain.validation_summary()),
            detail: "ffi bridge strategy selected and runtime bridge path resolved; bridge invocation attempted, but no real executable OpenVINO artifact has been produced yet".to_string(),
        })
    }
}

fn c_ptr_to_string(ptr: *const c_char) -> String {
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn c_ptr_to_owned_string(ptr: *mut c_char, free_fn: &BridgeFreeStringFn) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let value = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { free_fn(ptr) };
    Some(value)
}

fn io_other<E: std::fmt::Display>(error: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, error.to_string())
}

impl OpenVinoToolchainConfig {
    fn from_environment() -> Self {
        let install_root = env::var_os("LOCI_OPENVINO_ROOT").map(PathBuf::from);
        let text_materializer = env::var_os("LOCI_OPENVINO_TEXT_MATERIALIZER").map(PathBuf::from);
        let ffi_library = env::var_os("LOCI_OPENVINO_FFI_BRIDGE").map(PathBuf::from);
        let strategy = match env::var("LOCI_OPENVINO_TOOLCHAIN") {
            Ok(value) if value.eq_ignore_ascii_case("external_cli") => {
                OpenVinoToolchainStrategy::ExternalCli
            }
            Ok(value) if value.eq_ignore_ascii_case("ffi_bridge") => {
                OpenVinoToolchainStrategy::FfiBridge
            }
            _ => OpenVinoToolchainStrategy::Deferred,
        };

        Self {
            strategy,
            install_root,
            text_materializer,
            ffi_library,
        }
    }

    fn describe_text_materializer(&self, source_root: &Path, prepared_root: &Path) -> String {
        match &self.text_materializer {
            Some(path) => format!(
                "{} --source {} --output {}",
                path.display(),
                source_root.display(),
                prepared_root.display()
            ),
            None => format!(
                "openvino-text-materializer --source {} --output {}",
                source_root.display(),
                prepared_root.display()
            ),
        }
    }

    fn validation_summary(&self) -> String {
        format!(
            "text-only source asset accepted; strategy={:?}; install_root={}; text_materializer={}; ffi_bridge={}",
            self.strategy,
            self.install_root
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unset>".to_string()),
            self.text_materializer
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unset>".to_string()),
            self.ffi_library
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unset>".to_string())
        )
    }

    fn bridge_library_path(&self) -> Option<PathBuf> {
        self.ffi_library
            .clone()
            .or_else(default_bridge_library_path)
    }
}

fn default_bridge_library_path() -> Option<PathBuf> {
    let target_dir = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let file_name = if cfg!(target_os = "windows") {
        "loci_backend_openvino_bridge.dll"
    } else if cfg!(target_os = "macos") {
        "libloci_backend_openvino_bridge.dylib"
    } else {
        "libloci_backend_openvino_bridge.so"
    };
    let candidate = target_dir.join(profile).join(file_name);
    candidate.exists().then_some(candidate)
}

static RUNTIME_BOOTSTRAP: OnceLock<Option<RuntimeBootstrap>> = OnceLock::new();

impl Backend for OpenVinoBackend {
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "openvino".to_string(),
            runtime_family: BackendRuntimeFamily::OpenVino,
            supports_cpu: true,
            supports_gpu: true,
            supports_npu: true,
            supports_disk_tiering: true,
            supports_paged_kv: true,
            supports_multimodal: true,
        }
    }

    fn asset_capabilities(&self) -> BackendAssetCapabilities {
        BackendAssetCapabilities {
            backend: "openvino".to_string(),
            runtime_family: BackendRuntimeFamily::OpenVino,
            directly_supported_layouts: vec![
                ModelAssetLayout::OpenVinoGenAiExport,
                ModelAssetLayout::OpenVinoIr,
                ModelAssetLayout::OpenVinoBlob,
            ],
            ingestible_layouts: vec![
                ModelAssetLayout::OnnxModel,
                ModelAssetLayout::GgufFile,
                ModelAssetLayout::GgufDirectory,
                ModelAssetLayout::SafeTensorsFile,
                ModelAssetLayout::SafeTensorsDirectory,
                ModelAssetLayout::PytorchBinFile,
                ModelAssetLayout::PytorchCheckpointDirectory,
                ModelAssetLayout::TransformersCheckpoint,
                ModelAssetLayout::UnknownDirectory,
                ModelAssetLayout::UnknownFile,
            ],
            preferred_artifact: ExecutionArtifactKind::OpenVinoIr,
            requires_lowering_for_execution: true,
            notes: vec![
                "OpenVINO executes exported IR or GenAI layouts directly".to_string(),
                "raw checkpoints and foreign graph formats must be lowered or converted before real execution".to_string(),
            ],
        }
    }

    fn lowering_capabilities(&self) -> BackendLoweringCapabilities {
        BackendLoweringCapabilities {
            backend: "openvino".to_string(),
            runtime_family: BackendRuntimeFamily::OpenVino,
            granularity: LoweringGranularity::Subgraph,
            supports_real_execution: true,
            supports_graph_partitioning: true,
            supports_layer_affinity: false,
            supports_dynamic_reoffload: true,
            supports_custom_operators: false,
            operator_classes: vec![
                ChipOperatorClass::Attention,
                ChipOperatorClass::Matmul,
                ChipOperatorClass::Embedding,
                ChipOperatorClass::RmsNorm,
                ChipOperatorClass::VisionEncoder,
                ChipOperatorClass::Mlp,
                ChipOperatorClass::KvCache,
                ChipOperatorClass::Sampling,
            ],
            notes: vec![
                "real OpenVINO and OpenVINO GenAI execution are available".to_string(),
                "the current Loci integration plans placements at pipeline-stage granularity and does not yet lower them into explicit per-layer affinities".to_string(),
                "this backend is the primary Intel CPU/GPU/NPU integration path".to_string(),
            ],
        }
    }

    fn kernel_catalog(&self) -> BackendKernelCatalog {
        BackendKernelCatalog {
            backend: "openvino".to_string(),
            runtime_family: BackendRuntimeFamily::OpenVino,
            kernels: vec![
                KernelDescriptor {
                    backend: "openvino".to_string(),
                    kernel_name: "openvino_hetero_attention_partition".to_string(),
                    operator_class: ChipOperatorClass::Attention,
                    implementation: KernelImplementationKind::VendorRuntime,
                    maturity: KernelMaturity::Integrated,
                    origin: KernelOrigin {
                        project: "OpenVINO GenAI".to_string(),
                        component: "heterogeneous text generation".to_string(),
                        license: Some("Apache-2.0".to_string()),
                        notes: vec![
                            "current execution path relies on vendor graph/runtime partitioning"
                                .to_string(),
                        ],
                    },
                    supported_targets: vec![
                        AcceleratorKind::Cpu,
                        AcceleratorKind::Gpu,
                        AcceleratorKind::Npu,
                    ],
                    supported_formats: vec![
                        ModelFormat::OpenVinoIr,
                        ModelFormat::OpenVinoBlob,
                        ModelFormat::Directory,
                    ],
                    supported_architectures: vec![
                        "llama".to_string(),
                        "qwen".to_string(),
                        "phi".to_string(),
                        "vlm".to_string(),
                    ],
                    dispatch_keys: vec!["hetero".to_string(), "decode".to_string()],
                    notes: vec![
                        "operator availability is mediated by OpenVINO IR lowering, not a native Rust kernel"
                            .to_string(),
                    ],
                },
                KernelDescriptor {
                    backend: "openvino".to_string(),
                    kernel_name: "openvino_vlm_vision_encoder".to_string(),
                    operator_class: ChipOperatorClass::VisionEncoder,
                    implementation: KernelImplementationKind::IrGraph,
                    maturity: KernelMaturity::Integrated,
                    origin: KernelOrigin {
                        project: "OpenVINO GenAI".to_string(),
                        component: "VLM pipeline".to_string(),
                        license: Some("Apache-2.0".to_string()),
                        notes: vec![
                            "used for multimodal pipelines that already exist as executable IR assets"
                                .to_string(),
                        ],
                    },
                    supported_targets: vec![AcceleratorKind::Cpu, AcceleratorKind::Gpu],
                    supported_formats: vec![ModelFormat::OpenVinoIr, ModelFormat::Directory],
                    supported_architectures: vec![
                        "vlm".to_string(),
                        "qwen2-vl".to_string(),
                        "minicpm-v".to_string(),
                    ],
                    dispatch_keys: vec!["multimodal".to_string(), "prefill".to_string()],
                    notes: vec![
                        "depends on executable OpenVINO-exported vision-language assets".to_string(),
                    ],
                },
                KernelDescriptor {
                    backend: "openvino".to_string(),
                    kernel_name: "openvino_bridge_text_materializer".to_string(),
                    operator_class: ChipOperatorClass::Matmul,
                    implementation: KernelImplementationKind::ExternalBridge,
                    maturity: KernelMaturity::Stubbed,
                    origin: KernelOrigin {
                        project: "Loci backend-openvino-bridge".to_string(),
                        component: "text materialization bridge".to_string(),
                        license: Some("MIT".to_string()),
                        notes: vec![
                            "backend-managed preparation path for raw checkpoints".to_string(),
                        ],
                    },
                    supported_targets: vec![AcceleratorKind::Cpu, AcceleratorKind::Npu],
                    supported_formats: vec![ModelFormat::SafeTensors, ModelFormat::Directory],
                    supported_architectures: vec![
                        "llama".to_string(),
                        "mistral".to_string(),
                        "qwen".to_string(),
                    ],
                    dispatch_keys: vec!["materialize".to_string(), "export".to_string()],
                    notes: vec![
                        "describes the preparation bridge rather than a final runtime kernel".to_string(),
                    ],
                },
            ],
            notes: vec![
                "OpenVINO catalogs mostly describe vendor-runtime or IR-mediated kernels rather than native portable kernels".to_string(),
            ],
        }
    }

    fn discover_topology(&self) -> HardwareTopology {
        self.runtime.discover_topology()
    }

    fn supports_model(&self, model: &ModelDescriptor) -> bool {
        matches!(
            model.inferred_format(),
            ModelFormat::OpenVinoIr
                | ModelFormat::OpenVinoBlob
                | ModelFormat::Onnx
                | ModelFormat::Gguf
                | ModelFormat::Directory
        )
    }

    fn prepare(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
    ) -> BackendResult<PreparedModel> {
        let profile = openvino_profile(plan)?;
        self.runtime.compile_session(model, plan, profile)
    }

    fn execute(
        &self,
        prepared: &PreparedModel,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
    ) -> BackendResult<BackendOutput> {
        if request.prompt.trim().is_empty() {
            return Err(BackendError {
                message: "prompt must not be empty".to_string(),
            });
        }

        let profile = openvino_profile(plan)?;
        if prepared.session_key != profile.session_key {
            return Err(BackendError {
                message: format!(
                    "prepared OpenVINO session `{}` does not match plan `{}`",
                    prepared.session_key, profile.session_key
                ),
            });
        }

        self.runtime
            .run_session(prepared, model, request, plan, profile)
    }
}

impl OpenVinoRuntime {
    fn discover_topology(&self) -> HardwareTopology {
        discover_runtime_topology().unwrap_or_else(|_| synthetic_topology())
    }

    fn compile_session(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
        profile: &OpenVinoExecutionProfile,
    ) -> BackendResult<PreparedModel> {
        validate_openvino_plan(plan, profile)?;
        if profile.hetero_devices.is_empty() {
            return Err(BackendError {
                message: "OpenVINO execution profile must include at least one device".to_string(),
            });
        }

        let residency = derive_residency(plan);
        let estimated_memory_bytes = estimate_resident_memory_bytes(model, plan, residency);
        let session = self.build_session(model, plan, profile);
        self.sessions()?
            .insert(profile.session_key.clone(), session);

        Ok(PreparedModel {
            model_name: model.name.clone(),
            backend: "openvino".to_string(),
            session_key: profile.session_key.clone(),
            residency,
            estimated_memory_bytes,
        })
    }

    fn run_session(
        &self,
        prepared: &PreparedModel,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
        profile: &OpenVinoExecutionProfile,
    ) -> BackendResult<BackendOutput> {
        let mut sessions = self.sessions()?;
        let session = sessions
            .get_mut(&prepared.session_key)
            .ok_or_else(|| BackendError {
                message: format!(
                    "OpenVINO session `{}` was not prepared or was evicted",
                    prepared.session_key
                ),
            })?;

        match session {
            SessionSlot::Real(session) => {
                self.run_real_session(session, model, request, plan, profile)
            }
            SessionSlot::Fallback(session) => fallback_behavior(session, model, request, plan, profile),
        }
    }

    fn build_session(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
        profile: &OpenVinoExecutionProfile,
    ) -> SessionSlot {
        let device_name = runtime_device_name(profile);
        let preparation = PreparedArtifactResolver::new().inspect(model);
        let model_root = match &preparation {
            ModelPreparationState::ReadyOpenVinoArtifact { model_root, .. } => {
                Some(model_root.clone())
            }
            ModelPreparationState::MaterializationPlanned {
                prepared_root,
                source_root,
                expected_entrypoint,
                metadata_path,
                detail,
            } => {
                return SessionSlot::Fallback(FallbackSession {
                    reason: format!(
                        "text-only materialization planned for `{}`: source asset root `{}` is mapped to prepared root `{}` with metadata `{}`, but Intel execution still requires a real artifact `{}` ({})",
                        model.name,
                        source_root.display(),
                        prepared_root.display(),
                        metadata_path.display(),
                        expected_entrypoint,
                        detail
                    ),
                    device_name,
                    model_root: Some(source_root.clone()),
                    lowering_diagnostics: None,
                });
            }
            ModelPreparationState::MaterializedPlaceholder {
                source_root,
                expected_entrypoint,
                metadata_path,
                detail,
                ..
            } => {
                return SessionSlot::Fallback(FallbackSession {
                    reason: format!(
                        "backend-managed preparation placeholder created for `{}`: source asset root `{}` is now associated with metadata `{}` but Intel execution still requires a materialized artifact `{}` ({})",
                        model.name,
                        source_root.display(),
                        metadata_path.display(),
                        expected_entrypoint,
                        detail
                    ),
                    device_name,
                    model_root: Some(source_root.clone()),
                    lowering_diagnostics: None,
                });
            }
            ModelPreparationState::RequiresBackendAdaptation {
                source_root,
                expected_entrypoint,
                detail,
            } => {
                return SessionSlot::Fallback(FallbackSession {
                    reason: format!(
                        "backend-local adaptation not yet implemented for `{}`: source asset root `{}` contains a raw Transformers checkpoint and still needs preparation before Intel execution can consume it (expected prepared artifact `{}`; detail: {})",
                        model.name,
                        source_root.display(),
                        expected_entrypoint,
                        detail
                    ),
                    device_name,
                    model_root: Some(source_root.clone()),
                    lowering_diagnostics: None,
                });
            }
            ModelPreparationState::Unsupported { detail } => {
                return SessionSlot::Fallback(FallbackSession {
                    reason: detail.clone(),
                    device_name,
                    model_root: None,
                    lowering_diagnostics: None,
                });
            }
        };
        let Some(model_root) = model_root else {
            return SessionSlot::Fallback(FallbackSession {
                reason: format!(
                    "model path `{}` does not resolve to a usable OpenVINO GenAI directory",
                    model.path.display()
                ),
                device_name,
                model_root: None,
                lowering_diagnostics: None,
            });
        };

        let lowering_diagnostics = shadow_lowering_compile(model, &model_root, plan, profile).err();

        match create_pipeline(model, &model_root, &device_name, plan, profile) {
            Ok(pipeline) => SessionSlot::Real(RealSession {
                pipeline,
                lowering_diagnostics,
            }),
            Err(reason) => SessionSlot::Fallback(FallbackSession {
                reason,
                device_name,
                model_root: Some(model_root),
                lowering_diagnostics,
            }),
        }
    }

    fn run_real_session(
        &self,
        session: &mut RealSession,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
        profile: &OpenVinoExecutionProfile,
    ) -> BackendResult<BackendOutput> {
        let config = build_generation_config(request)?;
        let _lowering_diagnostics = session.lowering_diagnostics.as_deref();
        match &mut session.pipeline {
            RealPipeline::Llm(pipeline) => {
                if !request.images.is_empty() {
                    return Err(BackendError {
                        message: "OpenVINO text pipeline cannot accept image inputs; use a model with architecture `vlm`, `vision`, or `multimodal`".to_string(),
                    });
                }

                let results = pipeline
                    .generate(request.prompt.trim(), Some(&config), None)
                    .map_err(inference_error)?;
                let text = results.get_string().map_err(inference_error)?;
                let telemetry = telemetry_from_llm_results(&results, request, profile, model, plan);

                Ok(BackendOutput { text, telemetry })
            }
            RealPipeline::Onnx(session) => {
                if !request.images.is_empty() {
                    return Err(BackendError {
                        message: "OpenVINO ONNX text pipeline cannot accept image inputs".to_string(),
                    });
                }
                run_onnx_session(session, model, request, plan, profile)
            }
            RealPipeline::Vlm(pipeline) => {
                let image_tensors = load_image_tensors(&request.images)?;
                let image_ptrs: Vec<_> = image_tensors.iter().map(Tensor::as_c_ptr).collect();
                let results = pipeline
                    .generate(request.prompt.trim(), &image_ptrs, Some(&config), None)
                    .map_err(inference_error)?;
                let text = results.get_string().map_err(inference_error)?;
                let telemetry = telemetry_from_vlm_results(&results, request, profile, model, plan);

                Ok(BackendOutput { text, telemetry })
            }
        }
    }

    fn sessions(&self) -> BackendResult<std::sync::MutexGuard<'_, HashMap<String, SessionSlot>>> {
        self.sessions.lock().map_err(|_| BackendError {
            message: "OpenVINO session cache is poisoned".to_string(),
        })
    }
}

fn create_pipeline(
    model: &ModelDescriptor,
    model_root: &Path,
    device_name: &str,
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> Result<RealPipeline, String> {
    let _ = ensure_runtime_bootstrap();
    validate_model_root_layout(model, model_root)?;

    let properties = runtime_properties(plan, profile);
    let property_refs: Vec<(&str, &str)> = properties
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();

    if model_root.is_file() {
        if model.inferred_format() == ModelFormat::Onnx {
            return create_onnx_pipeline(model_root, device_name).map(RealPipeline::Onnx);
        }
        return LlmPipeline::with_properties(
            &model_root.to_string_lossy(),
            device_name,
            &property_refs,
        )
        .map(RealPipeline::Llm)
        .map_err(setup_error);
    }

    if model.is_multimodal_architecture() {
        VlmPipeline::with_properties(&model_root.to_string_lossy(), device_name, &property_refs)
            .map(RealPipeline::Vlm)
            .map_err(setup_error)
    } else {
        LlmPipeline::with_properties(&model_root.to_string_lossy(), device_name, &property_refs)
            .map(RealPipeline::Llm)
            .map_err(setup_error)
    }
}

fn create_onnx_pipeline(model_root: &Path, _device_name: &str) -> Result<OnnxSession, String> {
    if !model_root.is_file() {
        return Err(format!(
            "onnx model root `{}` is not a file",
            model_root.display()
        ));
    }
    Err(format!(
        "OpenVINO ONNX text execution is not implemented for `{}`: the runtime accepts ONNX as a direct asset layout, but Loci does not yet have a real tokenizer + decode loop on this path",
        model_root.display()
    ))
}

fn run_onnx_session(
    session: &mut OnnxSession,
    model: &ModelDescriptor,
    _request: &SessionRequest,
    _plan: &ExecutionPlan,
    _profile: &OpenVinoExecutionProfile,
) -> BackendResult<BackendOutput> {
    let _ = &session.compiled_model;
    Err(BackendError {
        message: format!(
            "OpenVINO ONNX execution for model `{}` is not implemented: the direct runtime lane exists, but Loci does not yet provide a real tokenizer + autoregressive decode loop for this backend",
            model.name
        ),
    })
}

fn validate_model_root_layout(model: &ModelDescriptor, model_root: &Path) -> Result<(), String> {
    if model_root.is_file() {
        return match model.inferred_format() {
            ModelFormat::Gguf | ModelFormat::Onnx => Ok(()),
            ModelFormat::OpenVinoIr | ModelFormat::OpenVinoBlob => Ok(()),
            ModelFormat::SafeTensors | ModelFormat::PytorchBin | ModelFormat::Unknown | ModelFormat::Directory => Err(format!(
                "model file `{}` is not directly executable by the OpenVINO text pipeline; expected GGUF, ONNX, or OpenVINO IR/Blob assets",
                model_root.display()
            )),
        };
    }

    let mut inspected_model = model.clone();
    inspected_model.path = model_root.to_path_buf();

    match PreparedArtifactResolver::new().inspect(&inspected_model) {
        ModelPreparationState::ReadyOpenVinoArtifact { model_root: prepared, .. } => {
            if prepared == model_root {
                Ok(())
            } else {
                Err(format!(
                    "prepared OpenVINO artifact root `{}` does not match requested root `{}`",
                    prepared.display(),
                    model_root.display()
                ))
            }
        }
        ModelPreparationState::MaterializedPlaceholder {
            source_root,
            expected_entrypoint,
            metadata_path,
            detail,
            ..
        } => Err(format!(
            "model directory `{}` contains a raw Transformers checkpoint and has backend-managed preparation metadata at `{}`, but Intel execution still requires a materialized artifact `{}` ({})",
            source_root.display(),
            metadata_path.display(),
            expected_entrypoint,
            detail
        )),
        ModelPreparationState::MaterializationPlanned {
            prepared_root,
            source_root,
            expected_entrypoint,
            metadata_path,
            detail,
        } => Err(format!(
            "model directory `{}` contains a raw Transformers checkpoint and has a planned backend-managed prepared root `{}` with metadata `{}`, but Intel execution still requires a real artifact `{}` ({})",
            source_root.display(),
            prepared_root.display(),
            metadata_path.display(),
            expected_entrypoint,
            detail
        )),
        ModelPreparationState::RequiresBackendAdaptation {
            source_root,
            expected_entrypoint,
            detail,
        } => Err(format!(
            "model directory `{}` contains a raw Transformers checkpoint and requires backend-local adaptation before Intel execution can consume it; expected prepared artifact `{}` ({})",
            source_root.display(),
            expected_entrypoint,
            detail
        )),
        ModelPreparationState::Unsupported { detail } => Err(detail),
    }
}

fn expected_openvino_entrypoint(model: &ModelDescriptor) -> &'static str {
    if model.is_multimodal_architecture() {
        "openvino_language_model.xml"
    } else {
        "openvino_model.xml"
    }
}

fn is_raw_transformers_checkpoint(model_root: &Path) -> bool {
    model_root.join("config.json").is_file()
        || model_root.join("model.safetensors.index.json").is_file()
        || model_root.join("pytorch_model.bin.index.json").is_file()
        || model_root
            .read_dir()
            .ok()
            .map(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .path()
                        .extension()
                        .and_then(|value| value.to_str())
                        .map(|value| {
                            value.eq_ignore_ascii_case("safetensors")
                                || value.eq_ignore_ascii_case("bin")
                        })
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false)
}

fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn is_placeholder_artifact(path: &Path) -> bool {
    fs::read_to_string(path)
        .map(|content| content.contains("status=placeholder_only"))
        .unwrap_or(false)
}

fn build_generation_config(request: &SessionRequest) -> BackendResult<GenerationConfig> {
    let mut config = GenerationConfig::new().map_err(inference_error)?;
    config
        .set_max_new_tokens(request.max_tokens as usize)
        .map_err(inference_error)?;
    config
        .set_temperature(request.temperature.max(0.0))
        .map_err(inference_error)?;
    config
        .set_do_sample(request.temperature > 0.0)
        .map_err(inference_error)?;
    config.set_num_beams(1).map_err(inference_error)?;
    config.validate().map_err(inference_error)?;
    Ok(config)
}

fn telemetry_from_llm_results(
    results: &DecodedResults,
    request: &SessionRequest,
    profile: &OpenVinoExecutionProfile,
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
) -> BackendTelemetry {
    let metrics = results.get_perf_metrics().map_err(inference_error);
    telemetry_from_metrics(metrics, request, profile, model, plan)
}

fn telemetry_from_vlm_results(
    results: &VlmDecodedResults,
    request: &SessionRequest,
    profile: &OpenVinoExecutionProfile,
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
) -> BackendTelemetry {
    let metrics = results.get_perf_metrics().map_err(inference_error);
    telemetry_from_metrics(metrics, request, profile, model, plan)
}

fn telemetry_from_metrics(
    metrics: Result<openvino_genai::PerfMetrics, BackendError>,
    request: &SessionRequest,
    profile: &OpenVinoExecutionProfile,
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
) -> BackendTelemetry {
    let fallback = BackendTelemetry {
        estimated_prefill_ms: estimate_prefill_ms(profile, model, plan),
        estimated_decode_ms: estimate_decode_ms(profile, plan),
        generated_tokens: request.max_tokens.min(128),
    };

    let Ok(metrics) = metrics else {
        return fallback;
    };

    let generated_tokens = metrics
        .get_num_generation_tokens()
        .ok()
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(fallback.generated_tokens);
    let estimated_prefill_ms = metrics
        .get_ttft()
        .ok()
        .map(|(mean, _)| f32_ms_to_u64(mean))
        .unwrap_or(fallback.estimated_prefill_ms);
    let estimated_decode_ms = metrics
        .get_tpot()
        .ok()
        .map(|(mean, _)| f32_ms_to_u64(mean))
        .unwrap_or(fallback.estimated_decode_ms);

    BackendTelemetry {
        estimated_prefill_ms,
        estimated_decode_ms,
        generated_tokens,
    }
}

fn run_fallback_session(
    session: &FallbackSession,
    model: &ModelDescriptor,
    request: &SessionRequest,
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> BackendOutput {
    let mode = match profile.execution_mode {
        OpenVinoExecutionMode::Hetero => "hetero",
        OpenVinoExecutionMode::NpuFirst => "npu-first",
    };
    let devices = profile.hetero_devices.join(">");
    let prefill = placement_summary(plan, PipelineStage::Prefill);
    let decode = placement_summary(plan, PipelineStage::Decode);
    let kv = placement_summary(plan, PipelineStage::KvCache);
    let weights = placement_summary(plan, PipelineStage::Weights);
    let lowering = lowering_summary(plan);
    let spill = plan
        .tiered_offload
        .as_ref()
        .map(|tier| {
            format!(
                "spill={}B profile={}",
                tier.spill_bytes,
                offload_profile_label(plan)
            )
        })
        .unwrap_or_else(|| "spill=0B".to_string());
    let model_root = session
        .model_root
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "unresolved".to_string());
    let lowering_diagnostics = session.lowering_diagnostics.as_deref().unwrap_or("none");

    BackendOutput {
        text: format!(
            "[openvino-fallback:{}] mode={} device={} devices={} model_root={} prefill={} decode={} kv={} weights={} lowering={} route={} {} image_count={} reason={} lowering_diagnostics={} prompt=`{}`",
            model.name,
            mode,
            session.device_name,
            devices,
            model_root,
            prefill,
            decode,
            kv,
            weights,
            lowering,
            plan.route.reason,
            spill,
            request.images.len(),
            session.reason,
            lowering_diagnostics,
            request.prompt.trim()
        ),
        telemetry: BackendTelemetry {
            estimated_prefill_ms: estimate_prefill_ms(profile, model, plan),
            estimated_decode_ms: estimate_decode_ms(profile, plan),
            generated_tokens: request.max_tokens.min(128),
        },
    }
}

fn fallback_behavior(
    session: &FallbackSession,
    model: &ModelDescriptor,
    request: &SessionRequest,
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> BackendResult<BackendOutput> {
    let allow_fallback = env::var("LOCI_OPENVINO_ALLOW_FALLBACK")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);

    if allow_fallback {
        return Ok(run_fallback_session(session, model, request, plan, profile));
    }

    Err(BackendError {
        message: format!(
            "OpenVINO real execution is unavailable for model `{}`: {}. Set LOCI_OPENVINO_ALLOW_FALLBACK=1 to re-enable diagnostic fallback output.",
            model.name, session.reason
        ),
    })
}

fn ensure_runtime_bootstrap() -> Option<&'static RuntimeBootstrap> {
    RUNTIME_BOOTSTRAP
        .get_or_init(bootstrap_runtime_environment)
        .as_ref()
}

fn bootstrap_runtime_environment() -> Option<RuntimeBootstrap> {
    let root_dir = discover_runtime_root()?;
    let lib_paths = collect_runtime_lib_paths(&root_dir);
    if lib_paths.is_empty() {
        return None;
    }

    let mut applied_environment = false;
    applied_environment |= set_env_path_if_missing("INTEL_OPENVINO_DIR", &root_dir);

    let cmake_dir = root_dir.join("runtime").join("cmake");
    if cmake_dir.is_dir() {
        applied_environment |= set_env_path_if_missing("OpenVINO_DIR", &cmake_dir);
        if cmake_dir.join("OpenVINOGenAIConfig.cmake").is_file() {
            applied_environment |= set_env_path_if_missing("OpenVINOGenAI_DIR", &cmake_dir);
        }
    }

    applied_environment |= prepend_env_paths("OPENVINO_LIB_PATHS", &lib_paths);
    applied_environment |= prepend_env_paths("PATH", &lib_paths);

    Some(RuntimeBootstrap {
        root_dir,
        lib_paths,
        applied_environment,
    })
}

fn discover_runtime_root() -> Option<PathBuf> {
    let env_root = env::var_os("INTEL_OPENVINO_DIR")
        .map(PathBuf::from)
        .filter(|path| has_runtime_layout(path));
    if env_root.is_some() {
        return env_root;
    }

    repo_runtime_root().filter(|path| has_runtime_layout(path))
}

fn repo_runtime_root() -> Option<PathBuf> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir.parent()?.parent()?;
    Some(repo_root.join("vendor").join("openvino-genai-runtime"))
}

fn has_runtime_layout(root_dir: &Path) -> bool {
    root_dir
        .join("runtime")
        .join("bin")
        .join("intel64")
        .join("Release")
        .join("openvino.dll")
        .is_file()
}

fn collect_runtime_lib_paths(root_dir: &Path) -> Vec<PathBuf> {
    let mut lib_paths = Vec::new();
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("bin")
            .join("intel64")
            .join("Release"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("bin")
            .join("intel64")
            .join("Debug"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("3rdparty")
            .join("tbb")
            .join("redist")
            .join("intel64")
            .join("vc14"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("3rdparty")
            .join("tbb")
            .join("bin")
            .join("intel64")
            .join("vc14"),
    );
    push_if_dir(
        &mut lib_paths,
        root_dir
            .join("runtime")
            .join("3rdparty")
            .join("tbb")
            .join("bin"),
    );
    lib_paths
}

fn push_if_dir(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if path.is_dir() && !paths.iter().any(|existing| same_path(existing, &path)) {
        paths.push(path);
    }
}

fn set_env_path_if_missing(name: &str, value: &Path) -> bool {
    match env::var_os(name) {
        Some(existing) if !existing.is_empty() => false,
        _ => {
            env::set_var(name, value);
            true
        }
    }
}

fn prepend_env_paths(name: &str, new_paths: &[PathBuf]) -> bool {
    if new_paths.is_empty() {
        return false;
    }

    let existing = env::var_os(name).unwrap_or_default();
    let existing_paths = env::split_paths(&existing).collect::<Vec<_>>();
    let mut merged = new_paths.to_vec();
    for path in &existing_paths {
        if !merged.iter().any(|candidate| same_path(candidate, &path)) {
            merged.push(path.clone());
        }
    }

    let had_missing = new_paths.iter().any(|path| {
        !existing_paths
            .iter()
            .any(|existing| same_path(path, existing))
    });

    match env::join_paths(&merged) {
        Ok(value) => {
            env::set_var(name, value);
            had_missing
        }
        Err(_) => false,
    }
}

fn same_path(left: &Path, right: &Path) -> bool {
    left.as_os_str()
        .to_string_lossy()
        .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
}

fn discover_runtime_topology() -> Result<HardwareTopology, String> {
    if let Some(bootstrap) = ensure_runtime_bootstrap() {
        let _ = (
            bootstrap.root_dir.as_os_str(),
            bootstrap.lib_paths.len(),
            bootstrap.applied_environment,
        );
    }
    let core = Core::new().map_err(setup_error)?;
    let devices = core.available_devices().map_err(openvino_error)?;
    let mut counts = HashMap::<String, usize>::new();
    let mut descriptors = Vec::new();

    for device in devices {
        let (kind, label, memory_bytes, power_watts) = match normalize_device_type(&device) {
            Some(spec) => spec,
            None => continue,
        };
        let index = counts.entry(label.to_string()).or_default();
        let device_id = format!("{}:{}", label.to_ascii_lowercase(), *index);
        *index += 1;

        descriptors.push(DeviceDescriptor {
            id: device_id,
            name: format!("openvino-{}", device),
            kind,
            platform: Some(std::env::consts::OS.to_string()),
            memory_bytes,
            compute_units: compute_units_for(kind),
            power_watts,
        });
    }

    if !descriptors
        .iter()
        .any(|device| device.kind == AcceleratorKind::Disk)
    {
        descriptors.push(disk_descriptor());
    }

    Ok(HardwareTopology {
        devices: descriptors,
        power: PowerState {
            battery_powered: false,
            battery_percent: None,
            thermal_state: ThermalState::Nominal,
            power_budget_watts: Some(45),
        },
    })
}

fn synthetic_topology() -> HardwareTopology {
    HardwareTopology {
        devices: vec![
            DeviceDescriptor {
                id: "cpu:0".to_string(),
                name: "host-cpu".to_string(),
                kind: AcceleratorKind::Cpu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(16 * 1024 * 1024 * 1024),
                compute_units: compute_units_for(AcceleratorKind::Cpu),
                power_watts: Some(25.0),
            },
            DeviceDescriptor {
                id: "gpu:0".to_string(),
                name: "integrated-gpu".to_string(),
                kind: AcceleratorKind::Gpu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(8 * 1024 * 1024 * 1024),
                compute_units: compute_units_for(AcceleratorKind::Gpu),
                power_watts: Some(20.0),
            },
            DeviceDescriptor {
                id: "npu:0".to_string(),
                name: "integrated-npu".to_string(),
                kind: AcceleratorKind::Npu,
                platform: Some(std::env::consts::OS.to_string()),
                memory_bytes: Some(2 * 1024 * 1024 * 1024),
                compute_units: compute_units_for(AcceleratorKind::Npu),
                power_watts: Some(5.0),
            },
            disk_descriptor(),
        ],
        power: PowerState {
            battery_powered: false,
            battery_percent: None,
            thermal_state: ThermalState::Nominal,
            power_budget_watts: Some(45),
        },
    }
}

fn disk_descriptor() -> DeviceDescriptor {
    DeviceDescriptor {
        id: "disk:0".to_string(),
        name: "nvme-tier".to_string(),
        kind: AcceleratorKind::Disk,
        platform: Some(std::env::consts::OS.to_string()),
        memory_bytes: Some(256 * 1024 * 1024 * 1024),
        compute_units: None,
        power_watts: None,
    }
}

#[allow(deprecated)]
fn normalize_device_type(
    device: &DeviceType<'_>,
) -> Option<(AcceleratorKind, &'static str, Option<u64>, Option<f32>)> {
    match device {
        DeviceType::CPU => Some((
            AcceleratorKind::Cpu,
            "cpu",
            Some(16 * 1024 * 1024 * 1024),
            Some(25.0),
        )),
        DeviceType::GPU => Some((
            AcceleratorKind::Gpu,
            "gpu",
            Some(8 * 1024 * 1024 * 1024),
            Some(20.0),
        )),
        DeviceType::NPU | DeviceType::GNA => Some((
            AcceleratorKind::Npu,
            "npu",
            Some(2 * 1024 * 1024 * 1024),
            Some(5.0),
        )),
        DeviceType::Other(name) => {
            let uppercase = name.to_ascii_uppercase();
            if uppercase.contains("NPU") {
                Some((
                    AcceleratorKind::Npu,
                    "npu",
                    Some(2 * 1024 * 1024 * 1024),
                    Some(5.0),
                ))
            } else if uppercase.contains("GPU") {
                Some((
                    AcceleratorKind::Gpu,
                    "gpu",
                    Some(8 * 1024 * 1024 * 1024),
                    Some(20.0),
                ))
            } else if uppercase.contains("CPU") {
                Some((
                    AcceleratorKind::Cpu,
                    "cpu",
                    Some(16 * 1024 * 1024 * 1024),
                    Some(25.0),
                ))
            } else {
                None
            }
        }
    }
}

fn compute_units_for(kind: AcceleratorKind) -> Option<u32> {
    match kind {
        AcceleratorKind::Cpu => std::thread::available_parallelism()
            .ok()
            .and_then(|value| u32::try_from(value.get()).ok()),
        AcceleratorKind::Gpu => Some(128),
        AcceleratorKind::Npu => Some(1),
        AcceleratorKind::Disk => None,
    }
}

fn resolve_model_root(model: &ModelDescriptor) -> Option<PathBuf> {
    if model.path.is_dir() || model.path.is_file() {
        Some(model.path.clone())
    } else {
        None
    }
}

fn runtime_device_name(profile: &OpenVinoExecutionProfile) -> String {
    if !profile.hetero_devices.is_empty() {
        if profile.hetero_devices.len() == 1 {
            profile.hetero_devices[0].clone()
        } else {
            format!("HETERO:{}", profile.hetero_devices.join(","))
        }
    } else if let Some(device) = profile.decode_device.as_deref() {
        device_id_to_openvino_name(device).to_string()
    } else {
        "CPU".to_string()
    }
}

fn shadow_lowering_compile(
    model: &ModelDescriptor,
    model_root: &Path,
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> Result<(), String> {
    let Some(entrypoint) = resolve_shadow_entrypoint(model, model_root) else {
        return Ok(());
    };

    if plan
        .lowering_plan
        .as_ref()
        .map(|lowering| lowering.backend.as_str() != "openvino")
        .unwrap_or(false)
    {
        return Err("OpenVINO lowering plan backend mismatch during shadow compile".to_string());
    }

    let _ = ensure_runtime_bootstrap();
    let mut core = Core::new().map_err(setup_error)?;
    let compile_device = lowering_compile_device(plan, profile);
    if let Some(priorities) = lowering_priorities(plan, profile) {
        let hetero = DeviceType::Other(Cow::Borrowed("HETERO"));
        core.set_property(
            &hetero,
            &openvino::RwPropertyKey::DevicePriorities,
            &priorities,
        )
        .map_err(openvino_error)?;
    }

    let ir_model = core
        .read_model_from_file(
            &entrypoint.xml_path.to_string_lossy(),
            &entrypoint.bin_path.to_string_lossy(),
        )
        .map_err(openvino_error)?;
    let _compiled = core
        .compile_model(&ir_model, compile_device)
        .map_err(openvino_error)?;
    Ok(())
}

fn lowering_compile_device(
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> DeviceType<'static> {
    match lowering_priority_devices(plan, profile).len() {
        0 => DeviceType::Other(Cow::Owned(runtime_device_name(profile))),
        1 => DeviceType::Other(Cow::Owned(
            lowering_priority_devices(plan, profile)
                .into_iter()
                .next()
                .unwrap_or_else(|| "CPU".to_string()),
        )),
        _ => DeviceType::Other(Cow::Borrowed("HETERO")).to_owned(),
    }
}

fn lowering_priorities(plan: &ExecutionPlan, profile: &OpenVinoExecutionProfile) -> Option<String> {
    let priorities = lowering_priority_devices(plan, profile);
    (priorities.len() > 1).then(|| priorities.join(","))
}

fn lowering_priority_devices(
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> Vec<String> {
    let mut devices = Vec::new();
    if let Some(lowering) = &plan.lowering_plan {
        let lowering_regions = if !lowering.partitions.is_empty() {
            lowering
                .partitions
                .iter()
                .map(|partition| partition.affinity_tag.as_ref())
                .collect::<Vec<_>>()
        } else {
            lowering
                .subgraphs
                .iter()
                .map(|subgraph| subgraph.affinity_tag.as_ref())
                .collect::<Vec<_>>()
        };

        for affinity in lowering_regions {
            if let Some(affinity) = affinity {
                if !devices.contains(affinity) {
                    devices.push(affinity.clone());
                }
            }
        }
    }

    if devices.is_empty() {
        for device in &profile.hetero_devices {
            if !devices.contains(device) {
                devices.push(device.clone());
            }
        }
    }

    devices
}

struct ShadowEntrypoint {
    xml_path: PathBuf,
    bin_path: PathBuf,
}

fn resolve_shadow_entrypoint(
    model: &ModelDescriptor,
    model_root: &Path,
) -> Option<ShadowEntrypoint> {
    let xml_name = if model.is_multimodal_architecture() {
        "openvino_language_model.xml"
    } else {
        "openvino_model.xml"
    };
    let xml_path = model_root.join(xml_name);
    let bin_path = xml_path.with_extension("bin");
    if xml_path.is_file() && bin_path.is_file() {
        Some(ShadowEntrypoint { xml_path, bin_path })
    } else {
        None
    }
}

fn runtime_properties(
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> Vec<(String, String)> {
    let mut properties = vec![(
        "PERFORMANCE_HINT".to_string(),
        match profile.execution_mode {
            OpenVinoExecutionMode::NpuFirst => "LATENCY".to_string(),
            OpenVinoExecutionMode::Hetero => "THROUGHPUT".to_string(),
        },
    )];

    if profile.dynamic_reoffload {
        properties.push(("ENABLE_CPU_PINNING".to_string(), "NO".to_string()));
    }

    if let Some(priorities) = lowering_priorities(plan, profile) {
        properties.push(("MULTI_DEVICE_PRIORITIES".to_string(), priorities));
    }

    properties
}

fn device_id_to_openvino_name(device_id: &str) -> &'static str {
    if device_id.starts_with("npu:") {
        "NPU"
    } else if device_id.starts_with("gpu:") {
        "GPU"
    } else {
        "CPU"
    }
}

fn setup_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn openvino_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn inference_error(error: impl std::fmt::Display) -> BackendError {
    BackendError {
        message: error.to_string(),
    }
}

fn f32_ms_to_u64(value: f32) -> u64 {
    if value.is_finite() && value >= 0.0 {
        value.round() as u64
    } else {
        0
    }
}

fn load_image_tensors(images: &[ImageInput]) -> BackendResult<Vec<Tensor>> {
    images.iter().map(load_image_tensor).collect()
}

fn load_image_tensor(image: &ImageInput) -> BackendResult<Tensor> {
    match image {
        ImageInput::Path { path } => {
            let bytes = fs::read(path).map_err(io_error)?;
            decode_image_tensor(&bytes, Some(path))
        }
        ImageInput::Url { url } => {
            if let Some(path) = file_url_to_path(url) {
                let bytes = fs::read(&path).map_err(io_error)?;
                decode_image_tensor(&bytes, Some(&path))
            } else if let Some(bytes) = decode_data_url(url)? {
                decode_image_tensor(&bytes, None)
            } else {
                Err(BackendError {
                    message: format!(
                        "unsupported image URL `{url}`; only file:// URLs and data URLs are supported"
                    ),
                })
            }
        }
        ImageInput::Base64 {
            data_base64,
            media_type: _,
        } => {
            let bytes = BASE64_STANDARD
                .decode(data_base64)
                .map_err(|error| BackendError {
                    message: format!("invalid base64 image payload: {error}"),
                })?;
            decode_image_tensor(&bytes, None)
        }
    }
}

fn decode_image_tensor(bytes: &[u8], source_path: Option<&Path>) -> BackendResult<Tensor> {
    let decoded = image::load_from_memory(bytes).map_err(|error| BackendError {
        message: match source_path {
            Some(path) => format!("failed to decode image `{}`: {error}", path.display()),
            None => format!("failed to decode image payload: {error}"),
        },
    })?;
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();
    let shape = Shape::new(&[height as i64, width as i64, 3]).map_err(inference_error)?;
    let mut tensor = Tensor::new(ElementType::U8, &shape).map_err(inference_error)?;
    tensor
        .get_raw_data_mut()
        .map_err(inference_error)?
        .copy_from_slice(&rgb.into_raw());
    Ok(tensor)
}

fn file_url_to_path(url: &str) -> Option<PathBuf> {
    let normalized = url.strip_prefix("file://")?;
    let trimmed = normalized.strip_prefix('/').unwrap_or(normalized);
    Some(PathBuf::from(trimmed))
}

fn decode_data_url(url: &str) -> BackendResult<Option<Vec<u8>>> {
    let Some(payload) = url.strip_prefix("data:") else {
        return Ok(None);
    };
    let Some((metadata, encoded)) = payload.split_once(',') else {
        return Err(BackendError {
            message: "data URL image payload is malformed".to_string(),
        });
    };
    if !metadata.ends_with(";base64") {
        return Err(BackendError {
            message: "data URL image payload must use base64 encoding".to_string(),
        });
    }
    let bytes = BASE64_STANDARD
        .decode(encoded)
        .map_err(|error| BackendError {
            message: format!("invalid base64 image payload: {error}"),
        })?;
    Ok(Some(bytes))
}

fn io_error(error: std::io::Error) -> BackendError {
    BackendError {
        message: error.to_string(),
    }
}

fn openvino_profile(plan: &ExecutionPlan) -> BackendResult<&OpenVinoExecutionProfile> {
    match &plan.backend_profile {
        BackendExecutionProfile::OpenVino(profile) => Ok(profile),
        _ => Err(BackendError {
            message: "execution plan is missing an OpenVINO backend profile".to_string(),
        }),
    }
}

fn validate_openvino_plan(
    plan: &ExecutionPlan,
    profile: &OpenVinoExecutionProfile,
) -> BackendResult<()> {
    let decode_target = placement_target(plan, PipelineStage::Decode);
    if decode_target == Some(AcceleratorKind::Disk) {
        return Err(BackendError {
            message: "OpenVINO decode stage cannot target disk".to_string(),
        });
    }

    if matches!(profile.execution_mode, OpenVinoExecutionMode::NpuFirst)
        && !profile
            .decode_device
            .as_deref()
            .map(|device| device.starts_with("npu:"))
            .unwrap_or(false)
    {
        return Err(BackendError {
            message: "OpenVINO npu-first mode requires an NPU decode device".to_string(),
        });
    }

    if let Some(weights_device) = &profile.weights_device {
        if weights_device.starts_with("disk:")
            && plan
                .tiered_offload
                .as_ref()
                .map(|tier| tier.target_device.as_str())
                != Some(weights_device.as_str())
        {
            return Err(BackendError {
                message: "OpenVINO weights device must match the tiered offload target".to_string(),
            });
        }
    }

    if let Some(lowering_plan) = &plan.lowering_plan {
        if lowering_plan.backend != "openvino" {
            return Err(BackendError {
                message: "OpenVINO execution received a lowering plan for a different backend"
                    .to_string(),
            });
        }
        for partition in &lowering_plan.partitions {
            if partition.target == AcceleratorKind::Disk && partition.affinity_tag.is_some() {
                return Err(BackendError {
                    message:
                        "disk-backed lowering partitions must not expose executable OpenVINO affinities"
                            .to_string(),
                });
            }
        }
        if lowering_plan.subgraphs.iter().any(|subgraph| {
            subgraph.target == AcceleratorKind::Disk && subgraph.affinity_tag.is_some()
        }) {
            return Err(BackendError {
                message:
                    "disk-backed lowering regions must not expose executable OpenVINO affinities"
                        .to_string(),
            });
        }
        for operator in &lowering_plan.operators {
            if operator.target == AcceleratorKind::Disk && operator.affinity_tag.is_some() {
                return Err(BackendError {
                    message:
                        "disk-backed lowering operators must not expose executable OpenVINO affinities"
                            .to_string(),
                });
            }
            if !lowering_plan
                .partitions
                .iter()
                .any(|partition| partition.id == operator.partition)
            {
                return Err(BackendError {
                    message:
                        "OpenVINO lowering operator references a partition that does not exist"
                            .to_string(),
                });
            }
        }
    }

    Ok(())
}

fn derive_residency(plan: &ExecutionPlan) -> PreparedResidency {
    let weights_on_disk =
        placement_target(plan, PipelineStage::Weights) == Some(AcceleratorKind::Disk);
    let kv_on_disk = placement_target(plan, PipelineStage::KvCache) == Some(AcceleratorKind::Disk);

    if weights_on_disk && kv_on_disk {
        PreparedResidency::DiskBacked
    } else if weights_on_disk || kv_on_disk || plan.tiered_offload.is_some() {
        PreparedResidency::Hybrid
    } else {
        PreparedResidency::Memory
    }
}

fn estimate_resident_memory_bytes(
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
    residency: PreparedResidency,
) -> Option<u64> {
    let model_bytes = model.memory_bytes?;
    let resident_weights = if let Some(tier) = &plan.tiered_offload {
        let memory_percent = 100u64.saturating_sub(tier.policy.weights.disk_percent as u64);
        model_bytes.saturating_mul(memory_percent) / 100
    } else {
        model_bytes
    };
    let kv_bytes = plan.kv_cache.max_cache_bytes.unwrap_or(0);
    let resident_kv = match placement_target(plan, PipelineStage::KvCache) {
        Some(AcceleratorKind::Disk) => kv_bytes / 4,
        Some(_) => kv_bytes,
        None => 0,
    };

    Some(match residency {
        PreparedResidency::Memory => resident_weights.saturating_add(resident_kv),
        PreparedResidency::Hybrid => resident_weights
            .saturating_mul(9)
            .saturating_div(10)
            .saturating_add(resident_kv),
        PreparedResidency::DiskBacked => resident_weights / 2 + resident_kv / 2,
    })
}

fn estimate_prefill_ms(
    profile: &OpenVinoExecutionProfile,
    model: &ModelDescriptor,
    plan: &ExecutionPlan,
) -> u64 {
    let base = if profile.prefill_device.as_deref() == Some("gpu:0") {
        10
    } else {
        16
    };
    let weight_penalty = match placement_target(plan, PipelineStage::Weights) {
        Some(AcceleratorKind::Disk) => 6,
        Some(AcceleratorKind::Cpu) => 2,
        _ => 0,
    };
    let compression_bonus = plan
        .tiered_offload
        .as_ref()
        .map(|tier| u64::from(tier.policy.compress_weights))
        .unwrap_or(0);

    base + weight_penalty + model.parameter_count.unwrap_or_default() / 2_000_000_000
        - compression_bonus.min(1)
}

fn estimate_decode_ms(profile: &OpenVinoExecutionProfile, plan: &ExecutionPlan) -> u64 {
    let base = if profile.decode_device.as_deref() == Some("npu:0") {
        5
    } else {
        8
    };
    let kv_penalty = match placement_target(plan, PipelineStage::KvCache) {
        Some(AcceleratorKind::Disk) => 5,
        Some(AcceleratorKind::Cpu) => 2,
        _ => 0,
    };
    let reoffload_penalty = if profile.dynamic_reoffload { 2 } else { 0 };
    let compression_penalty = plan
        .tiered_offload
        .as_ref()
        .map(|tier| u64::from(tier.policy.compress_kv_cache))
        .unwrap_or(0);

    base + kv_penalty + reoffload_penalty + compression_penalty
}

fn placement_target(plan: &ExecutionPlan, stage: PipelineStage) -> Option<AcceleratorKind> {
    plan.placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(|placement| placement.target)
}

fn placement_summary(plan: &ExecutionPlan, stage: PipelineStage) -> String {
    plan.placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(format_placement)
        .unwrap_or_else(|| "unassigned".to_string())
}

fn lowering_summary(plan: &ExecutionPlan) -> String {
    plan.lowering_plan
        .as_ref()
        .map(|lowering| {
            format!(
                "{}-regions:{}",
                lowering.subgraphs.len(),
                lowering
                    .subgraphs
                    .iter()
                    .map(|subgraph| {
                        let affinity = subgraph.affinity_tag.as_deref().unwrap_or("none");
                        format!("{}={}", subgraph.id, affinity)
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .unwrap_or_else(|| "none".to_string())
}

fn format_placement(placement: &PlacementDecision) -> String {
    let device = placement.device_id.as_deref().unwrap_or("none");
    format!("{}@{}", accelerator_label(placement.target), device)
}

fn accelerator_label(kind: AcceleratorKind) -> &'static str {
    match kind {
        AcceleratorKind::Cpu => "cpu",
        AcceleratorKind::Gpu => "gpu",
        AcceleratorKind::Npu => "npu",
        AcceleratorKind::Disk => "disk",
    }
}

fn offload_profile_label(plan: &ExecutionPlan) -> &'static str {
    match plan.tiered_offload.as_ref().map(|tier| tier.profile) {
        Some(loci_protocol::TieredOffloadProfile::Auto) => "auto",
        Some(loci_protocol::TieredOffloadProfile::GpuResident) => "gpu_resident",
        Some(loci_protocol::TieredOffloadProfile::Balanced) => "balanced",
        Some(loci_protocol::TieredOffloadProfile::DiskHeavy) => "disk_heavy",
        None => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_protocol::{
        BackendExecutionProfile, BackendLoweringPlan, CandleExecutionProfile,
        CandleTensorResidency, ChipOperatorClass, ExecutionPlan, GenericExecutionProfile,
        KvCachePlan, LoweringAffinityMode, LoweringGranularity, LoweringOperatorPlan,
        LoweringPartitionPlan, LoweringSubgraphPlan, PlacementDecision, RouteDecision,
        TieredOffloadPlan, TieredOffloadPolicy, TieredPlacementPercentages,
    };
    use std::{
        env, fs,
        path::{Path, PathBuf},
        time::{SystemTime, UNIX_EPOCH},
    };

    fn demo_model() -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.xml"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(1_000_000_000),
            context_length: Some(8192),
            preferred_backend: Some("openvino".to_string()),
        }
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-openvino-{label}-{suffix}"))
    }

    fn openvino_plan() -> ExecutionPlan {
        ExecutionPlan {
            backend: "openvino".to_string(),
            route: RouteDecision {
                selected_model: "demo".to_string(),
                reason: "npu-first".to_string(),
                alternatives: vec!["fallback".to_string()],
            },
            placements: vec![
                PlacementDecision {
                    stage: PipelineStage::Prefill,
                    target: AcceleratorKind::Gpu,
                    device_id: Some("gpu:0".to_string()),
                    memory_bytes: None,
                    rationale: "prefill".to_string(),
                },
                PlacementDecision {
                    stage: PipelineStage::Decode,
                    target: AcceleratorKind::Npu,
                    device_id: Some("npu:0".to_string()),
                    memory_bytes: None,
                    rationale: "decode".to_string(),
                },
                PlacementDecision {
                    stage: PipelineStage::KvCache,
                    target: AcceleratorKind::Disk,
                    device_id: Some("disk:0".to_string()),
                    memory_bytes: None,
                    rationale: "kv".to_string(),
                },
                PlacementDecision {
                    stage: PipelineStage::Weights,
                    target: AcceleratorKind::Disk,
                    device_id: Some("disk:0".to_string()),
                    memory_bytes: None,
                    rationale: "weights".to_string(),
                },
            ],
            lowering_plan: Some(BackendLoweringPlan {
                backend: "openvino".to_string(),
                granularity: LoweringGranularity::Subgraph,
                affinity_mode: LoweringAffinityMode::Planned,
                subgraphs: vec![
                    LoweringSubgraphPlan {
                        id: "prefill_attention_block".to_string(),
                        stage: PipelineStage::Prefill,
                        operator_class: ChipOperatorClass::Attention,
                        target: AcceleratorKind::Gpu,
                        device_id: Some("gpu:0".to_string()),
                        affinity_tag: Some("GPU".to_string()),
                        estimated_bytes: Some(256 << 20),
                        spillable: false,
                        rationale: "prefill".to_string(),
                    },
                    LoweringSubgraphPlan {
                        id: "decode_attention_block".to_string(),
                        stage: PipelineStage::Decode,
                        operator_class: ChipOperatorClass::Attention,
                        target: AcceleratorKind::Npu,
                        device_id: Some("npu:0".to_string()),
                        affinity_tag: Some("NPU".to_string()),
                        estimated_bytes: Some(128 << 20),
                        spillable: false,
                        rationale: "decode".to_string(),
                    },
                ],
                partitions: vec![
                    LoweringPartitionPlan {
                        id: "partition-1-gpu".to_string(),
                        target: AcceleratorKind::Gpu,
                        device_id: Some("gpu:0".to_string()),
                        affinity_tag: Some("GPU".to_string()),
                        operator_classes: vec![ChipOperatorClass::Attention],
                        subgraphs: vec!["prefill_attention_block".to_string()],
                        estimated_bytes: Some(256 << 20),
                        spillable: false,
                        rationale: "prefill attention stays on the gpu partition".to_string(),
                    },
                    LoweringPartitionPlan {
                        id: "partition-2-npu".to_string(),
                        target: AcceleratorKind::Npu,
                        device_id: Some("npu:0".to_string()),
                        affinity_tag: Some("NPU".to_string()),
                        operator_classes: vec![ChipOperatorClass::Attention],
                        subgraphs: vec!["decode_attention_block".to_string()],
                        estimated_bytes: Some(128 << 20),
                        spillable: false,
                        rationale: "decode attention stays on the npu partition".to_string(),
                    },
                ],
                operators: vec![
                    LoweringOperatorPlan {
                        id: "operator-prefill_attention_block".to_string(),
                        partition: "partition-1-gpu".to_string(),
                        subgraph: "prefill_attention_block".to_string(),
                        stage: PipelineStage::Prefill,
                        operator_class: ChipOperatorClass::Attention,
                        target: AcceleratorKind::Gpu,
                        device_id: Some("gpu:0".to_string()),
                        affinity_tag: Some("GPU".to_string()),
                        estimated_bytes: Some(256 << 20),
                        spillable: false,
                        rationale: "prefill".to_string(),
                    },
                    LoweringOperatorPlan {
                        id: "operator-decode_attention_block".to_string(),
                        partition: "partition-2-npu".to_string(),
                        subgraph: "decode_attention_block".to_string(),
                        stage: PipelineStage::Decode,
                        operator_class: ChipOperatorClass::Attention,
                        target: AcceleratorKind::Npu,
                        device_id: Some("npu:0".to_string()),
                        affinity_tag: Some("NPU".to_string()),
                        estimated_bytes: Some(128 << 20),
                        spillable: false,
                        rationale: "decode".to_string(),
                    },
                ],
                notes: Vec::new(),
            }),
            kv_cache: KvCachePlan {
                strategy: "paged".to_string(),
                shared_across_models: false,
                page_size_bytes: Some(1 << 20),
                block_size_tokens: Some(16),
                max_cache_bytes: Some(128 << 20),
                type_k: Some("f16".to_string()),
                type_v: Some("f16".to_string()),
                tiered: false,
            },
            tiered_offload: Some(TieredOffloadPlan {
                spill_bytes: 512 << 20,
                prefetch_window_bytes: 64 << 20,
                target_device: "disk:0".to_string(),
                profile: loci_protocol::TieredOffloadProfile::Balanced,
                policy: TieredOffloadPolicy {
                    weights: TieredPlacementPercentages {
                        gpu_percent: 20,
                        cpu_percent: 50,
                        disk_percent: 30,
                    },
                    kv_cache: TieredPlacementPercentages {
                        gpu_percent: 0,
                        cpu_percent: 50,
                        disk_percent: 50,
                    },
                    activations: TieredPlacementPercentages {
                        gpu_percent: 50,
                        cpu_percent: 50,
                        disk_percent: 0,
                    },
                    cpu_cache_compute: true,
                    compress_weights: true,
                    compress_kv_cache: true,
                },
            }),
            backend_profile: BackendExecutionProfile::OpenVino(OpenVinoExecutionProfile {
                session_key: "ov:gpu:0:npu:0".to_string(),
                execution_mode: OpenVinoExecutionMode::NpuFirst,
                genai_pipeline: true,
                hetero_devices: vec!["NPU".to_string(), "GPU".to_string(), "CPU".to_string()],
                prefill_device: Some("gpu:0".to_string()),
                decode_device: Some("npu:0".to_string()),
                kv_cache_device: Some("npu:0".to_string()),
                weights_device: Some("disk:0".to_string()),
                dynamic_reoffload: false,
            }),
        }
    }

    #[test]
    fn prepare_returns_hybrid_session_for_tiered_plan() {
        let backend = OpenVinoBackend::default();
        let prepared = backend
            .prepare(&demo_model(), &openvino_plan())
            .expect("prepared");

        assert_eq!(prepared.backend, "openvino");
        assert_eq!(prepared.residency, PreparedResidency::DiskBacked);
        assert!(prepared.estimated_memory_bytes.unwrap_or_default() > 0);
    }

    #[test]
    fn execute_rejects_non_openvino_profiles() {
        let backend = OpenVinoBackend::default();
        let plan = ExecutionPlan {
            backend: "openvino".to_string(),
            route: RouteDecision {
                selected_model: "demo".to_string(),
                reason: "fallback".to_string(),
                alternatives: Vec::new(),
            },
            placements: Vec::new(),
            lowering_plan: None,
            kv_cache: KvCachePlan {
                strategy: "paged".to_string(),
                shared_across_models: false,
                page_size_bytes: None,
                block_size_tokens: None,
                max_cache_bytes: None,
                type_k: None,
                type_v: None,
                tiered: false,
            },
            tiered_offload: None,
            backend_profile: BackendExecutionProfile::Candle(CandleExecutionProfile {
                session_key: "candle:cpu:0:cpu:0".to_string(),
                prefill_device: "cpu:0".to_string(),
                decode_device: "cpu:0".to_string(),
                kv_cache_device: "cpu:0".to_string(),
                tensor_residency: CandleTensorResidency::MemoryOnly,
                fallback_reason: "fallback".to_string(),
            }),
        };
        let prepared = PreparedModel {
            model_name: "demo".to_string(),
            backend: "openvino".to_string(),
            session_key: "bad".to_string(),
            residency: PreparedResidency::Memory,
            estimated_memory_bytes: None,
        };

        let error = backend
            .execute(
                &prepared,
                &demo_model(),
                &SessionRequest {
                    prompt: "hello".to_string(),
                    max_tokens: 8,
                    temperature: 0.2,
                    target_model: None,
                    images: Vec::new(),
                    structured_output: false,
                    tool_calling: false,
                },
                &plan,
            )
            .expect_err("error");

        assert!(error
            .message
            .contains("missing an OpenVINO backend profile"));
    }

    #[test]
    fn generic_profile_is_not_accepted() {
        let error = openvino_profile(&ExecutionPlan {
            backend: "openvino".to_string(),
            route: RouteDecision {
                selected_model: "demo".to_string(),
                reason: "generic".to_string(),
                alternatives: Vec::new(),
            },
            placements: Vec::new(),
            lowering_plan: None,
            kv_cache: KvCachePlan {
                strategy: "contiguous".to_string(),
                shared_across_models: false,
                page_size_bytes: None,
                block_size_tokens: None,
                max_cache_bytes: None,
                type_k: None,
                type_v: None,
                tiered: false,
            },
            tiered_offload: None,
            backend_profile: BackendExecutionProfile::Generic(GenericExecutionProfile {
                session_key: "generic".to_string(),
                summary: "generic".to_string(),
            }),
        })
        .expect_err("error");

        assert!(error.message.contains("OpenVINO"));
    }

    #[test]
    fn prepare_rejects_npu_first_without_npu_decode() {
        let backend = OpenVinoBackend::default();
        let mut plan = openvino_plan();
        if let BackendExecutionProfile::OpenVino(profile) = &mut plan.backend_profile {
            profile.decode_device = Some("cpu:0".to_string());
        }

        let error = backend.prepare(&demo_model(), &plan).expect_err("error");
        assert!(error.message.contains("npu-first"));
    }

    #[test]
    fn execute_rejects_cached_fallback_when_openvino_real_execution_is_unavailable() {
        env::remove_var("LOCI_OPENVINO_ALLOW_FALLBACK");

        let backend = OpenVinoBackend::default();
        let plan = openvino_plan();
        let prepared = backend.prepare(&demo_model(), &plan).expect("prepared");

        let error = backend
            .execute(
                &prepared,
                &demo_model(),
                &SessionRequest {
                    prompt: "hello from test".to_string(),
                    max_tokens: 16,
                    temperature: 0.0,
                    target_model: Some("demo".to_string()),
                    images: Vec::new(),
                    structured_output: false,
                    tool_calling: false,
                },
                &plan,
            )
            .expect_err("fallback should now be rejected by default");

        assert!(error.message.contains("OpenVINO real execution is unavailable"));
        assert!(error.message.contains("LOCI_OPENVINO_ALLOW_FALLBACK=1"));
    }

    #[test]
    fn execute_allows_cached_fallback_when_explicitly_enabled() {
        env::set_var("LOCI_OPENVINO_ALLOW_FALLBACK", "1");

        let backend = OpenVinoBackend::default();
        let plan = openvino_plan();
        let prepared = backend.prepare(&demo_model(), &plan).expect("prepared");

        let output = backend
            .execute(
                &prepared,
                &demo_model(),
                &SessionRequest {
                    prompt: "hello from test".to_string(),
                    max_tokens: 16,
                    temperature: 0.0,
                    target_model: Some("demo".to_string()),
                    images: Vec::new(),
                    structured_output: false,
                    tool_calling: false,
                },
                &plan,
            )
            .expect("output");

        assert!(output.text.contains("[openvino-fallback:demo]"));
        assert!(output.text.contains("reason="));
        assert!(output.telemetry.generated_tokens > 0);
        env::remove_var("LOCI_OPENVINO_ALLOW_FALLBACK");
    }

    #[test]
    fn multimodal_architecture_detection_matches_known_vlm_families() {
        let mut model = demo_model();
        model.architecture = "vlm".to_string();
        assert!(model.is_multimodal_architecture());

        model.architecture = "qwen2-vl".to_string();
        assert!(model.is_multimodal_architecture());

        model.architecture = "phi-3.5-vision".to_string();
        assert!(model.is_multimodal_architecture());

        model.architecture = "llama".to_string();
        assert!(!model.is_multimodal_architecture());
    }

    #[test]
    fn file_url_paths_are_normalized_for_windows_style_inputs() {
        let path = file_url_to_path("file:///D:/images/demo.png").expect("path");
        assert_eq!(path, PathBuf::from("D:/images/demo.png"));
    }

    #[test]
    fn data_urls_decode_into_image_bytes() {
        let bytes = decode_data_url(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0WQAAAAASUVORK5CYII=",
        )
        .expect("decoded")
        .expect("payload");

        assert!(!bytes.is_empty());
    }

    #[test]
    fn gguf_files_are_treated_as_direct_openvino_text_inputs() {
        let file = unique_temp_dir("gguf-file").with_extension("gguf");
        fs::write(&file, "gguf").expect("gguf");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: file.clone(),
            architecture: "llama".to_string(),
            memory_bytes: Some(1),
            parameter_count: Some(1),
            context_length: Some(128),
            preferred_backend: Some("openvino".to_string()),
        };

        let result = validate_model_root_layout(&model, &file);
        assert!(result.is_ok());

        fs::remove_file(file).expect("cleanup");
    }

    #[test]
    fn onnx_files_are_treated_as_direct_openvino_text_inputs() {
        let file = unique_temp_dir("onnx-file").with_extension("onnx");
        fs::write(&file, "onnx").expect("onnx");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: file.clone(),
            architecture: "llama".to_string(),
            memory_bytes: Some(1),
            parameter_count: Some(1),
            context_length: Some(128),
            preferred_backend: Some("openvino".to_string()),
        };

        let result = validate_model_root_layout(&model, &file);
        assert!(result.is_ok());

        fs::remove_file(file).expect("cleanup");
    }

    #[test]
    fn raw_transformers_multimodal_directories_report_a_clear_export_error() {
        let dir = unique_temp_dir("raw-vlm");
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join("config.json"), "{}").expect("config");
        fs::write(dir.join("model-00001-of-00001.safetensors"), "pointer").expect("weights");

        let mut model = demo_model();
        model.architecture = "vision".to_string();

        let error = validate_model_root_layout(&model, &dir).expect_err("validation should fail");
        assert!(error.contains("raw Transformers checkpoint"));
        assert!(error.contains("openvino_language_model.xml"));

        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn collect_runtime_lib_paths_finds_release_and_tbb_bins() {
        let dir = unique_temp_dir("runtime-layout");
        fs::create_dir_all(
            dir.join("runtime")
                .join("bin")
                .join("intel64")
                .join("Release"),
        )
        .expect("release");
        fs::create_dir_all(dir.join("runtime").join("3rdparty").join("tbb").join("bin"))
            .expect("tbb");

        let lib_paths = collect_runtime_lib_paths(&dir);
        assert_eq!(lib_paths.len(), 2);
        assert!(lib_paths.iter().any(|path| {
            path.ends_with(
                PathBuf::from("runtime")
                    .join("bin")
                    .join("intel64")
                    .join("Release"),
            )
        }));
        assert!(lib_paths.iter().any(|path| path.ends_with(
            PathBuf::from("runtime")
                .join("3rdparty")
                .join("tbb")
                .join("bin")
        )));

        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn set_env_path_if_missing_only_sets_empty_variables() {
        let key = "LOCI_TEST_OPENVINO_PATH";
        env::remove_var(key);
        let path = PathBuf::from("D:/demo/runtime");

        assert!(set_env_path_if_missing(key, &path));
        assert_eq!(env::var_os(key).as_deref(), Some(path.as_os_str()));
        assert!(!set_env_path_if_missing(key, Path::new("D:/other/runtime")));

        env::remove_var(key);
    }

    #[test]
    fn runtime_properties_include_lowering_priorities() {
        let plan = openvino_plan();
        let profile = match &plan.backend_profile {
            BackendExecutionProfile::OpenVino(profile) => profile,
            _ => panic!("expected openvino profile"),
        };

        let properties = runtime_properties(&plan, profile);
        assert!(properties
            .iter()
            .any(|(key, value)| { key == "MULTI_DEVICE_PRIORITIES" && value == "GPU,NPU" }));
    }
}

use crate::ModelDescriptor;
use libloading::Library;
use loci_protocol::ModelFormat;
use std::{
    ffi::{c_char, CString},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone)]
pub(super) enum ModelPreparationState {
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
pub(super) struct PreparedArtifactResolver {
    cache_root: PathBuf,
    toolchain: OpenVinoToolchainConfig,
}

#[derive(Debug, Clone)]
pub(super) struct MaterializedPreparation {
    prepared_root: PathBuf,
    metadata_path: PathBuf,
    expected_entrypoint: &'static str,
    placeholder_entrypoint: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum PreparationJobKind {
    DiscoverOnly,
    MaterializePlaceholder,
    ValidateTextSourceAsset,
    MaterializeTextExecutable,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum PreparationJobStatus {
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
pub(super) struct OpenVinoToolchainConfig {
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

impl PreparedArtifactResolver {
    pub(super) fn new() -> Self {
        Self {
            cache_root: std::env::temp_dir().join("loci-openvino-artifacts"),
            toolchain: OpenVinoToolchainConfig::from_environment(),
        }
    }

    pub(super) fn inspect(&self, model: &ModelDescriptor) -> ModelPreparationState {
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

pub(super) fn io_other<E: std::fmt::Display>(error: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, error.to_string())
}

impl OpenVinoToolchainConfig {
    fn from_environment() -> Self {
        let install_root = std::env::var_os("LOCI_OPENVINO_ROOT").map(PathBuf::from);
        let text_materializer =
            std::env::var_os("LOCI_OPENVINO_TEXT_MATERIALIZER").map(PathBuf::from);
        let ffi_library = std::env::var_os("LOCI_OPENVINO_FFI_BRIDGE").map(PathBuf::from);
        let strategy = match std::env::var("LOCI_OPENVINO_TOOLCHAIN") {
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
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
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

fn resolve_model_root(model: &ModelDescriptor) -> Option<PathBuf> {
    if model.path.is_dir() || model.path.is_file() {
        Some(model.path.clone())
    } else {
        None
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

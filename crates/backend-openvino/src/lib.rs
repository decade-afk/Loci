mod bootstrap;
mod io_utils;
mod planning;
mod prepare;
mod topology;

use self::bootstrap::ensure_runtime_bootstrap;
use self::io_utils::{
    build_generation_config, fallback_behavior, inference_error, load_image_tensors,
    openvino_error, setup_error, telemetry_from_llm_results, telemetry_from_vlm_results,
};
use self::planning::{
    derive_residency, estimate_resident_memory_bytes, openvino_profile, runtime_device_name,
    runtime_properties, shadow_lowering_compile, validate_openvino_plan,
};
use self::prepare::{ModelPreparationState, PreparedArtifactResolver};
use self::topology::{discover_runtime_topology, synthetic_topology};
use loci_protocol::{
    AcceleratorKind, Backend, BackendAssetCapabilities, BackendDescriptor, BackendError,
    BackendKernelCatalog, BackendLoweringCapabilities, BackendOutput, BackendResult,
    BackendRuntimeFamily, ChipOperatorClass, ExecutionArtifactKind, ExecutionPlan,
    HardwareTopology, KernelDescriptor, KernelImplementationKind, KernelMaturity, KernelOrigin,
    LoweringGranularity, ModelAssetLayout, ModelDescriptor, ModelFormat, OpenVinoExecutionProfile,
    PreparedModel, SessionRequest,
};
use openvino::{CompiledModel, Tensor};
use openvino_genai::{LlmPipeline, VlmPipeline};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
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

impl Default for OpenVinoRuntime {
    fn default() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }
}

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
            SessionSlot::Fallback(session) => {
                fallback_behavior(session, model, request, plan, profile)
            }
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
        let model_root = match preparation {
            ModelPreparationState::ReadyOpenVinoArtifact { model_root, .. } => Some(model_root),
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
                    model_root: Some(source_root),
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
                    model_root: Some(source_root),
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
                    model_root: Some(source_root),
                    lowering_diagnostics: None,
                });
            }
            ModelPreparationState::Unsupported { detail } => {
                return SessionSlot::Fallback(FallbackSession {
                    reason: detail,
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
                        message: "OpenVINO ONNX text pipeline cannot accept image inputs"
                            .to_string(),
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
        || model_root.read_dir().ok().is_some_and(|entries| {
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

fn resolve_model_root(model: &ModelDescriptor) -> Option<PathBuf> {
    if model.path.is_dir() || model.path.is_file() {
        Some(model.path.clone())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::{collect_runtime_lib_paths, set_env_path_if_missing};
    use crate::io_utils::{decode_data_url, file_url_to_path};
    use loci_protocol::{
        BackendExecutionProfile, BackendLoweringPlan, CandleExecutionProfile,
        CandleTensorResidency, ChipOperatorClass, ExecutionPlan, GenericExecutionProfile,
        KvCachePlan, LoweringAffinityMode, LoweringGranularity, LoweringOperatorPlan,
        LoweringPartitionPlan, LoweringSubgraphPlan, OpenVinoExecutionMode, PipelineStage,
        PlacementDecision, PreparedResidency, RouteDecision, TieredOffloadPlan,
        TieredOffloadPolicy, TieredPlacementPercentages,
    };
    use std::{
        env, fs,
        path::{Path, PathBuf},
        sync::Mutex,
        time::{SystemTime, UNIX_EPOCH},
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
                session_key: "ov:gpu:0:npu:0:npu:0:disk:0".to_string(),
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
    fn prepare_accepts_materialized_openvino_directory() {
        let backend = OpenVinoBackend::default();
        let root = unique_temp_dir("openvino-ready");
        fs::create_dir_all(&root).expect("create root");
        fs::write(root.join("openvino_model.xml"), "<xml/>").expect("xml");
        fs::write(root.join("openvino_model.bin"), [0_u8; 32]).expect("bin");

        let mut model = demo_model();
        model.path = root.clone();

        let prepared = backend.prepare(&model, &openvino_plan()).expect("prepared");
        assert_eq!(prepared.backend, "openvino");
        assert_eq!(prepared.residency, PreparedResidency::DiskBacked);

        let _ = fs::remove_dir_all(root);
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
    fn prepare_records_placeholder_models_but_execute_still_rejects_without_fallback_override() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        env::remove_var("LOCI_OPENVINO_ALLOW_FALLBACK");

        let backend = OpenVinoBackend::default();
        let plan = openvino_plan();
        let prepared = backend
            .prepare(&demo_model(), &plan)
            .expect("placeholder-backed session is still cached for diagnostics");

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
            .expect_err("fallback should still be rejected by default");

        assert!(error
            .message
            .contains("OpenVINO real execution is unavailable"));
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

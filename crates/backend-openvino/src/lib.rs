use loci_protocol::{
    AcceleratorKind, Backend, BackendDescriptor, BackendError, BackendExecutionProfile,
    BackendOutput, BackendResult, BackendTelemetry, DeviceDescriptor, ExecutionPlan,
    HardwareTopology, ModelDescriptor, ModelFormat, OpenVinoExecutionMode,
    OpenVinoExecutionProfile, PipelineStage, PlacementDecision, PowerState, PreparedModel,
    PreparedResidency, SessionRequest, ThermalState,
};

pub fn boxed_backend() -> Box<dyn Backend> {
    Box::new(OpenVinoBackend::default())
}

#[derive(Default)]
struct OpenVinoBackend {
    runtime: OpenVinoRuntime,
}

#[derive(Default)]
struct OpenVinoRuntime;

impl Backend for OpenVinoBackend {
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "openvino".to_string(),
            supports_cpu: true,
            supports_gpu: true,
            supports_npu: true,
            supports_disk_tiering: true,
            supports_paged_kv: true,
        }
    }

    fn discover_topology(&self) -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "host-cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    memory_bytes: Some(16 * 1024 * 1024 * 1024),
                    compute_units: Some(16),
                    power_watts: Some(25.0),
                },
                DeviceDescriptor {
                    id: "gpu:0".to_string(),
                    name: "integrated-gpu".to_string(),
                    kind: AcceleratorKind::Gpu,
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    compute_units: Some(128),
                    power_watts: Some(20.0),
                },
                DeviceDescriptor {
                    id: "npu:0".to_string(),
                    name: "integrated-npu".to_string(),
                    kind: AcceleratorKind::Npu,
                    memory_bytes: Some(2 * 1024 * 1024 * 1024),
                    compute_units: Some(1),
                    power_watts: Some(5.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "nvme-tier".to_string(),
                    kind: AcceleratorKind::Disk,
                    memory_bytes: Some(256 * 1024 * 1024 * 1024),
                    compute_units: None,
                    power_watts: None,
                },
            ],
            power: PowerState {
                battery_powered: false,
                battery_percent: None,
                thermal_state: ThermalState::Nominal,
                power_budget_watts: Some(45),
            },
        }
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
        let mode = match profile.execution_mode {
            OpenVinoExecutionMode::Hetero => "hetero",
            OpenVinoExecutionMode::NpuFirst => "npu-first",
        };
        let devices = profile.hetero_devices.join(">");
        let prefill = placement_summary(plan, PipelineStage::Prefill);
        let decode = placement_summary(plan, PipelineStage::Decode);
        let kv = placement_summary(plan, PipelineStage::KvCache);
        let weights = placement_summary(plan, PipelineStage::Weights);
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

        Ok(BackendOutput {
            text: format!(
                "[openvino:{}] mode={} devices={} prefill={} decode={} kv={} weights={} prepared={} route={} {} prompt=`{}`",
                model.name,
                mode,
                devices,
                prefill,
                decode,
                kv,
                weights,
                prepared.session_key,
                plan.route.reason,
                spill,
                request.prompt.trim()
            ),
            telemetry: BackendTelemetry {
                estimated_prefill_ms: estimate_prefill_ms(profile, model, plan),
                estimated_decode_ms: estimate_decode_ms(profile, plan),
                generated_tokens: request.max_tokens.min(128),
            },
        })
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
        BackendExecutionProfile, CandleExecutionProfile, CandleTensorResidency, ExecutionPlan,
        GenericExecutionProfile, KvCachePlan, PlacementDecision, RouteDecision, TieredOffloadPlan,
        TieredOffloadPolicy, TieredPlacementPercentages,
    };
    use std::path::PathBuf;

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
}

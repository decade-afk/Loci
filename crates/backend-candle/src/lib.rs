use loci_protocol::{
    AcceleratorKind, Backend, BackendDescriptor, BackendError, BackendExecutionProfile,
    BackendOutput, BackendResult, BackendTelemetry, CandleExecutionProfile, CandleTensorResidency,
    DeviceDescriptor, ExecutionPlan, HardwareTopology, ModelDescriptor, ModelFormat, PipelineStage,
    PlacementDecision, PowerState, PreparedModel, PreparedResidency, SessionRequest, ThermalState,
};

pub fn boxed_backend() -> Box<dyn Backend> {
    Box::new(CandleBackend::default())
}

#[derive(Default)]
struct CandleBackend {
    runtime: CandleRuntime,
}

#[derive(Default)]
struct CandleRuntime;

impl Backend for CandleBackend {
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor {
            name: "candle".to_string(),
            supports_cpu: true,
            supports_gpu: true,
            supports_npu: false,
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
                    name: "generic-gpu".to_string(),
                    kind: AcceleratorKind::Gpu,
                    memory_bytes: Some(12 * 1024 * 1024 * 1024),
                    compute_units: Some(96),
                    power_watts: Some(35.0),
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
                power_budget_watts: Some(60),
            },
        }
    }

    fn supports_model(&self, model: &ModelDescriptor) -> bool {
        matches!(
            model.inferred_format(),
            ModelFormat::Gguf
                | ModelFormat::SafeTensors
                | ModelFormat::PytorchBin
                | ModelFormat::Directory
        )
    }

    fn prepare(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
    ) -> BackendResult<PreparedModel> {
        let profile = candle_profile(plan)?;
        self.runtime.prepare_session(model, plan, profile)
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

        let profile = candle_profile(plan)?;
        if prepared.session_key != profile.session_key {
            return Err(BackendError {
                message: format!(
                    "prepared Candle session `{}` does not match plan `{}`",
                    prepared.session_key, profile.session_key
                ),
            });
        }

        self.runtime
            .run_session(prepared, model, request, plan, profile)
    }
}

impl CandleRuntime {
    fn prepare_session(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
        profile: &CandleExecutionProfile,
    ) -> BackendResult<PreparedModel> {
        validate_candle_plan(plan, profile)?;
        let residency = derive_residency(plan, profile);

        Ok(PreparedModel {
            model_name: model.name.clone(),
            backend: "candle".to_string(),
            session_key: profile.session_key.clone(),
            residency,
            estimated_memory_bytes: estimate_resident_memory_bytes(model, plan, residency),
        })
    }

    fn run_session(
        &self,
        prepared: &PreparedModel,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
        profile: &CandleExecutionProfile,
    ) -> BackendResult<BackendOutput> {
        validate_candle_plan(plan, profile)?;
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
        let prefill = placement_summary(plan, PipelineStage::Prefill);
        let decode = placement_summary(plan, PipelineStage::Decode);
        let kv = placement_summary(plan, PipelineStage::KvCache);
        let weights = placement_summary(plan, PipelineStage::Weights);

        Ok(BackendOutput {
            text: format!(
                "[candle:{}] prefill={} decode={} kv={} weights={} residency={:?} prepared={} {} prompt=`{}`",
                model.name,
                prefill,
                decode,
                kv,
                weights,
                profile.tensor_residency,
                prepared.session_key,
                spill,
                request.prompt.trim()
            ),
            telemetry: BackendTelemetry {
                estimated_prefill_ms: estimate_prefill_ms(profile, plan),
                estimated_decode_ms: estimate_decode_ms(profile, plan),
                generated_tokens: request.max_tokens.min(128),
            },
        })
    }
}

fn candle_profile(plan: &ExecutionPlan) -> BackendResult<&CandleExecutionProfile> {
    match &plan.backend_profile {
        BackendExecutionProfile::Candle(profile) => Ok(profile),
        _ => Err(BackendError {
            message: "execution plan is missing a Candle backend profile".to_string(),
        }),
    }
}

fn validate_candle_plan(
    plan: &ExecutionPlan,
    profile: &CandleExecutionProfile,
) -> BackendResult<()> {
    if plan
        .placements
        .iter()
        .any(|placement| placement.target == AcceleratorKind::Npu)
    {
        return Err(BackendError {
            message: "Candle fallback does not support NPU placements".to_string(),
        });
    }

    if profile.prefill_device.starts_with("npu:")
        || profile.decode_device.starts_with("npu:")
        || profile.kv_cache_device.starts_with("npu:")
    {
        return Err(BackendError {
            message: "Candle execution profile cannot reference NPU devices".to_string(),
        });
    }

    let uses_disk = plan
        .placements
        .iter()
        .any(|placement| placement.target == AcceleratorKind::Disk);
    if !uses_disk && matches!(profile.tensor_residency, CandleTensorResidency::Hybrid) {
        return Err(BackendError {
            message: "Candle hybrid residency requires a disk-backed placement".to_string(),
        });
    }

    Ok(())
}

fn derive_residency(plan: &ExecutionPlan, profile: &CandleExecutionProfile) -> PreparedResidency {
    let weights_on_disk =
        placement_target(plan, PipelineStage::Weights) == Some(AcceleratorKind::Disk);
    let kv_on_disk = placement_target(plan, PipelineStage::KvCache) == Some(AcceleratorKind::Disk);

    match profile.tensor_residency {
        CandleTensorResidency::MemoryOnly => PreparedResidency::Memory,
        CandleTensorResidency::Hybrid => {
            if weights_on_disk && kv_on_disk {
                PreparedResidency::DiskBacked
            } else {
                PreparedResidency::Hybrid
            }
        }
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
        Some(AcceleratorKind::Disk) => kv_bytes / 3,
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

fn estimate_prefill_ms(profile: &CandleExecutionProfile, plan: &ExecutionPlan) -> u64 {
    if profile.prefill_device.starts_with("gpu:") {
        14 + weight_penalty(plan)
    } else {
        22 + weight_penalty(plan)
    }
}

fn estimate_decode_ms(profile: &CandleExecutionProfile, plan: &ExecutionPlan) -> u64 {
    let kv_penalty = match placement_target(plan, PipelineStage::KvCache) {
        Some(AcceleratorKind::Disk) => 6,
        Some(AcceleratorKind::Cpu) => 2,
        _ => 0,
    };
    if profile.decode_device.starts_with("gpu:") {
        9 + kv_penalty
    } else {
        13 + kv_penalty
    }
}

fn weight_penalty(plan: &ExecutionPlan) -> u64 {
    match placement_target(plan, PipelineStage::Weights) {
        Some(AcceleratorKind::Disk) => 5,
        Some(AcceleratorKind::Cpu) => 2,
        _ => 0,
    }
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
        BackendExecutionProfile, ExecutionPlan, GenericExecutionProfile, KvCachePlan,
        PlacementDecision, RouteDecision, TieredOffloadPlan, TieredOffloadPolicy,
        TieredPlacementPercentages,
    };
    use std::path::PathBuf;

    fn demo_model() -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.gguf"),
            architecture: "llama".to_string(),
            memory_bytes: Some(2 * 1024 * 1024 * 1024),
            parameter_count: Some(1_000_000_000),
            context_length: Some(8192),
            preferred_backend: Some("candle".to_string()),
        }
    }

    fn candle_plan() -> ExecutionPlan {
        ExecutionPlan {
            backend: "candle".to_string(),
            route: RouteDecision {
                selected_model: "demo".to_string(),
                reason: "fallback".to_string(),
                alternatives: Vec::new(),
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
                    target: AcceleratorKind::Cpu,
                    device_id: Some("cpu:0".to_string()),
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
                tiered: true,
            },
            tiered_offload: Some(TieredOffloadPlan {
                spill_bytes: 512 << 20,
                prefetch_window_bytes: 64 << 20,
                target_device: "disk:0".to_string(),
                profile: loci_protocol::TieredOffloadProfile::Balanced,
                policy: TieredOffloadPolicy {
                    weights: TieredPlacementPercentages {
                        gpu_percent: 20,
                        cpu_percent: 40,
                        disk_percent: 40,
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
            backend_profile: BackendExecutionProfile::Candle(CandleExecutionProfile {
                session_key: "candle:gpu:0:cpu:0".to_string(),
                prefill_device: "gpu:0".to_string(),
                decode_device: "cpu:0".to_string(),
                kv_cache_device: "disk:0".to_string(),
                tensor_residency: CandleTensorResidency::Hybrid,
                fallback_reason: "fallback".to_string(),
            }),
        }
    }

    #[test]
    fn prepare_uses_plan_profile() {
        let backend = CandleBackend::default();
        let prepared = backend
            .prepare(&demo_model(), &candle_plan())
            .expect("prepared");

        assert_eq!(prepared.backend, "candle");
        assert_eq!(prepared.session_key, "candle:gpu:0:cpu:0");
        assert_eq!(prepared.residency, PreparedResidency::DiskBacked);
    }

    #[test]
    fn execute_rejects_non_candle_profile() {
        let backend = CandleBackend::default();
        let prepared = PreparedModel {
            model_name: "demo".to_string(),
            backend: "candle".to_string(),
            session_key: "candle:gpu:0:cpu:0".to_string(),
            residency: PreparedResidency::Memory,
            estimated_memory_bytes: None,
        };
        let error = backend
            .execute(
                &prepared,
                &demo_model(),
                &SessionRequest {
                    prompt: "hello".to_string(),
                    max_tokens: 16,
                    temperature: 0.2,
                    target_model: None,
                    structured_output: false,
                    tool_calling: false,
                },
                &ExecutionPlan {
                    backend: "candle".to_string(),
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
                },
            )
            .expect_err("error");

        assert!(error.message.contains("Candle backend profile"));
    }

    #[test]
    fn prepare_rejects_npu_placements() {
        let backend = CandleBackend::default();
        let mut plan = candle_plan();
        plan.placements[1].target = AcceleratorKind::Npu;

        let error = backend.prepare(&demo_model(), &plan).expect_err("error");
        assert!(error.message.contains("NPU"));
    }
}

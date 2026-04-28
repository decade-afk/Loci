use crate::config::EngineConfig;
use crate::error::{LociError, Result};
use loci_protocol::{
    AcceleratorKind, Backend, BackendDescriptor, BackendExecutionProfile, CandleExecutionProfile,
    CandleTensorResidency, DeviceDescriptor, ExecutionPlan, GenericExecutionProfile,
    HardwareTopology, KvCachePlan, ModelDescriptor, OpenVinoExecutionMode,
    OpenVinoExecutionProfile, PipelineStage, PlacementDecision, PowerState, RouteDecision,
    SessionRequest, ThermalState, TieredOffloadPlan,
};

pub fn merge_topologies(backends: &[Box<dyn Backend>]) -> HardwareTopology {
    let mut devices = Vec::<DeviceDescriptor>::new();
    let mut thermal_state = ThermalState::Nominal;
    let mut battery_powered = false;
    let mut battery_percent: Option<u8> = None;
    let mut power_budget_watts = None;

    for backend in backends {
        let topology = backend.discover_topology();
        // Merge backend-reported devices into a single logical topology and keep
        // the most conservative power state for planning decisions.
        for device in topology.devices {
            let duplicate = devices.iter().any(|existing| {
                existing.kind == device.kind
                    && existing.id == device.id
                    && existing.name == device.name
            });
            if !duplicate {
                devices.push(device);
            }
        }
        thermal_state = thermal_state.max(topology.power.thermal_state);
        battery_powered |= topology.power.battery_powered;
        battery_percent = match (battery_percent, topology.power.battery_percent) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        power_budget_watts = power_budget_watts.or(topology.power.power_budget_watts);
    }

    HardwareTopology {
        devices,
        power: PowerState {
            battery_powered,
            battery_percent,
            thermal_state,
            power_budget_watts,
        },
    }
}

pub fn choose_backend<'a>(
    backends: &'a [Box<dyn Backend>],
    model: &ModelDescriptor,
    preferred_backend: Option<&str>,
) -> Result<&'a dyn Backend> {
    for preferred in [preferred_backend, model.preferred_backend.as_deref()] {
        if let Some(name) = preferred {
            if let Some(backend) = backends.iter().find(|backend| {
                let descriptor = backend.descriptor();
                descriptor.name == name && backend.supports_model(model)
            }) {
                return Ok(backend.as_ref());
            }
        }
    }

    if let Some(backend) = backends.iter().find(|backend| {
        let descriptor = backend.descriptor();
        descriptor.supports_npu && backend.supports_model(model)
    }) {
        return Ok(backend.as_ref());
    }

    backends
        .iter()
        .find(|backend| backend.supports_model(model))
        .map(|backend| backend.as_ref())
        .ok_or_else(|| LociError::NoCompatibleBackend {
            model: model.name.clone(),
            format: model.inferred_format().as_str().to_string(),
        })
}

pub fn build_plan(
    config: &EngineConfig,
    backend: &BackendDescriptor,
    topology: &HardwareTopology,
    model: &ModelDescriptor,
    request: &SessionRequest,
    route: RouteDecision,
    kv_cache: KvCachePlan,
    tiered_offload: Option<TieredOffloadPlan>,
) -> ExecutionPlan {
    let prefill_target = pick_prefill_target(topology, backend);
    let decode_target = pick_decode_target(topology, backend, prefill_target);
    let kv_target = pick_kv_target(
        topology,
        backend,
        decode_target,
        &kv_cache,
        tiered_offload.as_ref(),
    );
    let weights_target = pick_weights_target(topology, backend, tiered_offload.as_ref());

    // The planner intentionally separates throughput-biased prefill from
    // power-biased decode so the engine can model heterogeneous execution.
    let mut placements = vec![
        PlacementDecision {
            stage: PipelineStage::Load,
            target: AcceleratorKind::Cpu,
            device_id: None,
            memory_bytes: model.memory_bytes,
            rationale: "model metadata is normalized on the CPU control path".to_string(),
        },
        PlacementDecision {
            stage: PipelineStage::Prefill,
            target: prefill_target,
            device_id: preferred_device_id(topology, prefill_target),
            memory_bytes: model.memory_bytes,
            rationale: prefill_reason(config, topology, prefill_target),
        },
        PlacementDecision {
            stage: PipelineStage::Decode,
            target: decode_target,
            device_id: preferred_device_id(topology, decode_target),
            memory_bytes: Some(request.max_tokens as u64 * 1024),
            rationale: decode_reason(topology, decode_target),
        },
        PlacementDecision {
            stage: PipelineStage::KvCache,
            target: kv_target,
            device_id: preferred_device_id(topology, kv_target),
            memory_bytes: kv_cache.max_cache_bytes,
            rationale: kv_reason(kv_target, decode_target, &kv_cache, tiered_offload.as_ref()),
        },
        PlacementDecision {
            stage: PipelineStage::Sampling,
            target: AcceleratorKind::Cpu,
            device_id: None,
            memory_bytes: Some(8 * 1024 * 1024),
            rationale: "sampling and response assembly stay on the CPU orchestration path"
                .to_string(),
        },
    ];

    if tiered_offload.is_some() {
        placements.push(PlacementDecision {
            stage: PipelineStage::Weights,
            target: weights_target,
            device_id: weight_device_id(topology, tiered_offload.as_ref(), weights_target),
            memory_bytes: tiered_offload.as_ref().map(|plan| plan.spill_bytes),
            rationale: weights_reason(weights_target, tiered_offload.as_ref()),
        });
    }

    let backend_profile = build_backend_profile(backend, topology, &placements);

    ExecutionPlan {
        backend: backend.name.clone(),
        route,
        placements,
        kv_cache,
        tiered_offload,
        backend_profile,
    }
}

fn build_backend_profile(
    backend: &BackendDescriptor,
    topology: &HardwareTopology,
    placements: &[PlacementDecision],
) -> BackendExecutionProfile {
    match backend.name.as_str() {
        "openvino" => {
            BackendExecutionProfile::OpenVino(build_openvino_profile(topology, placements))
        }
        "candle" => BackendExecutionProfile::Candle(build_candle_profile(topology, placements)),
        name => BackendExecutionProfile::Generic(GenericExecutionProfile {
            session_key: format!("{name}-session"),
            summary: format!("generic execution profile for backend `{name}`"),
        }),
    }
}

fn build_openvino_profile(
    topology: &HardwareTopology,
    placements: &[PlacementDecision],
) -> OpenVinoExecutionProfile {
    let prefill_device = placement_device_id(placements, PipelineStage::Prefill);
    let decode_device = placement_device_id(placements, PipelineStage::Decode);
    let kv_cache_device = placement_device_id(placements, PipelineStage::KvCache);
    let weights_device = placement_device_id(placements, PipelineStage::Weights);
    let hetero_devices = hetero_device_names(topology, placements);
    let decode_uses_npu = placements.iter().any(|placement| {
        placement.stage == PipelineStage::Decode && placement.target == AcceleratorKind::Npu
    });

    OpenVinoExecutionProfile {
        session_key: format!(
            "ov:{}:{}",
            prefill_device.as_deref().unwrap_or("cpu"),
            decode_device.as_deref().unwrap_or("cpu")
        ),
        execution_mode: if decode_uses_npu {
            OpenVinoExecutionMode::NpuFirst
        } else {
            OpenVinoExecutionMode::Hetero
        },
        genai_pipeline: true,
        hetero_devices,
        prefill_device,
        decode_device,
        kv_cache_device,
        weights_device,
        dynamic_reoffload: topology.power.thermal_state >= ThermalState::Hot,
    }
}

fn build_candle_profile(
    _topology: &HardwareTopology,
    placements: &[PlacementDecision],
) -> CandleExecutionProfile {
    let prefill_device = placement_device_id(placements, PipelineStage::Prefill)
        .unwrap_or_else(|| "cpu:0".to_string());
    let decode_device = placement_device_id(placements, PipelineStage::Decode)
        .unwrap_or_else(|| prefill_device.clone());
    let kv_cache_device = placement_device_id(placements, PipelineStage::KvCache)
        .unwrap_or_else(|| decode_device.clone());
    let tensor_residency = if placements
        .iter()
        .any(|placement| placement.target == AcceleratorKind::Disk)
    {
        CandleTensorResidency::Hybrid
    } else {
        CandleTensorResidency::MemoryOnly
    };

    CandleExecutionProfile {
        session_key: format!("candle:{prefill_device}:{decode_device}"),
        prefill_device,
        decode_device,
        kv_cache_device,
        tensor_residency,
        fallback_reason: "pure Rust fallback path for hosts without an OpenVINO NPU route"
            .to_string(),
    }
}

fn pick_prefill_target(
    topology: &HardwareTopology,
    backend: &BackendDescriptor,
) -> AcceleratorKind {
    let hot = matches!(
        topology.power.thermal_state,
        ThermalState::Hot | ThermalState::Critical
    );
    let low_battery = topology
        .power
        .battery_percent
        .map(|value| value < 20)
        .unwrap_or(false);

    if !hot && !low_battery && backend.supports_gpu && has_kind(topology, AcceleratorKind::Gpu) {
        AcceleratorKind::Gpu
    } else if backend.supports_cpu && has_kind(topology, AcceleratorKind::Cpu) {
        AcceleratorKind::Cpu
    } else if backend.supports_npu && has_kind(topology, AcceleratorKind::Npu) {
        AcceleratorKind::Npu
    } else {
        AcceleratorKind::Cpu
    }
}

fn pick_decode_target(
    topology: &HardwareTopology,
    backend: &BackendDescriptor,
    fallback: AcceleratorKind,
) -> AcceleratorKind {
    if backend.supports_npu && has_kind(topology, AcceleratorKind::Npu) {
        AcceleratorKind::Npu
    } else if backend.supports_gpu
        && has_kind(topology, AcceleratorKind::Gpu)
        && fallback != AcceleratorKind::Cpu
    {
        AcceleratorKind::Gpu
    } else {
        fallback
    }
}

fn pick_kv_target(
    topology: &HardwareTopology,
    backend: &BackendDescriptor,
    decode_target: AcceleratorKind,
    kv_cache: &KvCachePlan,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> AcceleratorKind {
    let Some(plan) = tiered_offload else {
        return decode_target;
    };
    if !kv_cache.tiered {
        return decode_target;
    }

    let policy = plan.policy.kv_cache;
    if policy.disk_percent >= 40 && has_kind(topology, AcceleratorKind::Disk) {
        AcceleratorKind::Disk
    } else if policy.cpu_percent >= policy.gpu_percent
        && backend.supports_cpu
        && has_kind(topology, AcceleratorKind::Cpu)
    {
        AcceleratorKind::Cpu
    } else {
        decode_target
    }
}

fn pick_weights_target(
    topology: &HardwareTopology,
    backend: &BackendDescriptor,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> AcceleratorKind {
    let Some(plan) = tiered_offload else {
        return AcceleratorKind::Cpu;
    };

    let policy = plan.policy.weights;
    if policy.disk_percent >= policy.cpu_percent
        && policy.disk_percent >= policy.gpu_percent
        && has_kind(topology, AcceleratorKind::Disk)
    {
        AcceleratorKind::Disk
    } else if policy.gpu_percent >= policy.cpu_percent
        && backend.supports_gpu
        && has_kind(topology, AcceleratorKind::Gpu)
    {
        AcceleratorKind::Gpu
    } else {
        AcceleratorKind::Cpu
    }
}

fn has_kind(topology: &HardwareTopology, kind: AcceleratorKind) -> bool {
    topology.devices.iter().any(|device| device.kind == kind)
}

fn preferred_device_id(topology: &HardwareTopology, kind: AcceleratorKind) -> Option<String> {
    topology
        .devices
        .iter()
        .find(|device| device.kind == kind)
        .map(|device| device.id.clone())
}

fn weight_device_id(
    topology: &HardwareTopology,
    tiered_offload: Option<&TieredOffloadPlan>,
    target: AcceleratorKind,
) -> Option<String> {
    if target == AcceleratorKind::Disk {
        tiered_offload.map(|plan| plan.target_device.clone())
    } else {
        preferred_device_id(topology, target)
    }
}

fn placement_device_id(placements: &[PlacementDecision], stage: PipelineStage) -> Option<String> {
    placements
        .iter()
        .find(|placement| placement.stage == stage)
        .and_then(|placement| placement.device_id.clone())
}

fn hetero_device_names(
    topology: &HardwareTopology,
    placements: &[PlacementDecision],
) -> Vec<String> {
    let mut devices = Vec::new();
    for placement in placements {
        if placement.target == AcceleratorKind::Disk {
            continue;
        }

        if let Some(device_id) = &placement.device_id {
            if let Some(device) = topology
                .devices
                .iter()
                .find(|candidate| candidate.id == *device_id)
            {
                let name = device.kind_label();
                if !devices.contains(&name) {
                    devices.push(name);
                }
                continue;
            }
        }

        let name = placement.target.kind_label();
        if !devices.contains(&name) {
            devices.push(name);
        }
    }
    devices
}

fn prefill_reason(
    config: &EngineConfig,
    topology: &HardwareTopology,
    target: AcceleratorKind,
) -> String {
    #[cfg(feature = "power-aware")]
    if config.routing.enabled || topology.power.battery_powered {
        return match target {
            AcceleratorKind::Gpu => {
                "prefill uses the GPU for throughput while the power budget remains healthy"
                    .to_string()
            }
            AcceleratorKind::Cpu => {
                "prefill falls back to CPU because the power-aware planner avoided GPU pressure"
                    .to_string()
            }
            _ => "prefill target selected by the planner".to_string(),
        };
    }

    match target {
        AcceleratorKind::Gpu => "prefill is GPU-biased for throughput".to_string(),
        AcceleratorKind::Cpu => {
            "prefill stays on CPU because no better accelerator is available".to_string()
        }
        _ => "prefill target selected by the planner".to_string(),
    }
}

fn decode_reason(topology: &HardwareTopology, target: AcceleratorKind) -> String {
    match target {
        AcceleratorKind::Npu => {
            "decode is pinned to the NPU to minimize power draw on token-by-token generation"
                .to_string()
        }
        AcceleratorKind::Gpu => "decode stays on GPU because no NPU path is available".to_string(),
        AcceleratorKind::Cpu => {
            if topology.devices.is_empty() {
                "decode stays on CPU because no hardware topology was discovered".to_string()
            } else {
                "decode falls back to CPU because no NPU or GPU decode path is available"
                    .to_string()
            }
        }
        AcceleratorKind::Disk => "decode never directly executes on disk".to_string(),
    }
}

fn kv_reason(
    target: AcceleratorKind,
    decode_target: AcceleratorKind,
    kv_cache: &KvCachePlan,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> String {
    if !kv_cache.tiered {
        return "kv cache remains colocated with the decode path".to_string();
    }

    match target {
        AcceleratorKind::Disk => "paged kv cache spills cold pages to disk-backed storage"
            .to_string(),
        AcceleratorKind::Cpu => format!(
            "kv cache stays on CPU memory while decode remains on {} to reduce accelerator pressure",
            decode_target.kind_label()
        ),
        _ => {
            if let Some(plan) = tiered_offload {
                format!(
                    "kv cache remains hot on {} while spill profile `{}` manages colder state",
                    target.kind_label(),
                    offload_profile_label(plan)
                )
            } else {
                "kv cache remains colocated with the decode path".to_string()
            }
        }
    }
}

fn weights_reason(target: AcceleratorKind, tiered_offload: Option<&TieredOffloadPlan>) -> String {
    let Some(plan) = tiered_offload else {
        return "weights remain resident in memory".to_string();
    };

    match target {
        AcceleratorKind::Disk => {
            "cold weights are spillable to disk through the tiered offload manager".to_string()
        }
        AcceleratorKind::Gpu => format!(
            "hot weights stay primarily on GPU while {} bytes remain spillable to {}",
            plan.spill_bytes, plan.target_device
        ),
        AcceleratorKind::Cpu => format!(
            "weights are anchored in CPU memory with disk spill available on {}",
            plan.target_device
        ),
        AcceleratorKind::Npu => "weights are not directly anchored on the NPU".to_string(),
    }
}

fn offload_profile_label(plan: &TieredOffloadPlan) -> &'static str {
    match plan.profile {
        loci_protocol::TieredOffloadProfile::Auto => "auto",
        loci_protocol::TieredOffloadProfile::GpuResident => "gpu_resident",
        loci_protocol::TieredOffloadProfile::Balanced => "balanced",
        loci_protocol::TieredOffloadProfile::DiskHeavy => "disk_heavy",
    }
}

trait AcceleratorKindLabel {
    fn kind_label(&self) -> String;
}

impl AcceleratorKindLabel for AcceleratorKind {
    fn kind_label(&self) -> String {
        match self {
            AcceleratorKind::Cpu => "CPU".to_string(),
            AcceleratorKind::Gpu => "GPU".to_string(),
            AcceleratorKind::Npu => "NPU".to_string(),
            AcceleratorKind::Disk => "DISK".to_string(),
        }
    }
}

impl AcceleratorKindLabel for DeviceDescriptor {
    fn kind_label(&self) -> String {
        self.kind.kind_label()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use loci_protocol::{
        BackendDescriptor, KvCachePlan, RouteDecision, TieredOffloadPolicy, TieredOffloadProfile,
        TieredPlacementPercentages,
    };
    use std::path::PathBuf;

    fn backend(name: &str, supports_npu: bool) -> BackendDescriptor {
        BackendDescriptor {
            name: name.to_string(),
            supports_cpu: true,
            supports_gpu: true,
            supports_npu,
            supports_disk_tiering: true,
            supports_paged_kv: true,
        }
    }

    fn topology() -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    memory_bytes: Some(16 * 1024 * 1024 * 1024),
                    compute_units: Some(16),
                    power_watts: Some(20.0),
                },
                DeviceDescriptor {
                    id: "gpu:0".to_string(),
                    name: "gpu".to_string(),
                    kind: AcceleratorKind::Gpu,
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    compute_units: Some(128),
                    power_watts: Some(30.0),
                },
                DeviceDescriptor {
                    id: "npu:0".to_string(),
                    name: "npu".to_string(),
                    kind: AcceleratorKind::Npu,
                    memory_bytes: Some(2 * 1024 * 1024 * 1024),
                    compute_units: Some(1),
                    power_watts: Some(5.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "disk".to_string(),
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

    fn model() -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.gguf"),
            architecture: "llama".to_string(),
            memory_bytes: Some(16 * 1024 * 1024 * 1024),
            parameter_count: Some(8_000_000_000),
            context_length: Some(8192),
            preferred_backend: None,
        }
    }

    fn request() -> SessionRequest {
        SessionRequest {
            prompt: "hello".to_string(),
            max_tokens: 128,
            temperature: 0.2,
            target_model: Some("demo".to_string()),
            structured_output: false,
            tool_calling: false,
        }
    }

    fn route() -> RouteDecision {
        RouteDecision {
            selected_model: "demo".to_string(),
            reason: "explicit".to_string(),
            alternatives: Vec::new(),
        }
    }

    #[test]
    fn build_plan_uses_disk_for_kv_when_disk_heavy_policy_demands_it() {
        let plan = build_plan(
            &EngineConfig::default(),
            &backend("openvino", true),
            &topology(),
            &model(),
            &request(),
            route(),
            KvCachePlan {
                strategy: "paged-prefix-cache".to_string(),
                shared_across_models: true,
                page_size_bytes: Some(1 << 20),
                block_size_tokens: Some(16),
                max_cache_bytes: Some(512 << 20),
                type_k: Some("q8_0".to_string()),
                type_v: Some("q8_0".to_string()),
                tiered: true,
            },
            Some(TieredOffloadPlan {
                spill_bytes: 8 << 30,
                prefetch_window_bytes: 128 << 20,
                target_device: "disk:0".to_string(),
                profile: TieredOffloadProfile::DiskHeavy,
                policy: TieredOffloadPolicy {
                    weights: TieredPlacementPercentages {
                        gpu_percent: 10,
                        cpu_percent: 30,
                        disk_percent: 60,
                    },
                    kv_cache: TieredPlacementPercentages {
                        gpu_percent: 10,
                        cpu_percent: 30,
                        disk_percent: 60,
                    },
                    activations: TieredPlacementPercentages {
                        gpu_percent: 40,
                        cpu_percent: 60,
                        disk_percent: 0,
                    },
                    cpu_cache_compute: true,
                    compress_weights: true,
                    compress_kv_cache: true,
                },
            }),
        );

        assert!(plan.placements.iter().any(|placement| {
            placement.stage == PipelineStage::KvCache
                && placement.target == AcceleratorKind::Disk
                && placement.device_id.as_deref() == Some("disk:0")
        }));
        assert!(plan.placements.iter().any(|placement| {
            placement.stage == PipelineStage::Weights && placement.target == AcceleratorKind::Disk
        }));
    }

    #[test]
    fn build_plan_keeps_kv_on_cpu_when_balanced_policy_avoids_disk_dominance() {
        let plan = build_plan(
            &EngineConfig::default(),
            &backend("candle", false),
            &topology(),
            &model(),
            &request(),
            route(),
            KvCachePlan {
                strategy: "paged-prefix-cache".to_string(),
                shared_across_models: true,
                page_size_bytes: Some(1 << 20),
                block_size_tokens: Some(16),
                max_cache_bytes: Some(512 << 20),
                type_k: Some("q8_0".to_string()),
                type_v: Some("q8_0".to_string()),
                tiered: true,
            },
            Some(TieredOffloadPlan {
                spill_bytes: 4 << 30,
                prefetch_window_bytes: 128 << 20,
                target_device: "disk:0".to_string(),
                profile: TieredOffloadProfile::Balanced,
                policy: TieredOffloadPolicy {
                    weights: TieredPlacementPercentages {
                        gpu_percent: 30,
                        cpu_percent: 50,
                        disk_percent: 20,
                    },
                    kv_cache: TieredPlacementPercentages {
                        gpu_percent: 20,
                        cpu_percent: 60,
                        disk_percent: 20,
                    },
                    activations: TieredPlacementPercentages {
                        gpu_percent: 60,
                        cpu_percent: 40,
                        disk_percent: 0,
                    },
                    cpu_cache_compute: true,
                    compress_weights: false,
                    compress_kv_cache: false,
                },
            }),
        );

        assert!(plan.placements.iter().any(|placement| {
            placement.stage == PipelineStage::KvCache
                && placement.target == AcceleratorKind::Cpu
                && placement.device_id.as_deref() == Some("cpu:0")
        }));
        assert!(plan.placements.iter().any(|placement| {
            placement.stage == PipelineStage::Weights
                && placement.target == AcceleratorKind::Cpu
                && placement.device_id.as_deref() == Some("cpu:0")
        }));
        assert!(matches!(
            plan.backend_profile,
            BackendExecutionProfile::Candle(_)
        ));
    }
}

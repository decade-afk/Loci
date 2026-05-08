use crate::config::EngineConfig;
use crate::snapshot::HostCapabilitySnapshot;
use loci_protocol::{
    AcceleratorKind, BackendDescriptor, DeviceDescriptor, HardwareTopology, KvCachePlan,
    ModelDescriptor, PipelineStage, PlacementDecision, ThermalState, TieredOffloadPlan,
};

/// Builds the coarse-grained stage placement decisions consumed by lowering and execution.
pub(super) fn build_stage_placements(
    config: &EngineConfig,
    topology: &HardwareTopology,
    host: &HostCapabilitySnapshot,
    model: &ModelDescriptor,
    backend: &BackendDescriptor,
    request_max_tokens: u32,
    kv_cache: &KvCachePlan,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> Vec<PlacementDecision> {
    let prefill_target = pick_prefill_target(topology, backend);
    let decode_target = pick_decode_target(topology, backend, prefill_target);
    let kv_target = pick_kv_target(
        topology,
        host,
        model,
        backend,
        decode_target,
        kv_cache,
        tiered_offload,
    );
    let weights_target = pick_weights_target(topology, host, model, backend, tiered_offload);

    vec![
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
            memory_bytes: Some(request_max_tokens as u64 * 1024),
            rationale: decode_reason(topology, decode_target),
        },
        PlacementDecision {
            stage: PipelineStage::KvCache,
            target: kv_target,
            device_id: preferred_device_id(topology, kv_target),
            memory_bytes: kv_cache.max_cache_bytes,
            rationale: kv_reason(kv_target, decode_target, kv_cache, tiered_offload),
        },
        PlacementDecision {
            stage: PipelineStage::Sampling,
            target: AcceleratorKind::Cpu,
            device_id: None,
            memory_bytes: Some(8 * 1024 * 1024),
            rationale: "sampling and response assembly stay on the CPU orchestration path"
                .to_string(),
        },
        PlacementDecision {
            stage: PipelineStage::Weights,
            target: weights_target,
            device_id: weight_device_id(topology, tiered_offload, weights_target),
            memory_bytes: weights_memory_bytes(model, tiered_offload, weights_target),
            rationale: weights_reason(topology, weights_target, tiered_offload),
        },
    ]
}

/// Chooses the preferred prefill target under throughput and power constraints.
fn pick_prefill_target(
    topology: &HardwareTopology,
    backend: &BackendDescriptor,
) -> AcceleratorKind {
    let hot = has_thermal_pressure(topology);
    let low_battery = has_low_battery(topology);
    let tight_budget = has_tight_power_budget(topology);

    if !hot
        && !low_battery
        && !tight_budget
        && backend.supports_gpu
        && has_kind(topology, AcceleratorKind::Gpu)
    {
        AcceleratorKind::Gpu
    } else if backend.supports_cpu && has_kind(topology, AcceleratorKind::Cpu) {
        AcceleratorKind::Cpu
    } else if backend.supports_npu && has_kind(topology, AcceleratorKind::Npu) {
        AcceleratorKind::Npu
    } else {
        AcceleratorKind::Cpu
    }
}

/// Chooses the decode target, preferring NPU token generation when available.
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

/// Chooses the KV-cache target based on the spill policy and power pressure.
fn pick_kv_target(
    topology: &HardwareTopology,
    host: &HostCapabilitySnapshot,
    model: &ModelDescriptor,
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
    if host_prefers_disk_for_cold_state(host, model)
        && has_kind(topology, AcceleratorKind::Disk)
        && policy.disk_percent > 0
    {
        AcceleratorKind::Disk
    } else if policy.disk_percent >= 40 && has_kind(topology, AcceleratorKind::Disk) {
        AcceleratorKind::Disk
    } else if should_rebalance_to_cpu(topology)
        && backend.supports_cpu
        && has_kind(topology, AcceleratorKind::Cpu)
    {
        AcceleratorKind::Cpu
    } else if policy.cpu_percent >= policy.gpu_percent
        && backend.supports_cpu
        && has_kind(topology, AcceleratorKind::Cpu)
    {
        AcceleratorKind::Cpu
    } else {
        decode_target
    }
}

/// Chooses the primary weight residency target for the request.
fn pick_weights_target(
    topology: &HardwareTopology,
    host: &HostCapabilitySnapshot,
    model: &ModelDescriptor,
    backend: &BackendDescriptor,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> AcceleratorKind {
    let prefer_cpu = should_rebalance_to_cpu(topology);
    let Some(plan) = tiered_offload else {
        if !prefer_cpu && backend.supports_gpu && has_kind(topology, AcceleratorKind::Gpu) {
            return AcceleratorKind::Gpu;
        }
        return AcceleratorKind::Cpu;
    };

    let policy = plan.policy.weights;
    if host_prefers_disk_for_cold_state(host, model)
        && has_kind(topology, AcceleratorKind::Disk)
        && plan.spill_bytes > 0
    {
        AcceleratorKind::Disk
    } else if policy.disk_percent >= policy.cpu_percent
        && policy.disk_percent >= policy.gpu_percent
        && has_kind(topology, AcceleratorKind::Disk)
    {
        AcceleratorKind::Disk
    } else if prefer_cpu && backend.supports_cpu && has_kind(topology, AcceleratorKind::Cpu) {
        AcceleratorKind::Cpu
    } else if policy.gpu_percent >= policy.cpu_percent
        && backend.supports_gpu
        && has_kind(topology, AcceleratorKind::Gpu)
    {
        AcceleratorKind::Gpu
    } else {
        AcceleratorKind::Cpu
    }
}

/// Returns whether the merged topology contains a device of the requested kind.
fn has_kind(topology: &HardwareTopology, kind: AcceleratorKind) -> bool {
    topology.devices.iter().any(|device| device.kind == kind)
}

/// Returns the preferred device identifier for a logical accelerator kind.
pub(super) fn preferred_device_id(
    topology: &HardwareTopology,
    kind: AcceleratorKind,
) -> Option<String> {
    topology
        .devices
        .iter()
        .find(|device| device.kind == kind)
        .map(|device| device.id.clone())
}

/// Resolves the device identifier that should own the weights placement.
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

/// Estimates how many bytes of model weights remain on the selected target.
fn weights_memory_bytes(
    model: &ModelDescriptor,
    tiered_offload: Option<&TieredOffloadPlan>,
    target: AcceleratorKind,
) -> Option<u64> {
    let model_bytes = model.memory_bytes?;
    let spill_bytes = tiered_offload
        .map(|plan| plan.spill_bytes.min(model_bytes))
        .unwrap_or(0);

    if target == AcceleratorKind::Disk {
        Some(spill_bytes)
    } else {
        Some(model_bytes.saturating_sub(spill_bytes))
    }
}

/// Finds the device identifier attached to a specific pipeline stage.
pub(super) fn placement_device_id(
    placements: &[PlacementDecision],
    stage: PipelineStage,
) -> Option<String> {
    placements
        .iter()
        .find(|placement| placement.stage == stage)
        .and_then(|placement| placement.device_id.clone())
}

/// Finds the logical accelerator target attached to a specific pipeline stage.
pub(super) fn stage_target(
    placements: &[PlacementDecision],
    stage: PipelineStage,
) -> Option<AcceleratorKind> {
    placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(|placement| placement.target)
}

/// Converts a logical accelerator target into the affinity token expected by hetero runtimes.
pub(super) fn affinity_tag(target: AcceleratorKind) -> Option<String> {
    match target {
        AcceleratorKind::Cpu => Some("CPU".to_string()),
        AcceleratorKind::Gpu => Some("GPU".to_string()),
        AcceleratorKind::Npu => Some("NPU".to_string()),
        AcceleratorKind::Disk => None,
    }
}

/// Returns the ordered list of non-disk devices participating in the plan.
pub(super) fn hetero_device_names(
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

/// Produces a human-readable explanation for the chosen prefill placement.
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

/// Produces a human-readable explanation for the chosen decode placement.
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

/// Produces a human-readable explanation for the chosen KV-cache placement.
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

/// Produces a human-readable explanation for the chosen weight placement.
fn weights_reason(
    topology: &HardwareTopology,
    target: AcceleratorKind,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> String {
    let Some(plan) = tiered_offload else {
        return match target {
            AcceleratorKind::Gpu => {
                "weights stay resident on the GPU to maximize prefill throughput".to_string()
            }
            AcceleratorKind::Cpu => {
                if should_rebalance_to_cpu(topology) {
                    "weights stay in CPU memory because the planner reduced accelerator pressure"
                        .to_string()
                } else {
                    "weights remain resident in CPU memory".to_string()
                }
            }
            AcceleratorKind::Disk => {
                "weights are disk-backed even without an explicit tiering policy".to_string()
            }
            AcceleratorKind::Npu => "weights are not directly anchored on the NPU".to_string(),
        };
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

/// Returns whether the topology is already in a thermally constrained state.
fn has_thermal_pressure(topology: &HardwareTopology) -> bool {
    matches!(
        topology.power.thermal_state,
        ThermalState::Hot | ThermalState::Critical
    )
}

/// Returns whether the host is running on a low battery reserve.
fn has_low_battery(topology: &HardwareTopology) -> bool {
    topology
        .power
        .battery_percent
        .map(|value| value < 20)
        .unwrap_or(false)
}

/// Returns whether the exposed power budget is low enough to change placement.
fn has_tight_power_budget(topology: &HardwareTopology) -> bool {
    topology
        .power
        .power_budget_watts
        .map(|budget| budget <= 18)
        .unwrap_or(false)
}

/// Determines whether the planner should shift hot state back toward CPU memory.
fn should_rebalance_to_cpu(topology: &HardwareTopology) -> bool {
    has_thermal_pressure(topology) || has_low_battery(topology) || has_tight_power_budget(topology)
}

/// Signals that runtime re-offload may be needed as conditions worsen.
pub(super) fn should_dynamic_reoffload(
    topology: &HardwareTopology,
    placements: &[PlacementDecision],
) -> bool {
    should_rebalance_to_cpu(topology)
        && placements.iter().any(|placement| {
            matches!(
                placement.stage,
                PipelineStage::KvCache | PipelineStage::Weights
            )
        })
}

fn host_prefers_disk_for_cold_state(
    host: &HostCapabilitySnapshot,
    model: &ModelDescriptor,
) -> bool {
    let Some(model_bytes) = model.memory_bytes else {
        return false;
    };
    host.available_memory_bytes > 0
        && (host.available_memory_bytes < model_bytes / 2
            || host.available_memory_bytes < 2 * 1024 * 1024 * 1024)
}

/// Returns the stable string label used in planner rationale messages.
pub(super) fn offload_profile_label(plan: &TieredOffloadPlan) -> &'static str {
    match plan.profile {
        loci_protocol::TieredOffloadProfile::Auto => "auto",
        loci_protocol::TieredOffloadProfile::GpuResident => "gpu_resident",
        loci_protocol::TieredOffloadProfile::Balanced => "balanced",
        loci_protocol::TieredOffloadProfile::DiskHeavy => "disk_heavy",
    }
}

/// Helper trait for producing display labels in planner rationale strings.
pub(super) trait AcceleratorKindLabel {
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

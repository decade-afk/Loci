//! Heterogeneous placement and backend-profile construction for a single request.

use crate::config::EngineConfig;
use crate::error::{LociError, Result};
use crate::model_inspector::inspect_model;
use crate::snapshot::HostCapabilitySnapshot;
use loci_protocol::{
    AcceleratorKind, Backend, BackendDescriptor, BackendExecutionProfile,
    BackendLoweringCapabilities, BackendLoweringPlan, BackendRuntimeFamily, CandleExecutionProfile,
    CandleTensorResidency, ChipOperatorClass, DeviceDescriptor, ExecutionPlan,
    GenericExecutionProfile, HardwareTopology, KvCachePlan, LoweringAffinityMode,
    LoweringOperatorPlan, LoweringPartitionPlan, LoweringSubgraphPlan, ModelDescriptor,
    OpenVinoExecutionMode, OpenVinoExecutionProfile, PipelineStage, PlacementDecision, PowerState,
    RouteDecision, SessionRequest, ThermalState, TieredOffloadPlan,
};
use std::collections::BTreeMap;

/// Merges backend-reported hardware views into a single planner topology.
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

/// Chooses the backend that should execute a model under the current policy.
pub fn choose_backend<'a>(
    backends: &'a [Box<dyn Backend>],
    model: &ModelDescriptor,
    request: &SessionRequest,
    preferred_backend: Option<&str>,
) -> Result<&'a dyn Backend> {
    let requires_multimodal = !request.images.is_empty();
    let inspection = inspect_model(model, backends);

    let readiness_for = |backend_name: &str| {
        inspection
            .backend_readiness
            .iter()
            .find(|readiness| readiness.backend == backend_name)
    };
    let matches_candidate = |descriptor: &BackendDescriptor,
                             require_ready: bool,
                             require_npu: bool,
                             name_filter: Option<&str>| {
        let Some(readiness) = readiness_for(&descriptor.name) else {
            return false;
        };
        if name_filter.is_some_and(|name| descriptor.name != name) {
            return false;
        }
        if require_ready && !readiness.ready {
            return false;
        }
        if require_npu && !descriptor.supports_npu {
            return false;
        }
        if requires_multimodal && !readiness.supports_multimodal {
            return false;
        }
        true
    };

    let preferred_candidates = [preferred_backend, model.preferred_backend.as_deref()];

    for preferred in preferred_candidates {
        if let Some(name) = preferred {
            if let Some(backend) = backends
                .iter()
                .find(|backend| matches_candidate(&backend.descriptor(), true, false, Some(name)))
            {
                return Ok(backend.as_ref());
            }
        }
    }

    if preferred_backend.is_none() && model.preferred_backend.is_none() {
        if let Some(backend) = backends
            .iter()
            .find(|backend| matches_candidate(&backend.descriptor(), true, true, None))
        {
            return Ok(backend.as_ref());
        }

        if let Some(backend) = backends
            .iter()
            .find(|backend| matches_candidate(&backend.descriptor(), true, false, None))
        {
            return Ok(backend.as_ref());
        }
    }

    for preferred in preferred_candidates {
        if let Some(name) = preferred {
            if let Some(backend) = backends
                .iter()
                .find(|backend| matches_candidate(&backend.descriptor(), true, false, Some(name)))
            {
                return Ok(backend.as_ref());
            }
        }
    }

    backends
        .iter()
        .find(|backend| matches_candidate(&backend.descriptor(), true, true, None))
        .or_else(|| {
            backends
                .iter()
                .find(|backend| matches_candidate(&backend.descriptor(), true, false, None))
        })
        .map(|backend| backend.as_ref())
        .ok_or_else(|| LociError::NoCompatibleBackend {
            model: model.name.clone(),
            format: model.inferred_format().as_str().to_string(),
        })
}

/// Builds the execution plan that the runtime and backend layers consume.
pub fn build_plan(
    config: &EngineConfig,
    backend: &BackendDescriptor,
    backend_lowering: &BackendLoweringCapabilities,
    topology: &HardwareTopology,
    host: &HostCapabilitySnapshot,
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
        host,
        model,
        backend,
        decode_target,
        &kv_cache,
        tiered_offload.as_ref(),
    );
    let weights_target =
        pick_weights_target(topology, host, model, backend, tiered_offload.as_ref());

    // The planner intentionally separates throughput-biased prefill from
    // power-biased decode so the engine can model heterogeneous execution.
    let placements = vec![
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
        PlacementDecision {
            stage: PipelineStage::Weights,
            target: weights_target,
            device_id: weight_device_id(topology, tiered_offload.as_ref(), weights_target),
            memory_bytes: weights_memory_bytes(model, tiered_offload.as_ref(), weights_target),
            rationale: weights_reason(topology, weights_target, tiered_offload.as_ref()),
        },
    ];
    let lowering_plan = build_lowering_plan(
        backend,
        backend_lowering,
        model,
        &placements,
        &kv_cache,
        tiered_offload.as_ref(),
    );

    let backend_profile = build_backend_profile(backend, topology, &placements);

    ExecutionPlan {
        backend: backend.name.clone(),
        route,
        placements,
        lowering_plan: Some(lowering_plan),
        kv_cache,
        tiered_offload,
        backend_profile,
    }
}

/// Converts planner placements into backend-facing subgraph affinity guidance.
fn build_lowering_plan(
    backend: &BackendDescriptor,
    backend_lowering: &BackendLoweringCapabilities,
    model: &ModelDescriptor,
    placements: &[PlacementDecision],
    kv_cache: &KvCachePlan,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> BackendLoweringPlan {
    let prefill_target =
        stage_target(placements, PipelineStage::Prefill).unwrap_or(AcceleratorKind::Cpu);
    let decode_target = stage_target(placements, PipelineStage::Decode).unwrap_or(prefill_target);
    let kv_target = stage_target(placements, PipelineStage::KvCache).unwrap_or(decode_target);
    let weights_target =
        stage_target(placements, PipelineStage::Weights).unwrap_or(AcceleratorKind::Cpu);

    let mut subgraphs = Vec::new();
    if model.is_multimodal_architecture() {
        subgraphs.push(LoweringSubgraphPlan {
            id: "vision_encoder".to_string(),
            stage: PipelineStage::Prefill,
            operator_class: ChipOperatorClass::VisionEncoder,
            target: prefill_target,
            device_id: placement_device_id(placements, PipelineStage::Prefill),
            affinity_tag: affinity_tag(prefill_target),
            estimated_bytes: model.memory_bytes.map(|bytes| bytes / 5),
            spillable: false,
            rationale: "multimodal models front-load image encoding into the prefill path"
                .to_string(),
        });
    }

    subgraphs.extend([
        LoweringSubgraphPlan {
            id: "embedding_lookup".to_string(),
            stage: PipelineStage::Prefill,
            operator_class: ChipOperatorClass::Embedding,
            target: prefill_target,
            device_id: placement_device_id(placements, PipelineStage::Prefill),
            affinity_tag: affinity_tag(prefill_target),
            estimated_bytes: model.memory_bytes.map(|bytes| bytes / 20),
            spillable: false,
            rationale: "token embedding lookup should stay close to the prefill compute path".to_string(),
        },
        LoweringSubgraphPlan {
            id: "prefill_attention_block".to_string(),
            stage: PipelineStage::Prefill,
            operator_class: ChipOperatorClass::Attention,
            target: prefill_target,
            device_id: placement_device_id(placements, PipelineStage::Prefill),
            affinity_tag: affinity_tag(prefill_target),
            estimated_bytes: model.memory_bytes.map(|bytes| bytes / 3),
            spillable: false,
            rationale: "prompt-side attention is throughput-sensitive and follows the prefill placement".to_string(),
        },
        LoweringSubgraphPlan {
            id: "prefill_mlp_block".to_string(),
            stage: PipelineStage::Prefill,
            operator_class: ChipOperatorClass::Mlp,
            target: prefill_target,
            device_id: placement_device_id(placements, PipelineStage::Prefill),
            affinity_tag: affinity_tag(prefill_target),
            estimated_bytes: model.memory_bytes.map(|bytes| bytes / 4),
            spillable: false,
            rationale: "prefill MLP kernels should remain colocated with prompt-side attention".to_string(),
        },
        LoweringSubgraphPlan {
            id: "decode_attention_block".to_string(),
            stage: PipelineStage::Decode,
            operator_class: ChipOperatorClass::Attention,
            target: decode_target,
            device_id: placement_device_id(placements, PipelineStage::Decode),
            affinity_tag: affinity_tag(decode_target),
            estimated_bytes: model.memory_bytes.map(|bytes| bytes / 4),
            spillable: false,
            rationale: "token-step attention is latency-sensitive and follows the decode placement".to_string(),
        },
        LoweringSubgraphPlan {
            id: "decode_mlp_block".to_string(),
            stage: PipelineStage::Decode,
            operator_class: ChipOperatorClass::Mlp,
            target: decode_target,
            device_id: placement_device_id(placements, PipelineStage::Decode),
            affinity_tag: affinity_tag(decode_target),
            estimated_bytes: model.memory_bytes.map(|bytes| bytes / 5),
            spillable: false,
            rationale: "decode-side MLP should remain in the same hot path as token generation".to_string(),
        },
        LoweringSubgraphPlan {
            id: "kv_state_region".to_string(),
            stage: PipelineStage::KvCache,
            operator_class: ChipOperatorClass::KvCache,
            target: kv_target,
            device_id: placement_device_id(placements, PipelineStage::KvCache),
            affinity_tag: affinity_tag(kv_target),
            estimated_bytes: kv_cache.max_cache_bytes,
            spillable: kv_target == AcceleratorKind::Disk || kv_cache.tiered,
            rationale: "kv cache placement is modeled as a stateful region that may spill independently".to_string(),
        },
        LoweringSubgraphPlan {
            id: "weights_residency_region".to_string(),
            stage: PipelineStage::Weights,
            operator_class: ChipOperatorClass::Matmul,
            target: weights_target,
            device_id: placement_device_id(placements, PipelineStage::Weights),
            affinity_tag: affinity_tag(weights_target),
            estimated_bytes: model.memory_bytes,
            spillable: weights_target == AcceleratorKind::Disk || tiered_offload.is_some(),
            rationale: "weight residency guidance constrains where backend-local matmul-heavy subgraphs should source parameters".to_string(),
        },
        LoweringSubgraphPlan {
            id: "sampling_head".to_string(),
            stage: PipelineStage::Sampling,
            operator_class: ChipOperatorClass::Sampling,
            target: AcceleratorKind::Cpu,
            device_id: placement_device_id(placements, PipelineStage::Sampling),
            affinity_tag: affinity_tag(AcceleratorKind::Cpu),
            estimated_bytes: Some(8 * 1024 * 1024),
            spillable: false,
            rationale: "sampling remains on the CPU orchestration path for response assembly".to_string(),
        },
    ]);
    let partitions = build_lowering_partitions(&subgraphs);
    let operators = build_lowering_operators(&subgraphs, &partitions);

    BackendLoweringPlan {
        backend: backend.name.clone(),
        granularity: backend_lowering.granularity,
        affinity_mode: if backend_lowering.supports_layer_affinity {
            LoweringAffinityMode::Explicit
        } else if backend_lowering.supports_graph_partitioning {
            LoweringAffinityMode::Planned
        } else {
            LoweringAffinityMode::Automatic
        },
        subgraphs,
        partitions,
        operators,
        notes: lowering_notes(backend, backend_lowering, tiered_offload),
    }
}

/// Normalizes subgraph guidance into execution partitions grouped by target affinity.
fn build_lowering_partitions(subgraphs: &[LoweringSubgraphPlan]) -> Vec<LoweringPartitionPlan> {
    let mut grouped = BTreeMap::<String, Vec<&LoweringSubgraphPlan>>::new();
    for subgraph in subgraphs {
        grouped
            .entry(partition_group_key(subgraph))
            .or_default()
            .push(subgraph);
    }

    grouped
        .into_values()
        .enumerate()
        .map(|(index, grouped_subgraphs)| {
            let first = grouped_subgraphs[0];
            let mut operator_classes = Vec::new();
            let mut subgraph_ids = Vec::new();
            let mut estimated_bytes = 0u64;
            let mut has_estimate = false;
            let mut spillable = false;

            for subgraph in grouped_subgraphs {
                if !operator_classes.contains(&subgraph.operator_class) {
                    operator_classes.push(subgraph.operator_class);
                }
                subgraph_ids.push(subgraph.id.clone());
                if let Some(bytes) = subgraph.estimated_bytes {
                    estimated_bytes = estimated_bytes.saturating_add(bytes);
                    has_estimate = true;
                }
                spillable |= subgraph.spillable;
            }

            let affinity_label = first
                .affinity_tag
                .as_deref()
                .map(str::to_ascii_lowercase)
                .unwrap_or_else(|| accelerator_partition_label(first.target).to_string());
            LoweringPartitionPlan {
                id: format!("partition-{}-{affinity_label}", index + 1),
                target: first.target,
                device_id: first.device_id.clone(),
                affinity_tag: first.affinity_tag.clone(),
                operator_classes,
                subgraphs: subgraph_ids.clone(),
                estimated_bytes: has_estimate.then_some(estimated_bytes),
                spillable,
                rationale: format!(
                    "{} lowering regions share the {} execution partition",
                    subgraph_ids.len(),
                    first
                        .affinity_tag
                        .as_deref()
                        .unwrap_or_else(|| accelerator_partition_label(first.target))
                ),
            }
        })
        .collect()
}

/// Normalizes each subgraph into an operator-facing record tied to a concrete partition.
fn build_lowering_operators(
    subgraphs: &[LoweringSubgraphPlan],
    partitions: &[LoweringPartitionPlan],
) -> Vec<LoweringOperatorPlan> {
    subgraphs
        .iter()
        .map(|subgraph| LoweringOperatorPlan {
            id: format!("operator-{}", subgraph.id),
            partition: partitions
                .iter()
                .find(|partition| partition.subgraphs.contains(&subgraph.id))
                .map(|partition| partition.id.clone())
                .unwrap_or_else(|| "unassigned-partition".to_string()),
            subgraph: subgraph.id.clone(),
            stage: subgraph.stage,
            operator_class: subgraph.operator_class,
            target: subgraph.target,
            device_id: subgraph.device_id.clone(),
            affinity_tag: subgraph.affinity_tag.clone(),
            estimated_bytes: subgraph.estimated_bytes,
            spillable: subgraph.spillable,
            rationale: subgraph.rationale.clone(),
        })
        .collect()
}

fn partition_group_key(subgraph: &LoweringSubgraphPlan) -> String {
    format!(
        "{:?}|{}|{}|{}",
        subgraph.target,
        subgraph.device_id.as_deref().unwrap_or("none"),
        subgraph.affinity_tag.as_deref().unwrap_or("none"),
        subgraph.spillable
    )
}

fn accelerator_partition_label(kind: AcceleratorKind) -> &'static str {
    match kind {
        AcceleratorKind::Cpu => "cpu",
        AcceleratorKind::Gpu => "gpu",
        AcceleratorKind::Npu => "npu",
        AcceleratorKind::Disk => "disk",
    }
}

/// Converts generic placements into a backend-specific execution profile.
fn build_backend_profile(
    backend: &BackendDescriptor,
    topology: &HardwareTopology,
    placements: &[PlacementDecision],
) -> BackendExecutionProfile {
    match backend.runtime_family {
        BackendRuntimeFamily::OpenVino => {
            BackendExecutionProfile::OpenVino(build_openvino_profile(topology, placements))
        }
        BackendRuntimeFamily::Candle => {
            BackendExecutionProfile::Candle(build_candle_profile(topology, placements))
        }
        _ => BackendExecutionProfile::Generic(GenericExecutionProfile {
            session_key: format!("{}-session", backend.name),
            summary: format!("generic execution profile for backend `{}`", backend.name),
        }),
    }
}

/// Constructs the OpenVINO execution profile from the generic plan.
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
        dynamic_reoffload: should_dynamic_reoffload(topology, placements),
    }
}

/// Constructs the Candle execution profile from the generic plan.
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
fn preferred_device_id(topology: &HardwareTopology, kind: AcceleratorKind) -> Option<String> {
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
fn placement_device_id(placements: &[PlacementDecision], stage: PipelineStage) -> Option<String> {
    placements
        .iter()
        .find(|placement| placement.stage == stage)
        .and_then(|placement| placement.device_id.clone())
}

/// Finds the logical accelerator target attached to a specific pipeline stage.
fn stage_target(placements: &[PlacementDecision], stage: PipelineStage) -> Option<AcceleratorKind> {
    placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(|placement| placement.target)
}

/// Converts a logical accelerator target into the affinity token expected by hetero runtimes.
fn affinity_tag(target: AcceleratorKind) -> Option<String> {
    match target {
        AcceleratorKind::Cpu => Some("CPU".to_string()),
        AcceleratorKind::Gpu => Some("GPU".to_string()),
        AcceleratorKind::Npu => Some("NPU".to_string()),
        AcceleratorKind::Disk => None,
    }
}

/// Returns the ordered list of non-disk devices participating in the plan.
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
fn should_dynamic_reoffload(topology: &HardwareTopology, placements: &[PlacementDecision]) -> bool {
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
fn offload_profile_label(plan: &TieredOffloadPlan) -> &'static str {
    match plan.profile {
        loci_protocol::TieredOffloadProfile::Auto => "auto",
        loci_protocol::TieredOffloadProfile::GpuResident => "gpu_resident",
        loci_protocol::TieredOffloadProfile::Balanced => "balanced",
        loci_protocol::TieredOffloadProfile::DiskHeavy => "disk_heavy",
    }
}

/// Explains how the planner expects the backend to consume the lowering plan.
fn lowering_notes(
    backend: &BackendDescriptor,
    backend_lowering: &BackendLoweringCapabilities,
    tiered_offload: Option<&TieredOffloadPlan>,
) -> Vec<String> {
    let mut notes = Vec::new();
    if backend_lowering.supports_graph_partitioning {
        notes.push(format!(
            "planner emitted subgraph guidance for backend `{}` at {:?} granularity",
            backend.name, backend_lowering.granularity
        ));
    } else {
        notes.push(format!(
            "backend `{}` does not report graph partitioning support, so subgraph guidance is advisory",
            backend.name
        ));
    }
    if let Some(plan) = tiered_offload {
        notes.push(format!(
            "tiered residency is active with profile `{}` and spill budget {} bytes",
            offload_profile_label(plan),
            plan.spill_bytes
        ));
    }
    notes
}

/// Helper trait for producing display labels in planner rationale strings.
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
    use crate::snapshot::{HostCapabilitySnapshot, HostDiskSnapshot, HostProbeSnapshot};
    use loci_protocol::{
        Backend, BackendAssetCapabilities, BackendDescriptor, BackendError,
        BackendLoweringCapabilities, BackendOutput, BackendResult, ExecutionArtifactKind,
        ExecutionPlan, KvCachePlan, ModelAssetLayout, PreparedModel, RouteDecision,
        TieredOffloadPolicy, TieredOffloadProfile, TieredPlacementPercentages,
    };
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[derive(Clone)]
    struct MockBackend {
        descriptor: BackendDescriptor,
        supports_model: bool,
    }

    impl Backend for MockBackend {
        fn descriptor(&self) -> BackendDescriptor {
            self.descriptor.clone()
        }

        fn asset_capabilities(&self) -> BackendAssetCapabilities {
            match self.descriptor.runtime_family {
                loci_protocol::BackendRuntimeFamily::OpenVino => BackendAssetCapabilities {
                    backend: self.descriptor.name.clone(),
                    runtime_family: self.descriptor.runtime_family,
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
                    notes: Vec::new(),
                },
                loci_protocol::BackendRuntimeFamily::Candle => BackendAssetCapabilities {
                    backend: self.descriptor.name.clone(),
                    runtime_family: self.descriptor.runtime_family,
                    directly_supported_layouts: vec![
                        ModelAssetLayout::GgufFile,
                        ModelAssetLayout::GgufDirectory,
                        ModelAssetLayout::SafeTensorsFile,
                        ModelAssetLayout::SafeTensorsDirectory,
                        ModelAssetLayout::PytorchBinFile,
                        ModelAssetLayout::PytorchCheckpointDirectory,
                        ModelAssetLayout::TransformersCheckpoint,
                    ],
                    ingestible_layouts: vec![
                        ModelAssetLayout::UnknownDirectory,
                        ModelAssetLayout::UnknownFile,
                    ],
                    preferred_artifact: ExecutionArtifactKind::NativeCheckpoint,
                    requires_lowering_for_execution: false,
                    notes: Vec::new(),
                },
                _ => BackendAssetCapabilities {
                    backend: self.descriptor.name.clone(),
                    runtime_family: self.descriptor.runtime_family,
                    directly_supported_layouts: Vec::new(),
                    ingestible_layouts: Vec::new(),
                    preferred_artifact: ExecutionArtifactKind::RuntimeDefined,
                    requires_lowering_for_execution: false,
                    notes: Vec::new(),
                },
            }
        }

        fn discover_topology(&self) -> HardwareTopology {
            HardwareTopology::default()
        }

        fn supports_model(&self, _model: &ModelDescriptor) -> bool {
            self.supports_model
        }

        fn prepare(
            &self,
            _model: &ModelDescriptor,
            _plan: &ExecutionPlan,
        ) -> BackendResult<PreparedModel> {
            Err(BackendError {
                message: "unused in planner tests".to_string(),
            })
        }

        fn execute(
            &self,
            _prepared: &PreparedModel,
            _model: &ModelDescriptor,
            _request: &SessionRequest,
            _plan: &ExecutionPlan,
        ) -> BackendResult<BackendOutput> {
            Err(BackendError {
                message: "unused in planner tests".to_string(),
            })
        }
    }

    fn backend(name: &str, supports_npu: bool) -> BackendDescriptor {
        BackendDescriptor {
            name: name.to_string(),
            runtime_family: match name {
                "openvino" => loci_protocol::BackendRuntimeFamily::OpenVino,
                "candle" => loci_protocol::BackendRuntimeFamily::Candle,
                _ => loci_protocol::BackendRuntimeFamily::Generic,
            },
            supports_cpu: true,
            supports_gpu: true,
            supports_npu,
            supports_disk_tiering: true,
            supports_paged_kv: true,
            supports_multimodal: name == "openvino",
        }
    }

    fn lowering(name: &str, supports_npu: bool) -> BackendLoweringCapabilities {
        BackendLoweringCapabilities {
            backend: name.to_string(),
            runtime_family: match name {
                "openvino" => loci_protocol::BackendRuntimeFamily::OpenVino,
                "candle" => loci_protocol::BackendRuntimeFamily::Candle,
                _ => loci_protocol::BackendRuntimeFamily::Generic,
            },
            granularity: if supports_npu {
                loci_protocol::LoweringGranularity::Subgraph
            } else {
                loci_protocol::LoweringGranularity::Graph
            },
            supports_real_execution: name == "openvino",
            supports_graph_partitioning: supports_npu,
            supports_layer_affinity: false,
            supports_dynamic_reoffload: supports_npu,
            supports_custom_operators: false,
            operator_classes: vec![
                loci_protocol::ChipOperatorClass::Attention,
                loci_protocol::ChipOperatorClass::Mlp,
                loci_protocol::ChipOperatorClass::KvCache,
            ],
            notes: Vec::new(),
        }
    }

    fn topology() -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(16 * 1024 * 1024 * 1024),
                    compute_units: Some(16),
                    power_watts: Some(20.0),
                },
                DeviceDescriptor {
                    id: "gpu:0".to_string(),
                    name: "gpu".to_string(),
                    kind: AcceleratorKind::Gpu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    compute_units: Some(128),
                    power_watts: Some(30.0),
                },
                DeviceDescriptor {
                    id: "npu:0".to_string(),
                    name: "npu".to_string(),
                    kind: AcceleratorKind::Npu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(2 * 1024 * 1024 * 1024),
                    compute_units: Some(1),
                    power_watts: Some(5.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "disk".to_string(),
                    kind: AcceleratorKind::Disk,
                    platform: Some(std::env::consts::OS.to_string()),
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
            images: Vec::new(),
            structured_output: false,
            tool_calling: false,
        }
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-planner-{label}-{suffix}"))
    }

    fn route() -> RouteDecision {
        RouteDecision {
            selected_model: "demo".to_string(),
            reason: "explicit".to_string(),
            alternatives: Vec::new(),
        }
    }

    fn host() -> HostCapabilitySnapshot {
        HostCapabilitySnapshot {
            target_family: std::env::consts::FAMILY.to_string(),
            target_os: std::env::consts::OS.to_string(),
            target_arch: std::env::consts::ARCH.to_string(),
            mobile_class: false,
            host_name: Some("test-host".to_string()),
            os_name: Some("test-os".to_string()),
            os_version: Some("1".to_string()),
            kernel_version: Some("1".to_string()),
            cpu_brand: Some("cpu".to_string()),
            cpu_vendor: Some("vendor".to_string()),
            cpu_frequency_mhz: Some(1000),
            physical_cores: Some(4),
            logical_cores: 8,
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
            total_swap_bytes: 0,
            free_swap_bytes: 0,
            uptime_secs: 1,
            load_average_one: 0.0,
            load_average_five: 0.0,
            load_average_fifteen: 0.0,
            disks: vec![HostDiskSnapshot {
                name: "disk".to_string(),
                mount_point: "D:\\".to_string(),
                file_system: "NTFS".to_string(),
                total_bytes: 256 * 1024 * 1024 * 1024,
                available_bytes: 128 * 1024 * 1024 * 1024,
                is_removable: false,
            }],
            probe: HostProbeSnapshot {
                cpu_scalar_gops: 1.0,
                memory_bandwidth_gbps: 10.0,
                disk_read_mbps: 2000.0,
                disk_write_mbps: 1500.0,
                probe_bytes: 16 * 1024 * 1024,
                probe_duration_ms: 10,
            },
        }
    }

    #[test]
    fn build_plan_uses_disk_for_kv_when_disk_heavy_policy_demands_it() {
        let plan = build_plan(
            &EngineConfig::default(),
            &backend("openvino", true),
            &lowering("openvino", true),
            &topology(),
            &host(),
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
            &lowering("candle", false),
            &topology(),
            &host(),
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

    #[test]
    fn build_plan_assigns_weight_placement_without_disk_tiering() {
        let plan = build_plan(
            &EngineConfig::default(),
            &backend("openvino", true),
            &lowering("openvino", true),
            &topology(),
            &host(),
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
                tiered: false,
            },
            None,
        );

        assert!(plan.placements.iter().any(|placement| {
            placement.stage == PipelineStage::Weights
                && placement.target == AcceleratorKind::Gpu
                && placement.device_id.as_deref() == Some("gpu:0")
                && placement.memory_bytes == model().memory_bytes
        }));
    }

    #[test]
    fn build_plan_rebalances_kv_and_weights_under_power_pressure() {
        let mut constrained_topology = topology();
        constrained_topology.power = PowerState {
            battery_powered: true,
            battery_percent: Some(10),
            thermal_state: ThermalState::Hot,
            power_budget_watts: Some(15),
        };

        let plan = build_plan(
            &EngineConfig::default(),
            &backend("openvino", true),
            &lowering("openvino", true),
            &constrained_topology,
            &host(),
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
                        gpu_percent: 45,
                        cpu_percent: 35,
                        disk_percent: 20,
                    },
                    kv_cache: TieredPlacementPercentages {
                        gpu_percent: 60,
                        cpu_percent: 30,
                        disk_percent: 10,
                    },
                    activations: TieredPlacementPercentages {
                        gpu_percent: 50,
                        cpu_percent: 50,
                        disk_percent: 0,
                    },
                    cpu_cache_compute: false,
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
            BackendExecutionProfile::OpenVino(OpenVinoExecutionProfile {
                dynamic_reoffload: true,
                ..
            })
        ));
    }

    #[test]
    fn build_plan_prefers_disk_for_cold_state_when_host_memory_is_tight() {
        let mut tight_host = host();
        tight_host.available_memory_bytes = 1024 * 1024 * 1024;

        let plan = build_plan(
            &EngineConfig::default(),
            &backend("openvino", true),
            &lowering("openvino", true),
            &topology(),
            &tight_host,
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
                profile: TieredOffloadProfile::Balanced,
                policy: TieredOffloadPolicy {
                    weights: TieredPlacementPercentages {
                        gpu_percent: 35,
                        cpu_percent: 35,
                        disk_percent: 30,
                    },
                    kv_cache: TieredPlacementPercentages {
                        gpu_percent: 20,
                        cpu_percent: 50,
                        disk_percent: 30,
                    },
                    activations: TieredPlacementPercentages {
                        gpu_percent: 60,
                        cpu_percent: 40,
                        disk_percent: 0,
                    },
                    cpu_cache_compute: true,
                    compress_weights: true,
                    compress_kv_cache: true,
                },
            }),
        );

        assert!(plan.placements.iter().any(|placement| {
            placement.stage == PipelineStage::Weights && placement.target == AcceleratorKind::Disk
        }));
    }

    #[test]
    fn build_plan_emits_backend_lowering_guidance() {
        let plan = build_plan(
            &EngineConfig::default(),
            &backend("openvino", true),
            &lowering("openvino", true),
            &topology(),
            &host(),
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
            None,
        );

        let lowering = plan.lowering_plan.expect("lowering plan");
        assert_eq!(lowering.backend, "openvino");
        assert!(lowering.subgraphs.iter().any(|subgraph| {
            subgraph.id == "decode_attention_block"
                && subgraph.affinity_tag.as_deref() == Some("NPU")
        }));
        assert!(!lowering.partitions.is_empty());
        assert!(!lowering.operators.is_empty());
        assert!(lowering.partitions.iter().any(|partition| {
            partition.affinity_tag.as_deref() == Some("NPU")
                && partition
                    .operator_classes
                    .contains(&ChipOperatorClass::Attention)
        }));
        assert!(lowering.operators.iter().any(|operator| {
            operator.subgraph == "decode_attention_block"
                && operator.partition.starts_with("partition-")
        }));
        assert!(lowering.subgraphs.iter().any(|subgraph| {
            subgraph.id == "kv_state_region"
                && subgraph.operator_class == ChipOperatorClass::KvCache
        }));
    }

    #[test]
    fn build_plan_dispatches_backend_profile_by_runtime_family() {
        let plan = build_plan(
            &EngineConfig::default(),
            &BackendDescriptor {
                name: "intel-openvino-main".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: true,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: true,
            },
            &BackendLoweringCapabilities {
                backend: "intel-openvino-main".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                granularity: loci_protocol::LoweringGranularity::Subgraph,
                supports_real_execution: true,
                supports_graph_partitioning: true,
                supports_layer_affinity: false,
                supports_dynamic_reoffload: true,
                supports_custom_operators: false,
                operator_classes: vec![
                    loci_protocol::ChipOperatorClass::Attention,
                    loci_protocol::ChipOperatorClass::Mlp,
                ],
                notes: Vec::new(),
            },
            &topology(),
            &host(),
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
            None,
        );

        assert!(matches!(
            plan.backend_profile,
            BackendExecutionProfile::OpenVino(_)
        ));
    }

    #[test]
    fn choose_backend_skips_non_multimodal_backends_for_image_requests() {
        let dir = unique_temp_dir("multimodal-openvino-ready");
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join("openvino_model.xml"), "<xml/>").expect("xml");

        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(MockBackend {
                descriptor: BackendDescriptor {
                    name: "candle".to_string(),
                    runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
                    supports_cpu: true,
                    supports_gpu: true,
                    supports_npu: false,
                    supports_disk_tiering: true,
                    supports_paged_kv: true,
                    supports_multimodal: false,
                },
                supports_model: true,
            }),
            Box::new(MockBackend {
                descriptor: BackendDescriptor {
                    name: "openvino".to_string(),
                    runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                    supports_cpu: true,
                    supports_gpu: true,
                    supports_npu: true,
                    supports_disk_tiering: true,
                    supports_paged_kv: true,
                    supports_multimodal: true,
                },
                supports_model: true,
            }),
        ];
        let multimodal_model = ModelDescriptor {
            name: "demo".to_string(),
            path: dir.clone(),
            architecture: "minicpm-v".to_string(),
            memory_bytes: Some(1),
            parameter_count: Some(1),
            context_length: Some(128),
            preferred_backend: Some("candle".to_string()),
        };

        let mut multimodal_request = request();
        multimodal_request
            .images
            .push(loci_protocol::ImageInput::Path {
                path: PathBuf::from("D:/images/demo.png"),
            });

        let selected = choose_backend(&backends, &multimodal_model, &multimodal_request, Some("candle"))
            .expect("backend");

        assert_eq!(selected.descriptor().name, "openvino");
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn choose_backend_prefers_ready_backend_over_partial_fallbacks() {
        let dir = unique_temp_dir("openvino-ready");
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join("openvino_model.xml"), "<xml/>").expect("xml");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: dir.clone(),
            architecture: "llama".to_string(),
            memory_bytes: Some(1),
            parameter_count: Some(1),
            context_length: Some(128),
            preferred_backend: Some("candle".to_string()),
        };

        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(MockBackend {
                descriptor: BackendDescriptor {
                    name: "candle".to_string(),
                    runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
                    supports_cpu: true,
                    supports_gpu: true,
                    supports_npu: false,
                    supports_disk_tiering: true,
                    supports_paged_kv: true,
                    supports_multimodal: false,
                },
                supports_model: true,
            }),
            Box::new(MockBackend {
                descriptor: BackendDescriptor {
                    name: "openvino".to_string(),
                    runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                    supports_cpu: true,
                    supports_gpu: true,
                    supports_npu: true,
                    supports_disk_tiering: true,
                    supports_paged_kv: true,
                    supports_multimodal: true,
                },
                supports_model: true,
            }),
        ];

        let selected =
            choose_backend(&backends, &model, &request(), Some("candle")).expect("backend");

        assert_eq!(selected.descriptor().name, "openvino");
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn choose_backend_uses_readiness_instead_of_supports_model_heuristics() {
        let dir = unique_temp_dir("readiness-over-supports-model");
        fs::create_dir_all(&dir).expect("dir");
        fs::write(dir.join("openvino_model.xml"), "<xml/>").expect("xml");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: dir.clone(),
            architecture: "llama".to_string(),
            memory_bytes: Some(1),
            parameter_count: Some(1),
            context_length: Some(128),
            preferred_backend: None,
        };

        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(MockBackend {
                descriptor: BackendDescriptor {
                    name: "openvino".to_string(),
                    runtime_family: loci_protocol::BackendRuntimeFamily::OpenVino,
                    supports_cpu: true,
                    supports_gpu: true,
                    supports_npu: true,
                    supports_disk_tiering: true,
                    supports_paged_kv: true,
                    supports_multimodal: true,
                },
                supports_model: false,
            }),
            Box::new(MockBackend {
                descriptor: BackendDescriptor {
                    name: "candle".to_string(),
                    runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
                    supports_cpu: true,
                    supports_gpu: true,
                    supports_npu: false,
                    supports_disk_tiering: true,
                    supports_paged_kv: true,
                    supports_multimodal: false,
                },
                supports_model: true,
            }),
        ];

        let selected = choose_backend(&backends, &model, &request(), None).expect("backend");
        assert_eq!(selected.descriptor().name, "openvino");

        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn choose_backend_rejects_non_ready_candle_checkpoint_paths() {
        let file = unique_temp_dir("torch-checkpoint").with_extension("pt");
        fs::write(&file, "weights").expect("weights");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: file.clone(),
            architecture: "llama".to_string(),
            memory_bytes: Some(1),
            parameter_count: Some(1),
            context_length: Some(128),
            preferred_backend: Some("candle".to_string()),
        };

        let backends: Vec<Box<dyn Backend>> = vec![Box::new(MockBackend {
            descriptor: BackendDescriptor {
                name: "candle".to_string(),
                runtime_family: loci_protocol::BackendRuntimeFamily::Candle,
                supports_cpu: true,
                supports_gpu: true,
                supports_npu: false,
                supports_disk_tiering: true,
                supports_paged_kv: true,
                supports_multimodal: false,
            },
            supports_model: true,
        })];

        let error = match choose_backend(&backends, &model, &request(), Some("candle")) {
            Ok(_) => panic!("non-ready backend should be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            LociError::NoCompatibleBackend { .. }
        ));

        fs::remove_file(file).expect("cleanup");
    }
}

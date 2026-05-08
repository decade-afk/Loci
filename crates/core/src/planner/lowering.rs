use super::placement::{affinity_tag, offload_profile_label, placement_device_id, stage_target};
use loci_protocol::{
    AcceleratorKind, BackendDescriptor, BackendLoweringCapabilities, BackendLoweringPlan,
    ChipOperatorClass, KvCachePlan, LoweringAffinityMode, LoweringOperatorPlan,
    LoweringPartitionPlan, LoweringSubgraphPlan, ModelDescriptor, PipelineStage, PlacementDecision,
    TieredOffloadPlan,
};
use std::collections::BTreeMap;

/// Converts stage placements into backend-facing subgraph affinity guidance.
pub(super) fn build_lowering_plan(
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

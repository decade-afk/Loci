use crate::bootstrap::ensure_runtime_bootstrap;
use crate::{openvino_error, setup_error, ModelDescriptor};
use loci_protocol::{
    AcceleratorKind, BackendError, BackendExecutionProfile, BackendResult, ExecutionPlan,
    OpenVinoExecutionMode, OpenVinoExecutionProfile, PipelineStage, PlacementDecision,
    PreparedResidency,
};
use openvino::{Core, DeviceType};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

pub(super) fn runtime_device_name(profile: &OpenVinoExecutionProfile) -> String {
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

pub(super) fn shadow_lowering_compile(
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

pub(super) fn runtime_properties(
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

pub(super) fn openvino_profile(plan: &ExecutionPlan) -> BackendResult<&OpenVinoExecutionProfile> {
    match &plan.backend_profile {
        BackendExecutionProfile::OpenVino(profile) => Ok(profile),
        _ => Err(BackendError {
            message: "execution plan is missing an OpenVINO backend profile".to_string(),
        }),
    }
}

pub(super) fn validate_openvino_plan(
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

pub(super) fn derive_residency(plan: &ExecutionPlan) -> PreparedResidency {
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

pub(super) fn estimate_resident_memory_bytes(
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

pub(super) fn estimate_prefill_ms(
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

pub(super) fn estimate_decode_ms(profile: &OpenVinoExecutionProfile, plan: &ExecutionPlan) -> u64 {
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

pub(super) fn placement_summary(plan: &ExecutionPlan, stage: PipelineStage) -> String {
    plan.placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(format_placement)
        .unwrap_or_else(|| "unassigned".to_string())
}

pub(super) fn lowering_summary(plan: &ExecutionPlan) -> String {
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

pub(super) fn offload_profile_label(plan: &ExecutionPlan) -> &'static str {
    match plan.tiered_offload.as_ref().map(|tier| tier.profile) {
        Some(loci_protocol::TieredOffloadProfile::Auto) => "auto",
        Some(loci_protocol::TieredOffloadProfile::GpuResident) => "gpu_resident",
        Some(loci_protocol::TieredOffloadProfile::Balanced) => "balanced",
        Some(loci_protocol::TieredOffloadProfile::DiskHeavy) => "disk_heavy",
        None => "none",
    }
}

fn placement_target(plan: &ExecutionPlan, stage: PipelineStage) -> Option<AcceleratorKind> {
    plan.placements
        .iter()
        .find(|placement| placement.stage == stage)
        .map(|placement| placement.target)
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

fn device_id_to_openvino_name(device_id: &str) -> &'static str {
    if device_id.starts_with("npu:") {
        "NPU"
    } else if device_id.starts_with("gpu:") {
        "GPU"
    } else {
        "CPU"
    }
}

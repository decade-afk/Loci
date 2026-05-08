use super::placement::{hetero_device_names, placement_device_id, should_dynamic_reoffload};
use loci_protocol::{
    AcceleratorKind, BackendDescriptor, BackendExecutionProfile, BackendRuntimeFamily,
    CandleExecutionProfile, CandleTensorResidency, GenericExecutionProfile, HardwareTopology,
    ModelDescriptor, OpenVinoExecutionMode, OpenVinoExecutionProfile, PipelineStage,
    PlacementDecision,
};

/// Converts generic placements into a backend-specific execution profile.
pub(super) fn build_backend_profile(
    backend: &BackendDescriptor,
    model: &ModelDescriptor,
    topology: &HardwareTopology,
    placements: &[PlacementDecision],
) -> BackendExecutionProfile {
    match backend.runtime_family {
        BackendRuntimeFamily::OpenVino => {
            BackendExecutionProfile::OpenVino(build_openvino_profile(model, topology, placements))
        }
        BackendRuntimeFamily::Candle => {
            BackendExecutionProfile::Candle(build_candle_profile(placements))
        }
        _ => BackendExecutionProfile::Generic(GenericExecutionProfile {
            session_key: format!("{}-session", backend.name),
            summary: format!("generic execution profile for backend `{}`", backend.name),
        }),
    }
}

/// Constructs the OpenVINO execution profile from the generic plan.
fn build_openvino_profile(
    model: &ModelDescriptor,
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
            "ov:{}:{}:{}:{}:{}",
            sanitize_session_component(&model.name),
            prefill_device.as_deref().unwrap_or("cpu"),
            decode_device.as_deref().unwrap_or("cpu"),
            kv_cache_device.as_deref().unwrap_or("cpu"),
            weights_device.as_deref().unwrap_or("cpu")
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
fn build_candle_profile(placements: &[PlacementDecision]) -> CandleExecutionProfile {
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

fn sanitize_session_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

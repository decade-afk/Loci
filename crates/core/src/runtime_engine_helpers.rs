use crate::config::EngineConfig;
use crate::snapshot::EngineFeatureSnapshot;
use loci_protocol::{Backend, BackendExecutionProfile, ExecutionPlan};

/// Instantiates the statically compiled backend set for this build.
pub(crate) fn builtin_backends() -> Vec<Box<dyn Backend>> {
    let mut backends: Vec<Box<dyn Backend>> = Vec::new();

    #[cfg(feature = "candle")]
    backends.push(loci_backend_candle::boxed_backend());

    #[cfg(feature = "openvino")]
    backends.push(loci_backend_openvino::boxed_backend());

    backends
}

/// Extracts the backend-specific session key from an execution plan.
pub(crate) fn session_key(plan: &ExecutionPlan) -> &str {
    match &plan.backend_profile {
        BackendExecutionProfile::OpenVino(profile) => &profile.session_key,
        BackendExecutionProfile::Candle(profile) => &profile.session_key,
        BackendExecutionProfile::Generic(profile) => &profile.session_key,
    }
}

/// Computes the feature snapshot for the current configuration.
pub(crate) fn feature_snapshot(config: &EngineConfig) -> EngineFeatureSnapshot {
    EngineFeatureSnapshot {
        openvino: cfg!(feature = "openvino"),
        candle: cfg!(feature = "candle"),
        gguf: cfg!(feature = "gguf"),
        kernels_llama: cfg!(feature = "kernels-llama"),
        tiered_offload: cfg!(feature = "tiered-offload") && config.tiered_offload.enabled,
        paged_kv: cfg!(feature = "paged-kv") && config.paged_kv.enabled,
        power_aware: cfg!(feature = "power-aware"),
        dynamic_routing: cfg!(feature = "dynamic-routing") && config.routing.enabled,
        mobile: cfg!(feature = "mobile"),
        neon: cfg!(feature = "neon"),
        coreml: cfg!(feature = "coreml"),
        qnn: cfg!(feature = "qnn"),
    }
}

#[cfg(feature = "tiered-offload")]
pub(crate) fn segment_bytes(
    session: &loci_tiered_offload::TieredSessionSnapshot,
    tensor: loci_tiered_offload::SpillTensorKind,
) -> u64 {
    session
        .segments
        .iter()
        .filter(|segment| segment.tensor == tensor)
        .map(|segment| segment.length_bytes)
        .sum()
}

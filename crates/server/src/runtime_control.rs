use loci_protocol::{RoutingStrategy, TieredOffloadProfile};
use serde::{Deserialize, Serialize};

/// Captures the mutable routing controls exposed by the server surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeRoutingConfig {
    pub enabled: bool,
    pub strategy: RoutingStrategy,
    pub max_loaded_models: Option<usize>,
}

/// Captures the mutable planner and routing knobs exposed by the server runtime-control API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeControlConfig {
    pub model_keep_alive_secs: u64,
    pub tiered_offload_enabled: bool,
    #[serde(rename = "tiered_offload_profile", alias = "large_model_mode")]
    pub large_model_mode: TieredOffloadProfile,
    pub spill_threshold_bytes: Option<u64>,
    pub max_disk_bytes: Option<u64>,
    pub prefetch_window_bytes: Option<u64>,
    pub kv_cache_enabled: bool,
    pub kv_block_size_tokens: u32,
    pub kv_page_size_bytes: u64,
    pub kv_prefix_cache_enabled: bool,
    pub kv_type_k: String,
    pub kv_type_v: String,
    pub routing: RuntimeRoutingConfig,
}

impl RuntimeControlConfig {
    /// Constructs a server-facing runtime control config from individual planner knobs.
    pub fn new(
        prefetch_window_bytes: Option<u64>,
        routing_enabled: bool,
        routing_strategy: RoutingStrategy,
        max_loaded_models: Option<usize>,
        model_keep_alive_secs: u64,
        tiered_offload_enabled: bool,
        large_model_mode: TieredOffloadProfile,
        spill_threshold_bytes: Option<u64>,
        max_disk_bytes: Option<u64>,
        kv_cache_enabled: bool,
        kv_block_size_tokens: u32,
        kv_page_size_bytes: u64,
        kv_prefix_cache_enabled: bool,
        kv_type_k: String,
        kv_type_v: String,
    ) -> Self {
        Self {
            model_keep_alive_secs,
            tiered_offload_enabled,
            large_model_mode,
            spill_threshold_bytes,
            max_disk_bytes,
            prefetch_window_bytes,
            kv_cache_enabled,
            kv_block_size_tokens,
            kv_page_size_bytes,
            kv_prefix_cache_enabled,
            kv_type_k,
            kv_type_v,
            routing: RuntimeRoutingConfig {
                enabled: routing_enabled,
                strategy: routing_strategy,
                max_loaded_models,
            },
        }
    }

    /// Builds the control view from a live engine snapshot plus the disk prefetch window override.
    pub(crate) fn from_engine_snapshot(
        snapshot: &loci_core::RuntimeSnapshot,
        prefetch_window_bytes: Option<u64>,
    ) -> Self {
        Self::new(
            prefetch_window_bytes,
            snapshot.routing.enabled,
            snapshot.routing.strategy.clone(),
            snapshot.routing.max_loaded_models,
            snapshot.config.model_keep_alive_secs,
            snapshot.config.tiered_offload_enabled,
            snapshot.config.tiered_offload_profile,
            snapshot.config.spill_threshold_bytes,
            snapshot.config.max_disk_bytes,
            snapshot.config.kv_cache_enabled,
            snapshot.config.kv_block_size_tokens,
            snapshot.config.kv_page_size_bytes,
            snapshot.config.kv_prefix_cache_enabled,
            snapshot.config.kv_type_k.clone(),
            snapshot.config.kv_type_v.clone(),
        )
    }
}

/// Snapshots the mutable runtime-control state alongside the live engine state.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeControlSnapshot {
    pub config: RuntimeControlConfig,
    pub model_pool: loci_core::ModelPoolSnapshot,
    pub tiered_offload_runtime: Option<loci_core::TieredOffloadRuntimeSnapshot>,
    pub features: loci_core::EngineFeatureSnapshot,
}

/// Keeps the current mutable runtime-control state alongside the engine loop.
#[derive(Debug, Clone)]
pub(crate) struct ServerRuntimeControlState {
    pub(crate) config: RuntimeControlConfig,
}

/// Combines the mutable runtime-control state with the current engine snapshot.
pub(crate) fn runtime_control_snapshot(
    engine: &loci_core::InferenceEngine,
    runtime_state: &ServerRuntimeControlState,
) -> RuntimeControlSnapshot {
    let snapshot = engine.runtime_snapshot();
    RuntimeControlSnapshot {
        config: runtime_state.config.clone(),
        model_pool: snapshot.model_pool,
        tiered_offload_runtime: snapshot.tiered_offload_runtime,
        features: snapshot.features,
    }
}

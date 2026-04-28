use loci_protocol::{
    BackendDescriptor, HardwareTopology, ModelDescriptor, PreparedModel, RoutingStrategy,
    TieredOffloadProfile,
};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct EngineFeatureSnapshot {
    pub openvino: bool,
    pub candle: bool,
    pub tiered_offload: bool,
    pub paged_kv: bool,
    pub power_aware: bool,
    pub dynamic_routing: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoutingSnapshot {
    pub enabled: bool,
    pub strategy: RoutingStrategy,
    pub max_loaded_models: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeConfigSnapshot {
    pub model_keep_alive_secs: u64,
    pub model_aliases: HashMap<String, String>,
    pub tiered_offload_enabled: bool,
    pub tiered_offload_profile: TieredOffloadProfile,
    pub spill_threshold_bytes: Option<u64>,
    pub max_disk_bytes: Option<u64>,
    pub kv_cache_enabled: bool,
    pub kv_block_size_tokens: u32,
    pub kv_page_size_bytes: u64,
    pub kv_prefix_cache_enabled: bool,
    pub kv_type_k: String,
    pub kv_type_v: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelPoolSnapshot {
    pub registered_models: usize,
    pub resident_models: Vec<String>,
    pub prepared_models: Vec<PreparedModel>,
    pub resident_memory_bytes: u64,
    pub resident_budget_bytes: u64,
    pub keep_alive_secs: u64,
    pub max_loaded_models: Option<usize>,
    pub last_routed_model: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSnapshot {
    pub backends: Vec<BackendDescriptor>,
    pub topology: HardwareTopology,
    pub models: Vec<ModelDescriptor>,
    pub preferred_backend: Option<String>,
    pub config: RuntimeConfigSnapshot,
    pub routing: RoutingSnapshot,
    pub model_pool: ModelPoolSnapshot,
    pub features: EngineFeatureSnapshot,
}

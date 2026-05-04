//! Serializable runtime snapshots exposed by the engine and server layers.

use loci_protocol::{
    BackendAssetCapabilities, BackendDescriptor, BackendKernelCatalog, BackendLoweringCapabilities,
    HardwareTopology, ModelDescriptor, ModelReadinessReport, PreparedModel, RoutingStrategy,
    TieredOffloadProfile,
};
use serde::Serialize;
use std::collections::HashMap;

/// Reports basic host-level storage information outside backend-specific topology.
#[derive(Debug, Clone, Serialize)]
pub struct HostDiskSnapshot {
    pub name: String,
    pub mount_point: String,
    pub file_system: String,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub is_removable: bool,
}

/// Summarizes one lightweight capability probe executed locally on the host.
#[derive(Debug, Clone, Serialize)]
pub struct HostProbeSnapshot {
    pub cpu_scalar_gops: f64,
    pub memory_bandwidth_gbps: f64,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
    pub probe_bytes: u64,
    pub probe_duration_ms: u64,
}

/// Captures backend-agnostic host capabilities discovered directly from the OS.
#[derive(Debug, Clone, Serialize)]
pub struct HostCapabilitySnapshot {
    pub target_family: String,
    pub target_os: String,
    pub target_arch: String,
    pub mobile_class: bool,
    pub host_name: Option<String>,
    pub os_name: Option<String>,
    pub os_version: Option<String>,
    pub kernel_version: Option<String>,
    pub cpu_brand: Option<String>,
    pub cpu_vendor: Option<String>,
    pub cpu_frequency_mhz: Option<u64>,
    pub physical_cores: Option<usize>,
    pub logical_cores: usize,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub total_swap_bytes: u64,
    pub free_swap_bytes: u64,
    pub uptime_secs: u64,
    pub load_average_one: f64,
    pub load_average_five: f64,
    pub load_average_fifteen: f64,
    pub disks: Vec<HostDiskSnapshot>,
    pub probe: HostProbeSnapshot,
}

/// Reports which optional Loci features are compiled and active at runtime.
#[derive(Debug, Clone, Serialize)]
pub struct EngineFeatureSnapshot {
    pub openvino: bool,
    pub candle: bool,
    pub gguf: bool,
    pub kernels_llama: bool,
    pub tiered_offload: bool,
    pub paged_kv: bool,
    pub power_aware: bool,
    pub dynamic_routing: bool,
    pub mobile: bool,
    pub neon: bool,
    pub coreml: bool,
    pub qnn: bool,
}

/// Summarizes the active routing configuration.
#[derive(Debug, Clone, Serialize)]
pub struct RoutingSnapshot {
    pub enabled: bool,
    pub strategy: RoutingStrategy,
    pub max_loaded_models: Option<usize>,
}

/// Captures planner-facing runtime configuration after all updates are applied.
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

/// Exposes the active spill runtime state managed by `loci-tiered-offload`.
#[derive(Debug, Clone, Serialize)]
pub struct TieredOffloadRuntimeSnapshot {
    pub root_dir: String,
    pub total_spill_bytes: u64,
    pub total_prefetched_bytes: u64,
    pub sessions: Vec<TieredOffloadSessionSnapshot>,
}

/// Exposes one spill session created from a tiered-offload plan.
#[derive(Debug, Clone, Serialize)]
pub struct TieredOffloadSessionSnapshot {
    pub session_key: String,
    pub model_name: String,
    pub spill_path: String,
    pub mapped_bytes: u64,
    pub prefetched_bytes: u64,
    pub scheduled_prefetch_requests: usize,
    pub completed_prefetch_requests: usize,
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub activations_bytes: u64,
}

/// Describes the current model-pool state managed by the runtime.
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

/// Top-level inspection payload returned by the runtime and server APIs.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSnapshot {
    pub backends: Vec<BackendDescriptor>,
    pub backend_assets: Vec<BackendAssetCapabilities>,
    pub backend_lowering: Vec<BackendLoweringCapabilities>,
    pub backend_kernels: Vec<BackendKernelCatalog>,
    pub host: HostCapabilitySnapshot,
    pub topology: HardwareTopology,
    pub models: Vec<ModelDescriptor>,
    pub model_diagnostics: Vec<ModelReadinessReport>,
    pub preferred_backend: Option<String>,
    pub config: RuntimeConfigSnapshot,
    pub routing: RoutingSnapshot,
    pub model_pool: ModelPoolSnapshot,
    pub tiered_offload_runtime: Option<TieredOffloadRuntimeSnapshot>,
    pub features: EngineFeatureSnapshot,
}

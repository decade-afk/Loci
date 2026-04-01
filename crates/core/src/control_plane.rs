use loci_plugin_api::{CoreComponent, LegacyRuntimeBridge, PluginSourceFormat};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SamplingHookSource {
    None,
    NativeRuntime,
    LegacyCompat,
    DynamicRegistration,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PluginRuntimeStatus {
    pub name: String,
    pub version: String,
    pub supports_ai_infra: bool,
    pub supports_ai_agent: bool,
    pub source_format: PluginSourceFormat,
    pub runtime_bridge: LegacyRuntimeBridge,
    pub declares_inference_rewriter: bool,
    pub declares_sampling_hook: bool,
    pub sampling_hook_source: SamplingHookSource,
    pub registered_sampling_hook: bool,
    pub effective_sampling_hook: bool,
    pub materialized_legacy_runtime: bool,
    pub active_inference_rewriter: bool,
    pub has_sampling_hook: bool,
    pub is_legacy_compat: bool,
    pub legacy_text_candidate: bool,
    pub active_legacy_text: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PluginRuntimeDetail {
    pub status: PluginRuntimeStatus,
    pub declared_core_rewriters: Vec<CoreComponent>,
    pub auto_activate_components: Vec<CoreComponent>,
    pub active_core_rewriters: Vec<CoreComponent>,
    pub model_providers: Vec<String>,
    pub inference_hooks: Vec<String>,
    pub commands: Vec<String>,
    pub legacy_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CoreRewriterStatus {
    pub component: CoreComponent,
    pub plugin_name: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RuntimeSnapshot {
    pub plugin_count: usize,
    pub loaded_plugin_names: Vec<String>,
    pub active_backend: Option<String>,
    pub active_model_path: Option<String>,
    pub active_model_info: Option<ModelRuntimeInfo>,
    pub active_inference: Option<String>,
    pub configured_core_rewriters: Vec<CoreRewriterStatus>,
    pub legacy_text_candidates: Vec<String>,
    pub active_legacy_text: Vec<String>,
    pub plugins: Vec<PluginRuntimeStatus>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ManagementHealthStatus {
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct InferenceActivationStatus {
    pub status: &'static str,
    pub component: CoreComponent,
    pub plugin_name: String,
    pub active_inference: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LegacyTextPluginActivationStatus {
    pub status: &'static str,
    pub plugin_name: String,
    pub active_legacy_text: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelRuntimeInfo {
    pub architecture: String,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_layer: u32,
    pub param_count: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ModelLoadStatus {
    pub status: &'static str,
    pub backend_name: String,
    pub model_path: String,
    pub active_backend: Option<String>,
    pub active_model_path: Option<String>,
    pub active_model_info: Option<ModelRuntimeInfo>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelLoadStrategyRequest {
    #[default]
    Strict,
    AutoReduceGpuLayers {
        step: u32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadSplitMode {
    None,
    #[default]
    Layer,
    Row,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelLoadConfig {
    pub model_path: String,
    #[serde(default = "default_model_n_ctx")]
    pub n_ctx: u32,
    #[serde(default)]
    pub n_threads: Option<u32>,
    #[serde(default = "default_model_n_batch")]
    pub n_batch: u32,
    #[serde(default = "default_model_use_gpu")]
    pub use_gpu: bool,
    #[serde(default = "default_model_n_gpu_layers")]
    pub n_gpu_layers: i32,
    #[serde(default = "default_model_use_mmap")]
    pub use_mmap: bool,
    #[serde(default)]
    pub use_mlock: bool,
    #[serde(default = "default_model_kv_offload")]
    pub kv_offload: bool,
    #[serde(default = "default_model_op_offload")]
    pub op_offload: bool,
    #[serde(default)]
    pub split_mode: ModelLoadSplitMode,
    #[serde(default)]
    pub main_gpu: u32,
    #[serde(default)]
    pub tensor_split: Option<Vec<f32>>,
    #[serde(default)]
    pub load_strategy: ModelLoadStrategyRequest,
}

impl Default for ModelLoadConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            n_ctx: default_model_n_ctx(),
            n_threads: None,
            n_batch: default_model_n_batch(),
            use_gpu: default_model_use_gpu(),
            n_gpu_layers: default_model_n_gpu_layers(),
            use_mmap: default_model_use_mmap(),
            use_mlock: false,
            kv_offload: default_model_kv_offload(),
            op_offload: default_model_op_offload(),
            split_mode: ModelLoadSplitMode::default(),
            main_gpu: 0,
            tensor_split: None,
            load_strategy: ModelLoadStrategyRequest::Strict,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelLoadRequest {
    pub backend_name: String,
    #[serde(default)]
    pub config: ModelLoadConfig,
}

const fn default_model_n_ctx() -> u32 {
    4096
}

const fn default_model_n_batch() -> u32 {
    512
}

const fn default_model_use_gpu() -> bool {
    true
}

const fn default_model_n_gpu_layers() -> i32 {
    -1
}

const fn default_model_use_mmap() -> bool {
    true
}

const fn default_model_kv_offload() -> bool {
    true
}

const fn default_model_op_offload() -> bool {
    true
}

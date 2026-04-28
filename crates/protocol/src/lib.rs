use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorKind {
    Cpu,
    Gpu,
    Npu,
    Disk,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ThermalState {
    Nominal,
    Warm,
    Hot,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceDescriptor {
    pub id: String,
    pub name: String,
    pub kind: AcceleratorKind,
    pub memory_bytes: Option<u64>,
    pub compute_units: Option<u32>,
    pub power_watts: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PowerState {
    pub battery_powered: bool,
    pub battery_percent: Option<u8>,
    pub thermal_state: ThermalState,
    pub power_budget_watts: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareTopology {
    pub devices: Vec<DeviceDescriptor>,
    pub power: PowerState,
}

impl Default for HardwareTopology {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            power: PowerState {
                battery_powered: false,
                battery_percent: None,
                thermal_state: ThermalState::Nominal,
                power_budget_watts: None,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredOffloadConfig {
    pub enabled: bool,
    pub max_disk_bytes: Option<u64>,
    pub spill_threshold_bytes: Option<u64>,
    pub prefetch_window_bytes: Option<u64>,
    pub profile: TieredOffloadProfile,
}

impl Default for TieredOffloadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_disk_bytes: Some(64 * 1024 * 1024 * 1024),
            spill_threshold_bytes: Some(8 * 1024 * 1024 * 1024),
            prefetch_window_bytes: Some(256 * 1024 * 1024),
            profile: TieredOffloadProfile::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TieredOffloadProfile {
    Auto,
    GpuResident,
    Balanced,
    DiskHeavy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PagedKvConfig {
    pub enabled: bool,
    pub page_size_bytes: u64,
    pub block_size_tokens: u32,
    pub prefix_cache_enabled: bool,
    pub max_cache_pages: u32,
    pub type_k: String,
    pub type_v: String,
}

impl Default for PagedKvConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            page_size_bytes: 1 << 20,
            block_size_tokens: 16,
            prefix_cache_enabled: true,
            max_cache_pages: 4096,
            type_k: "f16".to_string(),
            type_v: "f16".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingConfig {
    pub enabled: bool,
    pub max_loaded_models: Option<usize>,
    pub strategy: RoutingStrategy,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_loaded_models: Some(4),
            strategy: RoutingStrategy::PromptComplexity,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    PromptComplexity,
    LatencyAware,
    PowerAware,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDescriptor {
    pub name: String,
    pub path: PathBuf,
    pub architecture: String,
    pub memory_bytes: Option<u64>,
    pub parameter_count: Option<u64>,
    pub context_length: Option<u32>,
    pub preferred_backend: Option<String>,
}

impl ModelDescriptor {
    pub fn inferred_format(&self) -> ModelFormat {
        let extension = self
            .path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase());

        match extension.as_deref() {
            Some("xml") => ModelFormat::OpenVinoIr,
            Some("onnx") => ModelFormat::Onnx,
            Some("gguf") => ModelFormat::Gguf,
            Some("safetensors") => ModelFormat::SafeTensors,
            Some("bin") => ModelFormat::PytorchBin,
            Some("blob") => ModelFormat::OpenVinoBlob,
            Some(_) => ModelFormat::Unknown,
            None => {
                if self.path.file_name().is_none() {
                    ModelFormat::Directory
                } else {
                    ModelFormat::Unknown
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    OpenVinoIr,
    OpenVinoBlob,
    Onnx,
    Gguf,
    SafeTensors,
    PytorchBin,
    Directory,
    Unknown,
}

impl ModelFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelFormat::OpenVinoIr => "openvino_ir",
            ModelFormat::OpenVinoBlob => "openvino_blob",
            ModelFormat::Onnx => "onnx",
            ModelFormat::Gguf => "gguf",
            ModelFormat::SafeTensors => "safetensors",
            ModelFormat::PytorchBin => "pytorch_bin",
            ModelFormat::Directory => "directory",
            ModelFormat::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub target_model: Option<String>,
    pub structured_output: bool,
    pub tool_calling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionResponse {
    pub text: String,
    pub backend: String,
    pub model: String,
    pub plan: ExecutionPlan,
    pub telemetry: BackendTelemetry,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackendDescriptor {
    pub name: String,
    pub supports_cpu: bool,
    pub supports_gpu: bool,
    pub supports_npu: bool,
    pub supports_disk_tiering: bool,
    pub supports_paged_kv: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionPlan {
    pub backend: String,
    pub route: RouteDecision,
    pub placements: Vec<PlacementDecision>,
    pub kv_cache: KvCachePlan,
    pub tiered_offload: Option<TieredOffloadPlan>,
    pub backend_profile: BackendExecutionProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteDecision {
    pub selected_model: String,
    pub reason: String,
    pub alternatives: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlacementDecision {
    pub stage: PipelineStage,
    pub target: AcceleratorKind,
    pub device_id: Option<String>,
    pub memory_bytes: Option<u64>,
    pub rationale: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Load,
    Prefill,
    Decode,
    KvCache,
    Weights,
    Sampling,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvCachePlan {
    pub strategy: String,
    pub shared_across_models: bool,
    pub page_size_bytes: Option<u64>,
    pub block_size_tokens: Option<u32>,
    pub max_cache_bytes: Option<u64>,
    pub type_k: Option<String>,
    pub type_v: Option<String>,
    pub tiered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredOffloadPlan {
    pub spill_bytes: u64,
    pub prefetch_window_bytes: u64,
    pub target_device: String,
    pub profile: TieredOffloadProfile,
    pub policy: TieredOffloadPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredOffloadPolicy {
    pub weights: TieredPlacementPercentages,
    pub kv_cache: TieredPlacementPercentages,
    pub activations: TieredPlacementPercentages,
    pub cpu_cache_compute: bool,
    pub compress_weights: bool,
    pub compress_kv_cache: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredPlacementPercentages {
    pub gpu_percent: u8,
    pub cpu_percent: u8,
    pub disk_percent: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum BackendExecutionProfile {
    OpenVino(OpenVinoExecutionProfile),
    Candle(CandleExecutionProfile),
    Generic(GenericExecutionProfile),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenVinoExecutionProfile {
    pub session_key: String,
    pub execution_mode: OpenVinoExecutionMode,
    pub genai_pipeline: bool,
    pub hetero_devices: Vec<String>,
    pub prefill_device: Option<String>,
    pub decode_device: Option<String>,
    pub kv_cache_device: Option<String>,
    pub weights_device: Option<String>,
    pub dynamic_reoffload: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenVinoExecutionMode {
    Hetero,
    NpuFirst,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandleExecutionProfile {
    pub session_key: String,
    pub prefill_device: String,
    pub decode_device: String,
    pub kv_cache_device: String,
    pub tensor_residency: CandleTensorResidency,
    pub fallback_reason: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandleTensorResidency {
    MemoryOnly,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenericExecutionProfile {
    pub session_key: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedModel {
    pub model_name: String,
    pub backend: String,
    pub session_key: String,
    pub residency: PreparedResidency,
    pub estimated_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreparedResidency {
    Memory,
    Hybrid,
    DiskBacked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendTelemetry {
    pub estimated_prefill_ms: u64,
    pub estimated_decode_ms: u64,
    pub generated_tokens: u32,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct BackendError {
    pub message: String,
}

pub type BackendResult<T> = std::result::Result<T, BackendError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendOutput {
    pub text: String,
    pub telemetry: BackendTelemetry,
}

pub trait Backend: Send + Sync {
    fn descriptor(&self) -> BackendDescriptor;
    fn discover_topology(&self) -> HardwareTopology;
    fn supports_model(&self, model: &ModelDescriptor) -> bool;
    fn prepare(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
    ) -> BackendResult<PreparedModel>;
    fn execute(
        &self,
        prepared: &PreparedModel,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
    ) -> BackendResult<BackendOutput>;
}

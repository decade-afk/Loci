//! Stable protocol types shared across the Loci workspace.
//!
//! The documentation style in this crate follows the conventions used by
//! high-quality Rust projects: each public type explains its role in the
//! system, and the protocol intentionally stays runtime-agnostic so the core,
//! backends, CLI, and server can evolve independently.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Enumerates the logical resource tiers Loci can target during planning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorKind {
    Cpu,
    Gpu,
    Npu,
    Disk,
}

/// Identifies the execution runtime family behind a backend implementation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BackendRuntimeFamily {
    OpenVino,
    Candle,
    CoreMl,
    Qnn,
    Rknn,
    WasiNn,
    WebGpu,
    OnnxRuntime,
    Tract,
    Generic,
}

/// Captures coarse thermal pressure seen by the planner.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ThermalState {
    Nominal,
    Warm,
    Hot,
    Critical,
}

/// Describes a device that participates in the merged runtime topology.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceDescriptor {
    pub id: String,
    pub name: String,
    pub kind: AcceleratorKind,
    pub platform: Option<String>,
    pub memory_bytes: Option<u64>,
    pub compute_units: Option<u32>,
    pub power_watts: Option<f32>,
}

/// Represents host power constraints that influence routing and placement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PowerState {
    pub battery_powered: bool,
    pub battery_percent: Option<u8>,
    pub thermal_state: ThermalState,
    pub power_budget_watts: Option<u64>,
}

/// Aggregates the discovered execution resources and their power state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareTopology {
    pub devices: Vec<DeviceDescriptor>,
    pub power: PowerState,
}

impl Default for HardwareTopology {
    /// Creates an empty topology with a neutral power state.
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

/// Configures the policy layer for disk-backed model and KV spill.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredOffloadConfig {
    pub enabled: bool,
    pub max_disk_bytes: Option<u64>,
    pub spill_threshold_bytes: Option<u64>,
    pub prefetch_window_bytes: Option<u64>,
    pub profile: TieredOffloadProfile,
}

impl Default for TieredOffloadConfig {
    /// Chooses conservative defaults that enable tiering without forcing spill.
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

/// Expresses the high-level residency bias for tiered offload planning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TieredOffloadProfile {
    Auto,
    GpuResident,
    Balanced,
    DiskHeavy,
}

/// Configures the planner-facing shape of the paged KV cache.
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
    /// Enables paged KV with f16 defaults sized for medium edge workloads.
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

/// Controls optional model routing across a pool of registered models.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingConfig {
    pub enabled: bool,
    pub max_loaded_models: Option<usize>,
    pub strategy: RoutingStrategy,
}

impl Default for RoutingConfig {
    /// Leaves routing disabled unless explicitly enabled by the caller.
    fn default() -> Self {
        Self {
            enabled: false,
            max_loaded_models: Some(4),
            strategy: RoutingStrategy::PromptComplexity,
        }
    }
}

/// Selects the heuristic family used when dynamic routing is enabled.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    PromptComplexity,
    LatencyAware,
    PowerAware,
}

/// Identifies a model known to the runtime and enough metadata to plan around it.
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
    /// Infers the model artifact format from the model path.
    pub fn inferred_format(&self) -> ModelFormat {
        if self.path.is_dir() {
            return ModelFormat::Directory;
        }

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
            Some("bin") | Some("pt") | Some("pth") => ModelFormat::PytorchBin,
            Some("blob") => ModelFormat::OpenVinoBlob,
            Some(_) => ModelFormat::Unknown,
            None => ModelFormat::Unknown,
        }
    }

    /// Returns whether the model architecture should be treated as multimodal.
    pub fn is_multimodal_architecture(&self) -> bool {
        let architecture = self.architecture.to_ascii_lowercase();
        architecture.contains("vlm")
            || architecture.contains("vision")
            || architecture.contains("multimodal")
            || architecture.contains("minicpm-v")
            || architecture.contains("qwen2") && architecture.contains("vl")
            || architecture.contains("phi") && architecture.contains('v')
    }
}

/// Enumerates the model formats that Loci currently recognizes.
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
    /// Returns the stable protocol label used in responses and errors.
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

/// Identifies the concrete asset layout discovered on disk for a model path.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelAssetLayout {
    Missing,
    OpenVinoGenAiExport,
    OpenVinoIr,
    OpenVinoBlob,
    OnnxModel,
    GgufFile,
    GgufDirectory,
    SafeTensorsFile,
    SafeTensorsDirectory,
    PytorchBinFile,
    PytorchCheckpointDirectory,
    TransformersCheckpoint,
    UnknownDirectory,
    UnknownFile,
}

impl ModelAssetLayout {
    /// Returns the stable protocol label used in responses and diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelAssetLayout::Missing => "missing",
            ModelAssetLayout::OpenVinoGenAiExport => "openvino_genai_export",
            ModelAssetLayout::OpenVinoIr => "openvino_ir",
            ModelAssetLayout::OpenVinoBlob => "openvino_blob",
            ModelAssetLayout::OnnxModel => "onnx_model",
            ModelAssetLayout::GgufFile => "gguf_file",
            ModelAssetLayout::GgufDirectory => "gguf_directory",
            ModelAssetLayout::SafeTensorsFile => "safetensors_file",
            ModelAssetLayout::SafeTensorsDirectory => "safetensors_directory",
            ModelAssetLayout::PytorchBinFile => "pytorch_bin_file",
            ModelAssetLayout::PytorchCheckpointDirectory => "pytorch_checkpoint_directory",
            ModelAssetLayout::TransformersCheckpoint => "transformers_checkpoint",
            ModelAssetLayout::UnknownDirectory => "unknown_directory",
            ModelAssetLayout::UnknownFile => "unknown_file",
        }
    }
}

/// Classifies the role a shard or artifact plays inside a model asset bundle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelShardRole {
    Weights,
    Tokenizer,
    Config,
    Graph,
    Metadata,
    Unknown,
}

/// Describes one concrete file shard discovered under a model asset root.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelShardDescriptor {
    pub name: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub format: ModelFormat,
    pub role: ModelShardRole,
    pub mmap_candidate: bool,
}

/// Provides a format-agnostic inventory view over a model's on-disk assets.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelAssetInventory {
    pub root: PathBuf,
    pub layout: ModelAssetLayout,
    pub total_bytes: u64,
    pub shards: Vec<ModelShardDescriptor>,
}

/// Describes a single inference request entering the Loci runtime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionRequest {
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub target_model: Option<String>,
    #[serde(default)]
    pub images: Vec<ImageInput>,
    pub structured_output: bool,
    pub tool_calling: bool,
}

/// Describes one image attachment that can be consumed by multimodal backends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageInput {
    Path {
        path: PathBuf,
    },
    Url {
        url: String,
    },
    Base64 {
        data_base64: String,
        media_type: Option<String>,
    },
}

/// Returns the backend output together with the execution plan that produced it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionResponse {
    pub text: String,
    pub backend: String,
    pub model: String,
    pub plan: ExecutionPlan,
    pub telemetry: BackendTelemetry,
}

/// Declares the capabilities of a backend implementation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackendDescriptor {
    pub name: String,
    pub runtime_family: BackendRuntimeFamily,
    pub supports_cpu: bool,
    pub supports_gpu: bool,
    pub supports_npu: bool,
    pub supports_disk_tiering: bool,
    pub supports_paged_kv: bool,
    pub supports_multimodal: bool,
}

/// Describes the lowest execution granularity a backend can currently accept.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoweringGranularity {
    Graph,
    Subgraph,
    Layer,
    Tensor,
}

/// Groups low-level operator families that matter for chip-specific backends.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChipOperatorClass {
    Attention,
    Matmul,
    Embedding,
    RmsNorm,
    Convolution,
    VisionEncoder,
    Mlp,
    KvCache,
    Sampling,
}

/// Identifies the implementation family behind a kernel/operator realization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KernelImplementationKind {
    Rust,
    Cpp,
    VendorRuntime,
    IrGraph,
    ExternalBridge,
}

/// Describes how production-ready a registered kernel is inside the workspace.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum KernelMaturity {
    Planned,
    Stubbed,
    Integrated,
    Validated,
}

/// Attributes one kernel to its upstream origin so ports remain auditable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KernelOrigin {
    pub project: String,
    pub component: String,
    pub license: Option<String>,
    pub notes: Vec<String>,
}

/// Describes one statically-registered kernel that a backend can dispatch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KernelDescriptor {
    pub backend: String,
    pub kernel_name: String,
    pub operator_class: ChipOperatorClass,
    pub implementation: KernelImplementationKind,
    pub maturity: KernelMaturity,
    pub origin: KernelOrigin,
    pub supported_targets: Vec<AcceleratorKind>,
    pub supported_formats: Vec<ModelFormat>,
    pub supported_architectures: Vec<String>,
    pub dispatch_keys: Vec<String>,
    pub notes: Vec<String>,
}

/// Collects all kernels a backend wants to expose to the core for diagnostics and planning.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendKernelCatalog {
    pub backend: String,
    pub runtime_family: BackendRuntimeFamily,
    pub kernels: Vec<KernelDescriptor>,
    pub notes: Vec<String>,
}

/// Exposes the backend-lowering ABI surface that future chip backends must implement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendLoweringCapabilities {
    pub backend: String,
    pub runtime_family: BackendRuntimeFamily,
    pub granularity: LoweringGranularity,
    pub supports_real_execution: bool,
    pub supports_graph_partitioning: bool,
    pub supports_layer_affinity: bool,
    pub supports_dynamic_reoffload: bool,
    pub supports_custom_operators: bool,
    pub operator_classes: Vec<ChipOperatorClass>,
    pub notes: Vec<String>,
}

/// Identifies the execution artifact family a backend wants to consume.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionArtifactKind {
    OpenVinoIr,
    OpenVinoGenAi,
    NativeCheckpoint,
    GgufWeights,
    SafeTensorsBundle,
    PytorchCheckpoint,
    OnnxGraph,
    RuntimeDefined,
}

impl ExecutionArtifactKind {
    /// Returns the stable protocol label used in diagnostics and snapshots.
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionArtifactKind::OpenVinoIr => "openvino_ir",
            ExecutionArtifactKind::OpenVinoGenAi => "openvino_genai",
            ExecutionArtifactKind::NativeCheckpoint => "native_checkpoint",
            ExecutionArtifactKind::GgufWeights => "gguf_weights",
            ExecutionArtifactKind::SafeTensorsBundle => "safetensors_bundle",
            ExecutionArtifactKind::PytorchCheckpoint => "pytorch_checkpoint",
            ExecutionArtifactKind::OnnxGraph => "onnx_graph",
            ExecutionArtifactKind::RuntimeDefined => "runtime_defined",
        }
    }
}

/// Describes which on-disk model assets a backend can ingest directly or via lowering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendAssetCapabilities {
    pub backend: String,
    pub runtime_family: BackendRuntimeFamily,
    pub directly_supported_layouts: Vec<ModelAssetLayout>,
    pub ingestible_layouts: Vec<ModelAssetLayout>,
    pub preferred_artifact: ExecutionArtifactKind,
    pub requires_lowering_for_execution: bool,
    pub notes: Vec<String>,
}

/// Describes how explicit the planner's lowering guidance is for a backend.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoweringAffinityMode {
    Automatic,
    Planned,
    Explicit,
}

/// Captures one planner-produced subgraph or tensor-state region for backend lowering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoweringSubgraphPlan {
    pub id: String,
    pub stage: PipelineStage,
    pub operator_class: ChipOperatorClass,
    pub target: AcceleratorKind,
    pub device_id: Option<String>,
    pub affinity_tag: Option<String>,
    pub estimated_bytes: Option<u64>,
    pub spillable: bool,
    pub rationale: String,
}

/// Groups multiple lowering regions that should land on the same execution partition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoweringPartitionPlan {
    pub id: String,
    pub target: AcceleratorKind,
    pub device_id: Option<String>,
    pub affinity_tag: Option<String>,
    pub operator_classes: Vec<ChipOperatorClass>,
    pub subgraphs: Vec<String>,
    pub estimated_bytes: Option<u64>,
    pub spillable: bool,
    pub rationale: String,
}

/// Captures a normalized backend-facing operator placement derived from a subgraph plan.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoweringOperatorPlan {
    pub id: String,
    pub partition: String,
    pub subgraph: String,
    pub stage: PipelineStage,
    pub operator_class: ChipOperatorClass,
    pub target: AcceleratorKind,
    pub device_id: Option<String>,
    pub affinity_tag: Option<String>,
    pub estimated_bytes: Option<u64>,
    pub spillable: bool,
    pub rationale: String,
}

/// Carries the backend-facing subgraph partition and affinity guidance from the planner.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendLoweringPlan {
    pub backend: String,
    pub granularity: LoweringGranularity,
    pub affinity_mode: LoweringAffinityMode,
    pub subgraphs: Vec<LoweringSubgraphPlan>,
    pub partitions: Vec<LoweringPartitionPlan>,
    pub operators: Vec<LoweringOperatorPlan>,
    pub notes: Vec<String>,
}

/// Describes how ready one backend is to execute a particular model asset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelBackendReadiness {
    pub backend: String,
    pub runtime_family: BackendRuntimeFamily,
    pub format_supported: bool,
    pub preferred_artifact: ExecutionArtifactKind,
    pub ready: bool,
    pub real_execution: bool,
    pub requires_conversion: bool,
    pub supports_multimodal: bool,
    pub supports_graph_partitioning: bool,
    pub supports_low_level_ops: bool,
    pub reason: String,
}

/// Summarizes the discovered readiness state for one registered model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelReadinessReport {
    pub model_name: String,
    pub path: PathBuf,
    pub architecture: String,
    pub inferred_format: ModelFormat,
    pub asset_layout: ModelAssetLayout,
    pub asset_inventory: ModelAssetInventory,
    pub exists: bool,
    pub multimodal: bool,
    pub ready_for_inference: bool,
    pub recommended_backend: Option<String>,
    pub backend_readiness: Vec<ModelBackendReadiness>,
    pub notes: Vec<String>,
}

/// Captures the complete planning result for a request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionPlan {
    pub backend: String,
    pub route: RouteDecision,
    pub placements: Vec<PlacementDecision>,
    pub lowering_plan: Option<BackendLoweringPlan>,
    pub kv_cache: KvCachePlan,
    pub tiered_offload: Option<TieredOffloadPlan>,
    pub backend_profile: BackendExecutionProfile,
}

/// Explains which model was selected and why.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteDecision {
    pub selected_model: String,
    pub reason: String,
    pub alternatives: Vec<String>,
}

/// Assigns one pipeline stage to a logical target device.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlacementDecision {
    pub stage: PipelineStage,
    pub target: AcceleratorKind,
    pub device_id: Option<String>,
    pub memory_bytes: Option<u64>,
    pub rationale: String,
}

/// Enumerates the major stages of a single-model inference pipeline.
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

/// Describes the planned KV cache layout for a request.
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

/// Describes the spill strategy chosen for a model instance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredOffloadPlan {
    pub spill_bytes: u64,
    pub prefetch_window_bytes: u64,
    pub target_device: String,
    pub profile: TieredOffloadProfile,
    pub policy: TieredOffloadPolicy,
}

/// Breaks the spill strategy down by tensor category.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredOffloadPolicy {
    pub weights: TieredPlacementPercentages,
    pub kv_cache: TieredPlacementPercentages,
    pub activations: TieredPlacementPercentages,
    pub cpu_cache_compute: bool,
    pub compress_weights: bool,
    pub compress_kv_cache: bool,
}

/// Stores device percentages for a particular tensor category.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TieredPlacementPercentages {
    pub gpu_percent: u8,
    pub cpu_percent: u8,
    pub disk_percent: u8,
}

/// Provides backend-specific execution metadata while preserving a stable envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "backend", rename_all = "snake_case")]
pub enum BackendExecutionProfile {
    OpenVino(OpenVinoExecutionProfile),
    Candle(CandleExecutionProfile),
    Generic(GenericExecutionProfile),
}

/// Encodes the execution details required by the OpenVINO path.
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

/// Distinguishes between general heterogeneous execution and NPU-first decoding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenVinoExecutionMode {
    Hetero,
    NpuFirst,
}

/// Encodes the execution details required by the Candle fallback path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandleExecutionProfile {
    pub session_key: String,
    pub prefill_device: String,
    pub decode_device: String,
    pub kv_cache_device: String,
    pub tensor_residency: CandleTensorResidency,
    pub fallback_reason: String,
}

/// Summarizes whether tensors stay in memory or are partially disk-backed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CandleTensorResidency {
    MemoryOnly,
    Hybrid,
}

/// Provides a minimal profile for backends that do not have a richer schema yet.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GenericExecutionProfile {
    pub session_key: String,
    pub summary: String,
}

/// Records the reusable prepared state for a model/backend session pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreparedModel {
    pub model_name: String,
    pub backend: String,
    pub session_key: String,
    pub residency: PreparedResidency,
    pub estimated_memory_bytes: Option<u64>,
}

/// Describes how much of a prepared model remains memory resident.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreparedResidency {
    Memory,
    Hybrid,
    DiskBacked,
}

/// Reports lightweight backend timing estimates used by tests and demos.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendTelemetry {
    pub estimated_prefill_ms: u64,
    pub estimated_decode_ms: u64,
    pub generated_tokens: u32,
}

/// Represents a backend-local failure before it is translated into a core error.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct BackendError {
    pub message: String,
}

/// Standard result alias for backend operations.
pub type BackendResult<T> = std::result::Result<T, BackendError>;

/// Carries generated text and telemetry back from a backend execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendOutput {
    pub text: String,
    pub telemetry: BackendTelemetry,
}

/// Common interface implemented by all static Loci backends.
pub trait Backend: Send + Sync {
    /// Returns the immutable capability descriptor for this backend.
    fn descriptor(&self) -> BackendDescriptor;
    /// Reports which model asset layouts the backend can consume directly or via lowering.
    fn asset_capabilities(&self) -> BackendAssetCapabilities {
        let descriptor = self.descriptor();
        BackendAssetCapabilities {
            backend: descriptor.name,
            runtime_family: descriptor.runtime_family,
            directly_supported_layouts: Vec::new(),
            ingestible_layouts: Vec::new(),
            preferred_artifact: ExecutionArtifactKind::RuntimeDefined,
            requires_lowering_for_execution: false,
            notes: vec![
                "backend did not override asset_capabilities; treat model ingestion support as undefined".to_string(),
            ],
        }
    }
    /// Reports the lowering ABI surface exposed by this backend.
    fn lowering_capabilities(&self) -> BackendLoweringCapabilities {
        let descriptor = self.descriptor();
        BackendLoweringCapabilities {
            backend: descriptor.name,
            runtime_family: descriptor.runtime_family,
            granularity: LoweringGranularity::Graph,
            supports_real_execution: false,
            supports_graph_partitioning: false,
            supports_layer_affinity: false,
            supports_dynamic_reoffload: false,
            supports_custom_operators: false,
            operator_classes: Vec::new(),
            notes: vec![
                "backend did not override lowering_capabilities; treat it as a coarse graph-level integration only".to_string(),
            ],
        }
    }
    /// Reports the backend's statically registered kernel/operator catalog.
    fn kernel_catalog(&self) -> BackendKernelCatalog {
        let descriptor = self.descriptor();
        BackendKernelCatalog {
            backend: descriptor.name,
            runtime_family: descriptor.runtime_family,
            kernels: Vec::new(),
            notes: vec![
                "backend did not override kernel_catalog; treat low-level kernel availability as undefined".to_string(),
            ],
        }
    }
    /// Discovers the logical hardware topology visible to this backend.
    fn discover_topology(&self) -> HardwareTopology;
    /// Reports whether the backend can execute the supplied model artifact.
    fn supports_model(&self, model: &ModelDescriptor) -> bool;
    /// Prepares reusable backend state for a model and execution plan.
    fn prepare(
        &self,
        model: &ModelDescriptor,
        plan: &ExecutionPlan,
    ) -> BackendResult<PreparedModel>;
    /// Executes one request against an already prepared backend session.
    fn execute(
        &self,
        prepared: &PreparedModel,
        model: &ModelDescriptor,
        request: &SessionRequest,
        plan: &ExecutionPlan,
    ) -> BackendResult<BackendOutput>;
}

#[cfg(test)]
mod tests {
    use super::{ModelDescriptor, ModelFormat};
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn unique_temp_path(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-{label}-{suffix}"))
    }

    #[test]
    fn inferred_format_treats_existing_directories_as_directory_models() {
        let dir = unique_temp_path("model-dir");
        fs::create_dir_all(&dir).expect("dir");

        let model = ModelDescriptor {
            name: "demo".to_string(),
            path: dir.clone(),
            architecture: "vision".to_string(),
            memory_bytes: None,
            parameter_count: None,
            context_length: None,
            preferred_backend: None,
        };

        assert_eq!(model.inferred_format(), ModelFormat::Directory);
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn inferred_format_recognizes_pt_and_pth_as_pytorch_bin() {
        for extension in ["pt", "pth"] {
            let path = unique_temp_path(&format!("torch-{extension}")).with_extension(extension);
            fs::write(&path, "weights").expect("weights");

            let model = ModelDescriptor {
                name: "demo".to_string(),
                path: path.clone(),
                architecture: "llama".to_string(),
                memory_bytes: None,
                parameter_count: None,
                context_length: None,
                preferred_backend: None,
            };

            assert_eq!(model.inferred_format(), ModelFormat::PytorchBin);
            fs::remove_file(path).expect("cleanup");
        }
    }
}

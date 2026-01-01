# Loci API Reference

**Version**: 0.1.0
**Last Updated**: 2025-12-28

This document provides comprehensive API documentation for all public modules in the Loci engine.

---

## Table of Contents

1. [Core Engine API](#core-engine-api)
2. [Backend API](#backend-api)
3. [GGUF Loader API](#gguf-loader-api)
4. [Sampling API](#sampling-api)
5. [Paged Attention API](#paged-attention-api)
6. [Constraint Sampling API](#constraint-sampling-api)
7. [Suspend/Resume API](#suspendresume-api)
8. [Radix Tree Caching API](#radix-tree-caching-api)
9. [Plugin System API](#plugin-system-api)
10. [Model Registry API](#model-registry-api)
11. [LoRA API](#lora-api)
12. [Model Encryption API](#model-encryption-api)
13. [Multi-Tenancy API](#multi-tenancy-api)
14. [Multimodal API](#multimodal-api)
15. [Quantization API](#quantization-api)
16. [Kernel Fusion API](#kernel-fusion-api)
17. [Configuration API](#configuration-api)

---

## Core Engine API

### `LociEngine`

Main inference engine for running AI models. Built on llama.cpp FFI bindings for maximum performance.

#### Constructor

```rust
pub fn new(config: EngineConfig) -> anyhow::Result<Self>
```

Creates a new Loci engine instance with the specified configuration.

**Example:**
```rust
use loci::{LociEngine, EngineConfig};

let config = EngineConfig {
    model_path: "path/to/model.gguf".to_string(),
    n_gpu_layers: -1,       // Use all GPU layers
    temperature: 0.7,       // Sampling temperature
    top_k: 40,              // Top-K sampling
    top_p: 0.9,             // Top-P sampling
    ..Default::default()
};

let engine = LociEngine::new(config)?;
```

#### Methods

##### `generate`

```rust
pub fn generate(
    &self,
    prompt: &str,
    max_tokens: usize
) -> anyhow::Result<String>
```

Generates text based on the given prompt using the sampler configured in `EngineConfig`.

**Parameters:**
- `prompt`: Input text prompt
- `max_tokens`: Maximum number of tokens to generate

**Returns:** Generated text as `String`

**Example:**
```rust
let response = engine.generate("Once upon a time", 100)?;
println!("{}", response);
```

**Note:** Sampling parameters (temperature, top_k, top_p) are configured in `EngineConfig` during engine creation. The engine uses llama.cpp's internal sampler for optimal performance.

##### `stats`

```rust
pub fn stats(&self) -> PerformanceStats
```

Returns current performance statistics.

**Example:**
```rust
let stats = engine.stats();
println!("Prompt eval: {:.2} t/s", stats.prompt_tokens_per_second());
println!("Generation: {:.2} t/s", stats.eval_tokens_per_second());
```

##### `backend_name`

```rust
pub fn backend_name(&self) -> String
```

Returns the name of the active compute backend.

**Example:**
```rust
let backend = engine.backend_name();
println!("Using backend: {}", backend);
```

### `EngineConfig`

Configuration for the Loci engine.

```rust
pub struct EngineConfig {
    pub model_path: String,
    pub n_ctx: u32,           // Default: 2048
    pub n_gpu_layers: i32,    // Default: 0 (-1 = all layers)
    pub n_batch: u32,         // Default: 512
    pub n_threads: u32,       // Default: num_cpus
    pub temperature: f32,     // Default: 0.8
    pub top_k: i32,           // Default: 40
    pub top_p: f32,           // Default: 0.9
    pub repeat_penalty: f32,  // Default: 1.1
}
```

**Field Descriptions:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `String` | `""` | Path to the GGUF model file |
| `n_ctx` | `u32` | `2048` | Context length (max tokens in context) |
| `n_gpu_layers` | `i32` | `0` | Number of layers to offload to GPU (-1 = all, 0 = CPU only) |
| `n_batch` | `u32` | `512` | Batch size for prompt processing |
| `n_threads` | `u32` | auto | Number of CPU threads (0 = auto-detect) |
| `temperature` | `f32` | `0.8` | Sampling temperature (0.0 = deterministic, >1.0 = more random) |
| `top_k` | `i32` | `40` | Top-K sampling (0 = disabled) |
| `top_p` | `f32` | `0.9` | Top-P/nucleus sampling (1.0 = disabled) |
| `repeat_penalty` | `f32` | `1.1` | Repetition penalty (1.0 = no penalty) |

**Example:**
```rust
use loci::EngineConfig;

let config = EngineConfig {
    model_path: "llama-2-7b.gguf".to_string(),
    n_ctx: 4096,           // Longer context
    n_gpu_layers: -1,      // Use all GPU
    temperature: 0.7,      // Less random
    top_k: 40,
    top_p: 0.9,
    repeat_penalty: 1.15,  // Stronger penalty
    ..Default::default()
};
```

### `PerformanceStats`

Performance metrics for the engine.

```rust
pub struct PerformanceStats {
    pub prompt_eval_count: u64,
    pub prompt_eval_time_ms: u64,
    pub eval_count: u64,
    pub eval_time_ms: u64,
}

impl PerformanceStats {
    pub fn prompt_tokens_per_second(&self) -> f64;
    pub fn eval_tokens_per_second(&self) -> f64;
}
```

**Example:**
```rust
let stats = engine.stats();

println!("Prompt processing:");
println!("  Tokens: {}", stats.prompt_eval_count);
println!("  Time: {}ms", stats.prompt_eval_time_ms);
println!("  Speed: {:.2} t/s", stats.prompt_tokens_per_second());

println!("Generation:");
println!("  Tokens: {}", stats.eval_count);
println!("  Time: {}ms", stats.eval_time_ms);
println!("  Speed: {:.2} t/s", stats.eval_tokens_per_second());
```

---

## Backend API

### `ComputeBackend`

Trait for compute backend abstraction.

```rust
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn device_info(&self) -> DeviceInfo;
    fn is_available(&self) -> bool;
    fn forward(&self, input: &[f32]) -> Result<Vec<f32>>;
}
```

### `BackendType`

Enumeration of supported backend types.

```rust
pub enum BackendType {
    CPU,
    CUDA,
    Metal,
    ROCm,
    Vulkan,
}
```

### `detect_backend`

```rust
pub fn detect_backend() -> Box<dyn ComputeBackend>
```

Automatically detects and returns the best available backend.

**Example:**
```rust
use loci::detect_backend;

let backend = detect_backend();
println!("Using backend: {}", backend.name());
```

### `DeviceInfo`

Information about the compute device.

```rust
pub struct DeviceInfo {
    pub name: String,
    pub memory_mb: u64,
    pub compute_units: usize,
}
```

---

## GGUF Loader API

### `GGUFModel`

Zero-copy GGUF model loader using memory mapping.

#### Constructor

```rust
pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self>
```

Loads a GGUF model from a file using zero-copy memory mapping.

**Example:**
```rust
use loci::GGUFModel;

let model = GGUFModel::from_file("path/to/model.gguf")?;
println!("Model: {}", model.metadata().name);
```

#### Methods

##### `metadata`

```rust
pub fn metadata(&self) -> &GGUFMetadata
```

Returns model metadata.

##### `tensor_info`

```rust
pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo>
```

Gets information about a specific tensor.

### `GGUFMetadata`

Model metadata from GGUF file.

```rust
pub struct GGUFMetadata {
    pub name: String,
    pub architecture: String,
    pub vocab_size: usize,
    pub context_length: usize,
    pub embedding_dim: usize,
}
```

### `TensorInfo`

Information about a tensor in the model.

```rust
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub offset: u64,
    pub size: u64,
}
```

---

## Sampling API

### Integrated Sampling

Loci uses llama.cpp's high-performance integrated sampler. Sampling parameters are configured directly in `EngineConfig`.

**Supported Sampling Methods:**
- **Temperature Scaling**: Controls randomness (configured via `temperature`)
- **Top-K Sampling**: Samples from top K tokens (configured via `top_k`)
- **Top-P (Nucleus) Sampling**: Samples from cumulative probability mass (configured via `top_p`)
- **Repetition Penalty**: Penalizes repeated tokens (configured via `repeat_penalty`)

**Example:**
```rust
use loci::{LociEngine, EngineConfig};

let config = EngineConfig {
    model_path: "model.gguf".to_string(),
    temperature: 0.7,      // Moderate randomness
    top_k: 40,             // Consider top 40 tokens
    top_p: 0.9,            // Nucleus sampling threshold
    repeat_penalty: 1.1,   // Slight repetition penalty
    ..Default::default()
};

let engine = LociEngine::new(config)?;
let output = engine.generate("Once upon a time", 100)?;
```

**Sampling Parameter Guidelines:**

| Use Case | Temperature | Top-K | Top-P | Repeat Penalty |
|----------|-------------|-------|-------|----------------|
| **Deterministic** | 0.0-0.3 | 1-10 | 0.5-0.7 | 1.0-1.1 |
| **Balanced** | 0.7-0.8 | 40-50 | 0.9-0.95 | 1.1-1.15 |
| **Creative** | 0.9-1.2 | 100+ | 0.95-1.0 | 1.0-1.05 |

**Note:** The integrated sampler provides optimal performance by operating directly on llama.cpp's internal state. For advanced use cases requiring custom sampling logic, consider using the Plugin System with the `on_sample` hook.

---

## Paged Attention API

### `SessionManager`

Manages paged attention sessions with memory budgeting.

#### Constructor

```rust
pub fn new(vram_mb: u64, ram_mb: u64, block_size_kb: usize) -> Self
```

Creates a new session manager with memory budgets.

**Example:**
```rust
use loci::SessionManager;

let manager = SessionManager::new(
    4096,  // 4GB VRAM
    8192,  // 8GB RAM
    256,   // 256KB block size
);
```

#### Methods

##### `create_session`

```rust
pub fn create_session(&mut self) -> Result<SessionId>
```

Creates a new paged attention session.

##### `allocate_blocks`

```rust
pub fn allocate_blocks(
    &mut self,
    session_id: SessionId,
    num_blocks: usize
) -> Result<Vec<PhysicalBlockId>>
```

Allocates physical blocks for a session.

##### `free_session`

```rust
pub fn free_session(&mut self, session_id: SessionId) -> Result<()>
```

Frees all resources associated with a session.

### `BlockTable`

Maps logical blocks to physical blocks.

```rust
pub struct BlockTable {
    pub session_id: SessionId,
    pub mapping: HashMap<LogicalBlockId, PhysicalBlockId>,
}
```

### Constants

```rust
pub const BLOCK_SIZE: usize = 256;  // Tokens per block
```

---

## Constraint Sampling API

### `Constraint`

Trait for implementing sampling constraints.

```rust
pub trait Constraint: Send + Sync {
    fn apply(&self, ctx: &ConstraintContext) -> Result<TokenMask>;
    fn reset(&mut self);
}
```

### `RegexConstraint`

Constrains output to match a regular expression.

#### Constructor

```rust
pub fn new(pattern: &str) -> Result<Self>
```

**Example:**
```rust
use loci::RegexConstraint;

let constraint = RegexConstraint::new(r"^\d{3}-\d{4}$")?;
```

### `JsonSchemaConstraint`

Constrains output to valid JSON matching a schema.

#### Constructor

```rust
pub fn new(schema: JsonType) -> Self
```

**Example:**
```rust
use loci::{JsonSchemaConstraint, JsonType};

let schema = JsonType::Object(vec![
    ("name".to_string(), JsonType::String),
    ("age".to_string(), JsonType::Number),
]);

let constraint = JsonSchemaConstraint::new(schema);
```

### `JsonType`

JSON schema type enumeration.

```rust
pub enum JsonType {
    String,
    Number,
    Boolean,
    Null,
    Array(Box<JsonType>),
    Object(Vec<(String, JsonType)>),
}
```

### `TokenMask`

Efficient token filtering mask.

```rust
pub struct TokenMask {
    pub allowed_tokens: Vec<usize>,
}
```

---

## Suspend/Resume API

### `SuspendableSession`

Session with suspend/resume capabilities.

#### Constructor

```rust
pub fn new(session_id: SessionID) -> Self
```

#### Methods

##### `suspend`

```rust
pub fn suspend(&mut self, reason: SuspendReason) -> Result<ResumeContext>
```

Suspends the session and returns a resume context.

**Example:**
```rust
use loci::{SuspendableSession, SuspendReason};

let mut session = SuspendableSession::new(session_id);
let resume_ctx = session.suspend(SuspendReason::ToolCall {
    tool_name: "web_search".to_string(),
    arguments: args,
})?;
```

##### `resume`

```rust
pub fn resume(&mut self, ctx: ResumeContext) -> Result<()>
```

Resumes the session from a previous suspension.

### `SuspendReason`

Reason for suspending a session.

```rust
pub enum SuspendReason {
    ToolCall { tool_name: String, arguments: HashMap<String, String> },
    UserInput,
    Approval,
    Custom(String),
}
```

### `ResumeContext`

Context for resuming a suspended session.

```rust
pub struct ResumeContext {
    pub session_id: SessionID,
    pub state: SessionState,
    pub injection: Option<String>,
    pub injection_type: InjectionType,
}
```

---

## Radix Tree Caching API

### `RadixTree`

Prefix-sharing data structure for KV cache.

#### Constructor

```rust
pub fn new() -> Self
```

#### Methods

##### `insert`

```rust
pub fn insert(&mut self, tokens: &[TokenId]) -> NodeId
```

Inserts a token sequence and returns the leaf node ID.

##### `search`

```rust
pub fn search(&self, tokens: &[TokenId]) -> Option<NodeId>
```

Searches for a token sequence, returns node ID if found.

##### `get_cache_blocks`

```rust
pub fn get_cache_blocks(&self, node_id: NodeId) -> Vec<CacheBlockId>
```

Gets all cache blocks associated with a node.

##### `stats`

```rust
pub fn stats(&self) -> RadixTreeStats
```

Returns statistics about the tree.

### `RadixTreeStats`

Statistics for the radix tree.

```rust
pub struct RadixTreeStats {
    pub total_nodes: usize,
    pub total_tokens: usize,
    pub shared_tokens: usize,
    pub memory_saved_bytes: usize,
}
```

### `KVCacheManager`

Manages KV cache with radix tree integration.

#### Constructor

```rust
pub fn new() -> Self
```

#### Methods

##### `get_or_allocate`

```rust
pub fn get_or_allocate(
    &mut self,
    tokens: &[TokenId]
) -> Result<(NodeId, Vec<CacheBlockId>)>
```

Gets or allocates cache blocks for a token sequence.

---

## Plugin System API

### `Plugin`

Trait for implementing plugins.

```rust
pub trait Plugin: Send + Sync {
    fn metadata(&self) -> &PluginMetadata;
    fn on_load(&mut self) -> Result<()> { Ok(()) }
    fn on_sample(&self, ctx: &mut PluginContext) -> Result<PluginControlFlow>;
    fn on_unload(&mut self) -> Result<()> { Ok(()) }
}
```

### `PluginRegistry`

Dual-track plugin registry (Native + WASM).

#### Constructor

```rust
pub fn new() -> Self
```

#### Methods

##### `register_native`

```rust
pub fn register_native(
    &mut self,
    plugin: Box<dyn Plugin>
) -> Result<String>
```

Registers a native plugin.

**Example:**
```rust
use loci::PluginRegistry;

let mut registry = PluginRegistry::new();
let plugin_id = registry.register_native(Box::new(MyPlugin))?;
```

##### `register_wasm`

```rust
pub fn register_wasm(
    &mut self,
    path: impl AsRef<Path>,
    signature: Option<&[u8]>
) -> Result<String>
```

Registers a WASM plugin with optional signature verification.

##### `invoke_all`

```rust
pub fn invoke_all(&self, ctx: &mut PluginContext) -> Result<PluginControlFlow>
```

Invokes all registered plugins on the given context.

### `PluginMetadata`

Plugin metadata.

```rust
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
}
```

### `PluginContext`

Context passed to plugin hooks.

```rust
pub struct PluginContext<'a> {
    pub logits: LogitsView<'a>,
    pub tokens: &'a [usize],
    pub metadata: HashMap<String, String>,
}
```

---

## Model Registry API

### `ModelRegistry`

Global registry for managing multiple models and LoRA adapters.

#### Methods

##### `load_model`

```rust
pub fn load_model(
    &mut self,
    model_id: ModelID,
    path: impl AsRef<Path>
) -> Result<()>
```

Loads a model into the registry.

**Example:**
```rust
use loci::MODEL_REGISTRY;

MODEL_REGISTRY.load_model("llama-7b", "path/to/llama-7b.gguf")?;
```

##### `switch_model`

```rust
pub fn switch_model(&mut self, model_id: &ModelID) -> Result<()>
```

Switches the active model at runtime.

##### `load_lora`

```rust
pub fn load_lora(
    &mut self,
    lora_id: LoRAID,
    path: impl AsRef<Path>
) -> Result<()>
```

Loads a LoRA adapter.

##### `apply_lora`

```rust
pub fn apply_lora(
    &mut self,
    model_id: &ModelID,
    lora_id: &LoRAID,
    scale: f32
) -> Result<()>
```

Applies a LoRA adapter to a model.

### `LoRAConfig`

Configuration for LoRA adapter.

```rust
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
}
```

---

## LoRA API

### `LoRAManager`

Manages LoRA weight merging.

#### Constructor

```rust
pub fn new() -> Self
```

#### Methods

##### `load_lora`

```rust
pub fn load_lora(&mut self, path: impl AsRef<Path>) -> Result<LoRAModel>
```

Loads a LoRA model from GGUF format.

##### `merge_lora`

```rust
pub fn merge_lora(
    &self,
    base_weight: &[f32],
    lora: &LoRALayer,
    scale: f32
) -> Result<Vec<f32>>
```

Merges LoRA weights: `W' = W + scale * (A @ B)`

**Example:**
```rust
use loci::LoRAManager;

let manager = LoRAManager::new();
let lora = manager.load_lora("path/to/lora.gguf")?;
let merged = manager.merge_lora(&base_weights, &lora.layers[0], 1.0)?;
```

### `LoRAModel`

LoRA model structure.

```rust
pub struct LoRAModel {
    pub config: LoRAConfig,
    pub layers: Vec<LoRALayer>,
}
```

### `LoRALayer`

Individual LoRA layer.

```rust
pub struct LoRALayer {
    pub name: String,
    pub lora_a: LoRATensor,  // [rank, in_features]
    pub lora_b: LoRATensor,  // [out_features, rank]
}
```

---

## Model Encryption API

### `EncryptedModelLoader`

Loads and decrypts AES-256-GCM encrypted models.

#### Constructor

```rust
pub fn new(key_source: KeySource) -> Result<Self>
```

**Example:**
```rust
use loci::{EncryptedModelLoader, KeySource};

let loader = EncryptedModelLoader::new(
    KeySource::Environment("LOCI_ENCRYPTION_KEY".to_string())
)?;
```

#### Methods

##### `load_encrypted_model`

```rust
pub fn load_encrypted_model(&self, path: impl AsRef<Path>) -> Result<Vec<u8>>
```

Loads and decrypts a model file.

##### `encrypt_model`

```rust
pub fn encrypt_model(
    &self,
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>
) -> Result<()>
```

Encrypts a model file with AES-256-GCM.

### `KeySource`

Source for encryption keys.

```rust
pub enum KeySource {
    Environment(String),
    File(PathBuf),
    KMS { endpoint: String, key_id: String },
    Hardware,
    Direct(Zeroizing<Vec<u8>>),
}
```

### `generate_key`

```rust
pub fn generate_key() -> Result<Zeroizing<Vec<u8>>>
```

Generates a cryptographically secure 256-bit encryption key.

---

## Multi-Tenancy API

### `TenantManager`

Manages multi-tenant resource isolation.

#### Constructor

```rust
pub fn new() -> Self
```

#### Methods

##### `create_tenant`

```rust
pub fn create_tenant(
    &mut self,
    quota: TenantQuota
) -> Result<TenantID>
```

Creates a new tenant with specified quota.

**Example:**
```rust
use loci::{TenantManager, TenantQuota};

let mut manager = TenantManager::new();
let tenant_id = manager.create_tenant(TenantQuota::Enterprise {
    max_sessions: 100,
    max_memory_mb: 16384,
    max_tokens_per_session: 128000,
})?;
```

##### `get_context`

```rust
pub fn get_context(&self, tenant_id: &TenantID) -> Result<Arc<TenantContext>>
```

Gets the isolated context for a tenant.

##### `delete_tenant`

```rust
pub fn delete_tenant(&mut self, tenant_id: &TenantID) -> Result<()>
```

Deletes a tenant and frees all resources.

### `TenantQuota`

Quota tiers for tenants.

```rust
pub enum TenantQuota {
    Free {
        max_sessions: usize,
        max_memory_mb: u64,
        max_tokens_per_session: usize,
    },
    Default {
        max_sessions: usize,
        max_memory_mb: u64,
        max_tokens_per_session: usize,
    },
    Enterprise {
        max_sessions: usize,
        max_memory_mb: u64,
        max_tokens_per_session: usize,
    },
}
```

---

## Multimodal API

### `VisionEncoder`

Trait for vision encoders.

```rust
pub trait VisionEncoder: Send + Sync {
    fn encode_image(&self, image: &ImageBuffer) -> Result<Vec<f32>>;
    fn embedding_dim(&self) -> usize;
    fn supported_sizes(&self) -> Vec<(u32, u32)>;
}
```

### `CLIPVisionEncoder`

CLIP ViT-L/14@336 implementation.

#### Constructor

```rust
pub fn new() -> Self
```

#### Methods

##### `encode_image`

```rust
pub fn encode_image(&self, image: &ImageBuffer) -> Result<Vec<f32>>
```

Encodes an image into embeddings.

**Example:**
```rust
use loci::{CLIPVisionEncoder, ImageBuffer};

let encoder = CLIPVisionEncoder::new();
let image = ImageBuffer::from_file("image.jpg")?;
let embeddings = encoder.encode_image(&image)?;
```

### `MultimodalKVCache`

Unified KV cache for text and image tokens.

```rust
pub struct MultimodalKVCache {
    pub text_cache: Vec<(Vec<f32>, Vec<f32>)>,
    pub image_cache: Vec<(Vec<f32>, Vec<f32>)>,
    pub token_types: Vec<TokenType>,
}
```

### `TokenType`

Token type enumeration.

```rust
pub enum TokenType {
    Text,
    Image,
}
```

---

## Quantization API

### `QuantizationType`

Supported quantization formats.

```rust
pub enum QuantizationType {
    None,
    FP16,
    Q8_0,
    Q4_0,
    Q4_K_M,
    IQ2_XXS,      // 2-bit importance weighted
    BitNet158,    // Ternary {-1, 0, +1}
}
```

### `QuantizationManager`

Manages quantization schemes.

#### Methods

##### `quantize`

```rust
pub fn quantize(
    &self,
    data: &[f32],
    qtype: QuantizationType
) -> Result<QuantizedTensor>
```

Quantizes float32 data to specified format.

**Example:**
```rust
use loci::{QuantizationManager, QuantizationType};

let manager = QuantizationManager::new();
let quantized = manager.quantize(&data, QuantizationType::IQ2_XXS)?;
```

##### `dequantize`

```rust
pub fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>>
```

Dequantizes tensor back to float32.

### `IQ2_XXS`

2-bit importance-weighted quantization.

```rust
pub struct IQ2_XXS {
    pub block_size: usize,           // 32 elements
    pub importance_threshold: f32,
}
```

### `BitNet158`

Ternary quantization {-1, 0, +1}.

```rust
pub struct BitNet158 {
    pub zero_threshold: f32,
}
```

---

## Kernel Fusion API

### `KernelFusionManager`

Manages fused kernels for performance optimization.

#### Methods

##### `create_rmsnorm_rope`

```rust
pub fn create_rmsnorm_rope(
    rmsnorm_params: RMSNormParams,
    rope_params: RoPEParams
) -> Result<RMSNormRoPEFusion>
```

Creates a fused RMSNorm+RoPE kernel.

**Example:**
```rust
use loci::{KernelFusionManager, RMSNormParams, RoPEParams};

let fusion = KernelFusionManager::create_rmsnorm_rope(
    RMSNormParams { dim: 4096, eps: 1e-5 },
    RoPEParams { dim: 128, base: 10000.0, max_seq_len: 2048 }
)?;

let output = fusion.forward(&input, position)?;
```

### `RMSNormRoPEFusion`

Fused RMSNorm + RoPE kernel (-30% latency).

```rust
pub struct RMSNormRoPEFusion {
    pub rmsnorm_params: RMSNormParams,
    pub rope_params: RoPEParams,
}
```

### `MatMulAddFusion`

Fused MatMul + Bias Add (+18% throughput).

```rust
pub struct MatMulAddFusion {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}
```

---

## Configuration API

### `ConfigLoader`

Loads and validates configuration from files and environment.

#### Constructor

```rust
pub fn new() -> Self
```

#### Methods

##### `from_file`

```rust
pub fn from_file(path: impl AsRef<Path>) -> Result<Self>
```

Loads configuration from TOML or JSON file.

**Example:**
```rust
use loci::ConfigLoader;

let config = ConfigLoader::from_file("loci.toml")?
    .with_env_overrides()
    .build()?;
```

##### `with_env_overrides`

```rust
pub fn with_env_overrides(self) -> Self
```

Applies environment variable overrides.

**Environment Variables:**
- `LOCI_MODEL_PATH`: Model file path
- `LOCI_BACKEND`: Backend type (cpu/cuda/metal/rocm)
- `LOCI_N_GPU_LAYERS`: Number of GPU layers
- `LOCI_LOG_LEVEL`: Logging level

##### `validate`

```rust
pub fn validate(&self) -> Result<()>
```

Validates the configuration.

##### `build`

```rust
pub fn build(self) -> Result<LociConfig>
```

Builds the final configuration after validation.

### `LociConfig`

Global configuration structure.

```rust
pub struct LociConfig {
    pub engine: EngineSettings,
    pub backend: BackendSettings,
    pub memory: MemorySettings,
    pub plugins: PluginSettings,
    pub logging: LoggingSettings,
    pub server: ServerSettings,
}
```

---

## Error Handling

All API functions that can fail return `anyhow::Result<T>` for flexible error handling.

**Example:**
```rust
use anyhow::{Result, Context};

fn my_function() -> Result<()> {
    let engine = LociEngine::new(config)
        .context("Failed to initialize engine")?;

    let output = engine.generate(prompt, &mut sampler, 100)
        .context("Generation failed")?;

    Ok(())
}
```

---

## Type Aliases

Common type aliases used throughout the API:

```rust
pub type SessionId = u64;
pub type TokenId = u32;
pub type NodeId = usize;
pub type ModelID = String;
pub type LoRAID = String;
pub type TenantID = uuid::Uuid;
```

---

## Performance Tips

1. **Use mmap for large models**: Set `use_mmap: true` in `EngineConfig`
2. **Optimize batch size**: Adjust `batch_size` based on available memory
3. **Enable kernel fusion**: Set `enable_fusion: true` in backend config
4. **Use GPU layers**: Set `n_gpu_layers: -1` for maximum GPU utilization
5. **Enable radix tree caching**: Automatically enabled for prefix sharing
6. **Use appropriate quantization**: Q4_K_M for balance, IQ2_XXS for maximum compression

---

## Thread Safety

- All public APIs are thread-safe unless otherwise noted
- `LociEngine` uses internal locking for concurrent access
- `PluginRegistry` is thread-safe
- `SessionManager` is thread-safe
- `TenantManager` is thread-safe

---

## Version Compatibility

This API reference is for Loci version 0.1.0.

**Minimum Rust Version**: 1.85+

---

## Additional Resources

- [Quick Start Guide](./QUICK_START.md)
- [Configuration Guide](./CONFIGURATION.md)
- [Plugin Development](./PLUGIN_DEVELOPMENT.md)
- [Performance White Paper](./PERFORMANCE_WHITEPAPER.md)
- [GitHub Repository](https://github.com/decade-afk/Loci)

---

**Last Updated**: 2026-01-01
**License**: MIT

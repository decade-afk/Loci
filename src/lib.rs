










//! # Loci - High-Performance LLM Inference Engine
//!
//! Loci is a high-performance, feature-rich inference engine for large language models (LLMs).
//! It provides efficient model loading, execution, and management with advanced features such as:
//!
//! - **Paged Attention**: Efficient KV cache management for long-context inference
//! - **Multi-Tenancy**: Support for concurrent sessions with resource isolation
//! - **Plugin System**: Extensible architecture via native and WASM plugins
//! - **LoRA Support**: Low-Rank Adaptation for fine-tuned models
//! - **Model Encryption**: Secure model loading with encryption support
//! - **Multi-Modal**: Vision and text processing capabilities
//! - **Kernel Fusion**: Optimized compute operations for better performance
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use loci::quick_load;
//!
//! // Load a model with default settings
//! let engine = quick_load("path/to/model.gguf")?;
//! ```
//!
//! ## Core Modules
//!
//! - [`backend`]: Compute backend abstraction (CPU/GPU)
//! - [`engine`]: Main inference engine and execution context
//! - [`gguf`]: GGUF model format loading and parsing
//! - [`sampling`]: Token sampling strategies and generation
//! - [`paged_attention`]: Paged KV cache management
//! - [`constraints`]: Output constraints (regex, JSON schema)
//! - [`suspend`]: Session suspension and resumption
//! - [`radix_tree`]: Prefix caching using radix tree structure
//! - [`plugin_system`]: Plugin architecture for extensions
//! - [`model_registry`]: Centralized model and LoRA management
//! - [`lora`]: LoRA adapter implementation
//! - [`plugin_marketplace`]: Plugin marketplace integration
//! - [`model_encryption`]: Encrypted model loading
//! - [`multi_tenancy`]: Tenant-based resource management
//! - [`multimodal`]: Multi-modal (vision+text) support
//! - [`marketplace_client`]: HTTP client for marketplace API
//! - [`quantization`]: Model quantization schemes
//! - [`kernel_fusion`]: Compute kernel optimization
//! - [`config`]: Configuration management
//! - [`streaming`]: Streaming output handling

/// Compute backend module providing hardware abstraction.
///
/// This module defines the interface for different compute backends (CPU, GPU, etc.)
/// and handles device detection and initialization.
pub mod backend;








pub mod gguf;








pub mod engine;










pub mod sampling;









pub mod paged_attention;









pub mod constraints;









pub mod suspend;









pub mod radix_tree;










pub mod plugin_system;









#[cfg(any(target_os = "android", target_os = "ios", feature = "mobile-ffi"))]
pub mod mobile_ffi;









pub mod model_registry;









pub mod lora;










pub mod plugin_marketplace;








pub mod model_encryption;








pub mod multi_tenancy;








pub mod multimodal;








pub mod marketplace_client;








pub mod quantization;








pub mod kernel_fusion;








pub mod config;










pub mod streaming;



pub use backend::{
    ComputeBackend,
    DeviceInfo,
    BackendType,
    detect_backend,
};

pub use gguf::{
    GGUFModel,
    GGUFMetadata,
    TensorInfo,
};

pub use engine::{
    LociEngine,
    EngineConfig,
    PerformanceStats,
};

pub use sampling::{
    Sampler,
    SamplerConfig,
};

pub use paged_attention::{
    SessionManager,
    SessionId,
    PhysicalBlockId,
    LogicalBlockId,
    BlockTable,
    PhysicalBlock,
    BlockLocation,
    BLOCK_SIZE,
};

pub use constraints::{
    Constraint,
    ConstraintContext,
    TokenMask,
    RegexConstraint,
    JsonSchemaConstraint,
    JsonType,
    JsonState,
    AndConstraint,
    OrConstraint,
};

pub use suspend::{
    ControlFlow,
    SuspendReason,
    StopReason,
    SessionState,
    ResumeContext,
    InjectionType,
    SuspendableSession,
    SuspendableSessionManager,
    SessionInfo,
};

pub use radix_tree::{
    RadixNode,
    RadixTree,
    RadixTreeStats,
    KVCacheManager,
    PrefixCacheStats,
    TokenId,
    NodeId,
    CacheBlockId,
};

pub use plugin_system::{
    Plugin,
    PluginType,
    PluginMetadata,
    PluginControlFlow,
    PluginContext,
    LogitsView,
    NativePlugin,
    WasmPlugin,
    PluginRegistry,
    PluginRegistryStats,
    SignatureVerifier,
    ResourceQuota,
    Watchdog,
};

pub use model_registry::{
    ModelID,
    LoRAID,
    SessionID,
    ModelMetadata,
    LoRAConfig,
    LoRAAdapter,
    LoadedModel,
    ModelRegistry,
    MODEL_REGISTRY,
};

pub use lora::{
    LoRATensor,
    TensorDataType,
    LoRALayer,
    LoRAModel,
    LoRAStats,
    LoRAManager,
    create_example_lora_layer,
};

pub use plugin_marketplace::{
    PluginManifest,
    PluginAuthor,
    PluginKind,
    PluginDependency,
    PluginHooks,
    PluginLimits,
    PluginDownloadInfo,
    PluginRegistry as MarketplaceRegistry,
    InstalledPlugin,
};

pub use model_encryption::{
    EncryptedModelConfig,
    EncryptedModelLoader,
    KeySource,
    generate_key,
};

pub use multi_tenancy::{
    TenantID,
    TenantQuota,
    TenantResourceUsage,
    TenantContext,
    TenantManager,
    TenantSessionID,
};

pub use multimodal::{
    ImageBuffer,
    Tensor,
    VisionEncoder,
    CLIPVisionEncoder,
    TokenType,
    TypedToken,
    MultimodalKVCache,
};

pub use marketplace_client::{
    MarketplaceClient,
    MarketplaceClientConfig,
    PluginSearchResult,
    PluginSummary,
    PluginDetails,
    PluginUpdate,
};

pub use quantization::{
    QuantizationType,
    QuantizationScheme,
    QuantizedTensor,
    QuantizationMetadata,
    Iq2Xxs,
    BitNet158,
    QuantizationManager,
};

pub use kernel_fusion::{
    RMSNormParams,
    RoPEParams,
    RMSNormRoPEFusion,
    MatMulAddFusion,
    LayerNormParams,
    LayerNormLinearFusion,
    KernelFusionManager,
};

pub use config::{
    LociConfig,
    EngineSettings,
    BackendSettings,
    MemorySettings,
    PluginSettings,
    LoggingSettings,
    ServerSettings,
    ConfigLoader,
};

pub use streaming::{
    StreamCallback,
    StreamControlFlow,
    StreamToken,
    StreamStats,
    ClosureCallback,
    BatchedCallback,
    ConsoleCallback,
    AccumulatorCallback,
    safe_callback_invoke,
};




pub const VERSION: &str = env!("CARGO_PKG_VERSION");


pub const BUILD_INFO: &str = concat!(
    "Loci Phase 2 Week 1 - ",
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("CARGO_PKG_NAME"),
    ")"
);


pub fn print_banner() {
    eprintln!("╔════════════════════════════════════════╗");
    eprintln!("║         Loci Phase 2 Engine            ║");
    eprintln!("║  Paged Attention + Memory Budgeter     ║");
    eprintln!("╚════════════════════════════════════════╝");
    eprintln!("  Version: {}", VERSION);
    eprintln!("  Features: 128k+ Context, Multi-Session");
    eprintln!();
}













pub fn quick_load(model_path: &str) -> anyhow::Result<LociEngine> {
    print_banner();

    let config = EngineConfig {
        model_path: model_path.to_string(),
        n_gpu_layers: -1,  
        ..Default::default()
    };

    LociEngine::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        println!("Loci version: {}", VERSION);
    }

    #[test]
    fn test_backend_detection() {
        let backend = detect_backend();
        assert!(backend.is_available());
        println!("Detected backend: {}", backend.name());
    }
}

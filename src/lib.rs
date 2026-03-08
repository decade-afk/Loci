//! Loci - A cross-platform, plugin-based local LLM inference framework
//!
//! Loci provides a unified interface for running large language models locally
//! with support for multiple backends and plugin architectures.
//!
//! ## Architecture
//!
//! Loci uses a trait-based backend system with three planned phases:
//!
//! - **Phase 1 (Current)**: Native backend with hardcoded llama.cpp
//! - **Phase 2 (Planned)**: Dynamic backend loading via libloading
//! - **Phase 3 (Planned)**: WASM backends for sandboxed plugins
//!
//! ## Quick Start
//!
//! ```ignore
//! use loci::prelude::*;
//! use loci::backend::InferenceParams;
//!
//! // Create inference engine with llama.cpp backend
//! let mut engine = InferenceEngine::builder()
//!     .model_path("model.gguf")
//!     .build()?;
//!
//! // Generate text
//! let response = engine.generate("Hello, world!", &InferenceParams::default())?;
//! # Ok::<(), loci::error::LociError>(())
//! ```

mod ffi;

pub mod backend;
pub mod backends;
pub mod constraint;
pub mod constraint_complete;
pub mod device;
pub mod error;
pub mod hooks;
pub mod inference;
pub mod kv_cache;
pub mod kv_cache_advanced;
pub mod radix_cache;
pub mod model;
pub mod model_hot_swap;
pub mod model_registry;
pub mod adapter_system;
pub mod adapter_complete;
pub mod multimodal;
pub mod multimodal_plugin;
pub mod vision_clip;
pub mod multimodal_fusion;
pub mod plugin;
pub mod plugin_registry;
pub mod sampler;
pub mod session;
pub mod session_bus;
pub mod wasm_plugin;
pub mod c_api;
pub mod chat_template;
pub mod function_calling;
pub mod image_kernel;
pub mod batch_inference;
pub mod rag;
pub mod quantization;
pub mod inference_cache;
pub mod resource_manager;
pub mod timeout_controller;
pub mod concurrency_manager;

// Plugin examples (compiled as a feature)
#[cfg(feature = "plugin-examples")]
pub mod examples;

pub use error::{LociError, Result};
pub use inference::InferenceEngine;
pub use model::{ModelConfig, ModelLoader};
pub use chat_template::{ChatMessage, ChatTemplate, ChatTemplateBuilder, ChatTemplateType};
pub use function_calling::{
    FunctionCall, FunctionCallingManager, FunctionDefinition, FunctionParameter,
};
pub use image_kernel::{
    dynamic_image_plugin_from_opaque, dynamic_image_plugin_into_opaque,
    load_dynamic_image_plugin, DynamicImageKernel, DynamicImagePluginOpaque,
    ImageGenerationPlugin, ImageGenerationRequest, ImageGenerationResult,
};
pub use batch_inference::{
    BatchConfig, BatchInferenceBuilder, BatchInferenceProcessor, BatchResult, PromptBatch,
};
pub use rag::{
    ChunkingConfig, EmbeddingProvider, HashEmbeddingProvider, InMemoryVectorStore, RagChunk,
    RagDocument, RagEngine, RagPlugin, InMemoryRagPlugin, RetrievedChunk,
};
pub use quantization::{
    QuantizationReport, QuantizationScheme, QuantizationTool, QuantizedData, QuantizedTensor,
};
pub use inference_cache::{
    CacheConfig, CacheStats, InferenceCache,
};
pub use resource_manager::{
    ResourceLimits, ResourceManager, ResourceStats, ResourceGuard, MonitorConfig,
};
pub use timeout_controller::{
    TimeoutConfig, TimeoutController, TimeoutStats, TimeoutContext, CancellationHandle,
};
pub use concurrency_manager::{
    ConcurrencyConfig, ConcurrencyManager, ConcurrencyStats, ConnectionPool, PoolStats,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::backend::{
        BackendCapabilities, BackendParams, BackendRegistry, InferenceBackend, InferenceParams,
        Model, ModelMetadata,
    };
    pub use crate::backends::{
        CandleBackend, CandleModel, DynamicBackend, LlamaCppBackend, LlamaCppModel,
    };
    pub use crate::constraint::{
        Constraint, ConstraintMask, ConstraintManager, ConstraintCombinator, CombinatorMode,
        TokenWhitelistConstraint, TokenBlacklistConstraint, LengthConstraint,
        RegexConstraint, JsonConstraint, ConstraintBuilder, JsonSchema, ConstraintPlugin
    };
    pub use crate::device::{DeviceSelector, DeviceConfig, DeviceInfo, DeviceType};
    pub use crate::error::{LociError, Result};
    pub use crate::hooks::{
        DeepProgrammableHooks, HookManager, HookContext, HookPriority, HookControl,
        LifecycleHooks, ModelHooks, InferenceHooks, TokenHooks, MemoryHooks,
        BackendHooks, SessionHooks, ErrorHooks, ErrorRecovery, SpecialTokenType
    };
    pub use crate::inference::{InferenceEngine, InferenceEngineBuilder};
    pub use crate::kv_cache::{KVBlock, PagedKVCache, SessionKVCache, BLOCK_SIZE};
    pub use crate::kv_cache_advanced::{
        PhysicalBlockPool, BlockTable, BlockId,
        MemoryBudgetConfig, MemoryBudgeter, PoolStatistics
    };
    pub use crate::radix_cache::{
        ShardedRadixCache, RadixTree, RadixNode, RadixTreeStats,
        TokenId, BlockHash
    };
    pub use crate::model::{ModelConfig, ModelLoader};
    pub use crate::model_hot_swap::{
        HotSwapModelRegistry, LoRAConfig, ModelInfo as HotSwapModelInfo, LoadedModel
    };
    pub use crate::model_registry::{ModelId, ModelRegistry, ModelInfo};
    pub use crate::adapter_system::{
        AdapterRegistry, Adapter, AdapterId, AdapterType,
        LoRAAdapterConfig, QLoRAAdapterConfig, AdapterFusionConfig,
        QuantizationType, FusionStrategy,
        SimpleLoRAAdapter, SimpleQLoRAAdapter
    };
    pub use crate::multimodal::{
        Image, ImageFormat, Audio, MultimodalInput, ModalityToken,
        VisionEncoderConfig, AudioEncoderConfig, VisionEncoderType, AudioEncoderType,
        ProcessorConfig, MultimodalProcessor, MultimodalModelAdapter, Modality,
        ImagePatch
    };
    pub use crate::multimodal_plugin::{
        MultimodalPluginRegistry, ModalPluginId,
        VisionEncoderPlugin, AudioEncoderPlugin, FusionStrategyPlugin, PreprocessingPlugin
    };
    pub use crate::vision_clip::{
        CLIPViTL14Encoder, CLIPViTL14Config, ImageEmbedding, BatchCLIPEncoder
    };
    pub use crate::multimodal_fusion::{
        MultimodalFusion, FusedTokenSequence, TokenType, FusionConfig,
        VisionPosition, FusionStrategyType, SpecialToken
    };
    pub use crate::plugin::{Plugin, PluginManager};
    pub use crate::plugin_registry::{
        PluginRegistry, PluginConfig, PluginType, SharedRegistry,
        RegistryConfig, create_shared_registry
    };
    pub use crate::sampler::{LogitsView, SamplingParams, Sampler, sample_token, DefaultSampler, GreedySampler, TopKSampler, TopPSampler, MirostatSampler, TemperatureSampler};
    pub use crate::session::{SessionId, SessionManager, SessionHandle, SessionInfo, SessionState};
    pub use crate::session_bus::{SessionBus, SessionMessage, ControlMessage, BusError};
    pub use crate::wasm_plugin::{WasmPlugin, WasmPluginConfig, WasmPluginManager};
    pub use crate::chat_template::{
        ChatMessage, ChatTemplate, ChatTemplateBuilder, ChatTemplateType,
    };
    pub use crate::function_calling::{
        FunctionCall, FunctionCallingManager, FunctionDefinition, FunctionParameter,
    };
    pub use crate::image_kernel::{
        dynamic_image_plugin_from_opaque, dynamic_image_plugin_into_opaque,
        load_dynamic_image_plugin, DynamicImageKernel, DynamicImagePluginOpaque,
        ImageGenerationPlugin, ImageGenerationRequest, ImageGenerationResult,
    };
    pub use crate::batch_inference::{
        BatchConfig, BatchInferenceBuilder, BatchInferenceProcessor, BatchResult, PromptBatch,
    };
    pub use crate::rag::{
        ChunkingConfig, EmbeddingProvider, HashEmbeddingProvider, InMemoryVectorStore, RagChunk,
        RagDocument, RagEngine, RagPlugin, InMemoryRagPlugin, RetrievedChunk,
    };
    pub use crate::quantization::{
        QuantizationReport, QuantizationScheme, QuantizationTool, QuantizedData, QuantizedTensor,
    };
    pub use crate::inference_cache::{
        CacheConfig, CacheStats, InferenceCache,
    };
    pub use crate::resource_manager::{
        ResourceLimits, ResourceManager, ResourceStats, ResourceGuard, MonitorConfig,
    };
    pub use crate::timeout_controller::{
        TimeoutConfig, TimeoutController, TimeoutStats, TimeoutContext, CancellationHandle,
    };
    pub use crate::concurrency_manager::{
        ConcurrencyConfig, ConcurrencyManager, ConcurrencyStats, ConnectionPool, PoolStats,
    };
}

/// C API types for FFI interop
pub mod c_api_types {
    pub use crate::c_api::{
        LociEngine, LociStreamCallback, LociPluginRegistry,
        LociDeviceInfo, LociDeviceSelector
    };
}

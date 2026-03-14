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

pub mod adapter_complete;
pub mod adapter_system;
pub mod backend;
pub mod backends;
pub mod batch_inference;
pub mod c_api;
pub mod chat_template;
pub mod concurrency_manager;
pub mod constraint;
pub mod constraint_complete;
pub mod device;
pub mod error;
pub mod execution_policy_plugin;
pub mod function_calling;
pub mod hooks;
pub mod http_compat;
pub mod image_kernel;
pub mod inference;
pub mod inference_cache;
pub mod kv_cache;
pub mod kv_cache_advanced;
pub mod management_auth;
pub mod mcp;
pub mod mcp_registry;
pub mod model;
pub mod model_hot_swap;
pub mod model_pull_jobs;
pub mod model_pull_policy;
pub mod model_pull_verifier;
pub mod model_registry;
pub mod model_store;
pub mod multimodal;
pub mod multimodal_fusion;
pub mod multimodal_io;
pub mod multimodal_plugin;
pub mod plugin;
pub mod plugin_contract;
pub mod plugin_registry;
pub mod policy_registry;
pub mod quantization;
pub mod radix_cache;
pub mod rag;
pub mod resource_manager;
pub mod resource_planner;
pub mod runtime_events;
pub mod sampler;
pub mod serve_dispatch;
pub mod session;
pub mod session_bus;
pub mod session_store;
pub mod skills;
pub mod timeout_controller;
pub mod tool_plugin;
pub mod vision_clip;
#[cfg(feature = "wasm-plugins")]
pub mod wasm_plugin;
#[cfg(not(feature = "wasm-plugins"))]
#[path = "wasm_plugin_stub.rs"]
pub mod wasm_plugin;

// Plugin examples (compiled as a feature)
#[cfg(feature = "plugin-examples")]
pub mod examples;

pub use batch_inference::{
    BatchConfig, BatchInferenceBuilder, BatchInferenceProcessor, BatchResult, PromptBatch,
};
pub use chat_template::{ChatMessage, ChatTemplate, ChatTemplateBuilder, ChatTemplateType};
pub use concurrency_manager::{
    ConcurrencyConfig, ConcurrencyManager, ConcurrencyStats, ConnectionPool, PoolStats,
};
pub use error::{LociError, Result};
pub use execution_policy_plugin::{
    dynamic_execution_policy_from_opaque, dynamic_execution_policy_into_opaque,
    load_dynamic_execution_policy, DynamicExecutionPolicyOpaque, ExecutionPolicyDescriptor,
    ExecutionPolicyRegistry, LoadedExecutionPolicy,
};
pub use function_calling::{
    FunctionCall, FunctionCallingManager, FunctionDefinition, FunctionHandler, FunctionParameter,
};
pub use http_compat::{
    compatibility_created_at, estimate_token_count, normalize_openai_embedding_input,
    openai_chat_messages_to_prompt, OllamaGenerateOptions, OllamaGenerateRequest,
    OllamaGenerateResponse, OllamaModelTag, OllamaTagsResponse, OpenAiChatChoice,
    OpenAiChatCompletionsRequest, OpenAiChatCompletionsResponse, OpenAiChatMessage,
    OpenAiEmbeddingData, OpenAiEmbeddingInput, OpenAiEmbeddingsRequest, OpenAiEmbeddingsResponse,
    OpenAiModelDescriptor, OpenAiModelListResponse, OpenAiUsage,
};
pub use image_kernel::{
    dynamic_image_plugin_from_opaque, dynamic_image_plugin_into_opaque, load_dynamic_image_plugin,
    DynamicImageKernel, DynamicImagePluginOpaque, ImageGenerationPlugin, ImageGenerationRequest,
    ImageGenerationResult,
};
pub use inference::{DefaultExecutionPolicy, ExecutionPolicy, InferenceEngine};
pub use inference_cache::{CacheConfig, CacheStats, InferenceCache};
pub use management_auth::{
    dynamic_management_auth_policy_from_opaque, dynamic_management_auth_policy_into_opaque,
    load_dynamic_management_auth_policy, AllowAllManagementAuthPolicy,
    BearerTokenManagementAuthPolicy, DynamicManagementAuthPolicyOpaque, LoadedManagementAuthPolicy,
    LoopbackOnlyManagementAuthPolicy, ManagementAuthContext, ManagementAuthDecision,
    ManagementAuthPolicyDescriptor, ManagementAuthPolicyPlugin, ManagementAuthPolicyRegistry,
};
pub use mcp::{
    connect_and_register_stdio_server, register_mcp_client_tools, McpClient, McpInputSchema,
    McpRegistrationReport, McpStdioServerConfig, McpTool, McpToolRegistrationOptions,
    StdioMcpClient, DEFAULT_MCP_PROTOCOL_VERSION,
};
pub use mcp_registry::{McpServerConfig, McpServerRegistry};
pub use model::{ModelConfig, ModelLoader};
pub use model_pull_jobs::{
    ModelPullJobEvent, ModelPullJobManager, ModelPullJobRequest, ModelPullJobSnapshot,
    ModelPullJobState,
};
pub use model_pull_policy::{
    authorize_model_pull_request, dynamic_model_pull_policy_from_opaque,
    dynamic_model_pull_policy_into_opaque, load_dynamic_model_pull_policy, AllowAllModelPullPolicy,
    DynamicModelPullPolicyOpaque, LoadedModelPullPolicy, LocalOnlyModelPullPolicy,
    ModelPullPolicyContext, ModelPullPolicyDecision, ModelPullPolicyDescriptor,
    ModelPullPolicyPlugin, ModelPullPolicyRegistry, RequireChecksumForRemoteModelPullPolicy,
};
pub use model_pull_verifier::{
    dynamic_model_pull_verifier_from_opaque, dynamic_model_pull_verifier_into_opaque,
    load_dynamic_model_pull_verifier, verify_model_pull, AllowAllModelPullVerifier,
    DynamicModelPullVerifierOpaque, LoadedModelPullVerifier, ModelPullVerificationContext,
    ModelPullVerifierDecision, ModelPullVerifierDescriptor, ModelPullVerifierPlugin,
    ModelPullVerifierRegistry, SidecarSha256ModelPullVerifier,
};
pub use model_registry::{
    EnsembleCandidateResponse, EnsembleGeneration, EnsembleMergeStrategy, ModelBenchmark, ModelId,
    ModelInfo as RegistryModelInfo, ModelRegistry, ModelRoutingStrategy, RoutedGeneration,
    RoutingAttempt,
};
pub use model_store::{
    ModelPullOptions, ModelPullPhase, ModelPullProgress, ModelStore, StoredModel,
};
pub use multimodal_io::{
    dynamic_multimodal_io_plugin_from_opaque, dynamic_multimodal_io_plugin_into_opaque,
    DescriptorMultimodalIoPlugin, DynamicMultimodalIoPluginOpaque, MultimodalIoPlugin,
    MultimodalIoRegistry, MultimodalOutputPlan, MultimodalRequest, OutputModality,
};
pub use plugin_contract::{
    load_and_validate_plugin_contract, load_plugin_contract_manifest,
    validate_plugin_contract_manifest, validate_runtime_plugin_identity, PluginContractKind,
    PluginContractManifest, LOCI_PLUGIN_ABI_VERSION,
};
pub use policy_registry::{DynamicPolicyRegistry, DynamicPolicyRegistryFile};
pub use quantization::{
    QuantizationReport, QuantizationScheme, QuantizationTool, QuantizedData, QuantizedTensor,
};
pub use rag::{
    ChunkingConfig, EmbeddingProvider, HashEmbeddingProvider, InMemoryRagPlugin,
    InMemoryVectorStore, RagChunk, RagDocument, RagEngine, RagPlugin, RetrievedChunk,
};
pub use resource_manager::{
    MonitorConfig, ResourceGuard, ResourceLimits, ResourceManager, ResourceStats,
};
pub use resource_planner::{ModelResourceEstimate, ResourcePlan, ResourcePlanner};
pub use runtime_events::{
    RuntimeEvent, RuntimeEventBus, RuntimeEventCategory, RuntimeEventOutcome,
};
pub use serve_dispatch::{
    dynamic_serve_dispatch_policy_from_opaque, dynamic_serve_dispatch_policy_into_opaque,
    load_dynamic_serve_dispatch_policy, BlockServeDispatchPolicyPlugin,
    DynamicServeDispatchPolicyOpaque, LoadedServeDispatchPolicy, QueueFullAction,
    QueuePressureContext, RejectServeDispatchPolicyPlugin, ServeDispatchPolicyDescriptor,
    ServeDispatchPolicyPlugin, ServeDispatchPolicyRegistry,
};
pub use session_store::{
    dynamic_session_store_factory_from_opaque, dynamic_session_store_factory_into_opaque,
    DynamicSessionStoreFactoryOpaque, InMemorySessionStore, InMemorySessionStoreFactory,
    SessionStore, SessionStoreConfig, SessionStoreFactory, SessionStoreRegistry,
    SqliteSessionStore, SqliteSessionStoreFactory,
};
#[cfg(feature = "redis-store")]
pub use session_store::{RedisSessionStore, RedisSessionStoreFactory};
pub use skills::{Skill, SkillPack, SkillProvider, SkillRegistry, SkillToolPolicy};
pub use timeout_controller::{
    CancellationHandle, TimeoutConfig, TimeoutContext, TimeoutController, TimeoutStats,
};
pub use tool_plugin::{
    dynamic_tool_plugin_from_opaque, dynamic_tool_plugin_into_opaque, load_dynamic_tool_plugin,
    register_tool_plugin, DynamicToolPluginOpaque, LoadedToolPlugin, LoadedToolPluginDescriptor,
    ToolPlugin,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::adapter_system::{
        Adapter, AdapterFusionConfig, AdapterId, AdapterRegistry, AdapterType, FusionStrategy,
        LoRAAdapterConfig, QLoRAAdapterConfig, QuantizationType, SimpleLoRAAdapter,
        SimpleQLoRAAdapter,
    };
    pub use crate::backend::{
        BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceBackend,
        InferenceParams, Model, ModelMetadata,
    };
    pub use crate::backends::{
        CandleBackend, CandleModel, DynamicBackend, LlamaCppBackend, LlamaCppModel,
    };
    pub use crate::batch_inference::{
        BatchConfig, BatchInferenceBuilder, BatchInferenceProcessor, BatchResult, PromptBatch,
    };
    pub use crate::chat_template::{
        ChatMessage, ChatTemplate, ChatTemplateBuilder, ChatTemplateType,
    };
    pub use crate::concurrency_manager::{
        ConcurrencyConfig, ConcurrencyManager, ConcurrencyStats, ConnectionPool, PoolStats,
    };
    pub use crate::constraint::{
        CombinatorMode, Constraint, ConstraintBuilder, ConstraintCombinator, ConstraintManager,
        ConstraintMask, ConstraintPlugin, JsonConstraint, JsonSchema, LengthConstraint,
        RegexConstraint, TokenBlacklistConstraint, TokenWhitelistConstraint,
    };
    pub use crate::device::{DeviceConfig, DeviceInfo, DeviceSelector, DeviceType};
    pub use crate::error::{LociError, Result};
    pub use crate::execution_policy_plugin::{
        dynamic_execution_policy_from_opaque, dynamic_execution_policy_into_opaque,
        load_dynamic_execution_policy, DynamicExecutionPolicyOpaque, ExecutionPolicyDescriptor,
        ExecutionPolicyRegistry, LoadedExecutionPolicy,
    };
    pub use crate::function_calling::{
        FunctionCall, FunctionCallingManager, FunctionDefinition, FunctionHandler,
        FunctionParameter,
    };
    pub use crate::hooks::{
        BackendHooks, DeepProgrammableHooks, ErrorHooks, ErrorRecovery, HookContext, HookControl,
        HookManager, HookPriority, InferenceHooks, LifecycleHooks, MemoryHooks, ModelHooks,
        SessionHooks, SpecialTokenType, TokenHooks,
    };
    pub use crate::http_compat::{
        compatibility_created_at, estimate_token_count, normalize_openai_embedding_input,
        openai_chat_messages_to_prompt, OllamaGenerateOptions, OllamaGenerateRequest,
        OllamaGenerateResponse, OllamaModelTag, OllamaTagsResponse, OpenAiChatChoice,
        OpenAiChatCompletionsRequest, OpenAiChatCompletionsResponse, OpenAiChatMessage,
        OpenAiEmbeddingData, OpenAiEmbeddingInput, OpenAiEmbeddingsRequest,
        OpenAiEmbeddingsResponse, OpenAiModelDescriptor, OpenAiModelListResponse, OpenAiUsage,
    };
    pub use crate::image_kernel::{
        dynamic_image_plugin_from_opaque, dynamic_image_plugin_into_opaque,
        load_dynamic_image_plugin, DynamicImageKernel, DynamicImagePluginOpaque,
        ImageGenerationPlugin, ImageGenerationRequest, ImageGenerationResult,
    };
    pub use crate::inference::{
        DefaultExecutionPolicy, ExecutionPolicy, InferenceEngine, InferenceEngineBuilder,
    };
    pub use crate::inference_cache::{CacheConfig, CacheStats, InferenceCache};
    pub use crate::kv_cache::{KVBlock, PagedKVCache, SessionKVCache, BLOCK_SIZE};
    pub use crate::kv_cache_advanced::{
        BlockId, BlockTable, MemoryBudgetConfig, MemoryBudgeter, PhysicalBlockPool, PoolStatistics,
    };
    pub use crate::management_auth::{
        dynamic_management_auth_policy_from_opaque, dynamic_management_auth_policy_into_opaque,
        load_dynamic_management_auth_policy, AllowAllManagementAuthPolicy,
        BearerTokenManagementAuthPolicy, DynamicManagementAuthPolicyOpaque,
        LoadedManagementAuthPolicy, LoopbackOnlyManagementAuthPolicy, ManagementAuthContext,
        ManagementAuthDecision, ManagementAuthPolicyDescriptor, ManagementAuthPolicyPlugin,
        ManagementAuthPolicyRegistry,
    };
    pub use crate::mcp::{
        connect_and_register_stdio_server, register_mcp_client_tools, McpClient, McpInputSchema,
        McpRegistrationReport, McpStdioServerConfig, McpTool, McpToolRegistrationOptions,
        StdioMcpClient, DEFAULT_MCP_PROTOCOL_VERSION,
    };
    pub use crate::mcp_registry::{McpServerConfig, McpServerRegistry};
    pub use crate::model::{ModelConfig, ModelLoader};
    pub use crate::model_hot_swap::{
        HotSwapModelRegistry, LoRAConfig, LoadedModel, ModelInfo as HotSwapModelInfo,
    };
    pub use crate::model_pull_policy::{
        authorize_model_pull_request, dynamic_model_pull_policy_from_opaque,
        dynamic_model_pull_policy_into_opaque, load_dynamic_model_pull_policy,
        AllowAllModelPullPolicy, DynamicModelPullPolicyOpaque, LoadedModelPullPolicy,
        LocalOnlyModelPullPolicy, ModelPullPolicyContext, ModelPullPolicyDecision,
        ModelPullPolicyDescriptor, ModelPullPolicyPlugin, ModelPullPolicyRegistry,
        RequireChecksumForRemoteModelPullPolicy,
    };
    pub use crate::model_pull_verifier::{
        dynamic_model_pull_verifier_from_opaque, dynamic_model_pull_verifier_into_opaque,
        load_dynamic_model_pull_verifier, verify_model_pull, AllowAllModelPullVerifier,
        DynamicModelPullVerifierOpaque, LoadedModelPullVerifier, ModelPullVerificationContext,
        ModelPullVerifierDecision, ModelPullVerifierDescriptor, ModelPullVerifierPlugin,
        ModelPullVerifierRegistry, SidecarSha256ModelPullVerifier,
    };
    pub use crate::model_registry::{
        EnsembleCandidateResponse, EnsembleGeneration, EnsembleMergeStrategy, ModelBenchmark,
        ModelId, ModelInfo, ModelRegistry, ModelRoutingStrategy, RoutedGeneration, RoutingAttempt,
    };
    pub use crate::model_store::{
        ModelPullOptions, ModelPullPhase, ModelPullProgress, ModelStore, StoredModel,
    };
    pub use crate::multimodal::{
        Audio, AudioEncoderConfig, AudioEncoderType, Image, ImageFormat, ImagePatch, Modality,
        ModalityToken, MultimodalInput, MultimodalModelAdapter, MultimodalProcessor,
        ProcessorConfig, VisionEncoderConfig, VisionEncoderType,
    };
    pub use crate::multimodal_fusion::{
        FusedTokenSequence, FusionConfig, FusionStrategyType, MultimodalFusion, SpecialToken,
        TokenType, VisionPosition,
    };
    pub use crate::multimodal_io::{
        dynamic_multimodal_io_plugin_from_opaque, dynamic_multimodal_io_plugin_into_opaque,
        DescriptorMultimodalIoPlugin, DynamicMultimodalIoPluginOpaque, MultimodalIoPlugin,
        MultimodalIoRegistry, MultimodalOutputPlan, MultimodalRequest, OutputModality,
    };
    pub use crate::multimodal_plugin::{
        AudioEncoderPlugin, FusionStrategyPlugin, ModalPluginId, MultimodalPluginRegistry,
        PreprocessingPlugin, VisionEncoderPlugin,
    };
    pub use crate::plugin::{Plugin, PluginManager};
    pub use crate::plugin_contract::{
        load_and_validate_plugin_contract, load_plugin_contract_manifest,
        validate_plugin_contract_manifest, validate_runtime_plugin_identity, PluginContractKind,
        PluginContractManifest, LOCI_PLUGIN_ABI_VERSION,
    };
    pub use crate::plugin_registry::{
        create_shared_registry, PluginConfig, PluginRegistry, PluginType, RegistryConfig,
        SharedRegistry,
    };
    pub use crate::policy_registry::{DynamicPolicyRegistry, DynamicPolicyRegistryFile};
    pub use crate::quantization::{
        QuantizationReport, QuantizationScheme, QuantizationTool, QuantizedData, QuantizedTensor,
    };
    pub use crate::radix_cache::{
        BlockHash, RadixNode, RadixTree, RadixTreeStats, ShardedRadixCache, TokenId,
    };
    pub use crate::rag::{
        ChunkingConfig, EmbeddingProvider, HashEmbeddingProvider, InMemoryRagPlugin,
        InMemoryVectorStore, RagChunk, RagDocument, RagEngine, RagPlugin, RetrievedChunk,
    };
    pub use crate::resource_manager::{
        MonitorConfig, ResourceGuard, ResourceLimits, ResourceManager, ResourceStats,
    };
    pub use crate::resource_planner::{ModelResourceEstimate, ResourcePlan, ResourcePlanner};
    pub use crate::runtime_events::{
        RuntimeEvent, RuntimeEventBus, RuntimeEventCategory, RuntimeEventOutcome,
    };
    pub use crate::sampler::{
        sample_token, DefaultSampler, GreedySampler, LogitsView, MirostatSampler, Sampler,
        SamplingParams, TemperatureSampler, TopKSampler, TopPSampler,
    };
    pub use crate::serve_dispatch::{
        dynamic_serve_dispatch_policy_from_opaque, dynamic_serve_dispatch_policy_into_opaque,
        load_dynamic_serve_dispatch_policy, BlockServeDispatchPolicyPlugin,
        DynamicServeDispatchPolicyOpaque, LoadedServeDispatchPolicy, QueueFullAction,
        QueuePressureContext, RejectServeDispatchPolicyPlugin, ServeDispatchPolicyDescriptor,
        ServeDispatchPolicyPlugin, ServeDispatchPolicyRegistry,
    };
    pub use crate::session::{
        SessionHandle, SessionId, SessionInfo, SessionManager, SessionRecord, SessionRole,
        SessionSnapshot, SessionState, SessionSuspendedSnapshot,
    };
    pub use crate::session_bus::{BusError, ControlMessage, SessionBus, SessionMessage};
    pub use crate::session_store::{
        dynamic_session_store_factory_from_opaque, dynamic_session_store_factory_into_opaque,
        DynamicSessionStoreFactoryOpaque, InMemorySessionStore, InMemorySessionStoreFactory,
        SessionStore, SessionStoreConfig, SessionStoreFactory, SessionStoreRegistry,
        SqliteSessionStore, SqliteSessionStoreFactory,
    };
    #[cfg(feature = "redis-store")]
    pub use crate::session_store::{RedisSessionStore, RedisSessionStoreFactory};
    pub use crate::skills::{Skill, SkillPack, SkillProvider, SkillRegistry, SkillToolPolicy};
    pub use crate::timeout_controller::{
        CancellationHandle, TimeoutConfig, TimeoutContext, TimeoutController, TimeoutStats,
    };
    pub use crate::tool_plugin::{
        dynamic_tool_plugin_from_opaque, dynamic_tool_plugin_into_opaque, load_dynamic_tool_plugin,
        register_tool_plugin, DynamicToolPluginOpaque, LoadedToolPlugin,
        LoadedToolPluginDescriptor, ToolPlugin,
    };
    pub use crate::vision_clip::{
        BatchCLIPEncoder, CLIPViTL14Config, CLIPViTL14Encoder, ImageEmbedding,
    };
    pub use crate::wasm_plugin::{WasmPlugin, WasmPluginConfig, WasmPluginManager};
}

/// C API types for FFI interop
pub mod c_api_types {
    pub use crate::c_api::{
        LociDeviceInfo, LociDeviceSelector, LociEngine, LociPluginRegistry, LociStreamCallback,
    };
}

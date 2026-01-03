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
//! ```no_run
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
pub mod device;
pub mod error;
pub mod hooks;
pub mod inference;
pub mod kv_cache;
pub mod kv_cache_advanced;
pub mod model;
pub mod model_registry;
pub mod plugin;
pub mod plugin_registry;
pub mod sampler;
pub mod session;
pub mod session_bus;
pub mod wasm_plugin;
pub mod c_api;

pub use error::{LociError, Result};
pub use inference::InferenceEngine;
pub use model::{ModelConfig, ModelLoader};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::backend::{InferenceBackend, Model, InferenceParams};
    pub use crate::backends::LlamaCppBackend;
    pub use crate::constraint::{
        Constraint, ConstraintMask, ConstraintManager, ConstraintCombinator, CombinatorMode,
        TokenWhitelistConstraint, TokenBlacklistConstraint, LengthConstraint,
        RegexConstraint, JsonConstraint
    };
    pub use crate::device::{DeviceSelector, DeviceConfig, DeviceInfo, DeviceType};
    pub use crate::error::{LociError, Result};
    pub use crate::hooks::{
        DeepProgrammableHooks, HookManager, HookContext, HookPriority, HookControl,
        LifecycleHooks, ModelHooks, InferenceHooks, TokenHooks, MemoryHooks,
        BackendHooks, SessionHooks, ErrorHooks, ErrorRecovery, SpecialTokenType
    };
    pub use crate::inference::InferenceEngine;
    pub use crate::kv_cache::{KVBlock, PagedKVCache, SessionKVCache, BLOCK_SIZE};
    pub use crate::kv_cache_advanced::{
        PhysicalBlockPool, BlockTable, BlockId,
        MemoryBudgetConfig, MemoryBudgeter, PoolStatistics
    };
    pub use crate::model::{ModelConfig, ModelLoader};
    pub use crate::model_registry::{ModelId, ModelRegistry, ModelInfo};
    pub use crate::plugin::{Plugin, PluginManager};
    pub use crate::plugin_registry::{PluginRegistry, PluginConfig, PluginType, SharedRegistry};
    pub use crate::sampler::{LogitsView, SamplingParams, Sampler, sample_token, DefaultSampler, GreedySampler};
    pub use crate::session::{SessionId, SessionManager, SessionHandle, SessionInfo};
    pub use crate::session_bus::{SessionBus, SessionMessage, ControlMessage, BusError};
    pub use crate::wasm_plugin::{WasmPlugin, WasmPluginConfig, WasmPluginManager};
}

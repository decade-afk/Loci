pub mod backend;
pub mod backends;
pub mod core;
pub mod engine;
pub mod error;
pub mod model;
pub mod plugin;

pub use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceBackend,
    InferenceParams, Model, ModelMetadata,
};
pub use crate::core::{
    CoreRegistry, DefaultCoreRegistry, EventBus, ModelRepository, PluginManager, WorkflowEngine,
};
pub use crate::engine::{GenerationParams, InferenceEngine, InferenceEngineBuilder, ModelInfo};
pub use crate::error::{LociError, Result};
pub use crate::model::{ModelConfig, ModelLoadStrategy, ModelLoader};
pub use crate::plugin::{InMemoryPluginManager, RegisteredPlugin};

pub mod backend;
pub mod backends;
pub mod core;
pub mod error;
pub mod model;
pub mod plugin;
pub mod engine;

pub use loci_plugin_api::{CoreComponent, CoreRewriters, PlatformTrack, PluginManifest};
pub use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceBackend,
    InferenceParams, Model, ModelMetadata,
};
pub use crate::core::{
    CoreRegistry, DefaultCoreRegistry, EventBus, HardwareAbstraction, ModelRepository,
    PluginManager, UiHost, WorkflowEngine,
};
pub use crate::engine::{GenerationParams, InferenceEngine, InferenceEngineBuilder, ModelInfo};
pub use crate::error::{LociError, Result};
pub use crate::model::{ModelConfig, ModelLoadStrategy, ModelLoader};
pub use crate::plugin::{InMemoryPluginManager, RegisteredPlugin};

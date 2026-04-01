pub mod backend;
pub mod backends;
pub mod core;
pub mod engine;
pub mod error;
pub mod model;
pub mod plugin;
pub mod sampler;

pub use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceBackend,
    InferenceParams, Model, ModelMetadata,
};
pub use crate::core::{
    CoreRegistry, DefaultCoreRegistry, EventBus, HardwareAbstraction, ModelRepository,
    PluginManager, UiHost, WorkflowEngine,
};
pub use crate::engine::{
    CoreRewriterStatus, GenerationParams, InferenceEngine, InferenceEngineBuilder, ModelInfo,
    PluginRuntimeDetail, PluginRuntimeStatus, RuntimeSnapshot,
};
pub use crate::error::{LociError, Result};
pub use crate::model::{ModelConfig, ModelLoadStrategy, ModelLoader};
pub use crate::plugin::{
    discover_plugin_bundle_files, discover_plugin_manifest_files, load_plugin_bundle_file,
    load_plugin_manifest_file, InMemoryPluginManager, PluginSamplingRuntime, RegisteredPlugin,
    SamplingHook, SamplingHookProfile, SamplingLogitBias,
};
pub use crate::sampler::{sample_token, LogitsView, SamplingParams};
pub use loci_plugin_api::{
    CoreComponent, CoreRewriters, LegacyRuntimeBridge, PlatformTrack, PluginBootstrap,
    PluginCompatibility, PluginManifest, PluginRuntime, PluginSourceFormat,
};

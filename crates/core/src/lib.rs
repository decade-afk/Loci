pub mod backend;
pub mod backends;
pub mod control_plane;
pub mod core;
pub mod engine;
pub mod error;
pub mod management;
pub mod model;
pub mod plugin;
pub mod sampler;

pub use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceBackend,
    InferenceParams, Model, ModelMetadata,
};
pub use crate::control_plane::{
    CoreRewriterActivationRequest, CoreRewriterActivationStatus, CoreRewriterStatus,
    InferenceActivationStatus, LegacyTextPluginActivationStatus, ManagementHealthStatus,
    ModelLoadConfig, ModelLoadRequest, ModelLoadSplitMode, ModelLoadStatus,
    ModelLoadStrategyRequest, ModelRuntimeInfo, PluginLoadRequest, PluginLoadSourceKind,
    PluginLoadStatus, PluginRuntimeDetail, PluginRuntimeStatus, RuntimeSnapshot,
    SamplingHookSource, TextGenerationParams, TextGenerationRequest, TextGenerationResponse,
};
pub use crate::core::{
    CoreRegistry, DefaultCoreRegistry, EventBus, HardwareAbstraction, ModelRepository,
    PluginManager, UiHost, WorkflowEngine,
};
pub use crate::engine::{GenerationParams, InferenceEngine, InferenceEngineBuilder, ModelInfo};
pub use crate::error::{LociError, Result};
pub use crate::management::ManagementService;
pub use crate::model::{ModelConfig, ModelLoadStrategy, ModelLoader};
pub use crate::plugin::{
    discover_plugin_bundle_files, discover_plugin_manifest_files, load_plugin_bundle_file,
    load_plugin_manifest_file, InMemoryPluginManager, PluginSamplingRuntime, RegisteredPlugin,
    SamplingHook, SamplingHookProfile, SamplingLogitBias,
};
pub use crate::sampler::{sample_token, LogitsView, SamplingParams};
pub use loci_plugin_api::{
    ContributionPoints, CoreComponent, CoreRewriters, LegacyRuntimeBridge, PlatformTrack,
    PluginBootstrap, PluginCompatibility, PluginManifest, PluginRuntime, PluginSourceFormat,
};

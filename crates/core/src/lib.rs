pub mod backend;
pub mod backends;
pub mod engine;
pub mod error;
pub mod model;
pub mod pipeline;
pub mod plugin;
pub mod runtime;
pub mod sampler;

pub use crate::backend::{
    BackendCapabilities, BackendParams, BackendRegistry, GpuSplitMode, InferenceBackend,
    InferenceParams, Model, ModelMetadata,
};
pub use crate::engine::{InferenceEngine, InferenceEngineBuilder};
pub use crate::error::{LociError, Result};
pub use crate::model::{ModelConfig, ModelLoadStrategy, ModelLoader};
pub use crate::pipeline::{merge_inference_params, InferenceResponse};
pub use crate::plugin::{
    discover_plugin_manifest_files, load_plugin_manifest_file, PluginSamplingRuntime,
    PluginStatus, RegisteredPlugin,
};
pub use crate::runtime::{ActiveModelStatus, RuntimeSnapshot};
pub use crate::sampler::{sample_token, LogitsView, SamplingParams};
pub use loci_plugin_api::{
    PluginCapabilities, PluginKind, PluginManifest, PluginRuntime, HOST_PLUGIN_API_VERSION,
};

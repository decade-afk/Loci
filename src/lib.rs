//! Loci - A cross-platform, plugin-based local LLM inference framework
//!
//! Loci provides a unified interface for running large language models locally
//! with support for multiple backends and plugin architectures.

mod ffi;

pub mod error;
pub mod inference;
pub mod model;
pub mod plugin;
pub mod c_api;

pub use error::{LociError, Result};
pub use inference::InferenceEngine;
pub use model::{ModelConfig, ModelLoader};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::error::{LociError, Result};
    pub use crate::inference::InferenceEngine;
    pub use crate::model::{ModelConfig, ModelLoader};
    pub use crate::plugin::{Plugin, PluginManager};
}

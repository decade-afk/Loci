//! Model loading and configuration

use crate::error::{LociError, Result};
use std::path::{Path, PathBuf};

/// Configuration for loading a model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Path to the GGUF model file
    pub model_path: PathBuf,
    /// Context size (number of tokens)
    pub n_ctx: u32,
    /// Number of threads to use for generation
    pub n_threads: Option<u32>,
    /// Batch size for prompt processing
    pub n_batch: u32,
    /// Use GPU if available
    pub use_gpu: bool,
    /// GPU layers to offload (-1 for all)
    pub n_gpu_layers: i32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 4096,
            n_threads: None, // Auto-detect
            n_batch: 512,
            use_gpu: true,
            n_gpu_layers: -1, // All layers
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration with the given path
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Set the context size
    pub fn with_context_size(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// Set the number of threads
    pub fn with_threads(mut self, n_threads: u32) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    /// Set the batch size
    pub fn with_batch_size(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

    /// Disable GPU acceleration
    pub fn cpu_only(mut self) -> Self {
        self.use_gpu = false;
        self.n_gpu_layers = 0;
        self
    }

    /// Set GPU layers to offload
    pub fn with_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if !self.model_path.exists() {
            return Err(LociError::ConfigError(format!(
                "Model file not found: {}",
                self.model_path.display()
            )));
        }

        if self.n_ctx == 0 {
            return Err(LociError::ConfigError(
                "Context size must be greater than 0".to_string(),
            ));
        }

        if self.n_batch == 0 {
            return Err(LociError::ConfigError(
                "Batch size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Model loader interface
pub struct ModelLoader;

impl ModelLoader {
    /// Load a model with the given configuration
    pub fn load(config: &ModelConfig) -> Result<()> {
        config.validate()?;
        // Actual loading will be implemented in the inference module
        Ok(())
    }
}

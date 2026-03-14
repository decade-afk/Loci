//! Model loading and configuration

use crate::backend::GpuSplitMode;
use crate::error::{LociError, Result};
use std::path::{Path, PathBuf};

/// Model loading strategy for large-model placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelLoadStrategy {
    /// Fail immediately when the initial placement cannot be satisfied.
    #[default]
    Strict,
    /// Retry model loading with progressively fewer GPU-offloaded layers.
    AutoReduceGpuLayers { step: u32 },
}

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
    /// Use memory-mapped model loading when possible
    pub use_mmap: bool,
    /// Lock model pages into RAM when supported by the OS
    pub use_mlock: bool,
    /// Offload K/Q/V ops and KV cache placement to device
    pub kv_offload: bool,
    /// Offload host tensor ops to the active device
    pub op_offload: bool,
    /// Multi-GPU split strategy
    pub split_mode: GpuSplitMode,
    /// Primary GPU index used for single-GPU placement
    pub main_gpu: u32,
    /// Relative split weights for each GPU
    pub tensor_split: Option<Vec<f32>>,
    /// Strategy for retrying placement when the requested GPU residency does not fit.
    pub load_strategy: ModelLoadStrategy,
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
            use_mmap: true,
            use_mlock: false,
            kv_offload: true,
            op_offload: true,
            split_mode: GpuSplitMode::Layer,
            main_gpu: 0,
            tensor_split: None,
            load_strategy: ModelLoadStrategy::Strict,
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
        self.kv_offload = false;
        self.op_offload = false;
        self.split_mode = GpuSplitMode::None;
        self.main_gpu = 0;
        self.tensor_split = None;
        self
    }

    /// Set GPU layers to offload
    pub fn with_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Enable or disable memory-mapped model loading.
    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    /// Enable or disable memory locking for model pages.
    pub fn with_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    /// Enable or disable KV cache and K/Q/V offload.
    pub fn with_kv_offload(mut self, kv_offload: bool) -> Self {
        self.kv_offload = kv_offload;
        self
    }

    /// Enable or disable host op offload.
    pub fn with_op_offload(mut self, op_offload: bool) -> Self {
        self.op_offload = op_offload;
        self
    }

    /// Set the multi-GPU split strategy.
    pub fn with_gpu_split_mode(mut self, split_mode: GpuSplitMode) -> Self {
        self.split_mode = split_mode;
        self
    }

    /// Set the primary GPU index used for single-GPU placement.
    pub fn with_main_gpu(mut self, main_gpu: u32) -> Self {
        self.main_gpu = main_gpu;
        self
    }

    /// Set relative split weights across multiple GPUs.
    pub fn with_tensor_split(mut self, tensor_split: Vec<f32>) -> Self {
        self.tensor_split = Some(tensor_split);
        self
    }

    /// Set the model loading strategy.
    pub fn with_load_strategy(mut self, load_strategy: ModelLoadStrategy) -> Self {
        self.load_strategy = load_strategy;
        self
    }

    /// Retry model loading with fewer GPU layers when the requested placement does not fit.
    pub fn with_auto_gpu_layer_fallback(mut self, step: u32) -> Self {
        self.load_strategy = ModelLoadStrategy::AutoReduceGpuLayers { step };
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

        if matches!(
            self.load_strategy,
            ModelLoadStrategy::AutoReduceGpuLayers { step: 0 }
        ) {
            return Err(LociError::ConfigError(
                "GPU fallback step must be greater than 0".to_string(),
            ));
        }

        if let Some(tensor_split) = &self.tensor_split {
            if tensor_split.is_empty() {
                return Err(LociError::ConfigError(
                    "Tensor split must contain at least one value".to_string(),
                ));
            }
            if tensor_split
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(LociError::ConfigError(
                    "Tensor split values must be finite and non-negative".to_string(),
                ));
            }
            if !tensor_split.iter().any(|value| *value > 0.0) {
                return Err(LociError::ConfigError(
                    "Tensor split must contain at least one positive value".to_string(),
                ));
            }
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

use crate::backend::{BackendParams, GpuSplitMode};
use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelLoadStrategy {
    #[default]
    Strict,
    AutoReduceGpuLayers {
        step: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub n_threads: Option<u32>,
    pub n_batch: u32,
    pub use_gpu: bool,
    pub n_gpu_layers: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub kv_offload: bool,
    pub op_offload: bool,
    pub split_mode: GpuSplitMode,
    pub main_gpu: u32,
    pub tensor_split: Option<Vec<f32>>,
    pub load_strategy: ModelLoadStrategy,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 4096,
            n_threads: None,
            n_batch: 512,
            use_gpu: true,
            n_gpu_layers: -1,
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
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    pub fn with_context_size(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn with_threads(mut self, n_threads: u32) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    pub fn with_batch_size(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

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

    pub fn with_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    pub fn with_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    pub fn with_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    pub fn with_kv_offload(mut self, kv_offload: bool) -> Self {
        self.kv_offload = kv_offload;
        self
    }

    pub fn with_op_offload(mut self, op_offload: bool) -> Self {
        self.op_offload = op_offload;
        self
    }

    pub fn with_gpu_split_mode(mut self, split_mode: GpuSplitMode) -> Self {
        self.split_mode = split_mode;
        self
    }

    pub fn with_main_gpu(mut self, main_gpu: u32) -> Self {
        self.main_gpu = main_gpu;
        self
    }

    pub fn with_tensor_split(mut self, tensor_split: Vec<f32>) -> Self {
        self.tensor_split = Some(tensor_split);
        self
    }

    pub fn with_load_strategy(mut self, load_strategy: ModelLoadStrategy) -> Self {
        self.load_strategy = load_strategy;
        self
    }

    pub fn with_auto_gpu_layer_fallback(mut self, step: u32) -> Self {
        self.load_strategy = ModelLoadStrategy::AutoReduceGpuLayers { step };
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_path.as_os_str().is_empty() {
            return Err(LociError::ConfigError(
                "model path must not be empty".to_string(),
            ));
        }
        if self.n_ctx == 0 {
            return Err(LociError::ConfigError(
                "context size must be greater than 0".to_string(),
            ));
        }
        if self.n_batch == 0 {
            return Err(LociError::ConfigError(
                "batch size must be greater than 0".to_string(),
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
                    "tensor split must contain at least one value".to_string(),
                ));
            }
            if tensor_split
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err(LociError::ConfigError(
                    "tensor split values must be finite and non-negative".to_string(),
                ));
            }
            if !tensor_split.iter().any(|value| *value > 0.0) {
                return Err(LociError::ConfigError(
                    "tensor split must contain at least one positive value".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn to_backend_params(&self) -> BackendParams {
        let mut options = vec![
            ("n_ctx".to_string(), self.n_ctx.to_string()),
            ("n_batch".to_string(), self.n_batch.to_string()),
        ];
        if let Some(n_threads) = self.n_threads {
            options.push(("n_threads".to_string(), n_threads.to_string()));
        }

        BackendParams {
            n_gpu_layers: self.n_gpu_layers,
            use_gpu: self.use_gpu,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            kv_offload: self.kv_offload,
            op_offload: self.op_offload,
            split_mode: self.split_mode,
            main_gpu: self.main_gpu,
            tensor_split: self.tensor_split.clone(),
            options,
        }
    }
}

pub struct ModelLoader;

impl ModelLoader {
    pub fn load(config: &ModelConfig) -> Result<()> {
        config.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_rejects_empty_path() {
        let err = ModelConfig::default().validate().expect_err("should fail");
        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn model_config_validates_tensor_split() {
        let err = ModelConfig::new("demo.gguf")
            .with_tensor_split(vec![0.0, 0.0])
            .validate()
            .expect_err("should fail");
        assert!(matches!(err, LociError::ConfigError(_)));
    }

    #[test]
    fn model_config_converts_to_backend_params() {
        let params = ModelConfig::new("demo.gguf")
            .with_context_size(8192)
            .with_batch_size(1024)
            .with_threads(12)
            .cpu_only()
            .to_backend_params();

        assert_eq!(params.n_gpu_layers, 0);
        assert!(!params.use_gpu);
        assert!(params
            .options
            .iter()
            .any(|(k, v)| k == "n_ctx" && v == "8192"));
        assert!(params
            .options
            .iter()
            .any(|(k, v)| k == "n_batch" && v == "1024"));
        assert!(params
            .options
            .iter()
            .any(|(k, v)| k == "n_threads" && v == "12"));
    }
}

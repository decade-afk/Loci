use crate::backends;
use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GpuSplitMode {
    None,
    #[default]
    Layer,
    Row,
}

#[derive(Debug, Clone)]
pub struct BackendParams {
    pub n_gpu_layers: i32,
    pub use_gpu: bool,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub kv_offload: bool,
    pub op_offload: bool,
    pub split_mode: GpuSplitMode,
    pub main_gpu: u32,
    pub tensor_split: Option<Vec<f32>>,
    pub options: Vec<(String, String)>,
}

impl Default for BackendParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: -1,
            use_gpu: true,
            use_mmap: true,
            use_mlock: false,
            kv_offload: true,
            op_offload: true,
            split_mode: GpuSplitMode::Layer,
            main_gpu: 0,
            tensor_split: None,
            options: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: Option<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: None,
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub name: String,
    pub version: String,
    pub supports_text: bool,
    pub supports_multimodal: bool,
    pub supports_embeddings: bool,
    pub supports_streaming: bool,
    pub has_gpu_support: bool,
    pub supported_formats: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub architecture: String,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_layer: u32,
    pub param_count: Option<u64>,
}

pub trait InferenceBackend: Send + Sync {
    fn capabilities(&self) -> BackendCapabilities;
    fn load_model(
        &self,
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>>;

    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

pub trait Model: Send + Sync {
    fn metadata(&self) -> ModelMetadata;
    fn infer_text(&mut self, prompt: &str, params: &InferenceParams) -> Result<String>;

    fn attach_sampling_runtime(
        &mut self,
        _runtime: crate::plugin::PluginSamplingRuntime,
    ) -> Result<()> {
        Ok(())
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, backend: Box<dyn InferenceBackend>) {
        self.backends.insert(name, backend);
    }

    pub fn with_builtin_backends() -> Self {
        let mut registry = Self::new();
        backends::register_builtin_backends(&mut registry);
        registry
    }

    pub fn contains(&self, name: &str) -> bool {
        self.backends.contains_key(name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.backends.keys().map(String::as_str).collect()
    }

    pub fn list(&self) -> Vec<BackendCapabilities> {
        let mut backends = self
            .backends
            .values()
            .map(|backend| backend.capabilities())
            .collect::<Vec<_>>();
        backends.sort_by(|left, right| left.name.cmp(&right.name));
        backends
    }

    pub fn capabilities(&self, backend_name: &str) -> Option<BackendCapabilities> {
        self.backends
            .get(backend_name)
            .map(|backend| backend.capabilities())
    }

    pub fn load_model(
        &mut self,
        backend_name: &str,
        model_path: &Path,
        params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        let backend = self.backends.get_mut(backend_name).ok_or_else(|| {
            LociError::BackendNotAvailable(format!("backend not found: {backend_name}"))
        })?;
        backend.init()?;
        backend.load_model(model_path, params)
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::with_builtin_backends()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_include_mock_backend() {
        let registry = BackendRegistry::with_builtin_backends();
        assert!(registry.contains("mock"));
    }

    #[cfg(feature = "llama")]
    #[test]
    fn builtins_include_llama_backend_when_feature_enabled() {
        let registry = BackendRegistry::with_builtin_backends();
        assert!(registry.contains("llama.cpp"));
    }
}

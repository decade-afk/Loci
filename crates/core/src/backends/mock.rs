use crate::backend::{
    BackendCapabilities, BackendParams, InferenceBackend, InferenceParams, Model, ModelMetadata,
};
use crate::error::{LociError, Result};
use std::path::{Path, PathBuf};

pub struct MockBackend;

impl MockBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "mock".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            supports_text: true,
            supports_multimodal: false,
            supports_embeddings: false,
            supports_streaming: false,
            has_gpu_support: false,
            supported_formats: vec!["gguf".to_string(), "mock".to_string()],
        }
    }

    fn load_model(
        &self,
        model_path: &Path,
        _backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        Ok(Box::new(MockModel {
            model_path: model_path.to_path_buf(),
        }))
    }
}

pub struct MockModel {
    model_path: PathBuf,
}

impl Model for MockModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            architecture: "mock".to_string(),
            n_vocab: 0,
            n_ctx_train: 4096,
            n_embd: 0,
            n_layer: 0,
            param_count: None,
        }
    }

    fn infer_text(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        if prompt.trim().is_empty() {
            return Err(LociError::InvalidArgument(
                "prompt must not be empty".to_string(),
            ));
        }

        Ok(format!(
            "mock:{prompt} [model={}, max_tokens={}, temp={}]",
            self.model_path.display(),
            params.max_tokens,
            params.temperature
        ))
    }
}

use crate::backend::{
    BackendCapabilities, BackendParams, InferenceBackend, InferenceParams, Model, ModelMetadata,
};
use crate::error::{LociError, Result};
use std::path::{Path, PathBuf};

pub struct LlamaCppBackend {
    initialized: bool,
}

impl LlamaCppBackend {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "llama.cpp".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            supports_text: true,
            supports_multimodal: false,
            supports_embeddings: false,
            supports_streaming: false,
            has_gpu_support: true,
            supported_formats: vec!["gguf".to_string()],
        }
    }

    fn load_model(
        &self,
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        if !self.initialized {
            return Err(LociError::BackendError(
                "llama.cpp backend not initialized".to_string(),
            ));
        }

        Ok(Box::new(LlamaCppModel {
            model_path: model_path.to_path_buf(),
            backend_params,
        }))
    }

    fn init(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }
}

pub struct LlamaCppModel {
    model_path: PathBuf,
    backend_params: BackendParams,
}

impl Model for LlamaCppModel {
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            architecture: "llama".to_string(),
            n_vocab: 0,
            n_ctx_train: self
                .backend_params
                .options
                .iter()
                .find(|(key, _)| key == "n_ctx")
                .and_then(|(_, value)| value.parse().ok())
                .unwrap_or(4096),
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
            "llama.cpp-stub:{prompt} [model={}, gpu_layers={}, max_tokens={}]",
            self.model_path.display(),
            self.backend_params.n_gpu_layers,
            params.max_tokens
        ))
    }
}

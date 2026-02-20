use crate::backend::{
    BackendCapabilities, BackendParams, InferenceBackend, InferenceParams, Model, ModelExt,
    ModelMetadata,
};
use crate::error::{LociError, Result};
use std::path::{Path, PathBuf};

pub struct CandleBackend {
    initialized: bool,
}

impl CandleBackend {
    pub fn new() -> Self {
        Self { initialized: false }
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for CandleBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            name: "candle".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            supports_text: true,
            supports_multimodal: false,
            supports_embeddings: false,
            supports_streaming: true,
            has_gpu_support: false,
            supported_formats: vec!["safetensors".to_string(), "gguf".to_string()],
        }
    }

    fn load_model(
        &self,
        model_path: &Path,
        _backend_params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        if !self.initialized {
            return Err(LociError::BackendError(
                "Candle backend not initialized. Call init() first.".to_string(),
            ));
        }

        let model = CandleModel::load(model_path)?;
        Ok(Box::new(model))
    }

    fn init(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        self.initialized = false;
        Ok(())
    }
}

pub struct CandleModel {
    model_path: PathBuf,
    metadata: ModelMetadata,
}

impl CandleModel {
    pub fn load(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(LociError::ModelLoadError(format!(
                "Model not found: {}",
                model_path.display()
            )));
        }

        Ok(Self {
            model_path: model_path.to_path_buf(),
            metadata: ModelMetadata {
                architecture: "candle-transformer".to_string(),
                n_vocab: 32000,
                n_ctx_train: 4096,
                n_embd: 4096,
                n_layer: 32,
                param_count: None,
            },
        })
    }
}

impl Model for CandleModel {
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn infer_text(&mut self, prompt: &str, _params: &InferenceParams) -> Result<String> {
        if prompt.trim().is_empty() {
            return Err(LociError::InvalidArgument(
                "Prompt cannot be empty".to_string(),
            ));
        }

        Ok(format!(
            "[candle backend demo: {}] {}",
            self.model_path.display(),
            prompt
        ))
    }

    fn infer_stream(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        let response = self.infer_text(prompt, params)?;
        for token in response.split_whitespace() {
            if !callback(token) {
                break;
            }
            if !callback(" ") {
                break;
            }
        }
        Ok(())
    }
}

impl ModelExt for CandleModel {
    fn infer_stream<F>(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        let response = self.infer_text(prompt, params)?;
        for token in response.split_whitespace() {
            if !callback(token) {
                break;
            }
            if !callback(" ") {
                break;
            }
        }
        Ok(())
    }

    fn infer_multimodal_stream<F>(
        &mut self,
        _text: &str,
        _images: &[crate::backend::Image],
        _params: &InferenceParams,
        _callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        Err(LociError::UnsupportedOperation(
            "Candle multimodal streaming not supported".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_backend_capabilities() {
        let backend = CandleBackend::new();
        let capabilities = backend.capabilities();
        assert_eq!(capabilities.name, "candle");
        assert!(capabilities.supports_text);
        assert!(capabilities.supports_streaming);
    }
}

//! Core inference engine with plugin support
//!
//! This module provides the main `InferenceEngine` which orchestrates:
//! - Backend selection and model loading
//! - Plugin management (text processing hooks)
//! - Unified inference API

use crate::backend::{BackendParams, InferenceBackend, InferenceParams, Model, ModelExt};
use crate::backends::{LlamaCppBackend, LlamaCppModel};
use crate::error::{LociError, Result};
use crate::model::ModelConfig;
use crate::plugin::PluginManager;
use std::path::PathBuf;

/// Parameters for text generation (legacy compatibility)
///
/// This is a convenience wrapper around `InferenceParams` for backward compatibility.
/// New code should use `InferenceParams` directly.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling (0.0 = greedy, higher = more random)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Top-k sampling threshold
    pub top_k: u32,
    /// Repetition penalty
    pub repeat_penalty: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

impl From<GenerationParams> for InferenceParams {
    fn from(params: GenerationParams) -> Self {
        InferenceParams {
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repeat_penalty: params.repeat_penalty,
            ..Default::default()
        }
    }
}

/// Main inference engine
///
/// Orchestrates backend, model, and plugin management. Provides a unified
/// interface for text generation with plugin hooks.
///
/// Note: Currently hardcoded to use LlamaCppModel for Phase 1.
/// Future versions will support generic backends.
pub struct InferenceEngine {
    model: LlamaCppModel,
    plugin_manager: PluginManager,
}

impl InferenceEngine {
    /// Create a new inference engine with the given model configuration
    ///
    /// This uses the llama.cpp backend by default (Phase 1 implementation).
    pub fn new(config: ModelConfig) -> Result<Self> {
        config.validate()?;

        // Initialize llama.cpp backend
        let mut backend = LlamaCppBackend::new();
        backend.init()?;

        // Set up backend parameters
        let backend_params = BackendParams {
            n_gpu_layers: config.n_gpu_layers,
            use_gpu: config.use_gpu,
            options: Vec::new(),
        };

        // Load model directly as LlamaCppModel
        // Phase 1: Hardcoded to llama.cpp backend
        let model = LlamaCppModel::load(&config.model_path, backend_params)?;

        Ok(Self {
            model,
            plugin_manager: PluginManager::new(),
        })
    }

    /// Create a new builder for configuring the engine
    pub fn builder() -> InferenceEngineBuilder {
        InferenceEngineBuilder::new()
    }

    /// Get plugin manager (mutable)
    pub fn plugin_manager_mut(&mut self) -> &mut PluginManager {
        &mut self.plugin_manager
    }

    /// Get plugin manager (immutable)
    pub fn plugin_manager(&self) -> &PluginManager {
        &self.plugin_manager
    }

    /// Generate text from a prompt (legacy API)
    ///
    /// For new code, prefer using `generate_with_params` with `InferenceParams`.
    pub fn generate(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        let inference_params = InferenceParams::from(params);
        self.generate_with_params(prompt, &inference_params)
    }

    /// Generate text from a prompt with full parameter control
    pub fn generate_with_params(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
    ) -> Result<String> {
        // Apply pre-generate plugins
        let processed_prompt = self.plugin_manager.apply_pre_generate(prompt)?;

        // Perform inference
        let response = self.model.infer_text(&processed_prompt, params)?;

        // Apply post-generate plugins
        let final_response = self.plugin_manager.apply_post_generate(&response)?;

        Ok(final_response)
    }

    /// Generate text with streaming output (legacy API)
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        params: GenerationParams,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        let inference_params = InferenceParams::from(params);
        self.generate_stream_with_params(prompt, &inference_params, callback)
    }

    /// Generate text with streaming output
    pub fn generate_stream_with_params<F>(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        if !self.model.supports_streaming() {
            return Err(LociError::UnsupportedOperation(
                "Streaming not supported by current backend".to_string(),
            ));
        }

        // Apply pre-generate plugins
        let processed_prompt = self.plugin_manager.apply_pre_generate(prompt)?;

        // Wrap callback with plugin processing
        let plugin_manager = &self.plugin_manager;
        let wrapped_callback = |token: &str| -> bool {
            let processed_token = match plugin_manager.apply_on_token(token) {
                Ok(t) => t,
                Err(_) => return false,
            };
            callback(&processed_token)
        };

        // Perform streaming inference using ModelExt trait
        self.model
            .infer_stream(&processed_prompt, params, wrapped_callback)?;

        Ok(())
    }

    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        let metadata = self.model.metadata();
        ModelInfo {
            n_vocab: metadata.n_vocab,
            n_ctx_train: metadata.n_ctx_train,
            n_embd: metadata.n_embd,
        }
    }

    /// Get detailed model metadata
    pub fn model_metadata(&self) -> crate::backend::ModelMetadata {
        self.model.metadata()
    }
}

/// Information about the loaded model (legacy compatibility)
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Vocabulary size
    pub n_vocab: u32,
    /// Training context size
    pub n_ctx_train: u32,
    /// Embedding dimension
    pub n_embd: u32,
}

/// Builder for configuring InferenceEngine
pub struct InferenceEngineBuilder {
    model_path: Option<PathBuf>,
    n_ctx: u32,
    n_threads: Option<u32>,
    n_batch: u32,
    use_gpu: bool,
    n_gpu_layers: i32,
}

impl InferenceEngineBuilder {
    fn new() -> Self {
        Self {
            model_path: None,
            n_ctx: 4096,
            n_threads: None,
            n_batch: 512,
            use_gpu: true,
            n_gpu_layers: -1,
        }
    }

    /// Set the model path
    pub fn model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the context size
    pub fn context_size(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// Set the number of threads
    pub fn threads(mut self, n_threads: u32) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, n_batch: u32) -> Self {
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
    pub fn gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Build the inference engine
    pub fn build(self) -> Result<InferenceEngine> {
        let model_path = self
            .model_path
            .ok_or_else(|| LociError::ConfigError("Model path not specified".to_string()))?;

        let mut config = ModelConfig::new(model_path)
            .with_context_size(self.n_ctx)
            .with_batch_size(self.n_batch)
            .with_gpu_layers(self.n_gpu_layers);

        if !self.use_gpu {
            config = config.cpu_only();
        }

        if let Some(threads) = self.n_threads {
            config = config.with_threads(threads);
        }

        InferenceEngine::new(config)
    }
}

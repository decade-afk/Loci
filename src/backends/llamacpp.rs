//! llama.cpp backend implementation with stateless sampling
//!
//! This module provides the native llama.cpp backend, refactored to use
//! the new stateless sampling system with zero-copy logits manipulation.

use crate::backend::{
    BackendCapabilities, BackendParams, InferenceBackend, InferenceParams, Model, ModelMetadata,
};
use crate::error::{LociError, Result};
use crate::ffi;
use crate::sampler::{LogitsView, SamplingParams, sample_token};
use std::path::Path;

/// llama.cpp backend implementation
pub struct LlamaCppBackend {
    initialized: bool,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend
    pub fn new() -> Self {
        Self {
            initialized: false,
        }
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
            supports_streaming: true,
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
                "Backend not initialized. Call init() first.".to_string(),
            ));
        }

        let model = LlamaCppModel::load(model_path, backend_params)?;
        Ok(Box::new(model))
    }

    fn init(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        ffi::backend_init();
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }

        ffi::backend_free();
        self.initialized = false;
        Ok(())
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// llama.cpp model implementation with stateless sampling
pub struct LlamaCppModel {
    model: ffi::LlamaModel,
    context: ffi::LlamaContext,
    metadata: ModelMetadata,
}

// Safety: llama.cpp's context is thread-safe for single-threaded access
unsafe impl Send for LlamaCppModel {}
unsafe impl Sync for LlamaCppModel {}

impl LlamaCppModel {
    /// Load a model from file
    pub fn load(model_path: &Path, params: BackendParams) -> Result<Self> {
        // Set up model parameters
        let mut model_params = ffi::model_default_params();
        model_params.n_gpu_layers = params.n_gpu_layers as i32;

        // Load model
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| LociError::ConfigError("Invalid model path".to_string()))?;

        let model = ffi::LlamaModel::from_file(model_path_str, &model_params)
            .map_err(|e| LociError::ModelLoadError(e))?;

        // Extract metadata
        let metadata = ModelMetadata {
            architecture: "llama".to_string(),
            n_vocab: model.n_vocab() as u32,
            n_ctx_train: model.n_ctx_train() as u32,
            n_embd: model.n_embd() as u32,
            n_layer: 0,
            param_count: None,
        };

        // Create context
        let mut ctx_params = ffi::context_default_params();
        ctx_params.n_ctx = 512;

        let context = ffi::LlamaContext::new(&model, &ctx_params)
            .map_err(|e| LociError::InferenceError(e))?;

        Ok(Self {
            model,
            context,
            metadata,
        })
    }

    /// Recreate context with new parameters
    fn recreate_context(&mut self, params: &InferenceParams) -> Result<()> {
        let mut ctx_params = ffi::context_default_params();
        ctx_params.n_ctx = params.n_ctx;
        ctx_params.n_batch = params.n_batch;

        if let Some(n_threads) = params.n_threads {
            ctx_params.n_threads = n_threads as i32;
        }

        // Drop old context and create new one
        let new_context = ffi::LlamaContext::new(&self.model, &ctx_params)
            .map_err(|e| LociError::InferenceError(e))?;

        self.context = new_context;
        Ok(())
    }

    /// Sample token using stateless sampler with plugin hooks
    ///
    /// This is the core sampling function that integrates:
    /// - Zero-copy logits access
    /// - Plugin logits transformation
    /// - Stateless sampling
    /// - Plugin post-sample hooks
    fn sample_with_plugins(
        &mut self,
        logits_ptr: *mut f32,
        n_vocab: usize,
        sampling_params: &SamplingParams,
        context_tokens: &[i32],
        plugin_manager: Option<&crate::plugin::PluginManager>,
    ) -> Result<i32> {
        // Create zero-copy logits view
        let mut logits_view = unsafe { LogitsView::from_raw(logits_ptr, n_vocab) };

        // Apply plugin logits transformations
        if let Some(pm) = plugin_manager {
            pm.apply_transform_logits(&mut logits_view, context_tokens)?;
        }

        // Sample using stateless sampler
        let token_id = sample_token(&logits_view, sampling_params, context_tokens);

        // Apply plugin post-sample hooks
        let final_token = if let Some(pm) = plugin_manager {
            pm.apply_post_sample(token_id)?
        } else {
            token_id
        };

        Ok(final_token)
    }
}

impl Model for LlamaCppModel {
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn infer_text(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        // Recreate context with inference parameters
        self.recreate_context(params)?;

        // Tokenize prompt
        let tokens = self
            .model
            .tokenize(prompt, true, false)
            .map_err(|e| LociError::InferenceError(e))?;

        // Clear context
        self.context.kv_cache_clear();

        // Create batch
        let mut batch = ffi::batch_init(params.n_batch as i32, 0, 1);

        // Add tokens to batch
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            unsafe {
                batch.n_tokens = i as i32 + 1;
                *batch.token.add(i) = token;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                let seq_id_ptr = *batch.seq_id.add(i);
                *seq_id_ptr = 0;
                *batch.logits.add(i) = if is_last { 1 } else { 0 };
            }
        }

        // Decode the batch
        self.context
            .decode(&mut batch)
            .map_err(|e| LociError::InferenceError(e))?;

        // Convert InferenceParams to SamplingParams
        let sampling_params = SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            repeat_penalty: params.repeat_penalty,
            seed: 0,
        };

        // Generate tokens
        let mut result = String::new();
        let n_vocab = self.model.n_vocab() as usize;
        let mut n_cur = tokens.len();
        let mut context_tokens = tokens.clone();

        for _ in 0..params.max_tokens {
            // Get logits (zero-copy pointer)
            let logits_ptr = self.context.get_logits_ith(batch.n_tokens - 1);

            // Sample using new stateless sampler (no plugin hooks in direct mode)
            let new_token = {
                let logits_view = unsafe { LogitsView::from_raw(logits_ptr, n_vocab) };
                sample_token(&logits_view, &sampling_params, &context_tokens)
            };

            // Check for EOS
            if self.model.is_eog(new_token) {
                break;
            }

            // Convert token to string
            let token_str = self
                .model
                .token_to_str(new_token)
                .map_err(|e| LociError::InferenceError(e))?;

            result.push_str(&token_str);

            // Update context for repetition penalty
            context_tokens.push(new_token);
            if context_tokens.len() > 64 {
                context_tokens.remove(0);
            }

            // Prepare next batch
            unsafe {
                batch.n_tokens = 1;
                *batch.token = new_token;
                *batch.pos = n_cur as i32;
                *batch.n_seq_id = 1;
                let seq_id_ptr = *batch.seq_id;
                *seq_id_ptr = 0;
                *batch.logits = 1;
            }

            n_cur += 1;

            // Decode
            self.context
                .decode(&mut batch)
                .map_err(|e| LociError::InferenceError(e))?;
        }

        // Free batch
        ffi::batch_free(batch);

        Ok(result)
    }
}

impl crate::backend::ModelExt for LlamaCppModel {
    fn infer_stream<F>(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        // Recreate context with inference parameters
        self.recreate_context(params)?;

        // Tokenize prompt
        let tokens = self
            .model
            .tokenize(prompt, true, false)
            .map_err(|e| LociError::InferenceError(e))?;

        // Clear context
        self.context.kv_cache_clear();

        // Create batch
        let mut batch = ffi::batch_init(params.n_batch as i32, 0, 1);

        // Add tokens to batch
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            unsafe {
                batch.n_tokens = i as i32 + 1;
                *batch.token.add(i) = token;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                let seq_id_ptr = *batch.seq_id.add(i);
                *seq_id_ptr = 0;
                *batch.logits.add(i) = if is_last { 1 } else { 0 };
            }
        }

        // Decode the batch
        self.context
            .decode(&mut batch)
            .map_err(|e| LociError::InferenceError(e))?;

        // Convert InferenceParams to SamplingParams
        let sampling_params = SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            repeat_penalty: params.repeat_penalty,
            seed: 0,
        };

        // Generate tokens
        let n_vocab = self.model.n_vocab() as usize;
        let mut n_cur = tokens.len();
        let mut context_tokens = tokens.clone();

        for _ in 0..params.max_tokens {
            // Get logits (zero-copy pointer)
            let logits_ptr = self.context.get_logits_ith(batch.n_tokens - 1);

            // Sample using new stateless sampler
            let new_token = {
                let logits_view = unsafe { LogitsView::from_raw(logits_ptr, n_vocab) };
                sample_token(&logits_view, &sampling_params, &context_tokens)
            };

            // Check for EOS
            if self.model.is_eog(new_token) {
                break;
            }

            // Convert token to string
            let token_str = self
                .model
                .token_to_str(new_token)
                .map_err(|e| LociError::InferenceError(e))?;

            // Call callback
            if !callback(&token_str) {
                break;
            }

            // Update context for repetition penalty
            context_tokens.push(new_token);
            if context_tokens.len() > 64 {
                context_tokens.remove(0);
            }

            // Prepare next batch
            unsafe {
                batch.n_tokens = 1;
                *batch.token = new_token;
                *batch.pos = n_cur as i32;
                *batch.n_seq_id = 1;
                let seq_id_ptr = *batch.seq_id;
                *seq_id_ptr = 0;
                *batch.logits = 1;
            }

            n_cur += 1;

            // Decode
            self.context
                .decode(&mut batch)
                .map_err(|e| LociError::InferenceError(e))?;
        }

        // Free batch
        ffi::batch_free(batch);

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
            "Multimodal streaming not yet supported".to_string(),
        ))
    }
}

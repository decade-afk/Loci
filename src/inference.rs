//! Core inference engine using native llama.cpp

use crate::error::{LociError, Result};
use crate::ffi;
use crate::model::ModelConfig;
use crate::plugin::PluginManager;
use std::ffi::c_int;

/// Parameters for text generation
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

/// Main inference engine
pub struct InferenceEngine {
    model: ffi::LlamaModel,
    context: ffi::LlamaContext,
    plugin_manager: PluginManager,
}

impl InferenceEngine {
    /// Create a new inference engine with the given model configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        config.validate()?;

        // Initialize backend
        ffi::backend_init();

        // Set up model parameters
        let mut model_params = ffi::model_default_params();
        model_params.n_gpu_layers = config.n_gpu_layers as c_int;

        // Load model
        let model = ffi::LlamaModel::from_file(
            config.model_path.to_str().unwrap(),
            &model_params,
        )
        .map_err(|e| LociError::ModelLoadError(e))?;

        // Set up context parameters
        let mut ctx_params = ffi::context_default_params();
        ctx_params.n_ctx = config.n_ctx;
        ctx_params.n_batch = config.n_batch;

        if let Some(n_threads) = config.n_threads {
            ctx_params.n_threads = n_threads as i32;
        }

        // Create context
        let context = ffi::LlamaContext::new(&model, &ctx_params)
            .map_err(|e| LociError::InferenceError(e))?;

        Ok(Self {
            model,
            context,
            plugin_manager: PluginManager::new(),
        })
    }

    /// Get plugin manager (mutable)
    pub fn plugin_manager_mut(&mut self) -> &mut PluginManager {
        &mut self.plugin_manager
    }

    /// Get plugin manager (immutable)
    pub fn plugin_manager(&self) -> &PluginManager {
        &self.plugin_manager
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        // Apply pre-generate plugins
        let processed_prompt = self
            .plugin_manager
            .apply_pre_generate(prompt)?;

        // Tokenize prompt
        let tokens = self
            .model
            .tokenize(&processed_prompt, true, false)
            .map_err(|e| LociError::InferenceError(e))?;

        // Clear previous context
        self.context.kv_cache_clear();

        // Create batch
        let mut batch = ffi::batch_init(512, 0, 1);

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

        // Generate tokens
        let mut result = String::new();
        let n_vocab = self.model.n_vocab();
        let mut n_cur = tokens.len(); // Track current position

        for _ in 0..params.max_tokens {
            // Get logits
            let logits = self.context.get_logits_ith(batch.n_tokens - 1);

            // Sample token (greedy for now)
            let new_token = if params.temperature == 0.0 {
                self.context.sample_greedy(logits, n_vocab)
            } else {
                self.context.sample_greedy(logits, n_vocab) // TODO: Implement proper sampling
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

        // Apply post-generate plugins
        let final_response = self
            .plugin_manager
            .apply_post_generate(&result)?;

        Ok(final_response)
    }

    /// Generate text with streaming output
    pub fn generate_stream<F>(
        &mut self,
        prompt: &str,
        params: GenerationParams,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool,
    {
        // Apply pre-generate plugins
        let processed_prompt = self
            .plugin_manager
            .apply_pre_generate(prompt)?;

        // Tokenize prompt
        let tokens = self
            .model
            .tokenize(&processed_prompt, true, false)
            .map_err(|e| LociError::InferenceError(e))?;

        // Clear previous context
        self.context.kv_cache_clear();

        // Create batch
        let mut batch = ffi::batch_init(512, 0, 1);

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

        // Generate tokens
        let n_vocab = self.model.n_vocab();
        let mut n_cur = tokens.len();

        for _ in 0..params.max_tokens {
            // Get logits
            let logits = self.context.get_logits_ith(batch.n_tokens - 1);

            // Sample token
            let new_token = if params.temperature == 0.0 {
                self.context.sample_greedy(logits, n_vocab)
            } else {
                self.context.sample_greedy(logits, n_vocab) // TODO: Implement proper sampling
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

            // Apply token plugins
            let processed_token = self
                .plugin_manager
                .apply_on_token(&token_str)?;

            // Call callback
            if !callback(&processed_token) {
                break;
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

            // Decode
            self.context
                .decode(&mut batch)
                .map_err(|e| LociError::InferenceError(e))?;

            n_cur += 1;
        }

        // Free batch
        ffi::batch_free(batch);

        Ok(())
    }

    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            n_vocab: self.model.n_vocab() as u32,
            n_ctx_train: self.model.n_ctx_train() as u32,
            n_embd: self.model.n_embd() as u32,
        }
    }
}

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        // Cleanup is handled by LlamaModel and LlamaContext Drop implementations
        ffi::backend_free();
    }
}

/// Information about the loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Vocabulary size
    pub n_vocab: u32,
    /// Training context size
    pub n_ctx_train: u32,
    /// Embedding dimension
    pub n_embd: u32,
}

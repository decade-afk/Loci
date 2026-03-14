//! llama.cpp backend implementation with stateless sampling
//!
//! This module provides the native llama.cpp backend, refactored to use
//! the new stateless sampling system with zero-copy logits manipulation.

use crate::backend::{
    BackendCapabilities, BackendParams, GpuSplitMode, InferenceBackend, InferenceParams, Model,
    ModelMetadata,
};
use crate::error::{LociError, Result};
use crate::ffi;
use crate::sampler::{sample_token, LogitsView, SamplingParams};
use std::path::Path;
use std::sync::OnceLock;

const DEFAULT_MAX_PROMPT_BYTES: usize = 24 * 1024;
const TOKENIZE_CHUNK_BYTES: usize = 4096;
const GENERATION_HEADROOM_MIN_TOKENS: usize = 8;
const GENERATION_HEADROOM_MAX_TOKENS: usize = 64;

/// llama.cpp backend implementation
pub struct LlamaCppBackend {
    initialized: bool,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend
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
            supports_embeddings: true,
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
    // Drop order is significant: context must be released before model.
    context: ffi::LlamaContext,
    model: ffi::LlamaModel,
    metadata: ModelMetadata,
    current_n_ctx: u32,
    current_n_batch: u32,
    current_n_threads: Option<u32>,
    kv_offload: bool,
    op_offload: bool,
}

// Safety: llama.cpp's context is thread-safe for single-threaded access
unsafe impl Send for LlamaCppModel {}
unsafe impl Sync for LlamaCppModel {}

impl LlamaCppModel {
    fn max_prompt_bytes() -> usize {
        static MAX_PROMPT_BYTES: OnceLock<usize> = OnceLock::new();
        *MAX_PROMPT_BYTES.get_or_init(|| {
            std::env::var("LOCI_MAX_PROMPT_BYTES")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| v >= 1024)
                .unwrap_or(DEFAULT_MAX_PROMPT_BYTES)
        })
    }

    fn generation_headroom(max_tokens: u32) -> usize {
        let requested = usize::try_from(max_tokens).unwrap_or(GENERATION_HEADROOM_MAX_TOKENS);
        requested.clamp(
            GENERATION_HEADROOM_MIN_TOKENS,
            GENERATION_HEADROOM_MAX_TOKENS,
        )
    }

    fn max_decode_chunk(&self) -> usize {
        self.current_n_batch.max(1) as usize
    }

    fn enforce_context_window(
        &self,
        tokens: &mut Vec<i32>,
        n_ctx: u32,
        reserve_tokens: usize,
    ) -> Result<()> {
        let n_ctx_usize = n_ctx as usize;
        if n_ctx_usize <= reserve_tokens {
            return Err(LociError::InferenceError(format!(
                "n_ctx is too small for inference (requires > {})",
                reserve_tokens
            )));
        }

        let limit = n_ctx_usize - reserve_tokens;
        if tokens.len() > limit {
            let drop_count = tokens.len() - limit;
            tokens.drain(0..drop_count);
        }

        Ok(())
    }

    /// Load a model from file
    pub fn load(model_path: &Path, params: BackendParams) -> Result<Self> {
        let BackendParams {
            n_gpu_layers: requested_n_gpu_layers,
            use_gpu,
            use_mmap,
            use_mlock,
            kv_offload: requested_kv_offload,
            op_offload: requested_op_offload,
            split_mode: requested_split_mode,
            main_gpu,
            tensor_split: requested_tensor_split,
            options,
        } = params;

        let n_gpu_layers = if use_gpu { requested_n_gpu_layers } else { 0 };
        let kv_offload = if use_gpu { requested_kv_offload } else { false };
        let op_offload = if use_gpu { requested_op_offload } else { false };
        let split_mode = if use_gpu {
            requested_split_mode
        } else {
            GpuSplitMode::None
        };
        let tensor_split = if use_gpu && split_mode != GpuSplitMode::None {
            requested_tensor_split
        } else {
            None
        };

        // Set up model parameters
        let mut model_params = ffi::model_default_params();
        model_params.n_gpu_layers = n_gpu_layers as i32;
        model_params.split_mode = match split_mode {
            GpuSplitMode::None => ffi::llama_split_mode_LLAMA_SPLIT_MODE_NONE,
            GpuSplitMode::Layer => ffi::llama_split_mode_LLAMA_SPLIT_MODE_LAYER,
            GpuSplitMode::Row => ffi::llama_split_mode_LLAMA_SPLIT_MODE_ROW,
        };
        model_params.main_gpu = main_gpu as i32;
        model_params.tensor_split = tensor_split
            .as_ref()
            .map_or(std::ptr::null(), |values| values.as_ptr());
        model_params.use_mmap = use_mmap;
        model_params.use_mlock = use_mlock;

        // Load model
        let model_path_str = model_path
            .to_str()
            .ok_or_else(|| LociError::ConfigError("Invalid model path".to_string()))?;

        let model = ffi::LlamaModel::from_file(model_path_str, &model_params)
            .map_err(|e| LociError::ModelLoadError(e))?;

        if !model.has_decoder() {
            return Err(LociError::ModelLoadError(
                "Model does not expose a decoder path required for text generation".to_string(),
            ));
        }

        // Extract metadata
        let metadata = ModelMetadata {
            architecture: "llama".to_string(),
            n_vocab: model.n_vocab() as u32,
            n_ctx_train: model.n_ctx_train() as u32,
            n_embd: model.n_embd() as u32,
            n_layer: 0,
            param_count: None,
        };

        // Create context using backend options when provided
        let n_ctx = options
            .iter()
            .find(|(k, _)| k == "n_ctx")
            .and_then(|(_, v)| v.parse::<u32>().ok())
            .unwrap_or(4096);
        let n_batch = options
            .iter()
            .find(|(k, _)| k == "n_batch")
            .and_then(|(_, v)| v.parse::<u32>().ok())
            .unwrap_or(512);
        let n_threads = options
            .iter()
            .find(|(k, _)| k == "n_threads")
            .and_then(|(_, v)| v.parse::<u32>().ok());

        let mut ctx_params = ffi::context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_batch;
        ctx_params.offload_kqv = kv_offload;
        ctx_params.op_offload = op_offload;
        ctx_params.flash_attn_type = ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if let Some(n_threads) = n_threads {
            ctx_params.n_threads = n_threads as i32;
        }

        let context = ffi::LlamaContext::new(&model, &ctx_params)
            .map_err(|e| LociError::InferenceError(e))?;

        Ok(Self {
            model,
            context,
            metadata,
            current_n_ctx: n_ctx,
            current_n_batch: n_batch,
            current_n_threads: n_threads,
            kv_offload,
            op_offload,
        })
    }

    /// Recreate context with new parameters
    fn recreate_context(&mut self, params: &InferenceParams) -> Result<()> {
        if self.current_n_ctx == params.n_ctx
            && self.current_n_batch == params.n_batch
            && self.current_n_threads == params.n_threads
        {
            return Ok(());
        }

        let mut ctx_params = ffi::context_default_params();
        ctx_params.n_ctx = params.n_ctx;
        ctx_params.n_batch = params.n_batch;
        ctx_params.offload_kqv = self.kv_offload;
        ctx_params.op_offload = self.op_offload;
        ctx_params.flash_attn_type = ffi::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED;

        if let Some(n_threads) = params.n_threads {
            ctx_params.n_threads = n_threads as i32;
        }

        // Drop old context and create new one
        let new_context = ffi::LlamaContext::new(&self.model, &ctx_params)
            .map_err(|e| LociError::InferenceError(e))?;

        self.context = new_context;
        self.current_n_ctx = params.n_ctx;
        self.current_n_batch = params.n_batch;
        self.current_n_threads = params.n_threads;
        Ok(())
    }

    /// Sample token using stateless sampler with plugin hooks
    ///
    /// This is the core sampling function that integrates:
    /// - Zero-copy logits access
    /// - Plugin logits transformation
    /// - Stateless sampling
    /// - Plugin post-sample hooks
    #[allow(dead_code)]
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

    /// Decode tokens with explicit positions and request logits on last token.
    fn decode_tokens(&mut self, tokens: &[i32], pos_start: i32) -> Result<()> {
        if tokens.is_empty() {
            return Err(LociError::InferenceError(
                "Cannot decode an empty token batch".to_string(),
            ));
        }

        let n_tokens = tokens.len();
        if n_tokens > self.max_decode_chunk() {
            return Err(LociError::InferenceError(format!(
                "Decode batch too large: {} > n_batch({})",
                n_tokens,
                self.max_decode_chunk()
            )));
        }

        let n_tokens_i32 = i32::try_from(n_tokens).map_err(|_| {
            LociError::InferenceError("Token batch length exceeds i32 range".to_string())
        })?;

        let _end_pos = pos_start
            .checked_add(n_tokens_i32 - 1)
            .ok_or_else(|| LociError::InferenceError("Token position overflow".to_string()))?;

        let mut batch =
            ffi::OwnedBatch::new(n_tokens_i32, 0, 1).map_err(LociError::InferenceError)?;
        let batch_ref = batch.as_mut();

        unsafe {
            for (i, &token) in tokens.iter().enumerate() {
                let i_i32 = i32::try_from(i).map_err(|_| {
                    LociError::InferenceError("Token position overflow".to_string())
                })?;
                *batch_ref.token.add(i) = token;
                *batch_ref.pos.add(i) = pos_start + i_i32;
                *batch_ref.n_seq_id.add(i) = 1;
                let seq_id_ptr = *batch_ref.seq_id.add(i);
                if seq_id_ptr.is_null() {
                    return Err(LociError::InferenceError(
                        "llama batch seq_id buffer is null".to_string(),
                    ));
                }
                *seq_id_ptr = 0;
                *batch_ref.logits.add(i) = if i + 1 == n_tokens { 1 } else { 0 };
            }
            batch_ref.n_tokens = n_tokens_i32;
        }

        self.context
            .decode(batch_ref)
            .map_err(LociError::InferenceError)
    }

    /// Decode potentially long token sequences in n_batch-sized chunks.
    fn decode_tokens_chunked(&mut self, tokens: &[i32], pos_start: i32) -> Result<()> {
        if tokens.is_empty() {
            return Err(LociError::InferenceError(
                "Cannot decode an empty token batch".to_string(),
            ));
        }

        let chunk_size = self.max_decode_chunk();
        let mut processed: usize = 0;

        while processed < tokens.len() {
            let end = (processed + chunk_size).min(tokens.len());
            let chunk = &tokens[processed..end];
            let processed_i32 = i32::try_from(processed)
                .map_err(|_| LociError::InferenceError("Token position overflow".to_string()))?;
            let chunk_pos = pos_start
                .checked_add(processed_i32)
                .ok_or_else(|| LociError::InferenceError("Token position overflow".to_string()))?;
            self.decode_tokens(chunk, chunk_pos)?;
            processed = end;
        }

        Ok(())
    }

    /// Split a UTF-8 string into chunk-size slices without breaking codepoint boundaries.
    fn split_utf8_chunks<'a>(text: &'a str, chunk_bytes: usize) -> Vec<&'a str> {
        if text.is_empty() {
            return vec![];
        }
        if chunk_bytes == 0 || text.len() <= chunk_bytes {
            return vec![text];
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < text.len() {
            let mut end = (start + chunk_bytes).min(text.len());
            while end > start && !text.is_char_boundary(end) {
                end -= 1;
            }
            if end == start {
                break;
            }
            chunks.push(&text[start..end]);
            start = end;
        }

        if chunks.is_empty() {
            vec![text]
        } else {
            chunks
        }
    }

    /// Tokenize with chunked fallback to avoid tokenizer stack overflow on very long prompts.
    fn tokenize_stable(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<i32>> {
        if text.len() <= TOKENIZE_CHUNK_BYTES {
            return self
                .model
                .tokenize(text, add_bos, special)
                .map_err(LociError::InferenceError);
        }

        let chunks = Self::split_utf8_chunks(text, TOKENIZE_CHUNK_BYTES);
        let mut all_tokens = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            let mut tokens = self
                .model
                .tokenize(chunk, add_bos && idx == 0, special)
                .map_err(LociError::InferenceError)?;
            all_tokens.append(&mut tokens);
        }

        Ok(all_tokens)
    }
}

impl Model for LlamaCppModel {
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn generate_embeddings(&mut self, text: &str) -> Result<Vec<f32>> {
        let max_prompt_bytes = Self::max_prompt_bytes();
        if text.len() > max_prompt_bytes {
            return Err(LociError::InferenceError(format!(
                "Input text is too large ({} bytes > {} bytes limit)",
                text.len(),
                max_prompt_bytes
            )));
        }

        // Tokenize the input text
        let mut tokens = self.tokenize_stable(text, true, false)?;

        if tokens.is_empty() {
            return Err(LociError::InferenceError("Empty input text".to_string()));
        }

        self.enforce_context_window(&mut tokens, self.current_n_ctx, 0)?;

        self.context.kv_cache_clear();
        self.decode_tokens_chunked(&tokens, 0)?;

        // Get embeddings for the last token
        self.context
            .get_embeddings()
            .map_err(LociError::InferenceError)
    }

    fn infer_text(&mut self, prompt: &str, params: &InferenceParams) -> Result<String> {
        let max_prompt_bytes = Self::max_prompt_bytes();
        if prompt.len() > max_prompt_bytes {
            return Err(LociError::InferenceError(format!(
                "Prompt is too large ({} bytes > {} bytes limit)",
                prompt.len(),
                max_prompt_bytes
            )));
        }

        let trace_enabled = std::env::var("LOCI_TRACE").ok().as_deref() == Some("1");
        if trace_enabled {
            eprintln!(
                "[infer_text] start n_ctx={} n_batch={} max_tokens={}",
                params.n_ctx, params.n_batch, params.max_tokens
            );
        }

        // Recreate context with inference parameters
        self.recreate_context(params)?;
        if trace_enabled {
            eprintln!("[infer_text] context ready");
        }

        // Tokenize prompt
        let mut tokens = self.tokenize_stable(prompt, false, false)?;
        let reserve = Self::generation_headroom(params.max_tokens);
        self.enforce_context_window(&mut tokens, params.n_ctx, reserve)?;
        if trace_enabled {
            eprintln!(
                "[infer_text] prompt tokens={} values={:?}",
                tokens.len(),
                tokens
            );
        }

        // Clear context
        self.context.kv_cache_clear();

        if trace_enabled {
            eprintln!("[infer_text] before prompt decode");
        }
        self.decode_tokens_chunked(&tokens, 0)?;
        if trace_enabled {
            eprintln!("[infer_text] prompt decode ok");
        }

        // Convert InferenceParams to SamplingParams
        let sampling_params = SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            repeat_penalty: params.repeat_penalty,
            seed: 0,
        };

        // Generate tokens
        let mut result = String::new();
        let n_vocab = self.model.n_vocab() as usize;
        let mut n_cur = i32::try_from(tokens.len())
            .map_err(|_| LociError::InferenceError("Token count exceeds i32 range".to_string()))?;
        let mut context_tokens = tokens.clone();
        let last_logits_idx = -1;

        let n_ctx_usize = params.n_ctx as usize;
        let mut remaining_decode_slots = n_ctx_usize.saturating_sub(tokens.len());
        let max_steps = usize::try_from(params.max_tokens).unwrap_or(usize::MAX);

        for _ in 0..max_steps {
            // Get logits (zero-copy pointer)
            if trace_enabled {
                eprintln!("[infer_text] step get_logits");
            }
            let logits_ptr = self.context.get_logits_ith(last_logits_idx);
            if logits_ptr.is_null() {
                return Err(LociError::InferenceError(
                    "Failed to get logits from llama context".to_string(),
                ));
            }

            // Sample using new stateless sampler (no plugin hooks in direct mode)
            let new_token = {
                let logits_view = unsafe { LogitsView::from_raw(logits_ptr, n_vocab) };
                sample_token(&logits_view, &sampling_params, &context_tokens)
            };
            if trace_enabled {
                eprintln!("[infer_text] sampled token={}", new_token);
            }

            // Check for EOS
            if self.model.is_eog(new_token) {
                break;
            }

            // Convert token to string
            let token_str = self
                .model
                .token_to_str(new_token)
                .map_err(|e| LociError::InferenceError(e))?;
            if trace_enabled {
                eprintln!("[infer_text] token_to_str len={}", token_str.len());
            }

            result.push_str(&token_str);

            // Update context for repetition penalty
            context_tokens.push(new_token);
            if context_tokens.len() > 64 {
                context_tokens.remove(0);
            }

            // Prepare next batch
            if remaining_decode_slots == 0 {
                break;
            }
            let next_token = [new_token];
            self.decode_tokens(&next_token, n_cur)?;
            n_cur = n_cur
                .checked_add(1)
                .ok_or_else(|| LociError::InferenceError("Token position overflow".to_string()))?;
            remaining_decode_slots -= 1;
            if trace_enabled {
                eprintln!("[infer_text] step decode ok");
            }
        }

        Ok(result)
    }

    fn infer_stream(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        crate::backend::ModelExt::infer_stream(self, prompt, params, |token| callback(token))
    }

    fn infer_multimodal_stream(
        &mut self,
        text: &str,
        images: &[crate::backend::Image],
        params: &InferenceParams,
        callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        crate::backend::ModelExt::infer_multimodal_stream(self, text, images, params, |token| {
            callback(token)
        })
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
        let max_prompt_bytes = Self::max_prompt_bytes();
        if prompt.len() > max_prompt_bytes {
            return Err(LociError::InferenceError(format!(
                "Prompt is too large ({} bytes > {} bytes limit)",
                prompt.len(),
                max_prompt_bytes
            )));
        }

        // Recreate context with inference parameters
        self.recreate_context(params)?;

        // Tokenize prompt
        let mut tokens = self.tokenize_stable(prompt, false, false)?;
        let reserve = Self::generation_headroom(params.max_tokens);
        self.enforce_context_window(&mut tokens, params.n_ctx, reserve)?;

        // Clear context
        self.context.kv_cache_clear();

        self.decode_tokens_chunked(&tokens, 0)?;

        // Convert InferenceParams to SamplingParams
        let sampling_params = SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            min_p: params.min_p,
            repeat_penalty: params.repeat_penalty,
            seed: 0,
        };

        // Generate tokens
        let n_vocab = self.model.n_vocab() as usize;
        let mut n_cur = i32::try_from(tokens.len())
            .map_err(|_| LociError::InferenceError("Token count exceeds i32 range".to_string()))?;
        let mut context_tokens = tokens.clone();
        let last_logits_idx = -1;

        let n_ctx_usize = params.n_ctx as usize;
        let mut remaining_decode_slots = n_ctx_usize.saturating_sub(tokens.len());
        let max_steps = usize::try_from(params.max_tokens).unwrap_or(usize::MAX);

        for _ in 0..max_steps {
            // Get logits (zero-copy pointer)
            let logits_ptr = self.context.get_logits_ith(last_logits_idx);
            if logits_ptr.is_null() {
                return Err(LociError::InferenceError(
                    "Failed to get logits from llama context".to_string(),
                ));
            }

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
            if remaining_decode_slots == 0 {
                break;
            }
            let next_token = [new_token];
            self.decode_tokens(&next_token, n_cur)?;
            n_cur = n_cur
                .checked_add(1)
                .ok_or_else(|| LociError::InferenceError("Token position overflow".to_string()))?;
            remaining_decode_slots -= 1;
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
            "Multimodal streaming not yet supported".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaCppModel, TOKENIZE_CHUNK_BYTES};

    #[test]
    fn split_utf8_chunks_preserves_data_for_multibyte_text() {
        let text = "深度审查与修复内存安全。".repeat(900);
        let chunks = LlamaCppModel::split_utf8_chunks(&text, TOKENIZE_CHUNK_BYTES);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.len() <= TOKENIZE_CHUNK_BYTES));
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn split_utf8_chunks_handles_empty_input() {
        let chunks = LlamaCppModel::split_utf8_chunks("", TOKENIZE_CHUNK_BYTES);
        assert!(chunks.is_empty());
    }
}

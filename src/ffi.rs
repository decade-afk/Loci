//! FFI bindings to llama.cpp

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CString;

/// Safe wrapper for llama_model
pub struct LlamaModel {
    ptr: *mut llama_model,
}

impl LlamaModel {
    /// Load a model from file
    pub fn from_file(path: &str, params: &llama_model_params) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;

        let ptr = unsafe { llama_model_load_from_file(c_path.as_ptr(), *params) };

        if ptr.is_null() {
            return Err(format!("Failed to load model from {}", path));
        }

        Ok(Self { ptr })
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut llama_model {
        self.ptr
    }

    /// Get vocabulary
    fn get_vocab(&self) -> *const llama_vocab {
        unsafe { llama_model_get_vocab(self.ptr) }
    }

    /// Get vocabulary size
    pub fn n_vocab(&self) -> i32 {
        unsafe { llama_n_vocab(self.get_vocab()) }
    }

    /// Get training context size
    pub fn n_ctx_train(&self) -> i32 {
        unsafe { llama_model_n_ctx_train(self.ptr) }
    }

    /// Get embedding dimension
    pub fn n_embd(&self) -> i32 {
        unsafe { llama_model_n_embd(self.ptr) }
    }

    /// Tokenize text
    pub fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<i32>, String> {
        let c_text = CString::new(text).map_err(|e| e.to_string())?;
        let text_len = text.len() as i32;

        // Allocate buffer for tokens
        let mut tokens = vec![0i32; (text_len + 16) as usize];

        let n_tokens = unsafe {
            llama_tokenize(
                self.get_vocab(),
                c_text.as_ptr(),
                text_len,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_bos,
                special,
            )
        };

        if n_tokens < 0 {
            return Err("Tokenization failed".to_string());
        }

        tokens.truncate(n_tokens as usize);
        Ok(tokens)
    }

    /// Convert token to string
    pub fn token_to_str(&self, token: i32) -> Result<String, String> {
        let mut buffer = vec![0u8; 32];
        let n_chars = unsafe {
            llama_detokenize(
                self.get_vocab(),
                &token,
                1,
                buffer.as_mut_ptr() as *mut i8,
                buffer.len() as i32,
                false,  // remove_special
                false,  // unparse_special
            )
        };

        if n_chars < 0 {
            return Err("Failed to convert token".to_string());
        }

        buffer.truncate(n_chars as usize);
        String::from_utf8(buffer).map_err(|e| e.to_string())
    }

    /// Check if token is end-of-generation
    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { llama_token_is_eog(self.get_vocab(), token) }
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                llama_model_free(self.ptr);
            }
        }
    }
}

unsafe impl Send for LlamaModel {}

/// Safe wrapper for llama_context
pub struct LlamaContext {
    ptr: *mut llama_context,
}

impl LlamaContext {
    /// Create a new context
    pub fn new(model: &LlamaModel, params: &llama_context_params) -> Result<Self, String> {
        let ptr = unsafe { llama_init_from_model(model.as_ptr(), *params) };

        if ptr.is_null() {
            return Err("Failed to create context".to_string());
        }

        Ok(Self { ptr })
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut llama_context {
        self.ptr
    }

    /// Decode a batch
    pub fn decode(&mut self, batch: &mut llama_batch) -> Result<(), String> {
        let ret = unsafe { llama_decode(self.ptr, *batch) };

        if ret != 0 {
            return Err(format!("Decode failed with code {}", ret));
        }

        Ok(())
    }

    /// Get logits for a specific token position
    pub fn get_logits_ith(&self, i: i32) -> *mut f32 {
        unsafe { llama_get_logits_ith(self.ptr, i) }
    }

    /// Sample a token using greedy sampling
    pub fn sample_greedy(&self, logits: *const f32, n_vocab: i32) -> i32 {
        unsafe {
            let mut max_idx = 0;
            let mut max_val = *logits;

            for i in 1..n_vocab {
                let val = *logits.offset(i as isize);
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            max_idx
        }
    }

    /// Clear the KV cache (no-op if not available in new API)
    pub fn kv_cache_clear(&mut self) {
        // Note: KV cache API may have changed in new llama.cpp
        // This is a placeholder that does nothing for now
        // TODO: Find the correct API for clearing KV cache in new version
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                llama_free(self.ptr);
            }
        }
    }
}

unsafe impl Send for LlamaContext {}

/// Initialize llama backend
pub fn backend_init() {
    unsafe {
        llama_backend_init();
        llama_numa_init(ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED);
    }
}

/// Free llama backend
pub fn backend_free() {
    unsafe {
        llama_backend_free();
    }
}

/// Get default model params
pub fn model_default_params() -> llama_model_params {
    unsafe { llama_model_default_params() }
}

/// Get default context params
pub fn context_default_params() -> llama_context_params {
    unsafe { llama_context_default_params() }
}

/// Create a new batch
pub fn batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch {
    unsafe { llama_batch_init(n_tokens, embd, n_seq_max) }
}

/// Free a batch
pub fn batch_free(batch: llama_batch) {
    unsafe { llama_batch_free(batch) }
}

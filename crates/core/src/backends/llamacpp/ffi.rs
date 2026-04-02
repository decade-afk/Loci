//! FFI bindings to llama.cpp

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

// Keep bindgen-generated items isolated so clippy can focus on hand-written code.
#[allow(clippy::all)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/llama_bindings.rs"));
}

pub use bindings::*;

use std::ffi::{c_char, CString};
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::{Mutex, OnceLock};

unsafe extern "C" {
    fn loci_llama_model_default_params(out_params: *mut llama_model_params);
    fn loci_llama_context_default_params(out_params: *mut llama_context_params);
    fn loci_llama_model_load_from_file(
        path: *const c_char,
        params: *const llama_model_params,
    ) -> *mut llama_model;
    fn loci_llama_init_from_model(
        model: *mut llama_model,
        params: *const llama_context_params,
    ) -> *mut llama_context;
    fn loci_llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32, out_batch: *mut llama_batch);
    fn loci_llama_batch_free(batch: *const llama_batch);
    fn loci_llama_encode(ctx: *mut llama_context, batch: *const llama_batch) -> i32;
    fn loci_llama_decode(ctx: *mut llama_context, batch: *const llama_batch) -> i32;
    fn loci_llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
}

/// Safe wrapper for llama_model

pub struct LlamaBackendHandle;

impl LlamaBackendHandle {
    pub fn acquire() -> Self {
        backend_init();
        Self
    }
}

impl Drop for LlamaBackendHandle {
    fn drop(&mut self) {
        backend_free();
    }
}

pub struct LlamaModel {
    ptr: *mut llama_model,
}

impl LlamaModel {
    /// Load a model from file
    pub fn from_file(path: &str, params: &llama_model_params) -> Result<Self, String> {
        let c_path = CString::new(path).map_err(|e| e.to_string())?;

        let ptr = unsafe { loci_llama_model_load_from_file(c_path.as_ptr(), params as *const _) };

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
        unsafe { llama_n_embd(self.ptr) }
    }

    /// Check whether model has an encoder
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.ptr) }
    }

    /// Check whether model has a decoder
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.ptr) }
    }

    /// Tokenize text
    pub fn tokenize(&self, text: &str, add_bos: bool, special: bool) -> Result<Vec<i32>, String> {
        self.tokenize_bytes(text.as_bytes(), add_bos, special)
    }

    /// Tokenize bytes with explicit length (binary-safe, supports interior NUL).
    pub fn tokenize_bytes(
        &self,
        text: &[u8],
        add_bos: bool,
        special: bool,
    ) -> Result<Vec<i32>, String> {
        let text_len = i32::try_from(text.len())
            .map_err(|_| "Input text exceeds i32 length limit".to_string())?;

        let mut capacity = (text_len.max(1) as usize) + 16;
        let mut tokens = vec![0i32; capacity];
        let text_ptr = text.as_ptr().cast::<c_char>();

        loop {
            let n_tokens = unsafe {
                loci_llama_tokenize(
                    self.get_vocab(),
                    text_ptr,
                    text_len,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    add_bos,
                    special,
                )
            };

            if n_tokens >= 0 {
                tokens.truncate(n_tokens as usize);
                return Ok(tokens);
            }

            let required = (-n_tokens) as usize;
            if required <= tokens.len() {
                return Err("Tokenization failed".to_string());
            }

            capacity = required;
            tokens.resize(capacity, 0);
        }
    }

    /// Convert token to string
    pub fn token_to_str(&self, token: i32) -> Result<String, String> {
        let mut buffer = vec![0u8; 32];

        loop {
            let n_chars = unsafe {
                llama_detokenize(
                    self.get_vocab(),
                    &token,
                    1,
                    buffer.as_mut_ptr().cast::<c_char>(),
                    buffer.len() as i32,
                    false, // remove_special
                    false, // unparse_special
                )
            };

            if n_chars >= 0 {
                buffer.truncate(n_chars as usize);
                return String::from_utf8(buffer).map_err(|e| e.to_string());
            }

            let required = (-n_chars) as usize;
            if required <= buffer.len() {
                return Err("Failed to convert token".to_string());
            }

            buffer.resize(required, 0);
        }
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
unsafe impl Sync for LlamaModel {}

/// Safe wrapper for llama_context
pub struct LlamaContext {
    ptr: *mut llama_context,
    n_embd: usize,
}

impl LlamaContext {
    /// Create a new context
    pub fn new(model: &LlamaModel, params: &llama_context_params) -> Result<Self, String> {
        let ptr = unsafe { loci_llama_init_from_model(model.as_ptr(), params as *const _) };

        if ptr.is_null() {
            return Err("Failed to create context".to_string());
        }

        Ok(Self {
            ptr,
            n_embd: model.n_embd() as usize,
        })
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *mut llama_context {
        self.ptr
    }

    /// Decode a batch
    pub fn decode(&mut self, batch: &mut llama_batch) -> Result<(), String> {
        let ret = unsafe { loci_llama_decode(self.ptr, batch as *const llama_batch) };

        if ret != 0 {
            return Err(format!("Decode failed with code {}", ret));
        }

        Ok(())
    }

    /// Get logits for a specific token position
    pub fn get_logits_ith(&self, i: i32) -> *mut f32 {
        unsafe { llama_get_logits_ith(self.ptr, i) }
    }

    /// Get embeddings for last position
    pub fn get_embeddings(&self) -> Result<Vec<f32>, String> {
        unsafe {
            let embd_ptr = llama_get_embeddings(self.ptr);
            if embd_ptr.is_null() {
                return Err(
                    "Embeddings not available. Make sure embeddings are enabled in model params."
                        .to_string(),
                );
            }

            let embeddings = std::slice::from_raw_parts(embd_ptr, self.n_embd).to_vec();
            Ok(embeddings)
        }
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

    pub fn kv_cache_clear(&mut self) {
        unsafe {
            let mem = llama_get_memory(self.ptr as *const llama_context);
            if !mem.is_null() {
                // Clear KV state and underlying cache data to ensure clean generation boundaries.
                llama_memory_clear(mem, true);
            }
        }
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
unsafe impl Sync for LlamaContext {}

/// Initialize llama backend
pub fn backend_init() {
    let mut refcount = backend_refcount()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if *refcount == 0 {
        unsafe {
            llama_backend_init();
            llama_numa_init(ggml_numa_strategy_GGML_NUMA_STRATEGY_DISABLED);
        }
    }
    *refcount += 1;
}

/// Free llama backend
pub fn backend_free() {
    let mut refcount = backend_refcount()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if *refcount == 0 {
        return;
    }

    *refcount -= 1;
    if *refcount == 0 {
        unsafe {
            llama_backend_free();
        }
    }
}

fn backend_refcount() -> &'static Mutex<usize> {
    static BACKEND_REFCOUNT: OnceLock<Mutex<usize>> = OnceLock::new();
    BACKEND_REFCOUNT.get_or_init(|| Mutex::new(0))
}

/// Get default model params
pub fn model_default_params() -> llama_model_params {
    let mut params = MaybeUninit::<llama_model_params>::uninit();
    unsafe {
        loci_llama_model_default_params(params.as_mut_ptr());
        params.assume_init()
    }
}

/// Get default context params
pub fn context_default_params() -> llama_context_params {
    let mut params = MaybeUninit::<llama_context_params>::uninit();
    unsafe {
        loci_llama_context_default_params(params.as_mut_ptr());
        params.assume_init()
    }
}

/// Create a new batch
pub fn batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch {
    let mut batch = MaybeUninit::<llama_batch>::uninit();
    unsafe {
        loci_llama_batch_init(n_tokens, embd, n_seq_max, batch.as_mut_ptr());
        batch.assume_init()
    }
}

/// RAII wrapper for llama_batch allocations.
///
/// Keeps batch memory ownership localized and guarantees release on all paths.
pub struct OwnedBatch {
    inner: llama_batch,
}

impl OwnedBatch {
    pub fn new(n_tokens: i32, embd: i32, n_seq_max: i32) -> Result<Self, String> {
        if n_tokens <= 0 {
            return Err("batch size must be positive".to_string());
        }

        let inner = batch_init(n_tokens, embd, n_seq_max);
        let has_payload = !inner.token.is_null() || !inner.embd.is_null();
        if !has_payload
            || inner.pos.is_null()
            || inner.n_seq_id.is_null()
            || inner.seq_id.is_null()
            || inner.logits.is_null()
        {
            batch_free(inner);
            return Err("llama_batch_init returned null buffers".to_string());
        }

        Ok(Self { inner })
    }

    pub fn as_mut(&mut self) -> &mut llama_batch {
        &mut self.inner
    }
}

/// Free a batch
pub fn batch_free(batch: llama_batch) {
    unsafe { loci_llama_batch_free(&batch as *const llama_batch) }
}

impl Drop for OwnedBatch {
    fn drop(&mut self) {
        batch_free(self.inner);
        self.inner = llama_batch {
            n_tokens: 0,
            token: ptr::null_mut(),
            embd: ptr::null_mut(),
            pos: ptr::null_mut(),
            n_seq_id: ptr::null_mut(),
            seq_id: ptr::null_mut(),
            logits: ptr::null_mut(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_batch_rejects_zero_size() {
        assert!(OwnedBatch::new(0, 0, 1).is_err());
    }

    #[test]
    fn owned_batch_allocates_required_buffers() {
        let mut batch = OwnedBatch::new(1, 0, 1).expect("batch should allocate");
        let batch_ref = batch.as_mut();
        assert!(!batch_ref.token.is_null());
        assert!(!batch_ref.pos.is_null());
        assert!(!batch_ref.n_seq_id.is_null());
        assert!(!batch_ref.seq_id.is_null());
        assert!(!batch_ref.logits.is_null());
    }

    #[test]
    fn backend_refcount_is_balanced() {
        // Balance the reference count back to zero before this test starts.
        while {
            let guard = backend_refcount()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *guard > 0
        } {
            backend_free();
        }

        backend_init();
        backend_init();
        {
            let guard = backend_refcount()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert_eq!(*guard, 2);
        }

        backend_free();
        {
            let guard = backend_refcount()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert_eq!(*guard, 1);
        }

        backend_free();
        {
            let guard = backend_refcount()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert_eq!(*guard, 0);
        }
    }
}

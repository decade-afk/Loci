//! C API for using Loci from other languages

use crate::error::Result;
use crate::inference::{GenerationParams, InferenceEngine};
use crate::model::ModelConfig;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;

/// Opaque handle to InferenceEngine
pub struct LociEngine {
    engine: InferenceEngine,
}

/// Create a new inference engine
///
/// # Safety
/// model_path must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn loci_engine_new(
    model_path: *const c_char,
    n_ctx: u32,
    n_gpu_layers: i32,
) -> *mut LociEngine {
    if model_path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let config = ModelConfig {
        model_path: PathBuf::from(path_str),
        n_ctx,
        n_batch: 512,
        n_threads: None,
        n_gpu_layers,
        use_gpu: n_gpu_layers != 0,
    };

    match InferenceEngine::new(config) {
        Ok(engine) => Box::into_raw(Box::new(LociEngine { engine })),
        Err(_) => ptr::null_mut(),
    }
}

/// Generate text from a prompt
///
/// # Safety
/// - engine must be a valid pointer from loci_engine_new
/// - prompt must be a valid null-terminated C string
/// - The returned string must be freed with loci_free_string
#[no_mangle]
pub unsafe extern "C" fn loci_generate(
    engine: *mut LociEngine,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
) -> *mut c_char {
    if engine.is_null() || prompt.is_null() {
        return ptr::null_mut();
    }

    let engine = &mut (*engine).engine;
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let params = GenerationParams {
        max_tokens,
        temperature,
        top_p: 0.95,
        top_k: 40,
        repeat_penalty: 1.1,
    };

    match engine.generate(prompt_str, params) {
        Ok(result) => match CString::new(result) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by loci_generate
///
/// # Safety
/// s must be a valid pointer from loci_generate
#[no_mangle]
pub unsafe extern "C" fn loci_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Destroy an inference engine
///
/// # Safety
/// engine must be a valid pointer from loci_engine_new
#[no_mangle]
pub unsafe extern "C" fn loci_engine_free(engine: *mut LociEngine) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine);
    }
}

/// Get model information
///
/// # Safety
/// engine must be a valid pointer from loci_engine_new
/// Returns vocab size, or 0 on error
#[no_mangle]
pub unsafe extern "C" fn loci_get_vocab_size(engine: *const LociEngine) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &(*engine).engine;
    engine.model_info().n_vocab
}

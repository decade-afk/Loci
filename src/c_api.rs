//! C API for embedding Loci into third-party applications
//!
//! This module provides a complete C-compatible API that allows:
//! - Embedding Loci into C/C++/Python/Node.js applications
//! - Full control over inference parameters
//! - Plugin system integration
//! - Streaming inference support

use crate::inference::{GenerationParams, InferenceEngine};
use crate::model::ModelConfig;
use crate::plugin_registry::PluginRegistry;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;
use std::slice;
use std::str;
use std::sync::{Arc, Mutex, OnceLock, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

/// Opaque handle to InferenceEngine
pub struct LociEngine {
    // Serialized access protects llama context from concurrent FFI calls.
    engine: Mutex<InferenceEngine>,
}

/// Callback function type for streaming inference
pub type LociStreamCallback =
    unsafe extern "C" fn(token: *const c_char, user_data: *mut std::ffi::c_void) -> bool;

const ENGINE_BUSY_MSG: &str = "engine is busy (another inference call is in progress)";
const ENGINE_TIMEOUT_MSG: &str = "engine lock timeout while waiting for in-flight inference";
const ENGINE_POISONED_MSG: &str = "engine lock is poisoned";
const ENGINE_OR_PROMPT_NULL_MSG: &str = "engine or prompt is null";
const ENGINE_LOCK_POLL_MS: u64 = 2;
const ENGINE_FREE_WAIT_MS: u32 = 30_000;
const C_API_DEFAULT_MAX_PROMPT_BYTES: usize = 24 * 1024;

fn c_api_max_prompt_bytes() -> usize {
    static MAX_PROMPT_BYTES: OnceLock<usize> = OnceLock::new();
    *MAX_PROMPT_BYTES.get_or_init(|| {
        std::env::var("LOCI_MAX_PROMPT_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 1024)
            .unwrap_or(C_API_DEFAULT_MAX_PROMPT_BYTES)
    })
}

fn build_generation_params(max_tokens: u32, temperature: f32) -> GenerationParams {
    GenerationParams {
        max_tokens,
        temperature,
        top_p: 0.95,
        min_p: 0.0,
        top_k: 40,
        repeat_penalty: 1.1,
    }
}

unsafe fn prompt_from_cstr_bounded(prompt: *const c_char) -> std::result::Result<String, ()> {
    if prompt.is_null() {
        set_last_error(ENGINE_OR_PROMPT_NULL_MSG);
        return Err(());
    }

    // Bounded scan avoids unbounded C-string walks for malformed input.
    let max_prompt_bytes = c_api_max_prompt_bytes();
    let mut nul_pos = None;
    for i in 0..=(max_prompt_bytes + 1) {
        if *prompt.add(i) == 0 {
            nul_pos = Some(i);
            break;
        }
    }

    let len = match nul_pos {
        Some(v) => v,
        None => {
            set_last_error("prompt is too large for current native safety limit");
            return Err(());
        }
    };

    if len > max_prompt_bytes {
        set_last_error("prompt is too large for current native safety limit");
        return Err(());
    }

    let bytes = slice::from_raw_parts(prompt as *const u8, len);
    let prompt_str = match str::from_utf8(bytes) {
        Ok(v) => v,
        Err(_) => {
            set_last_error("prompt is not valid UTF-8");
            return Err(());
        }
    };

    Ok(prompt_str.to_owned())
}

unsafe fn prompt_from_ptr_len(
    prompt: *const c_char,
    prompt_len: u32,
) -> std::result::Result<String, ()> {
    let len = prompt_len as usize;
    if len > c_api_max_prompt_bytes() {
        set_last_error("prompt is too large for current native safety limit");
        return Err(());
    }

    if len == 0 {
        return Ok(String::new());
    }

    if prompt.is_null() {
        set_last_error(ENGINE_OR_PROMPT_NULL_MSG);
        return Err(());
    }

    let bytes = slice::from_raw_parts(prompt as *const u8, len);
    let prompt_str = match str::from_utf8(bytes) {
        Ok(v) => v,
        Err(_) => {
            set_last_error("prompt is not valid UTF-8");
            return Err(());
        }
    };

    Ok(prompt_str.to_owned())
}

fn with_engine_lock<T, F>(
    engine: *mut LociEngine,
    wait_timeout_ms: Option<u32>,
    f: F,
) -> std::result::Result<T, ()>
where
    F: FnOnce(&mut InferenceEngine) -> T,
{
    let mutex = unsafe { &(*engine).engine };

    let mut guard = if let Some(wait_ms) = wait_timeout_ms {
        let timeout = Duration::from_millis(wait_ms as u64);
        let start = Instant::now();
        loop {
            match mutex.try_lock() {
                Ok(guard) => break guard,
                Err(TryLockError::WouldBlock) => {
                    if start.elapsed() >= timeout {
                        set_last_error(ENGINE_TIMEOUT_MSG);
                        return Err(());
                    }
                    thread::sleep(Duration::from_millis(ENGINE_LOCK_POLL_MS));
                }
                Err(TryLockError::Poisoned(_)) => {
                    set_last_error(ENGINE_POISONED_MSG);
                    return Err(());
                }
            }
        }
    } else {
        match mutex.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::WouldBlock) => {
                set_last_error(ENGINE_BUSY_MSG);
                return Err(());
            }
            Err(TryLockError::Poisoned(_)) => {
                set_last_error(ENGINE_POISONED_MSG);
                return Err(());
            }
        }
    };

    Ok(f(&mut guard))
}

unsafe fn free_engine_inner(engine: *mut LociEngine) -> bool {
    if engine.is_null() {
        return true;
    }

    // Guard against freeing while another inference call is still holding the lock.
    if with_engine_lock(engine, Some(ENGINE_FREE_WAIT_MS), |_engine| ()).is_err() {
        return false;
    }

    let _ = Box::from_raw(engine);
    true
}

unsafe fn loci_generate_inner_text(
    engine: *mut LociEngine,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    wait_timeout_ms: Option<u32>,
) -> *mut c_char {
    if engine.is_null() {
        set_last_error(ENGINE_OR_PROMPT_NULL_MSG);
        return ptr::null_mut();
    }

    if prompt.len() > c_api_max_prompt_bytes() {
        set_last_error("prompt is too large for current native safety limit");
        return ptr::null_mut();
    }

    let params = build_generation_params(max_tokens, temperature);

    let result = match with_engine_lock(engine, wait_timeout_ms, |engine| {
        engine.generate(prompt, params)
    }) {
        Ok(r) => r,
        Err(_) => return ptr::null_mut(),
    };

    match result {
        Ok(result) => match CString::new(result) {
            Ok(c_str) => c_str.into_raw(),
            Err(_) => {
                set_last_error("generated text contains interior NUL byte");
                ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(&format!("generation failed: {}", err));
            ptr::null_mut()
        }
    }
}

unsafe fn loci_generate_inner(
    engine: *mut LociEngine,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    wait_timeout_ms: Option<u32>,
) -> *mut c_char {
    let prompt_owned = match prompt_from_cstr_bounded(prompt) {
        Ok(s) => s,
        Err(_) => {
            return ptr::null_mut();
        }
    };
    loci_generate_inner_text(
        engine,
        &prompt_owned,
        max_tokens,
        temperature,
        wait_timeout_ms,
    )
}

unsafe fn loci_generate_with_len_inner(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    max_tokens: u32,
    temperature: f32,
    wait_timeout_ms: Option<u32>,
) -> *mut c_char {
    let prompt_owned = match prompt_from_ptr_len(prompt, prompt_len) {
        Ok(s) => s,
        Err(_) => {
            return ptr::null_mut();
        }
    };
    loci_generate_inner_text(
        engine,
        &prompt_owned,
        max_tokens,
        temperature,
        wait_timeout_ms,
    )
}

unsafe fn loci_generate_stream_inner_text(
    engine: *mut LociEngine,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    callback: LociStreamCallback,
    user_data: *mut std::ffi::c_void,
    wait_timeout_ms: Option<u32>,
) -> i32 {
    if engine.is_null() {
        set_last_error(ENGINE_OR_PROMPT_NULL_MSG);
        return -1;
    }

    if prompt.len() > c_api_max_prompt_bytes() {
        set_last_error("prompt is too large for current native safety limit");
        return -1;
    }

    let params = build_generation_params(max_tokens, temperature);

    let result = match with_engine_lock(engine, wait_timeout_ms, |engine| {
        engine.generate_stream(prompt, params, |token| {
            let c_token = match CString::new(token) {
                Ok(s) => s,
                Err(_) => return false,
            };
            callback(c_token.as_ptr(), user_data)
        })
    }) {
        Ok(r) => r,
        Err(_) => return -1,
    };

    match result {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(&format!("stream generation failed: {}", err));
            -1
        }
    }
}

unsafe fn loci_generate_stream_inner(
    engine: *mut LociEngine,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    callback: LociStreamCallback,
    user_data: *mut std::ffi::c_void,
    wait_timeout_ms: Option<u32>,
) -> i32 {
    let prompt_owned = match prompt_from_cstr_bounded(prompt) {
        Ok(s) => s,
        Err(_) => {
            return -1;
        }
    };

    loci_generate_stream_inner_text(
        engine,
        &prompt_owned,
        max_tokens,
        temperature,
        callback,
        user_data,
        wait_timeout_ms,
    )
}

unsafe fn loci_generate_stream_with_len_inner(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    max_tokens: u32,
    temperature: f32,
    callback: LociStreamCallback,
    user_data: *mut std::ffi::c_void,
    wait_timeout_ms: Option<u32>,
) -> i32 {
    let prompt_owned = match prompt_from_ptr_len(prompt, prompt_len) {
        Ok(s) => s,
        Err(_) => {
            return -1;
        }
    };

    loci_generate_stream_inner_text(
        engine,
        &prompt_owned,
        max_tokens,
        temperature,
        callback,
        user_data,
        wait_timeout_ms,
    )
}

/// Generate text from a UTF-8 prompt buffer with explicit byte length.
///
/// This API avoids C-string termination/scanning issues and is preferred for
/// FFI callers that already track prompt byte length.
#[no_mangle]
pub unsafe extern "C" fn loci_generate_with_len(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    max_tokens: u32,
    temperature: f32,
) -> *mut c_char {
    clear_last_error();
    loci_generate_with_len_inner(engine, prompt, prompt_len, max_tokens, temperature, None)
}

/// Generate text from a UTF-8 prompt buffer with explicit byte length and lock waiting.
#[no_mangle]
pub unsafe extern "C" fn loci_generate_wait_with_len(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    max_tokens: u32,
    temperature: f32,
    wait_timeout_ms: u32,
) -> *mut c_char {
    clear_last_error();
    loci_generate_with_len_inner(
        engine,
        prompt,
        prompt_len,
        max_tokens,
        temperature,
        Some(wait_timeout_ms),
    )
}

/// Stream generation from a UTF-8 prompt buffer with explicit byte length.
#[no_mangle]
pub unsafe extern "C" fn loci_generate_stream_with_len(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    max_tokens: u32,
    temperature: f32,
    callback: LociStreamCallback,
    user_data: *mut std::ffi::c_void,
) -> i32 {
    clear_last_error();
    loci_generate_stream_with_len_inner(
        engine,
        prompt,
        prompt_len,
        max_tokens,
        temperature,
        callback,
        user_data,
        None,
    )
}

/// Stream generation from a UTF-8 prompt buffer with explicit byte length and lock waiting.
#[no_mangle]
pub unsafe extern "C" fn loci_generate_stream_wait_with_len(
    engine: *mut LociEngine,
    prompt: *const c_char,
    prompt_len: u32,
    max_tokens: u32,
    temperature: f32,
    callback: LociStreamCallback,
    user_data: *mut std::ffi::c_void,
    wait_timeout_ms: u32,
) -> i32 {
    clear_last_error();
    loci_generate_stream_with_len_inner(
        engine,
        prompt,
        prompt_len,
        max_tokens,
        temperature,
        callback,
        user_data,
        Some(wait_timeout_ms),
    )
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
    clear_last_error();
    if model_path.is_null() {
        set_last_error("model_path is null");
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("model_path is not valid UTF-8");
            return ptr::null_mut();
        }
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
        Ok(engine) => Box::into_raw(Box::new(LociEngine {
            engine: Mutex::new(engine),
        })),
        Err(err) => {
            set_last_error(&format!("failed to create inference engine: {}", err));
            ptr::null_mut()
        }
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
    clear_last_error();
    loci_generate_inner(engine, prompt, max_tokens, temperature, None)
}

/// Generate text from a prompt with lock waiting.
///
/// `wait_timeout_ms` controls how long to wait for another in-flight call on the
/// same engine handle. `0` means "do not wait".
#[no_mangle]
pub unsafe extern "C" fn loci_generate_wait(
    engine: *mut LociEngine,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    wait_timeout_ms: u32,
) -> *mut c_char {
    clear_last_error();
    loci_generate_inner(
        engine,
        prompt,
        max_tokens,
        temperature,
        Some(wait_timeout_ms),
    )
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
    clear_last_error();
    let _ = free_engine_inner(engine);
}

/// Destroy an inference engine and set caller pointer to NULL.
///
/// # Safety
/// - engine_ptr must point to a valid `LociEngine*` variable.
/// - This function is idempotent for NULL pointers.
#[no_mangle]
pub unsafe extern "C" fn loci_engine_free_safe(engine_ptr: *mut *mut LociEngine) {
    clear_last_error();
    if engine_ptr.is_null() {
        return;
    }

    let engine = *engine_ptr;
    if !engine.is_null() {
        if free_engine_inner(engine) {
            *engine_ptr = ptr::null_mut();
        }
    }
}

/// Get model information
///
/// # Safety
/// engine must be a valid pointer from loci_engine_new
/// Returns vocab size, or 0 on error
#[no_mangle]
pub unsafe extern "C" fn loci_get_vocab_size(engine: *const LociEngine) -> u32 {
    clear_last_error();
    if engine.is_null() {
        set_last_error("engine is null");
        return 0;
    }
    match with_engine_lock(engine as *mut LociEngine, None, |engine| {
        engine.model_info().n_vocab
    }) {
        Ok(v) => v,
        Err(_) => 0,
    }
}

/// Get model context size
#[no_mangle]
pub unsafe extern "C" fn loci_get_context_size(engine: *const LociEngine) -> u32 {
    clear_last_error();
    if engine.is_null() {
        set_last_error("engine is null");
        return 0;
    }
    match with_engine_lock(engine as *mut LociEngine, None, |engine| {
        engine.model_info().n_ctx_train
    }) {
        Ok(v) => v,
        Err(_) => 0,
    }
}

/// Generate text with streaming output
///
/// # Safety
/// - engine must be a valid pointer
/// - prompt must be a valid null-terminated C string
/// - callback will be called for each token
/// - Returns 0 on success, -1 on error
#[no_mangle]
pub unsafe extern "C" fn loci_generate_stream(
    engine: *mut LociEngine,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    callback: LociStreamCallback,
    user_data: *mut std::ffi::c_void,
) -> i32 {
    clear_last_error();
    loci_generate_stream_inner(
        engine,
        prompt,
        max_tokens,
        temperature,
        callback,
        user_data,
        None,
    )
}

/// Generate text with streaming output and lock waiting.
///
/// `wait_timeout_ms` controls how long to wait for another in-flight call on the
/// same engine handle. `0` means "do not wait".
#[no_mangle]
pub unsafe extern "C" fn loci_generate_stream_wait(
    engine: *mut LociEngine,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    callback: LociStreamCallback,
    user_data: *mut std::ffi::c_void,
    wait_timeout_ms: u32,
) -> i32 {
    clear_last_error();
    loci_generate_stream_inner(
        engine,
        prompt,
        max_tokens,
        temperature,
        callback,
        user_data,
        Some(wait_timeout_ms),
    )
}

// ============================================================
// Plugin Registry C API
// ============================================================

/// Opaque handle to PluginRegistry
pub struct LociPluginRegistry {
    registry: Arc<Mutex<PluginRegistry>>,
}

/// Create a new plugin registry
#[no_mangle]
pub unsafe extern "C" fn loci_registry_new() -> *mut LociPluginRegistry {
    clear_last_error();
    let registry = PluginRegistry::new();
    Box::into_raw(Box::new(LociPluginRegistry {
        registry: Arc::new(Mutex::new(registry)),
    }))
}

/// Load a dynamic plugin from a shared library
///
/// # Safety
/// - registry must be a valid pointer
/// - plugin_path must be a valid null-terminated C string
/// - Returns 0 on success, -1 on error
#[no_mangle]
pub unsafe extern "C" fn loci_registry_load_plugin(
    registry: *mut LociPluginRegistry,
    plugin_path: *const c_char,
) -> i32 {
    clear_last_error();
    if registry.is_null() || plugin_path.is_null() {
        set_last_error("registry or plugin_path is null");
        return -1;
    }

    let path_str = match CStr::from_ptr(plugin_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("plugin_path is not valid UTF-8");
            return -1;
        }
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    match reg.load_dynamic_plugin(path_str) {
        Ok(_) => 0,
        Err(dynamic_err) => {
            // If path looks like WASM, try sandbox plugin loader first.
            if path_str
                .rsplit_once('.')
                .map(|(_, ext)| ext.eq_ignore_ascii_case("wasm"))
                .unwrap_or(false)
            {
                match reg.load_wasm_plugin(path_str) {
                    Ok(_) => 0,
                    Err(wasm_err) => {
                        set_last_error(&format!(
                            "load plugin failed (wasm): {}; dynamic fallback error: {}",
                            wasm_err, dynamic_err
                        ));
                        -1
                    }
                }
            } else {
                set_last_error(&format!("load plugin failed: {}", dynamic_err));
                -1
            }
        }
    }
}

/// Unload a hot-swappable plugin by name.
#[no_mangle]
pub unsafe extern "C" fn loci_registry_unload_plugin(
    registry: *mut LociPluginRegistry,
    plugin_name: *const c_char,
) -> i32 {
    clear_last_error();
    if registry.is_null() || plugin_name.is_null() {
        set_last_error("registry or plugin_name is null");
        return -1;
    }

    let name_str = match CStr::from_ptr(plugin_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("plugin_name is not valid UTF-8");
            return -1;
        }
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    match reg.unload(name_str) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(&format!("unload plugin failed: {}", err));
            -1
        }
    }
}

/// Reload a hot-swappable plugin by name.
#[no_mangle]
pub unsafe extern "C" fn loci_registry_reload_plugin(
    registry: *mut LociPluginRegistry,
    plugin_name: *const c_char,
) -> i32 {
    clear_last_error();
    if registry.is_null() || plugin_name.is_null() {
        set_last_error("registry or plugin_name is null");
        return -1;
    }

    let name_str = match CStr::from_ptr(plugin_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("plugin_name is not valid UTF-8");
            return -1;
        }
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    match reg.reload(name_str) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(&format!("reload plugin failed: {}", err));
            -1
        }
    }
}

/// Return plugin list as JSON string.
///
/// Caller must free returned pointer via `loci_free_string`.
#[no_mangle]
pub unsafe extern "C" fn loci_registry_list_json(
    registry: *const LociPluginRegistry,
) -> *mut c_char {
    clear_last_error();
    if registry.is_null() {
        set_last_error("registry is null");
        return ptr::null_mut();
    }

    let registry = &(*registry).registry;
    let reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return ptr::null_mut();
        }
    };

    let json = match serde_json::to_string(&reg.list_detailed()) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(&format!("failed to serialize plugin list: {}", err));
            return ptr::null_mut();
        }
    };

    match CString::new(json) {
        Ok(s) => s.into_raw(),
        Err(_) => {
            set_last_error("plugin list json contains interior NUL byte");
            ptr::null_mut()
        }
    }
}

/// Enable a plugin by name
#[no_mangle]
pub unsafe extern "C" fn loci_registry_enable_plugin(
    registry: *mut LociPluginRegistry,
    plugin_name: *const c_char,
) -> i32 {
    clear_last_error();
    if registry.is_null() || plugin_name.is_null() {
        set_last_error("registry or plugin_name is null");
        return -1;
    }

    let name_str = match CStr::from_ptr(plugin_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("plugin_name is not valid UTF-8");
            return -1;
        }
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    match reg.enable(name_str) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(&format!("enable plugin failed: {}", err));
            -1
        }
    }
}

/// Disable a plugin by name
#[no_mangle]
pub unsafe extern "C" fn loci_registry_disable_plugin(
    registry: *mut LociPluginRegistry,
    plugin_name: *const c_char,
) -> i32 {
    clear_last_error();
    if registry.is_null() || plugin_name.is_null() {
        set_last_error("registry or plugin_name is null");
        return -1;
    }

    let name_str = match CStr::from_ptr(plugin_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("plugin_name is not valid UTF-8");
            return -1;
        }
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    match reg.disable(name_str) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(&format!("disable plugin failed: {}", err));
            -1
        }
    }
}

/// Get plugin count
#[no_mangle]
pub unsafe extern "C" fn loci_registry_count(registry: *const LociPluginRegistry) -> i32 {
    clear_last_error();
    if registry.is_null() {
        set_last_error("registry is null");
        return -1;
    }

    let registry = &(*registry).registry;
    let reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    reg.count() as i32
}

/// Save registry configuration to file
#[no_mangle]
pub unsafe extern "C" fn loci_registry_save(
    registry: *mut LociPluginRegistry,
    config_path: *const c_char,
) -> i32 {
    clear_last_error();
    if registry.is_null() || config_path.is_null() {
        set_last_error("registry or config_path is null");
        return -1;
    }

    let path_str = match CStr::from_ptr(config_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("config_path is not valid UTF-8");
            return -1;
        }
    };

    let registry = &(*registry).registry;
    let reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    match reg.save_to_file(path_str) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(&format!("save registry failed: {}", err));
            -1
        }
    }
}

/// Load registry configuration from file
#[no_mangle]
pub unsafe extern "C" fn loci_registry_load(
    registry: *mut LociPluginRegistry,
    config_path: *const c_char,
) -> i32 {
    clear_last_error();
    if registry.is_null() || config_path.is_null() {
        set_last_error("registry or config_path is null");
        return -1;
    }

    let path_str = match CStr::from_ptr(config_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("config_path is not valid UTF-8");
            return -1;
        }
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to lock plugin registry");
            return -1;
        }
    };

    match reg.load_from_file(path_str) {
        Ok(_) => 0,
        Err(err) => {
            set_last_error(&format!("load registry failed: {}", err));
            -1
        }
    }
}

/// Free a plugin registry
#[no_mangle]
pub unsafe extern "C" fn loci_registry_free(registry: *mut LociPluginRegistry) {
    if !registry.is_null() {
        let _ = Box::from_raw(registry);
    }
}

// ============================================================
// Version and Info API
// ============================================================

/// Get Loci version string
#[no_mangle]
pub unsafe extern "C" fn loci_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

/// Check if GPU support is available
#[no_mangle]
pub extern "C" fn loci_has_gpu_support() -> bool {
    use crate::device::{DeviceSelector, DeviceType};

    let selector = DeviceSelector::new();
    selector.devices().iter().any(|d| {
        d.available && d.device_type != DeviceType::CPU
    })
}

// Get last error message (thread-local)
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = std::cell::RefCell::new(None);
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Set last error (internal)
#[allow(dead_code)]
fn set_last_error(msg: &str) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Get last error message
#[no_mangle]
pub unsafe extern "C" fn loci_get_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

// ============================================================
// Device Detection and Auto-Selection C API
// ============================================================

use crate::device::{DeviceSelector, DeviceType};

/// C-compatible device info structure
#[repr(C)]
pub struct LociDeviceInfo {
    pub device_id: i32,
    pub name: [c_char; 256],
    pub memory_bytes: u64,
    pub device_type: i32,  // 0=CPU, 1=CUDA, 2=Metal, 3=Vulkan, 4=ROCm, 5=OpenCL
    pub compute_capability: f32,
    pub available: bool,
}

/// Opaque handle to DeviceSelector
pub struct LociDeviceSelector {
    selector: DeviceSelector,
}

/// Create a new device selector
#[no_mangle]
pub unsafe extern "C" fn loci_device_selector_new() -> *mut LociDeviceSelector {
    clear_last_error();
    let selector = DeviceSelector::new();
    Box::into_raw(Box::new(LociDeviceSelector { selector }))
}

/// Free device selector
#[no_mangle]
pub unsafe extern "C" fn loci_device_selector_free(selector: *mut LociDeviceSelector) {
    if !selector.is_null() {
        let _ = Box::from_raw(selector);
    }
}

/// Get number of detected devices
#[no_mangle]
pub unsafe extern "C" fn loci_get_device_count(selector: *const LociDeviceSelector) -> i32 {
    clear_last_error();
    if selector.is_null() {
        set_last_error("selector is null");
        return -1;
    }
    let selector = &(*selector).selector;
    selector.device_count() as i32
}

/// Get device information by index
///
/// Returns 0 on success, -1 on error
#[no_mangle]
pub unsafe extern "C" fn loci_get_device_info(
    selector: *const LociDeviceSelector,
    index: i32,
    info: *mut LociDeviceInfo,
) -> i32 {
    clear_last_error();
    if selector.is_null() || info.is_null() || index < 0 {
        set_last_error("selector/info is null or index is negative");
        return -1;
    }

    let selector = &(*selector).selector;
    if let Some(device) = selector.device(index as usize) {
        // Convert Rust DeviceInfo to C DeviceInfo
        let device_info = &mut *info;
        device_info.device_id = device.id;

        // Copy name (null-terminated)
        let name_bytes = device.name.as_bytes();
        let copy_len = name_bytes.len().min(255);
        for (i, &byte) in name_bytes.iter().take(copy_len).enumerate() {
            device_info.name[i] = byte as c_char;
        }
        device_info.name[copy_len] = 0; // Null terminator

        device_info.memory_bytes = device.memory_bytes;
        device_info.device_type = device.device_type as i32;
        device_info.compute_capability = device.compute_capability;
        device_info.available = device.available;

        0 // Success
    } else {
        set_last_error("device index out of bounds");
        -1 // Index out of bounds
    }
}

/// Automatically select the best device
///
/// Returns device ID, or -1 on error
#[no_mangle]
pub unsafe extern "C" fn loci_auto_select_device(selector: *const LociDeviceSelector) -> i32 {
    clear_last_error();
    if selector.is_null() {
        set_last_error("selector is null");
        return -1;
    }

    let selector = &(*selector).selector;
    let config = selector.auto_select();
    config.device_id
}

/// Get recommended device configuration for model size
///
/// model_size_gb: Estimated model size in GB
/// Returns device ID, or -1 on error
#[no_mangle]
pub unsafe extern "C" fn loci_recommend_device_for_model(
    selector: *const LociDeviceSelector,
    model_size_gb: f32,
) -> i32 {
    clear_last_error();
    if selector.is_null() {
        set_last_error("selector is null");
        return -1;
    }

    let selector = &(*selector).selector;
    let config = selector.recommend_for_model(model_size_gb);
    config.device_id
}

/// Check if specific backend is available
///
/// device_type: 0=CPU, 1=CUDA, 2=Metal, 3=Vulkan, 4=ROCm, 5=OpenCL
#[no_mangle]
pub unsafe extern "C" fn loci_has_backend(
    selector: *const LociDeviceSelector,
    device_type: i32,
) -> bool {
    clear_last_error();
    if selector.is_null() || device_type < 0 || device_type > 5 {
        set_last_error("selector is null or device_type out of range [0,5]");
        return false;
    }

    let selector = &(*selector).selector;
    let dtype = match device_type {
        0 => DeviceType::CPU,
        1 => DeviceType::CUDA,
        2 => DeviceType::Metal,
        3 => DeviceType::Vulkan,
        4 => DeviceType::ROCm,
        5 => DeviceType::OpenCL,
        _ => return false,
    };

    selector.has_backend(dtype)
}

/// Create inference engine with automatic device selection
///
/// # Safety
/// model_path must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn loci_engine_new_auto(
    model_path: *const c_char,
    n_ctx: u32,
) -> *mut LociEngine {
    clear_last_error();
    if model_path.is_null() {
        set_last_error("model_path is null");
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("model_path is not valid UTF-8");
            return ptr::null_mut();
        }
    };

    // Auto-detect best device
    let selector = DeviceSelector::new();
    let device_config = selector.auto_select();

    let config = ModelConfig {
        model_path: PathBuf::from(path_str),
        n_ctx,
        n_batch: 512,
        n_threads: None,
        n_gpu_layers: device_config.n_gpu_layers,
        use_gpu: device_config.device_type != DeviceType::CPU,
    };

    match InferenceEngine::new(config) {
        Ok(engine) => Box::into_raw(Box::new(LociEngine {
            engine: Mutex::new(engine),
        })),
        Err(err) => {
            set_last_error(&format!("failed to create inference engine: {}", err));
            ptr::null_mut()
        }
    }
}

/// Create inference engine with specific device
///
/// # Safety
/// model_path must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn loci_engine_new_with_device(
    model_path: *const c_char,
    n_ctx: u32,
    _device_id: i32,
    n_gpu_layers: i32,
) -> *mut LociEngine {
    clear_last_error();
    if model_path.is_null() {
        set_last_error("model_path is null");
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_last_error("model_path is not valid UTF-8");
            return ptr::null_mut();
        }
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
        Ok(engine) => Box::into_raw(Box::new(LociEngine {
            engine: Mutex::new(engine),
        })),
        Err(err) => {
            set_last_error(&format!("failed to create inference engine: {}", err));
            ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn c_api_sets_last_error_on_null_model_path() {
        unsafe {
            let engine = loci_engine_new(ptr::null(), 512, 0);
            assert!(engine.is_null());
            let err = loci_get_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy().to_string();
            assert!(msg.contains("model_path is null"));
        }
    }

    #[test]
    fn c_api_sets_last_error_on_null_registry() {
        unsafe {
            let rc = loci_registry_count(ptr::null());
            assert_eq!(rc, -1);
            let err = loci_get_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy().to_string();
            assert!(msg.contains("registry is null"));
        }
    }

    #[test]
    fn c_api_registry_new_hot_swap_apis_validate_null() {
        unsafe {
            let unload_rc = loci_registry_unload_plugin(ptr::null_mut(), ptr::null());
            assert_eq!(unload_rc, -1);

            let reload_rc = loci_registry_reload_plugin(ptr::null_mut(), ptr::null());
            assert_eq!(reload_rc, -1);

            let list_json = loci_registry_list_json(ptr::null());
            assert!(list_json.is_null());
            let err = loci_get_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy().to_string();
            assert!(msg.contains("registry is null"));
        }
    }

    #[test]
    fn c_api_registry_list_json_empty_registry() {
        unsafe {
            let registry = loci_registry_new();
            assert!(!registry.is_null());

            let list_ptr = loci_registry_list_json(registry as *const LociPluginRegistry);
            assert!(!list_ptr.is_null());

            let payload = CStr::from_ptr(list_ptr).to_string_lossy().to_string();
            assert_eq!(payload, "[]");

            loci_free_string(list_ptr as *mut c_char);
            loci_registry_free(registry);
        }
    }

    #[test]
    fn c_api_free_safe_handles_null_inputs() {
        unsafe {
            loci_engine_free_safe(ptr::null_mut());
            let mut engine_ptr: *mut LociEngine = ptr::null_mut();
            loci_engine_free_safe(&mut engine_ptr as *mut *mut LociEngine);
            assert!(engine_ptr.is_null());
        }
    }

    #[test]
    fn c_api_wait_variants_validate_null_input() {
        unsafe {
            let rc = loci_generate_wait(ptr::null_mut(), ptr::null(), 8, 0.7, 100);
            assert!(rc.is_null());
            let err = loci_get_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy().to_string();
            assert!(msg.contains("engine or prompt is null"));

            let stream_rc = loci_generate_stream_wait(
                ptr::null_mut(),
                ptr::null(),
                8,
                0.7,
                dummy_stream_cb,
                ptr::null_mut(),
                100,
            );
            assert_eq!(stream_rc, -1);
        }
    }

    #[test]
    fn c_api_with_len_variants_validate_input() {
        unsafe {
            let rc = loci_generate_with_len(
                ptr::null_mut(),
                ptr::null(),
                1,
                8,
                0.7,
            );
            assert!(rc.is_null());
            let err = loci_get_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy().to_string();
            assert!(msg.contains("engine or prompt is null"));

            let too_large = loci_generate_with_len(
                ptr::null_mut(),
                b"abc".as_ptr() as *const c_char,
                (c_api_max_prompt_bytes() as u32) + 1,
                8,
                0.7,
            );
            assert!(too_large.is_null());
            let err2 = loci_get_last_error();
            assert!(!err2.is_null());
            let msg2 = CStr::from_ptr(err2).to_string_lossy().to_string();
            assert!(msg2.contains("prompt is too large"));

            let nul_payload = [b'a', 0, b'b'];
            let parsed = prompt_from_ptr_len(
                nul_payload.as_ptr() as *const c_char,
                nul_payload.len() as u32,
            );
            assert!(parsed.is_ok());
            let parsed_text = parsed.unwrap();
            assert_eq!(parsed_text.as_bytes(), &nul_payload);

            let rc_nul = loci_generate_with_len(
                ptr::null_mut(),
                nul_payload.as_ptr() as *const c_char,
                nul_payload.len() as u32,
                8,
                0.7,
            );
            assert!(rc_nul.is_null());
            let err3 = loci_get_last_error();
            assert!(!err3.is_null());
            let msg3 = CStr::from_ptr(err3).to_string_lossy().to_string();
            assert!(msg3.contains(ENGINE_OR_PROMPT_NULL_MSG));

            let stream_rc = loci_generate_stream_with_len(
                ptr::null_mut(),
                ptr::null(),
                1,
                8,
                0.7,
                dummy_stream_cb,
                ptr::null_mut(),
            );
            assert_eq!(stream_rc, -1);
        }
    }

    unsafe extern "C" fn dummy_stream_cb(
        _token: *const c_char,
        _user_data: *mut std::ffi::c_void,
    ) -> bool {
        true
    }
}

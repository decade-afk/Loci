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
use std::sync::{Arc, Mutex};

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

/// Get model context size
#[no_mangle]
pub unsafe extern "C" fn loci_get_context_size(engine: *const LociEngine) -> u32 {
    if engine.is_null() {
        return 0;
    }
    let engine = &(*engine).engine;
    engine.model_info().n_ctx_train
}

/// Callback function type for streaming inference
pub type LociStreamCallback = unsafe extern "C" fn(token: *const c_char, user_data: *mut std::ffi::c_void) -> bool;

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
    if engine.is_null() || prompt.is_null() {
        return -1;
    }

    let engine = &mut (*engine).engine;
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let params = GenerationParams {
        max_tokens,
        temperature,
        top_p: 0.95,
        top_k: 40,
        repeat_penalty: 1.1,
    };

    let result = engine.generate_stream(prompt_str, params, |token| {
        let c_token = match CString::new(token) {
            Ok(s) => s,
            Err(_) => return false,
        };
        callback(c_token.as_ptr(), user_data)
    });

    match result {
        Ok(_) => 0,
        Err(_) => -1,
    }
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
    if registry.is_null() || plugin_path.is_null() {
        return -1;
    }

    let path_str = match CStr::from_ptr(plugin_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => return -1,
    };

    match reg.load_dynamic_plugin(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Enable a plugin by name
#[no_mangle]
pub unsafe extern "C" fn loci_registry_enable_plugin(
    registry: *mut LociPluginRegistry,
    plugin_name: *const c_char,
) -> i32 {
    if registry.is_null() || plugin_name.is_null() {
        return -1;
    }

    let name_str = match CStr::from_ptr(plugin_name).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => return -1,
    };

    match reg.enable(name_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Disable a plugin by name
#[no_mangle]
pub unsafe extern "C" fn loci_registry_disable_plugin(
    registry: *mut LociPluginRegistry,
    plugin_name: *const c_char,
) -> i32 {
    if registry.is_null() || plugin_name.is_null() {
        return -1;
    }

    let name_str = match CStr::from_ptr(plugin_name).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => return -1,
    };

    match reg.disable(name_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Get plugin count
#[no_mangle]
pub unsafe extern "C" fn loci_registry_count(registry: *const LociPluginRegistry) -> i32 {
    if registry.is_null() {
        return -1;
    }

    let registry = &(*registry).registry;
    let reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => return -1,
    };

    reg.count() as i32
}

/// Save registry configuration to file
#[no_mangle]
pub unsafe extern "C" fn loci_registry_save(
    registry: *mut LociPluginRegistry,
    config_path: *const c_char,
) -> i32 {
    if registry.is_null() || config_path.is_null() {
        return -1;
    }

    let path_str = match CStr::from_ptr(config_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let registry = &(*registry).registry;
    let reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => return -1,
    };

    match reg.save_to_file(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Load registry configuration from file
#[no_mangle]
pub unsafe extern "C" fn loci_registry_load(
    registry: *mut LociPluginRegistry,
    config_path: *const c_char,
) -> i32 {
    if registry.is_null() || config_path.is_null() {
        return -1;
    }

    let path_str = match CStr::from_ptr(config_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let registry = &(*registry).registry;
    let mut reg = match registry.lock() {
        Ok(r) => r,
        Err(_) => return -1,
    };

    match reg.load_from_file(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
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
    if selector.is_null() {
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
    if selector.is_null() || info.is_null() || index < 0 {
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
        -1 // Index out of bounds
    }
}

/// Automatically select the best device
///
/// Returns device ID, or -1 on error
#[no_mangle]
pub unsafe extern "C" fn loci_auto_select_device(selector: *const LociDeviceSelector) -> i32 {
    if selector.is_null() {
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
    if selector.is_null() {
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
    if selector.is_null() || device_type < 0 || device_type > 5 {
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
    if model_path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
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
        Ok(engine) => Box::into_raw(Box::new(LociEngine { engine })),
        Err(_) => ptr::null_mut(),
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

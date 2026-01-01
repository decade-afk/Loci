//! # Mobile C FFI Module
//!
//! This module provides C Foreign Function Interface (FFI) bindings for mobile platforms
//! (Android and iOS) to interact with the Loci engine. It exposes a C-compatible API
//! that can be called from Java/Kotlin (Android) and Objective-C/Swift (iOS) applications.
//!
//! The module manages engine instances using a handle-based system where each engine
//! is assigned a unique integer handle. This allows multiple concurrent engine instances
//! to be managed from mobile applications.
//!
//! ## Features
//! - Engine initialization and lifecycle management
//! - Synchronous text generation
//! - Streaming text generation with callbacks
//! - Platform-specific JNI bindings for Android
//! - Platform-specific Objective-C bindings for iOS
//! - Thread-safe engine registry with mutex protection

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use once_cell::sync::Lazy;

use crate::engine::{LociEngine, EngineConfig};

/// Global registry mapping engine handles to their corresponding engine instances.
///
/// This static HashMap stores all active engine instances, allowing them to be
/// retrieved and managed using integer handles. The registry is protected by
/// a Mutex to ensure thread-safe access across multiple mobile threads.
static ENGINE_REGISTRY: Lazy<Mutex<HashMap<usize, Arc<Mutex<LociEngine>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Counter for generating unique engine handles.
///
/// This atomic counter increments each time a new engine is created,
/// ensuring each engine receives a unique identifier.
static HANDLE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| Mutex::new(0));

// ============================================================================
// Error Codes
// ============================================================================

/// Success status code - operation completed successfully
pub const LOCI_OK: c_int = 0;
/// Error code - invalid engine handle provided
pub const LOCI_ERR_INVALID_HANDLE: c_int = -1;
/// Error code - null pointer argument provided
pub const LOCI_ERR_NULL_POINTER: c_int = -2;
/// Error code - engine initialization failed
pub const LOCI_ERR_INIT_FAILED: c_int = -3;
/// Error code - text generation failed
pub const LOCI_ERR_GENERATION_FAILED: c_int = -4;
/// Error code - model file not found
pub const LOCI_ERR_MODEL_NOT_FOUND: c_int = -5;
/// Error code - insufficient memory available
pub const LOCI_ERR_OUT_OF_MEMORY: c_int = -6;
/// Error code - invalid argument provided
pub const LOCI_ERR_INVALID_ARG: c_int = -7;

// ============================================================================
// Type Definitions
// ============================================================================

/// Callback function type for streaming text generation.
///
/// This function pointer is called for each token generated during streaming mode.
/// It allows the mobile application to process tokens in real-time as they are generated.
///
/// # Parameters
/// - `user_data`: Opaque pointer to user-provided data, passed through from the caller
/// - `token`: Pointer to the generated token string (null-terminated)
/// - `token_len`: Length of the token in bytes
///
/// # Returns
/// - Non-zero value to continue generation
/// - Zero to stop generation early
///
/// # Safety
/// This function is marked unsafe as it involves raw pointers and is called from C code.
pub type StreamCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    token: *const c_char,
    token_len: c_int,
) -> c_int;

// ============================================================================
// Core FFI Functions
// ============================================================================

/// Initializes a new Loci engine instance with the specified configuration.
///
/// This function creates a new engine, loads the model from the specified path,
/// and returns a handle that can be used for subsequent operations. The handle
/// must be destroyed using `loci_destroy` when no longer needed.
///
/// # Parameters
/// - `model_path`: Null-terminated C string path to the GGUF model file
/// - `n_threads`: Number of CPU threads to use (-1 for automatic detection)
/// - `n_gpu_layers`: Number of model layers to offload to GPU (0 for CPU-only)
///
/// # Returns
/// - Non-null pointer: Engine handle (cast to `*mut c_void`)
/// - Null pointer: Initialization failed (check error logs)
///
/// # Safety
/// This function is unsafe as it involves raw pointers and C string handling.
/// The caller must ensure `model_path` is a valid null-terminated string.
#[no_mangle]
pub unsafe extern "C" fn loci_init(
    model_path: *const c_char,
    n_threads: c_int,
    n_gpu_layers: c_int,
) -> *mut c_void {
    // Validate model_path pointer
    if model_path.is_null() {
        eprintln!("[loci_init] ERROR: model_path is NULL");
        return std::ptr::null_mut();
    }

    // Convert C string to Rust string
    let model_path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[loci_init] ERROR: Invalid UTF-8 in model_path: {}", e);
            return std::ptr::null_mut();
        }
    };

    // Create engine configuration
    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        n_threads: if n_threads < 0 { num_cpus::get() as u32 } else { n_threads as u32 },
        n_gpu_layers: n_gpu_layers as i32,
        ..Default::default()
    };

    // Initialize the engine
    let engine = match LociEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[loci_init] ERROR: Failed to initialize engine: {}", e);
            return std::ptr::null_mut();
        }
    };

    // Generate unique handle for this engine instance
    let mut counter = HANDLE_COUNTER.lock()
        .map_err(|e| {
            eprintln!("[loci_init] ERROR: Failed to acquire handle counter lock: {}", e);
            e
        })
        .unwrap_or_else(|_| return std::ptr::null_mut());
    *counter += 1;
    let handle = *counter;
    drop(counter);

    // Store engine in the global registry
    let mut registry = ENGINE_REGISTRY.lock()
        .map_err(|e| {
            eprintln!("[loci_init] ERROR: Failed to acquire engine registry lock: {}", e);
            e
        })
        .unwrap_or_else(|_| return std::ptr::null_mut());
    registry.insert(handle, Arc::new(Mutex::new(engine)));
    drop(registry);

    println!("[loci_init] Engine initialized with handle: {}", handle);
    handle as *mut c_void
}

/// Generates text synchronously from the given prompt.
///
/// This function performs non-streaming text generation, returning the complete
/// generated text in a single call. The output is written to the provided buffer.
///
/// # Parameters
/// - `engine`: Engine handle returned by `loci_init`
/// - `prompt`: Null-terminated C string containing the input prompt
/// - `max_tokens`: Maximum number of tokens to generate
/// - `out_text`: Output buffer to receive the generated text (must be null-terminated)
/// - `out_len`: Size of the output buffer in bytes (including space for null terminator)
///
/// # Returns
/// - `LOCI_OK`: Success
/// - `LOCI_ERR_INVALID_HANDLE`: Invalid engine handle
/// - `LOCI_ERR_NULL_POINTER`: Null pointer argument
/// - `LOCI_ERR_GENERATION_FAILED`: Text generation failed
/// - `LOCI_ERR_INVALID_ARG`: Invalid buffer size
///
/// # Safety
/// This function is unsafe as it involves raw pointers and buffer manipulation.
/// The caller must ensure all pointers are valid and the output buffer has sufficient space.
#[no_mangle]
pub unsafe extern "C" fn loci_generate(
    engine: *mut c_void,
    prompt: *const c_char,
    max_tokens: c_int,
    out_text: *mut c_char,
    out_len: c_int,
) -> c_int {
    // Validate pointers
    if engine.is_null() {
        return LOCI_ERR_INVALID_HANDLE;
    }
    if prompt.is_null() || out_text.is_null() {
        return LOCI_ERR_NULL_POINTER;
    }

    let handle = engine as usize;

    // Retrieve engine from registry
    let registry = ENGINE_REGISTRY.lock()
        .map_err(|e| {
            eprintln!("[loci_generate] ERROR: Failed to acquire engine registry lock: {}", e);
            e
        })
        .unwrap_or_else(|_| return LOCI_ERR_INTERNAL);
    let engine_arc = match registry.get(&handle) {
        Some(e) => e.clone(),
        None => return LOCI_ERR_INVALID_HANDLE,
    };
    drop(registry);

    // Convert prompt C string to Rust string
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return LOCI_ERR_NULL_POINTER,
    };

    // Perform text generation
    let engine = engine_arc.lock()
        .map_err(|e| {
            eprintln!("[loci_generate] ERROR: Failed to acquire engine lock: {}", e);
            e
        })
        .unwrap_or_else(|_| return LOCI_ERR_INTERNAL);
    let result = match engine.generate(prompt_str, max_tokens as usize) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("[loci_generate] ERROR: {}", e);
            return LOCI_ERR_GENERATION_FAILED;
        }
    };

    // Validate output buffer size
    if out_len <= 0 {
        eprintln!("[loci_generate] ERROR: Invalid buffer size: {}", out_len);
        return LOCI_ERR_INVALID_ARG;
    }

    // Copy generated text to output buffer
    let result_bytes = result.as_bytes();
    let available_space = (out_len - 1) as usize; // Reserve space for null terminator
    let copy_len = std::cmp::min(result_bytes.len(), available_space);

    std::ptr::copy_nonoverlapping(result_bytes.as_ptr(), out_text as *mut u8, copy_len);
    *out_text.add(copy_len) = 0; // Null-terminate the output

    LOCI_OK
}

/// Generates text with streaming callback support.
///
/// This function performs streaming text generation, calling the provided callback
/// for each token as it is generated. This allows for real-time display of generated
/// text and early termination if needed.
///
/// # Parameters
/// - `engine`: Engine handle returned by `loci_init`
/// - `prompt`: Null-terminated C string containing the input prompt
/// - `max_tokens`: Maximum number of tokens to generate
/// - `callback`: Function pointer called for each generated token
/// - `user_data`: Opaque pointer passed to the callback function
///
/// # Returns
/// - `LOCI_OK`: Success
/// - `LOCI_ERR_INVALID_HANDLE`: Invalid engine handle
/// - `LOCI_ERR_NULL_POINTER`: Null pointer argument
/// - `LOCI_ERR_GENERATION_FAILED`: Text generation failed
///
/// # Safety
/// This function is unsafe as it involves raw pointers and callback invocation.
/// The callback must be thread-safe and handle all tokens correctly.
#[no_mangle]
pub unsafe extern "C" fn loci_generate_stream(
    engine: *mut c_void,
    prompt: *const c_char,
    max_tokens: c_int,
    callback: StreamCallback,
    user_data: *mut c_void,
) -> c_int {
    // Validate pointers
    if engine.is_null() || prompt.is_null() {
        return LOCI_ERR_INVALID_HANDLE;
    }

    let handle = engine as usize;

    // Retrieve engine from registry
    let registry = ENGINE_REGISTRY.lock()
        .map_err(|e| {
            eprintln!("[loci_generate_stream] ERROR: Failed to acquire engine registry lock: {}", e);
            e
        })
        .unwrap_or_else(|_| return LOCI_ERR_INTERNAL);
    let engine_arc = match registry.get(&handle) {
        Some(e) => e.clone(),
        None => return LOCI_ERR_INVALID_HANDLE,
    };
    drop(registry);

    // Convert prompt C string to Rust string
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return LOCI_ERR_NULL_POINTER,
    };

    // Wrapper struct to bridge C callback with Rust streaming trait
    struct CFfiCallback {
        callback: StreamCallback,
        user_data: *mut c_void,
    }

    // Mark callback as Send-safe for threading
    unsafe impl Send for CFfiCallback {}

    // Implement Rust streaming callback trait for C FFI callback
    impl crate::streaming::StreamCallback for CFfiCallback {
        fn on_token(&mut self, token: &str, _token_id: i32, _position: usize) -> crate::streaming::StreamControlFlow {
            // Convert token to C string
            let token_cstr = match CString::new(token) {
                Ok(s) => s,
                Err(_) => return crate::streaming::StreamControlFlow::Stop,
            };

            // Invoke C callback
            let should_continue = unsafe {
                (self.callback)(
                    self.user_data,
                    token_cstr.as_ptr(),
                    token.len() as c_int,
                )
            };

            // Convert callback return value to flow control
            if should_continue == 0 {
                crate::streaming::StreamControlFlow::Stop
            } else {
                crate::streaming::StreamControlFlow::Continue
            }
        }
    }

    // Perform streaming generation
    let engine = engine_arc.lock()
        .map_err(|e| {
            eprintln!("[loci_generate_stream] ERROR: Failed to acquire engine lock: {}", e);
            e
        })
        .unwrap_or_else(|_| return LOCI_ERR_INTERNAL);
    let mut ffi_callback = CFfiCallback {
        callback,
        user_data,
    };

    match engine.generate_stream(prompt_str, max_tokens as usize, &mut ffi_callback) {
        Ok(_stats) => LOCI_OK,
        Err(e) => {
            eprintln!("[loci_generate_stream] ERROR: {}", e);
            LOCI_ERR_GENERATION_FAILED
        }
    }
}

/// Destroys an engine instance and releases its resources.
///
/// This function removes the engine from the registry and allows its resources
/// to be freed. After calling this function, the engine handle becomes invalid
/// and should not be used.
///
/// # Parameters
/// - `engine`: Engine handle returned by `loci_init`
///
/// # Returns
/// - `LOCI_OK`: Engine successfully destroyed
/// - `LOCI_ERR_INVALID_HANDLE`: Invalid or already destroyed handle
///
/// # Safety
/// This function is unsafe as it involves raw pointer handling.
/// The caller must ensure the handle is valid.
#[no_mangle]
pub unsafe extern "C" fn loci_destroy(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return LOCI_ERR_INVALID_HANDLE;
    }

    let handle = engine as usize;

    // Remove engine from registry
    let mut registry = ENGINE_REGISTRY.lock()
        .map_err(|e| {
            eprintln!("[loci_destroy] ERROR: Failed to acquire engine registry lock: {}", e);
            e
        })
        .unwrap_or_else(|_| return LOCI_ERR_INTERNAL);
    if registry.remove(&handle).is_some() {
        println!("[loci_destroy] Engine {} destroyed", handle);
        LOCI_OK
    } else {
        LOCI_ERR_INVALID_HANDLE
    }
}

/// Returns a description of the last error that occurred.
///
/// This function provides a human-readable error message that can be used
/// for debugging and error reporting in mobile applications.
///
/// # Returns
/// - Pointer to a null-terminated C string containing the error message
/// - The string is static and should not be freed by the caller
///
/// # Safety
/// This function is unsafe as it returns a raw pointer.
/// The caller must ensure the pointer is valid before use.
#[no_mangle]
pub unsafe extern "C" fn loci_last_error() -> *const c_char {
    // TODO: Implement proper error tracking and message storage
    "Unknown error\0".as_ptr() as *const c_char
}

// ============================================================================
// Android JNI Bindings
// ============================================================================

/// Android-specific JNI bindings for Java/Kotlin integration.
///
/// This module provides JNI functions that can be called from Android Java/Kotlin
/// code. The JNI layer handles string conversions and type mapping between
/// Java and C, delegating the actual work to the core FFI functions.
///
/// The JNI functions follow the naming convention: `Java_<package>_<class>_<method>`
#[cfg(target_os = "android")]
pub mod android_jni {
    use super::*;
    use jni::JNIEnv;
    use jni::objects::{JClass, JString};
    use jni::sys::{jlong, jint, jstring};

    /// JNI function to initialize a new engine instance from Android Java/Kotlin.
    ///
    /// This function is called from Java/Kotlin code via JNI and creates a new
    /// Loci engine with the specified configuration. The engine handle is returned
    /// as a jlong for use in subsequent JNI calls.
    ///
    /// # JNI Signature
    /// `nativeInit(Ljava/lang/String;II)J`
    ///
    /// # Parameters
    /// - `env`: JNI environment pointer
    /// - `_class`: Java class object (unused)
    /// - `model_path`: Java string containing the path to the model file
    /// - `n_threads`: Number of CPU threads to use
    /// - `n_gpu_layers`: Number of GPU layers to offload
    ///
    /// # Returns
    /// - Non-zero: Engine handle (as jlong)
    /// - Zero: Initialization failed
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_loci_LociEngine_nativeInit(
        mut env: JNIEnv,
        _class: JClass,
        model_path: JString,
        n_threads: jint,
        n_gpu_layers: jint,
    ) -> jlong {
        // Convert Java string to Rust string
        let model_path_str: String = match env.get_string(&model_path) {
            Ok(s) => s.into(),
            Err(e) => {
                eprintln!("[JNI] Failed to convert model_path: {:?}", e);
                return 0;
            }
        };

        // Convert to C string for FFI call
        let model_path_cstr = match CString::new(model_path_str) {
            Ok(s) => s,
            Err(_) => return 0,
        };

        // Call core FFI initialization function
        let handle = loci_init(
            model_path_cstr.as_ptr(),
            n_threads,
            n_gpu_layers,
        );

        handle as jlong
    }

    /// JNI function to generate text from Android Java/Kotlin.
    ///
    /// This function is called from Java/Kotlin code via JNI and performs
    /// synchronous text generation. The generated text is returned as a Java string.
    ///
    /// # JNI Signature
    /// `nativeGenerate(JLjava/lang/String;I)Ljava/lang/String;`
    ///
    /// # Parameters
    /// - `env`: JNI environment pointer
    /// - `_class`: Java class object (unused)
    /// - `engine_handle`: Engine handle returned by nativeInit
    /// - `prompt`: Java string containing the input prompt
    /// - `max_tokens`: Maximum number of tokens to generate
    ///
    /// # Returns
    /// - Non-null: Java string containing the generated text
    /// - Null: Generation failed
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_loci_LociEngine_nativeGenerate(
        mut env: JNIEnv,
        _class: JClass,
        engine_handle: jlong,
        prompt: JString,
        max_tokens: jint,
    ) -> jstring {
        // Convert Java prompt string to C string
        let prompt_str: String = match env.get_string(&prompt) {
            Ok(s) => s.into(),
            Err(_) => return std::ptr::null_mut(),
        };

        let prompt_cstr = match CString::new(prompt_str) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        // Allocate output buffer
        let mut output = vec![0u8; 8192];

        // Call core FFI generation function
        let result = loci_generate(
            engine_handle as *mut c_void,
            prompt_cstr.as_ptr(),
            max_tokens,
            output.as_mut_ptr() as *mut c_char,
            output.len() as c_int,
        );

        if result != LOCI_OK {
            return std::ptr::null_mut();
        }

        // Convert output C string to Java string
        let output_str = CStr::from_ptr(output.as_ptr() as *const c_char)
            .to_string_lossy();

        match env.new_string(output_str.as_ref()) {
            Ok(s) => s.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    }

    /// JNI function to destroy an engine instance from Android Java/Kotlin.
    ///
    /// This function is called from Java/Kotlin code via JNI and releases
    /// the resources associated with the specified engine handle.
    ///
    /// # JNI Signature
    /// `nativeDestroy(J)V`
    ///
    /// # Parameters
    /// - `_env`: JNI environment pointer (unused)
    /// - `_class`: Java class object (unused)
    /// - `engine_handle`: Engine handle to destroy
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_loci_LociEngine_nativeDestroy(
        _env: JNIEnv,
        _class: JClass,
        engine_handle: jlong,
    ) {
        loci_destroy(engine_handle as *mut c_void);
    }
}

// ============================================================================
// iOS Objective-C Bindings
// ============================================================================

/// iOS-specific Objective-C bindings for Swift/Obj-C integration.
///
/// This module provides Objective-C compatible functions that can be called
/// from iOS applications written in Swift or Objective-C. This module is
/// compiled only when targeting iOS.
///
/// # Note
/// This module is currently a placeholder. Implementations should be added
/// as needed for iOS integration, following the pattern used in the Android
/// JNI module.
#[cfg(target_os = "ios")]
pub mod ios_objc {
    use super::*;

    // TODO: Implement Objective-C bindings for iOS
    // This should include:
    // - Engine initialization functions
    // - Text generation functions
    // - Streaming generation functions
    // - Resource cleanup functions
    // - Error handling utilities
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Tests the complete lifecycle of an FFI engine instance.
    ///
    /// This test verifies that engines can be created and destroyed properly
    /// through the FFI interface.
    #[test]
    fn test_ffi_lifecycle() {
        unsafe {
            // Create engine with test model path
            let model_path = CString::new("/tmp/test.gguf")
                .expect("Failed to create test model path CString");
            let engine = loci_init(model_path.as_ptr(), 4, 0);

            // Note: Actual engine initialization may fail if model doesn't exist
            // This test primarily validates the FFI interface mechanics

            if !engine.is_null() {
                loci_destroy(engine);
            }
        }
    }

    /// Tests that error codes are properly defined and distinct.
    ///
    /// This test verifies the error code constants are set correctly.
    #[test]
    fn test_error_codes() {
        assert_eq!(LOCI_OK, 0);
        assert_ne!(LOCI_ERR_INVALID_HANDLE, LOCI_OK);
    }
}
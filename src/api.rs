//! Api Module
//!
//! This module provides core functionality for the Loci project.
//!



pub use crate::engine::{AIService, AIConfig, GenerateRequest, GenerateResponse};
pub use crate::agent::{AgentSystem, ModelConfig, AgentConfig, AgentGenerateRequest, AgentGenerateResponse};
pub use crate::sysinfo::{SystemInfo, get_system_info};
pub use crate::errors::LociError;


use flutter_rust_bridge::frb;







#[frb(dart_metadata=("dart:ffi"))]
    /// LociAPI structure
pub struct LociAPI {
    engine: AIService,
    agent_system: AgentSystem,
}

#[frb(sync)]
#[frb(dart_metadata=("freezed"))]
// Implementation for LociAPI
impl LociAPI {
    
    #[frb(init)]
    /// new function
    pub fn new() -> Self {
        Self {
            engine: AIService::new(),
            agent_system: AgentSystem::new().expect("Failed to initialize AgentSystem"),
        }
    }
    
    
    /// update_config function
    pub fn update_config(&mut self, config: AIConfig) -> Result<(), LociError> {
        self.engine.update_config(config);
        Ok(())
    }
    
    
    /// get_config function
    pub fn get_config(&self) -> AIConfig {
        self.engine.get_config()
    }
    
    
    /// load_model function
    pub fn load_model(&mut self) -> Result<(), LociError> {
        self.engine.load_model()
    }
    
    
    /// unload_model function
    pub fn unload_model(&mut self) {
        self.engine.unload_model();
    }
    
    
    /// is_model_loaded function
    pub fn is_model_loaded(&self) -> bool {
        self.engine.is_model_loaded()
    }
    
    
    /// generate function
    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, LociError> {
        self.engine.generate(request)
    }
    
    
    /// get_system_info function
    pub fn get_system_info() -> Result<SystemInfo, LociError> {
        crate::sysinfo::get_system_info()
    }
    
    
    /// recommend_model function
    pub fn recommend_model(system_info: SystemInfo) -> crate::sysinfo::ModelRecommendation {
        crate::sysinfo::recommend_model(&system_info)
    }
    
    
    
    
    /// agent_load_model function
    pub fn agent_load_model(&mut self, config: ModelConfig) -> Result<(), LociError> {
        self.agent_system.load_model(config)
    }
    
    
    /// agent_create function
    pub fn agent_create(&mut self, config: AgentConfig) -> Result<(), LociError> {
        self.agent_system.create_agent(config)
    }
    
    
    /// agent_generate function
    pub fn agent_generate(&self, request: AgentGenerateRequest) -> Result<AgentGenerateResponse, LociError> {
        self.agent_system.generate(request)
    }
    
    
    /// dispose function
    pub fn dispose(&mut self) {
        self.engine.unload_model();
        
    }
}







use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};


#[repr(C)]
#[derive(Debug, Clone, Copy)]
    /// LociErrorCode enumeration
pub enum LociErrorCode {
    Success = 0,
    InvalidArgument = 1,
    ModelNotLoaded = 2,
    ModelLoadFailed = 3,
    GenerationFailed = 4,
    OutOfMemory = 5,
    Unknown = 999,
}


#[repr(C)]
/// Result structure returned by Loci C API functions.
///
/// This structure contains error information for API calls. When an error occurs,
/// `error_code` will be non-zero and `error_message` will point to a null-terminated
/// C string describing the error.
///
/// # Memory Management
/// **IMPORTANT**: If `error_message` is not null, the caller MUST call
/// `loci_free_result()` to free the allocated memory after processing the result.
/// Failure to do so will result in a memory leak.
///
/// # Example
/// ```c
/// LociResult result = loci_load_model("/path/to/model.gguf");
/// if (result.error_code != LOCI_SUCCESS) {
///     printf("Error: %s\n", result.error_message);
///     loci_free_result(&result);  // Must free the error message
///     return;
/// }
/// loci_free_result(&result);  // Still safe to call even on success
/// ```
pub struct LociResult {
    /// Error code indicating success or failure
    pub error_code: LociErrorCode,
    /// Error message (null-terminated C string), must be freed with loci_free_result()
    pub error_message: *const c_char,
}

// Implementation for LociResult
impl LociResult {
    fn success() -> Self {
        Self {
            error_code: LociErrorCode::Success,
            error_message: std::ptr::null(),
        }
    }
    
    fn error(error: LociError) -> Self {
        let message = CString::new(error.to_string()).unwrap_or_default();
        Self {
            error_code: LociErrorCode::Unknown,
            error_message: message.into_raw(),
        }
    }
}


static mut GLOBAL_API: Option<LociAPI> = None;
static INIT: std::sync::Once = std::sync::Once::new();


#[no_mangle]
pub extern "C" fn loci_init() -> LociResult {
    INIT.call_once(|| {
        unsafe {
            GLOBAL_API = Some(LociAPI::new());
        }
    });
    LociResult::success()
}


#[no_mangle]
pub extern "C" fn loci_dispose() -> LociResult {
    unsafe {
        if let Some(api) = GLOBAL_API.as_mut() {
            api.dispose();
            GLOBAL_API = None;
        }
    }
    LociResult::success()
}


#[no_mangle]
pub extern "C" fn loci_load_model(model_path: *const c_char) -> LociResult {
    if model_path.is_null() {
        return LociResult::error(LociError::InvalidArgument("model_path is null".to_string()));
    }
    
    let path = unsafe {
        CStr::from_ptr(model_path).to_string_lossy().into_owned()
    };
    
    unsafe {
        if let Some(api) = GLOBAL_API.as_mut() {
            let mut config = api.get_config();
            config.model_path = path;
            api.update_config(config);
            
            match api.load_model() {
                Ok(_) => LociResult::success(),
                Err(e) => LociResult::error(e),
            }
        } else {
            LociResult::error(LociError::InvalidArgument("API not initialized".to_string()))
        }
    }
}


#[no_mangle]
pub extern "C" fn loci_generate(
    prompt: *const c_char,
    max_tokens: c_uint,
    temperature: f32,
    output: *mut *const c_char,
) -> LociResult {
    if prompt.is_null() || output.is_null() {
        return LociResult::error(LociError::InvalidArgument("Invalid arguments".to_string()));
    }
    
    let prompt_str = unsafe {
        CStr::from_ptr(prompt).to_string_lossy().into_owned()
    };
    
    unsafe {
        if let Some(api) = GLOBAL_API.as_ref() {
            let request = GenerateRequest {
                prompt: prompt_str,
                max_tokens: max_tokens as usize,
                temperature: Some(temperature),
                ..Default::default()
            };
            
            match api.generate(request) {
                Ok(response) => {
                    let c_str = CString::new(response.content).unwrap_or_default();
                    *output = c_str.into_raw();
                    LociResult::success()
                }
                Err(e) => LociResult::error(e),
            }
        } else {
            LociResult::error(LociError::InvalidArgument("API not initialized".to_string()))
        }
    }
}


#[no_mangle]
pub extern "C" fn loci_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Frees resources associated with a LociResult structure.
///
/// This function should be called after processing a LociResult to free
/// any dynamically allocated memory, including the error message string.
///
/// # Parameters
/// - `result`: Pointer to the LociResult structure to free
///
/// # Safety
/// This function is unsafe as it involves raw pointer manipulation.
/// The caller must ensure the pointer is valid and points to a LociResult
/// that was previously allocated by Loci API functions.
#[no_mangle]
pub unsafe extern "C" fn loci_free_result(result: *mut LociResult) {
    if result.is_null() {
        return;
    }

    let result_ref = &mut *result;

    // Free the error message if it's not null
    if !result_ref.error_message.is_null() {
        let _ = CString::from_raw(result_ref.error_message);
        result_ref.error_message = std::ptr::null();
    }

    // Reset error code
    result_ref.error_code = LociErrorCode::Success;
}




    /// LociBuilder structure
pub struct LociBuilder {
    config: AIConfig,
}

// Implementation for LociBuilder
impl LociBuilder {
    /// new function
    pub fn new() -> Self {
        Self {
            config: AIConfig::default(),
        }
    }
    
    /// model_path function
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model_path = path.into();
        self
    }
    
    /// context_size function
    pub fn context_size(mut self, size: usize) -> Self {
        self.config.context_size = size;
        self
    }
    
    /// gpu_layers function
    pub fn gpu_layers(mut self, layers: i32) -> Self {
        self.config.gpu_layers = layers;
        self
    }
    
    /// temperature function
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }
    
    /// build function
    pub fn build(self) -> LociAPI {
        let mut api = LociAPI::new();
        api.update_config(self.config);
        api
    }
}

// Implementation for Default
impl Default for LociBuilder {
    fn default() -> Self {
        Self::new()
    }
}


    /// create_loci function
pub fn create_loci() -> LociBuilder {
    LociBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_api_creation() {
        let api = LociAPI::new();
        assert!(!api.is_model_loaded());
    }
    
    #[test]
    fn test_builder_pattern() {
        let api = create_loci()
            .model_path("/test/model.gguf")
            .context_size(2048)
            .temperature(0.8)
            .build();
        
        let config = api.get_config();
        assert_eq!(config.model_path, "/test/model.gguf");
        assert_eq!(config.context_size, 2048);
        assert_eq!(config.temperature, Some(0.8));
    }
}
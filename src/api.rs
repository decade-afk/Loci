/**
 * Loci API - 统一对外接口
 * 
 * 这个文件作为 loci 库的统一入口，提供：
 * 1. Flutter Rust Bridge 的 FFI 接口
 * 2. C 风格的 FFI 接口（用于其他语言）
 * 3. 高级 Rust API 封装
 */

// 导出核心模块
pub use crate::engine::{AIService, AIConfig, GenerateRequest, GenerateResponse};
pub use crate::agent::{AgentSystem, ModelConfig, AgentConfig, AgentGenerateRequest, AgentGenerateResponse};
pub use crate::sysinfo::{SystemInfo, get_system_info};
pub use crate::errors::LociError;

// Flutter Rust Bridge 相关
use flutter_rust_bridge::frb;

// ===== FRB API (Flutter Rust Bridge) =====

/// Flutter Rust Bridge 主入口
/// 
/// 这个函数会被 Flutter Rust Bridge 代码生成器处理，
/// 生成对应的 Dart 绑定代码。
#[frb(dart_metadata=("dart:ffi"))]
pub struct LociAPI {
    engine: AIService,
    agent_system: AgentSystem,
}

#[frb(sync)]
#[frb(dart_metadata=("freezed"))]
impl LociAPI {
    /// 创建新的 Loci API 实例
    #[frb(init)]
    pub fn new() -> Self {
        Self {
            engine: AIService::new(),
            agent_system: AgentSystem::new().expect("Failed to initialize AgentSystem"),
        }
    }
    
    /// 更新 AI 配置
    pub fn update_config(&mut self, config: AIConfig) -> Result<(), LociError> {
        self.engine.update_config(config);
        Ok(())
    }
    
    /// 获取当前 AI 配置
    pub fn get_config(&self) -> AIConfig {
        self.engine.get_config()
    }
    
    /// 加载 AI 模型
    pub fn load_model(&mut self) -> Result<(), LociError> {
        self.engine.load_model()
    }
    
    /// 卸载 AI 模型
    pub fn unload_model(&mut self) {
        self.engine.unload_model();
    }
    
    /// 检查模型是否已加载
    pub fn is_model_loaded(&self) -> bool {
        self.engine.is_model_loaded()
    }
    
    /// 生成文本（同步，适用于 Flutter Isolate）
    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, LociError> {
        self.engine.generate(request)
    }
    
    /// 获取系统信息
    pub fn get_system_info() -> Result<SystemInfo, LociError> {
        crate::sysinfo::get_system_info()
    }
    
    /// 获取模型推荐
    pub fn recommend_model(system_info: SystemInfo) -> crate::sysinfo::ModelRecommendation {
        crate::sysinfo::recommend_model(&system_info)
    }
    
    // Agent 系统方法
    
    /// 加载模型到模型池
    pub fn agent_load_model(&mut self, config: ModelConfig) -> Result<(), LociError> {
        self.agent_system.load_model(config)
    }
    
    /// 创建新的 Agent
    pub fn agent_create(&mut self, config: AgentConfig) -> Result<(), LociError> {
        self.agent_system.create_agent(config)
    }
    
    /// 使用 Agent 生成文本
    pub fn agent_generate(&self, request: AgentGenerateRequest) -> Result<AgentGenerateResponse, LociError> {
        self.agent_system.generate(request)
    }
    
    /// 释放资源
    pub fn dispose(&mut self) {
        self.engine.unload_model();
        // Agent 系统会在 Drop 时自动清理
    }
}

// ===== C FFI 接口 =====

/// C 风格的 FFI 接口，用于其他语言集成
/// 
/// 这些函数使用 C ABI，可以被 C、C++、Swift、Kotlin 等语言调用

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};

/// 错误码定义
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum LociErrorCode {
    Success = 0,
    InvalidArgument = 1,
    ModelNotLoaded = 2,
    ModelLoadFailed = 3,
    GenerationFailed = 4,
    OutOfMemory = 5,
    Unknown = 999,
}

/// C 风格的结果
#[repr(C)]
pub struct LociResult {
    pub error_code: LociErrorCode,
    pub error_message: *const c_char,
}

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

/// 全局 API 实例（注意：线程安全性）
static mut GLOBAL_API: Option<LociAPI> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// 初始化 Loci API
#[no_mangle]
pub extern "C" fn loci_init() -> LociResult {
    INIT.call_once(|| {
        unsafe {
            GLOBAL_API = Some(LociAPI::new());
        }
    });
    LociResult::success()
}

/// 释放 Loci API
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

/// 加载模型
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

/// 生成文本
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

/// 释放字符串内存
#[no_mangle]
pub extern "C" fn loci_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

// ===== Rust 高级 API =====

/// 便捷的 Builder 模式 API
pub struct LociBuilder {
    config: AIConfig,
}

impl LociBuilder {
    pub fn new() -> Self {
        Self {
            config: AIConfig::default(),
        }
    }
    
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model_path = path.into();
        self
    }
    
    pub fn context_size(mut self, size: usize) -> Self {
        self.config.context_size = size;
        self
    }
    
    pub fn gpu_layers(mut self, layers: i32) -> Self {
        self.config.gpu_layers = layers;
        self
    }
    
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }
    
    pub fn build(self) -> LociAPI {
        let mut api = LociAPI::new();
        api.update_config(self.config);
        api
    }
}

impl Default for LociBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 便捷函数
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
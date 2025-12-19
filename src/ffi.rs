/**
 * FFI (Foreign Function Interface) 接口层
 *
 * 本模块提供 C ABI 兼容的接口，支持：
 * - Tauri Desktop 动态链接
 * - Flutter FFI Bridge
 * - 其他跨语言调用场景
 *
 * ## 设计原则
 *
 * 1. **C 兼容性**：所有导出函数使用 C ABI（`extern "C"`）
 * 2. **内存安全**：明确的所有权转移和生命周期管理
 * 3. **错误处理**：通过返回值和输出参数传递错误信息
 * 4. **版本检查**：提供版本查询函数，确保 ABI 兼容
 *
 * ## 内存管理
 *
 * - 创建函数（`*_new`）：分配内存并返回 opaque pointer
 * - 销毁函数（`*_free`）：释放内存，调用者必须调用
 * - 字符串：使用 C 字符串（`*const c_char`），调用者负责释放
 *
 * ## 错误码
 *
 * ```
 * 0   - 成功
 * -1  - 空指针错误
 * -2  - 模型未加载
 * -3  - 配置错误
 * -4  - 推理错误
 * -5  - IO 错误
 * -99 - 未知错误
 * ```
 *
 * ## 使用示例（Rust）
 *
 * ```rust
 * use loci::ffi::*;
 *
 * unsafe {
 *     // 创建 AI 服务
 *     let service = loci_ai_service_new();
 *
 *     // 配置模型
 *     let config = r#"{"model_path":"/path/to/model.gguf","context_size":4096}"#;
 *     loci_ai_service_update_config(service, config.as_ptr() as *const i8);
 *
 *     // 加载模型
 *     let mut error_msg: *mut i8 = std::ptr::null_mut();
 *     let result = loci_ai_service_load_model(service, &mut error_msg);
 *     if result != 0 {
 *         let msg = std::ffi::CStr::from_ptr(error_msg).to_str().unwrap();
 *         println!("Error: {}", msg);
 *         loci_string_free(error_msg);
 *     }
 *
 *     // 生成文本
 *     let request = r#"{"prompt":"你好","max_tokens":100}"#;
 *     let mut response: *mut i8 = std::ptr::null_mut();
 *     loci_ai_service_generate(service, request.as_ptr() as *const i8, &mut response, &mut error_msg);
 *
 *     // 释放资源
 *     loci_string_free(response);
 *     loci_ai_service_free(service);
 * }
 * ```
 *
 * ## 使用示例（Flutter/Dart）
 *
 * ```dart
 * import 'dart:ffi';
 * import 'package:ffi/ffi.dart';
 *
 * // 加载动态库
 * final dylib = DynamicLibrary.open('libloci.so');
 *
 * // 定义函数签名
 * typedef LociAIServiceNew = Pointer Function();
 * final lociAIServiceNew = dylib.lookupFunction<
 *     Pointer Function(),
 *     LociAIServiceNew>('loci_ai_service_new');
 *
 * // 调用
 * final service = lociAIServiceNew();
 * ```
 */

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::engine::{AIService, AIConfig, GenerateRequest};
use crate::agent::AgentSystem;
use crate::sysinfo::{SystemInfo, get_system_info};

// ==================== 错误码定义 ====================

pub const LOCI_SUCCESS: i32 = 0;
pub const LOCI_ERROR_NULL_POINTER: i32 = -1;
pub const LOCI_ERROR_MODEL_NOT_LOADED: i32 = -2;
pub const LOCI_ERROR_CONFIG_ERROR: i32 = -3;
pub const LOCI_ERROR_INFERENCE_ERROR: i32 = -4;
pub const LOCI_ERROR_IO_ERROR: i32 = -5;
pub const LOCI_ERROR_UNKNOWN: i32 = -99;

// ==================== 辅助函数 ====================

/// 将 Rust String 转换为 C 字符串（调用者负责释放）
unsafe fn rust_string_to_c(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c_str) => c_str.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// 将 C 字符串转换为 Rust &str
unsafe fn c_str_to_rust<'a>(s: *const c_char) -> Result<&'a str, &'static str> {
    if s.is_null() {
        return Err("Null pointer");
    }
    CStr::from_ptr(s).to_str().map_err(|_| "Invalid UTF-8")
}

/// 设置错误消息（通过输出参数）
unsafe fn set_error_message(error_out: *mut *mut c_char, message: String) {
    if !error_out.is_null() {
        *error_out = rust_string_to_c(message);
    }
}

// ==================== 字符串管理 ====================

/// 释放 loci 分配的 C 字符串
///
/// # 安全性
/// - 必须传入由 loci FFI 函数返回的字符串指针
/// - 不要多次释放同一个指针
#[no_mangle]
pub unsafe extern "C" fn loci_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

// ==================== AI 服务 FFI ====================

/// 创建新的 AI 服务实例
///
/// 返回：opaque pointer，使用完毕后必须调用 `loci_ai_service_free`
#[no_mangle]
pub extern "C" fn loci_ai_service_new() -> *mut AIService {
    Box::into_raw(Box::new(AIService::new()))
}

/// 释放 AI 服务实例
///
/// # 安全性
/// - 必须传入由 `loci_ai_service_new` 返回的指针
/// - 释放后不要再使用该指针
#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_free(service: *mut AIService) {
    if !service.is_null() {
        drop(Box::from_raw(service));
    }
}

/// 更新 AI 服务配置
///
/// # 参数
/// - `service`: AI 服务实例
/// - `config_json`: JSON 格式的配置字符串
/// - `error_out`: 错误消息输出参数（可选，传入 null 则忽略）
///
/// # 返回
/// - 0: 成功
/// - 负数: 错误码
#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_update_config(
    service: *mut AIService,
    config_json: *const c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() || config_json.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    let json_str = match c_str_to_rust(config_json) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, format!("Invalid config string: {}", e));
            return LOCI_ERROR_CONFIG_ERROR;
        }
    };

    match serde_json::from_str::<AIConfig>(json_str) {
        Ok(config) => {
            service_ref.update_config(config);
            LOCI_SUCCESS
        }
        Err(e) => {
            set_error_message(error_out, format!("JSON parse error: {}", e));
            LOCI_ERROR_CONFIG_ERROR
        }
    }
}

/// 加载模型
///
/// # 参数
/// - `service`: AI 服务实例
/// - `error_out`: 错误消息输出参数（可选）
///
/// # 返回
/// - 0: 成功
/// - 负数: 错误码
#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_load_model(
    service: *mut AIService,
    error_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    match service_ref.load_model() {
        Ok(_) => LOCI_SUCCESS,
        Err(e) => {
            set_error_message(error_out, e);
            LOCI_ERROR_INFERENCE_ERROR
        }
    }
}

/// 卸载模型
///
/// # 参数
/// - `service`: AI 服务实例
///
/// # 返回
/// - 0: 成功
/// - 负数: 错误码
#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_unload_model(service: *mut AIService) -> i32 {
    if service.is_null() {
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    service_ref.unload_model();
    LOCI_SUCCESS
}

/// 检查模型是否已加载
///
/// # 参数
/// - `service`: AI 服务实例
///
/// # 返回
/// - 1: 已加载
/// - 0: 未加载
/// - 负数: 错误码
#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_is_model_loaded(service: *mut AIService) -> i32 {
    if service.is_null() {
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    if service_ref.is_model_loaded() {
        1
    } else {
        0
    }
}

/// 生成文本
///
/// # 参数
/// - `service`: AI 服务实例
/// - `request_json`: JSON 格式的请求字符串
/// - `response_out`: 响应 JSON 字符串输出参数（调用者负责释放）
/// - `error_out`: 错误消息输出参数（可选）
///
/// # 返回
/// - 0: 成功
/// - 负数: 错误码
#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_generate(
    service: *mut AIService,
    request_json: *const c_char,
    response_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() || request_json.is_null() || response_out.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    let json_str = match c_str_to_rust(request_json) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, format!("Invalid request string: {}", e));
            return LOCI_ERROR_CONFIG_ERROR;
        }
    };

    let request: GenerateRequest = match serde_json::from_str(json_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, format!("JSON parse error: {}", e));
            return LOCI_ERROR_CONFIG_ERROR;
        }
    };

    match service_ref.generate(request) {
        Ok(response) => {
            match serde_json::to_string(&response) {
                Ok(json) => {
                    *response_out = rust_string_to_c(json);
                    LOCI_SUCCESS
                }
                Err(e) => {
                    set_error_message(error_out, format!("JSON serialize error: {}", e));
                    LOCI_ERROR_UNKNOWN
                }
            }
        }
        Err(e) => {
            set_error_message(error_out, e);
            LOCI_ERROR_INFERENCE_ERROR
        }
    }
}

/// 获取当前配置
///
/// # 参数
/// - `service`: AI 服务实例
/// - `config_out`: 配置 JSON 字符串输出参数（调用者负责释放）
///
/// # 返回
/// - 0: 成功
/// - 负数: 错误码
#[no_mangle]
pub unsafe extern "C" fn loci_ai_service_get_config(
    service: *mut AIService,
    config_out: *mut *mut c_char,
) -> i32 {
    if service.is_null() || config_out.is_null() {
        return LOCI_ERROR_NULL_POINTER;
    }

    let service_ref = &*service;
    let config = service_ref.get_config();

    match serde_json::to_string(&config) {
        Ok(json) => {
            *config_out = rust_string_to_c(json);
            LOCI_SUCCESS
        }
        Err(_) => LOCI_ERROR_UNKNOWN,
    }
}

// ==================== Agent 系统 FFI ====================

/// 创建新的 Agent 系统实例
///
/// 返回：opaque pointer，如果创建失败则返回 null
#[no_mangle]
pub extern "C" fn loci_agent_system_new() -> *mut AgentSystem {
    match AgentSystem::new() {
        Ok(system) => Box::into_raw(Box::new(system)),
        Err(_) => ptr::null_mut(),
    }
}

/// 释放 Agent 系统实例
#[no_mangle]
pub unsafe extern "C" fn loci_agent_system_free(system: *mut AgentSystem) {
    if !system.is_null() {
        drop(Box::from_raw(system));
    }
}

// ==================== 系统信息 FFI ====================

/// 获取系统信息
///
/// # 参数
/// - `info_out`: 系统信息 JSON 字符串输出参数（调用者负责释放）
/// - `error_out`: 错误消息输出参数（可选）
///
/// # 返回
/// - 0: 成功
/// - 负数: 错误码
#[no_mangle]
pub unsafe extern "C" fn loci_get_system_info(
    info_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> i32 {
    if info_out.is_null() {
        set_error_message(error_out, "Null pointer".to_string());
        return LOCI_ERROR_NULL_POINTER;
    }

    match get_system_info() {
        Ok(info) => {
            match serde_json::to_string(&info) {
                Ok(json) => {
                    *info_out = rust_string_to_c(json);
                    LOCI_SUCCESS
                }
                Err(e) => {
                    set_error_message(error_out, format!("JSON serialize error: {}", e));
                    LOCI_ERROR_UNKNOWN
                }
            }
        }
        Err(e) => {
            set_error_message(error_out, e);
            LOCI_ERROR_UNKNOWN
        }
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_roundtrip() {
        unsafe {
            let original = "Hello, FFI!".to_string();
            let c_str = rust_string_to_c(original.clone());
            assert!(!c_str.is_null());

            let rust_str = c_str_to_rust(c_str).unwrap();
            assert_eq!(rust_str, original);

            loci_string_free(c_str);
        }
    }

    #[test]
    fn test_ai_service_lifecycle() {
        unsafe {
            let service = loci_ai_service_new();
            assert!(!service.is_null());

            let is_loaded = loci_ai_service_is_model_loaded(service);
            assert_eq!(is_loaded, 0);

            loci_ai_service_free(service);
        }
    }

    #[test]
    fn test_error_handling() {
        unsafe {
            let result = loci_ai_service_is_model_loaded(ptr::null_mut());
            assert_eq!(result, LOCI_ERROR_NULL_POINTER);
        }
    }
}

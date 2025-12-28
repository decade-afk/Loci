/**
 * Loci Phase 3 Week 1: 移动端 C FFI 接口
 *
 * 核心特性：
 * 1. 标准 C FFI 导出（跨平台兼容）
 * 2. Android JNI 接口（Java 互操作）
 * 3. iOS Objective-C 兼容接口
 * 4. 流式生成回调支持
 * 5. 线程安全保证
 *
 * ABI 稳定性：
 * - 所有导出函数使用 C 调用约定 (extern "C")
 * - 不透明指针传递（Opaque Pointer Pattern）
 * - 错误码返回（避免跨语言异常）
 */

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use once_cell::sync::Lazy;

use crate::engine::{LociEngine, EngineConfig};

// ==================== 全局状态管理 ====================

/// 全局引擎注册表（线程安全）
static ENGINE_REGISTRY: Lazy<Mutex<HashMap<usize, Arc<Mutex<LociEngine>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// 引擎句柄计数器（用于生成唯一 ID）
static HANDLE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| Mutex::new(0));

// ==================== 错误码定义 ====================

pub const LOCI_OK: c_int = 0;
pub const LOCI_ERR_INVALID_HANDLE: c_int = -1;
pub const LOCI_ERR_NULL_POINTER: c_int = -2;
pub const LOCI_ERR_INIT_FAILED: c_int = -3;
pub const LOCI_ERR_GENERATION_FAILED: c_int = -4;
pub const LOCI_ERR_MODEL_NOT_FOUND: c_int = -5;
pub const LOCI_ERR_OUT_OF_MEMORY: c_int = -6;

// ==================== 流式生成回调 ====================

/// 流式生成回调函数类型（C 兼容）
///
/// 参数：
/// - user_data: 用户自定义数据指针
/// - token: 当前生成的 token 字符串（UTF-8）
/// - token_len: token 长度
///
/// 返回值：
/// - 1: 继续生成
/// - 0: 停止生成
pub type StreamCallback = unsafe extern "C" fn(
    user_data: *mut c_void,
    token: *const c_char,
    token_len: c_int,
) -> c_int;

// ==================== 核心 C FFI 接口 ====================

/// 初始化 Loci 引擎
///
/// # 参数
/// - model_path: GGUF 模型文件路径（UTF-8 C 字符串）
/// - n_threads: CPU 线程数（-1 表示自动检测）
/// - n_gpu_layers: GPU 层数（-1 表示全部，0 表示纯 CPU）
///
/// # 返回值
/// - 成功：返回引擎句柄（非零）
/// - 失败：返回 0
///
/// # 示例
/// ```c
/// void* engine = loci_init("/path/to/model.gguf", -1, -1);
/// if (engine == NULL) {
///     fprintf(stderr, "Failed to initialize Loci\n");
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn loci_init(
    model_path: *const c_char,
    n_threads: c_int,
    n_gpu_layers: c_int,
) -> *mut c_void {
    // 参数校验
    if model_path.is_null() {
        eprintln!("[loci_init] ERROR: model_path is NULL");
        return std::ptr::null_mut();
    }

    // 转换 C 字符串
    let model_path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[loci_init] ERROR: Invalid UTF-8 in model_path: {}", e);
            return std::ptr::null_mut();
        }
    };

    // 创建引擎配置
    let config = EngineConfig {
        model_path: model_path_str.to_string(),
        n_threads: if n_threads < 0 { num_cpus::get() as u32 } else { n_threads as u32 },
        n_gpu_layers: n_gpu_layers as i32,
        ..Default::default()
    };

    // 初始化引擎
    let engine = match LociEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[loci_init] ERROR: Failed to initialize engine: {}", e);
            return std::ptr::null_mut();
        }
    };

    // 生成唯一句柄
    let mut counter = HANDLE_COUNTER.lock().unwrap();
    *counter += 1;
    let handle = *counter;
    drop(counter);

    // 注册到全局注册表
    let mut registry = ENGINE_REGISTRY.lock().unwrap();
    registry.insert(handle, Arc::new(Mutex::new(engine)));
    drop(registry);

    println!("[loci_init] Engine initialized with handle: {}", handle);
    handle as *mut c_void
}

/// 生成文本（阻塞式）
///
/// # 参数
/// - engine: 引擎句柄（由 loci_init 返回）
/// - prompt: 输入提示词（UTF-8 C 字符串）
/// - max_tokens: 最大生成 token 数
/// - out_text: 输出文本缓冲区（调用者负责分配和释放）
/// - out_len: 输出缓冲区大小
///
/// # 返回值
/// - LOCI_OK: 成功
/// - 错误码: 失败
///
/// # 示例
/// ```c
/// char output[4096];
/// int result = loci_generate(engine, "Hello", 50, output, 4096);
/// if (result == LOCI_OK) {
///     printf("Generated: %s\n", output);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn loci_generate(
    engine: *mut c_void,
    prompt: *const c_char,
    max_tokens: c_int,
    out_text: *mut c_char,
    out_len: c_int,
) -> c_int {
    // 参数校验
    if engine.is_null() {
        return LOCI_ERR_INVALID_HANDLE;
    }
    if prompt.is_null() || out_text.is_null() {
        return LOCI_ERR_NULL_POINTER;
    }

    let handle = engine as usize;

    // 获取引擎实例
    let registry = ENGINE_REGISTRY.lock().unwrap();
    let engine_arc = match registry.get(&handle) {
        Some(e) => e.clone(),
        None => return LOCI_ERR_INVALID_HANDLE,
    };
    drop(registry);

    // 转换输入
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return LOCI_ERR_NULL_POINTER,
    };

    // 生成文本
    let engine = engine_arc.lock().unwrap();
    let result = match engine.generate(prompt_str, max_tokens as usize) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("[loci_generate] ERROR: {}", e);
            return LOCI_ERR_GENERATION_FAILED;
        }
    };

    // 写入输出缓冲区
    let result_bytes = result.as_bytes();
    let copy_len = std::cmp::min(result_bytes.len(), (out_len - 1) as usize);
    std::ptr::copy_nonoverlapping(result_bytes.as_ptr(), out_text as *mut u8, copy_len);
    *out_text.add(copy_len) = 0; // 添加空字符终止符

    LOCI_OK
}

/// 流式生成文本（回调式）
///
/// # 参数
/// - engine: 引擎句柄
/// - prompt: 输入提示词
/// - max_tokens: 最大生成 token 数
/// - callback: 流式回调函数
/// - user_data: 传递给回调的用户数据
///
/// # 返回值
/// - LOCI_OK: 成功
/// - 错误码: 失败
#[no_mangle]
pub unsafe extern "C" fn loci_generate_stream(
    engine: *mut c_void,
    prompt: *const c_char,
    max_tokens: c_int,
    callback: StreamCallback,
    user_data: *mut c_void,
) -> c_int {
    if engine.is_null() || prompt.is_null() {
        return LOCI_ERR_INVALID_HANDLE;
    }

    let handle = engine as usize;

    // 获取引擎实例
    let registry = ENGINE_REGISTRY.lock().unwrap();
    let engine_arc = match registry.get(&handle) {
        Some(e) => e.clone(),
        None => return LOCI_ERR_INVALID_HANDLE,
    };
    drop(registry);

    // 转换输入
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s,
        Err(_) => return LOCI_ERR_NULL_POINTER,
    };

    // 流式生成（使用回调）
    let engine = engine_arc.lock().unwrap();

    // TODO: 实现真正的流式生成（需要引擎支持）
    // 当前简化实现：先生成完整文本，再逐 token 回调
    let result = match engine.generate(prompt_str, max_tokens as usize) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("[loci_generate_stream] ERROR: {}", e);
            return LOCI_ERR_GENERATION_FAILED;
        }
    };

    // 逐 token 回调（简化：按空格分割）
    for token in result.split_whitespace() {
        let token_cstr = match CString::new(token) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let should_continue = callback(
            user_data,
            token_cstr.as_ptr(),
            token.len() as c_int,
        );

        if should_continue == 0 {
            break; // 用户请求停止
        }
    }

    LOCI_OK
}

/// 销毁引擎
///
/// # 参数
/// - engine: 引擎句柄
///
/// # 返回值
/// - LOCI_OK: 成功
/// - 错误码: 失败
#[no_mangle]
pub unsafe extern "C" fn loci_destroy(engine: *mut c_void) -> c_int {
    if engine.is_null() {
        return LOCI_ERR_INVALID_HANDLE;
    }

    let handle = engine as usize;

    // 从注册表移除
    let mut registry = ENGINE_REGISTRY.lock().unwrap();
    if registry.remove(&handle).is_some() {
        println!("[loci_destroy] Engine {} destroyed", handle);
        LOCI_OK
    } else {
        LOCI_ERR_INVALID_HANDLE
    }
}

/// 获取最后一次错误信息（线程局部）
///
/// # 返回值
/// - 错误信息 C 字符串（静态生命周期）
#[no_mangle]
pub unsafe extern "C" fn loci_last_error() -> *const c_char {
    // TODO: 实现线程局部错误存储
    "Unknown error\0".as_ptr() as *const c_char
}

// ==================== Android JNI 接口 ====================

#[cfg(target_os = "android")]
pub mod android_jni {
    use super::*;
    use jni::JNIEnv;
    use jni::objects::{JClass, JString};
    use jni::sys::{jlong, jint, jstring};

    /// Java 类名：com.loci.LociEngine
    ///
    /// Java 示例：
    /// ```java
    /// LociEngine engine = new LociEngine("/sdcard/model.gguf", -1, -1);
    /// String output = engine.generate("Hello", 50);
    /// engine.destroy();
    /// ```

    /// JNI: 初始化引擎
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_loci_LociEngine_nativeInit(
        mut env: JNIEnv,
        _class: JClass,
        model_path: JString,
        n_threads: jint,
        n_gpu_layers: jint,
    ) -> jlong {
        // 转换 Java String -> Rust String
        let model_path_str: String = match env.get_string(&model_path) {
            Ok(s) => s.into(),
            Err(e) => {
                eprintln!("[JNI] Failed to convert model_path: {:?}", e);
                return 0;
            }
        };

        // 转换为 C 字符串
        let model_path_cstr = match CString::new(model_path_str) {
            Ok(s) => s,
            Err(_) => return 0,
        };

        // 调用核心 FFI
        let handle = loci_init(
            model_path_cstr.as_ptr(),
            n_threads,
            n_gpu_layers,
        );

        handle as jlong
    }

    /// JNI: 生成文本
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_loci_LociEngine_nativeGenerate(
        mut env: JNIEnv,
        _class: JClass,
        engine_handle: jlong,
        prompt: JString,
        max_tokens: jint,
    ) -> jstring {
        // 转换 Java String -> C String
        let prompt_str: String = match env.get_string(&prompt) {
            Ok(s) => s.into(),
            Err(_) => return std::ptr::null_mut(),
        };

        let prompt_cstr = match CString::new(prompt_str) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        // 分配输出缓冲区
        let mut output = vec![0u8; 8192];

        // 调用核心 FFI
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

        // 转换输出为 Java String
        let output_str = CStr::from_ptr(output.as_ptr() as *const c_char)
            .to_string_lossy();

        match env.new_string(output_str.as_ref()) {
            Ok(s) => s.into_raw(),
            Err(_) => std::ptr::null_mut(),
        }
    }

    /// JNI: 销毁引擎
    #[no_mangle]
    pub unsafe extern "C" fn Java_com_loci_LociEngine_nativeDestroy(
        _env: JNIEnv,
        _class: JClass,
        engine_handle: jlong,
    ) {
        loci_destroy(engine_handle as *mut c_void);
    }
}

// ==================== iOS Objective-C 接口 ====================

#[cfg(target_os = "ios")]
pub mod ios_objc {
    use super::*;

    // iOS 使用标准 C FFI 接口
    //
    // Objective-C 示例：
    // ```objc
    // void* engine = loci_init("/path/to/model.gguf", -1, -1);
    // char output[4096];
    // loci_generate(engine, "Hello", 50, output, 4096);
    // NSLog(@"Generated: %s", output);
    // loci_destroy(engine);
    // ```
    //
    // 或使用 Swift：
    // ```swift
    // let engine = loci_init("/path/to/model.gguf", -1, -1)
    // var output = [CChar](repeating: 0, count: 4096)
    // loci_generate(engine, "Hello", 50, &output, 4096)
    // print("Generated: \(String(cString: output))")
    // loci_destroy(engine)
    // ```

    // iOS 不需要额外的 JNI 封装，直接使用 C FFI
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_lifecycle() {
        unsafe {
            // 初始化（使用虚拟路径，预期失败）
            let model_path = CString::new("/tmp/test.gguf").unwrap();
            let engine = loci_init(model_path.as_ptr(), 4, 0);

            // 注意：实际测试需要真实模型文件
            // 这里只验证 API 调用不会崩溃

            if !engine.is_null() {
                loci_destroy(engine);
            }
        }
    }

    #[test]
    fn test_error_codes() {
        assert_eq!(LOCI_OK, 0);
        assert_ne!(LOCI_ERR_INVALID_HANDLE, LOCI_OK);
    }
}

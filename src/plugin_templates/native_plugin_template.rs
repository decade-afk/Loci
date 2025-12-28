//! # Loci Native 插件开发模板
//!
//! 本文件提供了 Native 插件的开发模板和 C API 定义。
//!
//! ## 编译方式
//!
//! ### Rust 插件
//! ```bash
//! cargo build --release --lib
//! ```
//!
//! ### C/C++ 插件
//! ```bash
//! gcc -shared -fPIC -o plugin.so plugin.c
//! clang -shared -fPIC -o plugin.dylib plugin.c  # macOS
//! cl /LD plugin.c  # Windows
//! ```

// ==================== C API 定义 ====================

/// Loci Native 插件 C API
///
/// 所有函数都是可选的，插件可以只实现需要的钩子。
///
/// # 返回值约定
/// - `0`: Continue（继续执行）
/// - `1`: Suspend（挂起，等待外部输入）
/// - `2`: Break（停止执行）

#[repr(C)]
pub struct LociPluginAPI {
    /// 插件初始化（可选）
    ///
    /// 返回值：0 = 成功，非 0 = 失败
    pub loci_initialize: Option<extern "C" fn() -> i32>,

    /// 插件清理（可选）
    ///
    /// 返回值：0 = 成功，非 0 = 失败
    pub loci_cleanup: Option<extern "C" fn() -> i32>,

    /// 转换 logits（采样前钩子）
    ///
    /// # 参数
    /// - `logits_ptr`: logits 数组指针（可修改）
    /// - `logits_len`: logits 数组长度
    ///
    /// # 返回值
    /// - `0`: Continue
    /// - `1`: Suspend
    /// - `2`: Break
    pub loci_transform_logits: Option<extern "C" fn(*mut f32, usize) -> i32>,

    /// Token 生成回调（采样后钩子）
    ///
    /// # 参数
    /// - `token_id`: 生成的 token ID
    /// - `token_text_ptr`: token 文本指针（只读）
    /// - `token_text_len`: token 文本长度
    ///
    /// # 返回值
    /// - `0`: Continue
    /// - `1`: Suspend
    /// - `2`: Break
    pub loci_on_token_generated: Option<extern "C" fn(i32, *const u8, usize) -> i32>,
}

// ==================== Rust 插件示例 ====================

/// 示例：Logit Bias 插件（Rust 实现）
///
/// 编译方式：
/// ```toml
/// [lib]
/// crate-type = ["cdylib"]
/// ```

#[no_mangle]
pub extern "C" fn loci_initialize() -> i32 {
    // 初始化逻辑
    println!("Logit Bias Plugin initialized");
    0 // 成功
}

#[no_mangle]
pub extern "C" fn loci_cleanup() -> i32 {
    // 清理逻辑
    println!("Logit Bias Plugin cleaned up");
    0 // 成功
}

#[no_mangle]
pub extern "C" fn loci_transform_logits(logits_ptr: *mut f32, logits_len: usize) -> i32 {
    unsafe {
        let logits = std::slice::from_raw_parts_mut(logits_ptr, logits_len);

        // 示例：对 token 0 和 1 施加偏置
        if logits.len() > 0 {
            logits[0] += 1.5; // 增加 token 0 的概率
        }
        if logits.len() > 1 {
            logits[1] -= 2.0; // 降低 token 1 的概率
        }
    }

    0 // Continue
}

#[no_mangle]
pub extern "C" fn loci_on_token_generated(
    token_id: i32,
    token_text_ptr: *const u8,
    token_text_len: usize,
) -> i32 {
    unsafe {
        let token_text = std::str::from_utf8_unchecked(
            std::slice::from_raw_parts(token_text_ptr, token_text_len)
        );

        println!("Generated token: {} (ID: {})", token_text, token_id);

        // 示例：检测特定 token 并挂起
        if token_text.contains("STOP") {
            return 1; // Suspend
        }
    }

    0 // Continue
}

// ==================== C 插件示例 ====================

/*
// plugin.c - Logit Bias Plugin (C 实现)

#include <stdio.h>
#include <string.h>

// 初始化插件
__attribute__((visibility("default")))
int loci_initialize() {
    printf("Logit Bias Plugin initialized (C)\n");
    return 0;
}

// 清理插件
__attribute__((visibility("default")))
int loci_cleanup() {
    printf("Logit Bias Plugin cleaned up (C)\n");
    return 0;
}

// 转换 logits
__attribute__((visibility("default")))
int loci_transform_logits(float* logits, size_t logits_len) {
    if (logits_len > 0) {
        logits[0] += 1.5f;  // 增加 token 0 的概率
    }
    if (logits_len > 1) {
        logits[1] -= 2.0f;  // 降低 token 1 的概率
    }
    return 0;  // Continue
}

// Token 生成回调
__attribute__((visibility("default")))
int loci_on_token_generated(int token_id, const unsigned char* token_text, size_t token_text_len) {
    printf("Generated token: %.*s (ID: %d)\n", (int)token_text_len, token_text, token_id);

    // 检测 "STOP" 并挂起
    if (token_text_len >= 4 && strncmp((const char*)token_text, "STOP", 4) == 0) {
        return 1;  // Suspend
    }

    return 0;  // Continue
}

// 编译方式：
// gcc -shared -fPIC -o logit_bias.so plugin.c
// clang -shared -fPIC -o logit_bias.dylib plugin.c  # macOS
// cl /LD /Fe:logit_bias.dll plugin.c  # Windows
*/

// ==================== C++ 插件示例 ====================

/*
// plugin.cpp - Logit Bias Plugin (C++ 实现)

#include <iostream>
#include <string>

extern "C" {

// 初始化插件
__attribute__((visibility("default")))
int loci_initialize() {
    std::cout << "Logit Bias Plugin initialized (C++)" << std::endl;
    return 0;
}

// 清理插件
__attribute__((visibility("default")))
int loci_cleanup() {
    std::cout << "Logit Bias Plugin cleaned up (C++)" << std::endl;
    return 0;
}

// 转换 logits
__attribute__((visibility("default")))
int loci_transform_logits(float* logits, size_t logits_len) {
    if (logits_len > 0) {
        logits[0] += 1.5f;
    }
    if (logits_len > 1) {
        logits[1] -= 2.0f;
    }
    return 0;
}

// Token 生成回调
__attribute__((visibility("default")))
int loci_on_token_generated(int token_id, const unsigned char* token_text, size_t token_text_len) {
    std::string text(reinterpret_cast<const char*>(token_text), token_text_len);
    std::cout << "Generated token: " << text << " (ID: " << token_id << ")" << std::endl;

    if (text.find("STOP") != std::string::npos) {
        return 1;  // Suspend
    }

    return 0;
}

}  // extern "C"

// 编译方式：
// g++ -shared -fPIC -o logit_bias.so plugin.cpp
// clang++ -shared -fPIC -o logit_bias.dylib plugin.cpp  # macOS
// cl /LD /Fe:logit_bias.dll plugin.cpp  # Windows
*/

// ==================== 签名生成脚本 ====================

/*
#!/usr/bin/env python3
# sign_plugin.py - 插件签名生成工具

import sys
import ed25519

def sign_plugin(plugin_path, private_key_path):
    """
    为插件生成 Ed25519 签名

    Args:
        plugin_path: 插件文件路径
        private_key_path: 私钥文件路径
    """
    # 读取私钥
    with open(private_key_path, 'rb') as f:
        private_key_bytes = f.read()

    private_key = ed25519.SigningKey(private_key_bytes)

    # 读取插件数据
    with open(plugin_path, 'rb') as f:
        plugin_data = f.read()

    # 生成签名
    signature = private_key.sign(plugin_data)

    # 保存签名文件
    sig_path = plugin_path + '.sig'
    with open(sig_path, 'wb') as f:
        f.write(signature)

    print(f"Signature saved to: {sig_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python sign_plugin.py <plugin_path> <private_key_path>")
        sys.exit(1)

    plugin_path = sys.argv[1]
    private_key_path = sys.argv[2]

    sign_plugin(plugin_path, private_key_path)

# 使用方式：
# python sign_plugin.py logit_bias.so private_key.bin
*/

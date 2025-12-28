//! # Loci WASM 插件开发模板
//!
//! 本文件提供了 WASM 插件的开发模板和 ABI 定义。
//!
//! ## 编译方式
//!
//! ### Rust → WASM
//! ```bash
//! cargo build --target wasm32-unknown-unknown --release
//! wasm-opt -O3 target/wasm32-unknown-unknown/release/plugin.wasm -o plugin_opt.wasm
//! ```
//!
//! ### C/C++ → WASM (使用 Emscripten)
//! ```bash
//! emcc plugin.c -o plugin.wasm -s STANDALONE_WASM=1 -O3
//! ```
//!
//! ### AssemblyScript → WASM
//! ```bash
//! asc plugin.ts -o plugin.wasm --optimize
//! ```

// ==================== WASM ABI 定义 ====================

/// Loci WASM 插件 ABI
///
/// 所有函数都是可选的，插件可以只实现需要的钩子。
///
/// # 内存布局
/// - 线性内存从偏移 0 开始
/// - Logits 数据写入偏移 0
/// - Token 文本写入偏移 0
///
/// # 返回值约定
/// - `0`: Continue（继续执行）
/// - `1`: Suspend（挂起，等待外部输入）
/// - `2`: Break（停止执行）

// ==================== Rust WASM 插件示例 ====================

/// 示例：Logit Bias 插件（Rust → WASM）
///
/// Cargo.toml 配置：
/// ```toml
/// [lib]
/// crate-type = ["cdylib"]
///
/// [dependencies]
/// # 无需依赖，纯 no_std
/// ```

#![no_std]

use core::panic::PanicInfo;
use core::slice;

// Panic handler（WASM 必需）
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// 导出内存（WASM 必需）
#[no_mangle]
pub static mut MEMORY: [u8; 65536] = [0; 65536]; // 64KB

// 初始化插件（可选）
#[no_mangle]
pub extern "C" fn loci_initialize() -> i32 {
    // 初始化逻辑
    0 // 成功
}

// 清理插件（可选）
#[no_mangle]
pub extern "C" fn loci_cleanup() -> i32 {
    // 清理逻辑
    0 // 成功
}

// 转换 logits（采样前钩子）
#[no_mangle]
pub extern "C" fn loci_transform_logits(logits_ptr: i32, logits_len: i32) -> i32 {
    unsafe {
        // 从线性内存读取 logits
        let logits = slice::from_raw_parts_mut(
            logits_ptr as *mut f32,
            logits_len as usize,
        );

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

// Token 生成回调（采样后钩子）
#[no_mangle]
pub extern "C" fn loci_on_token_generated(
    token_id: i32,
    token_text_ptr: i32,
    token_text_len: i32,
) -> i32 {
    unsafe {
        // 从线性内存读取 token 文本
        let token_text = slice::from_raw_parts(
            token_text_ptr as *const u8,
            token_text_len as usize,
        );

        // 检测 "STOP" 并挂起
        if token_text.len() >= 4 {
            if token_text[0] == b'S'
                && token_text[1] == b'T'
                && token_text[2] == b'O'
                && token_text[3] == b'P'
            {
                return 1; // Suspend
            }
        }
    }

    0 // Continue
}

// ==================== C WASM 插件示例 (Emscripten) ====================

/*
// plugin.c - Logit Bias Plugin (C → WASM)

#include <emscripten.h>
#include <string.h>

// 初始化插件
EMSCRIPTEN_KEEPALIVE
int loci_initialize() {
    return 0;
}

// 清理插件
EMSCRIPTEN_KEEPALIVE
int loci_cleanup() {
    return 0;
}

// 转换 logits
EMSCRIPTEN_KEEPALIVE
int loci_transform_logits(float* logits, int logits_len) {
    if (logits_len > 0) {
        logits[0] += 1.5f;
    }
    if (logits_len > 1) {
        logits[1] -= 2.0f;
    }
    return 0;
}

// Token 生成回调
EMSCRIPTEN_KEEPALIVE
int loci_on_token_generated(int token_id, const unsigned char* token_text, int token_text_len) {
    if (token_text_len >= 4 && strncmp((const char*)token_text, "STOP", 4) == 0) {
        return 1;  // Suspend
    }
    return 0;
}

// 编译方式：
// emcc plugin.c -o plugin.wasm \
//   -s STANDALONE_WASM=1 \
//   -s EXPORTED_FUNCTIONS='["_loci_initialize","_loci_cleanup","_loci_transform_logits","_loci_on_token_generated"]' \
//   -O3
*/

// ==================== AssemblyScript WASM 插件示例 ====================

/*
// plugin.ts - Logit Bias Plugin (AssemblyScript → WASM)

// 初始化插件
export function loci_initialize(): i32 {
    return 0;
}

// 清理插件
export function loci_cleanup(): i32 {
    return 0;
}

// 转换 logits
export function loci_transform_logits(logits_ptr: i32, logits_len: i32): i32 {
    const logits = new Float32Array(logits_len);

    // 从内存加载 logits
    for (let i = 0; i < logits_len; i++) {
        logits[i] = load<f32>(logits_ptr + i * 4);
    }

    // 施加偏置
    if (logits_len > 0) {
        logits[0] += 1.5;
    }
    if (logits_len > 1) {
        logits[1] -= 2.0;
    }

    // 写回内存
    for (let i = 0; i < logits_len; i++) {
        store<f32>(logits_ptr + i * 4, logits[i]);
    }

    return 0;
}

// Token 生成回调
export function loci_on_token_generated(
    token_id: i32,
    token_text_ptr: i32,
    token_text_len: i32
): i32 {
    const token_text = new Uint8Array(token_text_len);

    // 从内存加载 token 文本
    for (let i = 0; i < token_text_len; i++) {
        token_text[i] = load<u8>(token_text_ptr + i);
    }

    // 检测 "STOP"
    if (token_text_len >= 4) {
        if (
            token_text[0] == 83 && // 'S'
            token_text[1] == 84 && // 'T'
            token_text[2] == 79 && // 'O'
            token_text[3] == 80    // 'P'
        ) {
            return 1; // Suspend
        }
    }

    return 0;
}

// 编译方式：
// asc plugin.ts -o plugin.wasm --optimize
*/

// ==================== WASM 插件最佳实践 ====================

/*
# WASM 插件开发指南

## 1. 性能优化
- 使用 `wasm-opt -O3` 优化 WASM 文件
- 避免频繁的内存分配
- 使用 SIMD 指令加速（如果目标支持）

## 2. 安全建议
- 不要访问超出 logits/token_text 范围的内存
- 不要使用线程（wasmtime 禁用）
- 不要尝试访问文件系统或网络

## 3. 调试技巧
- 使用 `wasm-objdump` 查看导出函数
- 使用 `wasmtime run --invoke` 单独测试函数
- 使用 `console.log` (Emscripten) 或自定义日志函数

## 4. 签名生成
```bash
# 生成签名（与 Native 插件相同）
python sign_plugin.py plugin.wasm private_key.bin
```

## 5. 测试
```bash
# 使用 wasmtime 测试 WASM 插件
wasmtime run plugin.wasm --invoke loci_initialize
```

## 6. 内存限制
- 默认线性内存: 64KB
- 可扩展至 4GB（通过 WASM 内存配置）
- Loci 会自动检测并分配足够内存

## 7. 调用约定
- 所有指针都是 i32（线性内存偏移）
- 所有长度都是 i32
- 返回值都是 i32（0/1/2）

## 8. 示例项目结构
```
wasm-plugin/
├── Cargo.toml          # Rust 项目配置
├── src/
│   └── lib.rs          # 插件源码
├── build.sh            # 构建脚本
└── plugin.wasm.sig     # 签名文件
```

## 9. build.sh 示例
```bash
#!/bin/bash
set -e

# 构建 WASM
cargo build --target wasm32-unknown-unknown --release

# 优化 WASM
wasm-opt -O3 \
    target/wasm32-unknown-unknown/release/plugin.wasm \
    -o plugin_opt.wasm

# 生成签名
python3 ../sign_plugin.py plugin_opt.wasm ../private_key.bin

echo "Build complete: plugin_opt.wasm + plugin_opt.wasm.sig"
```
*/

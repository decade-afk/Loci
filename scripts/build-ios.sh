#!/bin/bash
# Loci Phase 3: iOS Universal Library 编译脚本
# 支持 iOS 设备 + 模拟器，生成 XCFramework

set -e

# ==================== 配置 ====================

# 检查 Xcode 环境
if ! command -v xcrun &> /dev/null; then
    echo "错误: 未找到 Xcode 命令行工具"
    echo "请运行: xcode-select --install"
    exit 1
fi

XCODE_VERSION=$(xcrun xcodebuild -version | head -n 1)
echo "使用 Xcode: $XCODE_VERSION"

# iOS 最低版本（支持 Metal）
IOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-14.0}"
echo "目标 iOS 版本: $IOS_DEPLOYMENT_TARGET"

# 输出目录
OUTPUT_DIR="target/ios"
mkdir -p "$OUTPUT_DIR"

# ==================== 目标架构列表 ====================

# iOS 设备架构（真机）
DEVICE_TARGETS=(
    "aarch64-apple-ios"           # ARM64 (iPhone/iPad)
)

# iOS 模拟器架构
SIMULATOR_TARGETS=(
    "aarch64-apple-ios-sim"       # ARM64 Simulator (Apple Silicon Mac)
    "x86_64-apple-ios"            # x86_64 Simulator (Intel Mac)
)

# ==================== 安装 Rust 目标 ====================

echo "安装 Rust 交叉编译目标..."
for target in "${DEVICE_TARGETS[@]}" "${SIMULATOR_TARGETS[@]}"; do
    rustup target add "$target"
done

# ==================== 编译 iOS 设备库 ====================

echo ""
echo "========================================="
echo "编译 iOS 设备库 (真机)"
echo "========================================="

DEVICE_LIBS=()

for target in "${DEVICE_TARGETS[@]}"; do
    echo "编译目标: $target"

    IPHONEOS_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
    cargo build \
        --target "$target" \
        --release \
        --lib

    DEVICE_LIBS+=("target/$target/release/libloci.a")
    echo "✓ $target 编译完成"
done

# 创建 Universal Library（如果有多个架构）
if [ ${#DEVICE_LIBS[@]} -gt 1 ]; then
    echo "合并设备库..."
    lipo -create "${DEVICE_LIBS[@]}" \
        -output "$OUTPUT_DIR/libloci-device.a"
else
    cp "${DEVICE_LIBS[0]}" "$OUTPUT_DIR/libloci-device.a"
fi

echo "✓ iOS 设备库: $OUTPUT_DIR/libloci-device.a"

# ==================== 编译 iOS 模拟器库 ====================

echo ""
echo "========================================="
echo "编译 iOS 模拟器库"
echo "========================================="

SIMULATOR_LIBS=()

for target in "${SIMULATOR_TARGETS[@]}"; do
    echo "编译目标: $target"

    IPHONEOS_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
    cargo build \
        --target "$target" \
        --release \
        --lib

    SIMULATOR_LIBS+=("target/$target/release/libloci.a")
    echo "✓ $target 编译完成"
done

# 创建 Universal Library
if [ ${#SIMULATOR_LIBS[@]} -gt 1 ]; then
    echo "合并模拟器库..."
    lipo -create "${SIMULATOR_LIBS[@]}" \
        -output "$OUTPUT_DIR/libloci-simulator.a"
else
    cp "${SIMULATOR_LIBS[0]}" "$OUTPUT_DIR/libloci-simulator.a"
fi

echo "✓ iOS 模拟器库: $OUTPUT_DIR/libloci-simulator.a"

# ==================== 生成 C 头文件 ====================

echo ""
echo "========================================="
echo "生成 C 头文件"
echo "========================================="

# 使用 cbindgen 生成头文件（如果已安装）
if command -v cbindgen &> /dev/null; then
    cbindgen \
        --config cbindgen.toml \
        --crate loci \
        --output "$OUTPUT_DIR/loci.h" \
        2>/dev/null || {
        echo "警告: cbindgen 失败，使用手动头文件"
    }
fi

# 如果 cbindgen 未安装或失败，创建手动头文件
if [ ! -f "$OUTPUT_DIR/loci.h" ]; then
    cat > "$OUTPUT_DIR/loci.h" << 'EOF'
// Loci Phase 3: iOS C Header
// Generated for Objective-C bridging

#ifndef LOCI_H
#define LOCI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ==================== 初始化与销毁 ====================

/// 初始化 Loci 引擎
/// @param model_path GGUF 模型路径
/// @param backend 后端类型 (0=CPU, 1=Metal, 2=CUDA, 3=Vulkan)
/// @param n_threads CPU 线程数 (0=自动)
/// @return 0=成功, -1=失败
int loci_init(const char* model_path, int backend, int n_threads);

/// 释放引擎资源
void loci_destroy(void);

// ==================== 文本生成 ====================

/// 生成文本（同步）
/// @param prompt 输入提示词
/// @param max_tokens 最大 token 数
/// @param temperature 采样温度
/// @param output_buffer 输出缓冲区
/// @param buffer_size 缓冲区大小
/// @return 生成的字符数 (>=0), 失败=-1
int loci_generate(
    const char* prompt,
    int max_tokens,
    float temperature,
    char* output_buffer,
    int buffer_size
);

/// 流式生成回调函数
/// @param token 生成的 token
/// @param user_data 用户数据
/// @return 0=继续, 非0=停止
typedef int (*loci_stream_callback)(const char* token, void* user_data);

/// 流式生成文本
/// @param prompt 输入提示词
/// @param max_tokens 最大 token 数
/// @param temperature 采样温度
/// @param callback 回调函数
/// @param user_data 用户数据
/// @return 0=成功, -1=失败
int loci_generate_stream(
    const char* prompt,
    int max_tokens,
    float temperature,
    loci_stream_callback callback,
    void* user_data
);

// ==================== 模型信息 ====================

/// 获取模型信息（JSON 格式）
/// @param info_buffer 输出缓冲区
/// @param buffer_size 缓冲区大小
/// @return 信息字符数 (>=0), 失败=-1
int loci_get_model_info(char* info_buffer, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif // LOCI_H
EOF
fi

echo "✓ C 头文件: $OUTPUT_DIR/loci.h"

# ==================== 创建 XCFramework ====================

echo ""
echo "========================================="
echo "创建 XCFramework"
echo "========================================="

XCFRAMEWORK_DIR="$OUTPUT_DIR/Loci.xcframework"
rm -rf "$XCFRAMEWORK_DIR"

# 创建临时框架目录
DEVICE_FRAMEWORK="$OUTPUT_DIR/tmp/Loci-device.framework"
SIMULATOR_FRAMEWORK="$OUTPUT_DIR/tmp/Loci-simulator.framework"

mkdir -p "$DEVICE_FRAMEWORK/Headers"
mkdir -p "$SIMULATOR_FRAMEWORK/Headers"

# 复制库文件
cp "$OUTPUT_DIR/libloci-device.a" "$DEVICE_FRAMEWORK/Loci"
cp "$OUTPUT_DIR/libloci-simulator.a" "$SIMULATOR_FRAMEWORK/Loci"

# 复制头文件
cp "$OUTPUT_DIR/loci.h" "$DEVICE_FRAMEWORK/Headers/"
cp "$OUTPUT_DIR/loci.h" "$SIMULATOR_FRAMEWORK/Headers/"

# 创建 module.modulemap
cat > "$DEVICE_FRAMEWORK/Headers/module.modulemap" << 'EOF'
module Loci {
    header "loci.h"
    export *
}
EOF
cp "$DEVICE_FRAMEWORK/Headers/module.modulemap" "$SIMULATOR_FRAMEWORK/Headers/"

# 使用 xcodebuild 创建 XCFramework
xcodebuild -create-xcframework \
    -library "$DEVICE_FRAMEWORK/Loci" \
    -headers "$DEVICE_FRAMEWORK/Headers" \
    -library "$SIMULATOR_FRAMEWORK/Loci" \
    -headers "$SIMULATOR_FRAMEWORK/Headers" \
    -output "$XCFRAMEWORK_DIR"

# 清理临时文件
rm -rf "$OUTPUT_DIR/tmp"

echo "✓ XCFramework: $XCFRAMEWORK_DIR"

# ==================== 完成 ====================

echo ""
echo "========================================="
echo "iOS 编译完成！"
echo "========================================="
echo "产物位置:"
echo "  设备库:       $OUTPUT_DIR/libloci-device.a"
echo "  模拟器库:     $OUTPUT_DIR/libloci-simulator.a"
echo "  C 头文件:     $OUTPUT_DIR/loci.h"
echo "  XCFramework:  $XCFRAMEWORK_DIR"
echo ""
echo "集成到 Xcode 项目："
echo "1. 将 Loci.xcframework 拖入项目"
echo "2. 在 Build Phases 中添加 Framework（Embed & Sign）"
echo "3. 在 Objective-C/Swift 中导入：#import <Loci/loci.h>"
echo "4. 调用 C 函数：loci_init(), loci_generate() 等"
echo ""
echo "Swift 使用示例:"
echo "  loci_init(modelPath, 1, 0)  // Metal backend"
echo "  var buffer = [CChar](repeating: 0, count: 4096)"
echo "  loci_generate(prompt, 50, 0.7, &buffer, 4096)"
echo "  let output = String(cString: buffer)"
echo ""

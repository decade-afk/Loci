#!/bin/bash
# Loci Phase 3: Android 4-ABI 构建脚本
# 生成 4 种 Android ABI 产物：arm64-v8a, armeabi-v7a, x86_64, x86
#
# 使用方法：
# ./scripts/build_android.sh [release|debug]

set -e  # 遇到错误立即退出

BUILD_MODE="${1:-release}"  # 默认 release 模式
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/target/android"

echo "╔══════════════════════════════════════════════╗"
echo "║   Loci Android 4-ABI 构建脚本 (Phase 3)     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "构建模式: ${BUILD_MODE}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 检查依赖
if ! command -v rustc &> /dev/null; then
    echo "❌ 错误: 未安装 Rust 工具链"
    echo "请访问 https://rustup.rs/ 安装"
    exit 1
fi

if [ -z "${ANDROID_NDK_ROOT}" ] && [ -z "${NDK_HOME}" ]; then
    echo "❌ 错误: 未设置 ANDROID_NDK_ROOT 或 NDK_HOME 环境变量"
    echo "请下载 Android NDK r26+ 并设置环境变量"
    exit 1
fi

# 安装 Android 编译目标（如果尚未安装）
echo "📦 检查 Rust Android targets..."
rustup target add aarch64-linux-android    # arm64-v8a
rustup target add armv7-linux-androideabi  # armeabi-v7a
rustup target add x86_64-linux-android     # x86_64
rustup target add i686-linux-android       # x86

# 清理旧产物
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"/{arm64-v8a,armeabi-v7a,x86_64,x86}

# 构建函数
build_abi() {
    local TARGET=$1
    local ABI=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔨 构建 Android $ABI ($TARGET)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "$BUILD_MODE" = "release" ]; then
        cargo build --target "$TARGET" --release --lib
        cp "${PROJECT_ROOT}/target/${TARGET}/release/libloci.so" \
           "${OUTPUT_DIR}/${ABI}/libloci.so"
    else
        cargo build --target "$TARGET" --lib
        cp "${PROJECT_ROOT}/target/${TARGET}/debug/libloci.so" \
           "${OUTPUT_DIR}/${ABI}/libloci.so"
    fi

    # 检查产物
    if [ -f "${OUTPUT_DIR}/${ABI}/libloci.so" ]; then
        local SIZE=$(du -h "${OUTPUT_DIR}/${ABI}/libloci.so" | cut -f1)
        echo "✅ 成功: ${ABI}/libloci.so (${SIZE})"
    else
        echo "❌ 失败: ${ABI}/libloci.so"
        exit 1
    fi
}

# 构建所有 ABI（按优先级排序）
build_abi "aarch64-linux-android"   "arm64-v8a"     # 64位 ARM（主流旗舰机）
build_abi "armv7-linux-androideabi" "armeabi-v7a"   # 32位 ARM（老设备兼容）
build_abi "x86_64-linux-android"    "x86_64"        # 64位 x86（模拟器）
build_abi "i686-linux-android"      "x86"           # 32位 x86（老模拟器）

# 生成产物清单
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Android 产物清单"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tree "${OUTPUT_DIR}" 2>/dev/null || find "${OUTPUT_DIR}" -type f

# 计算总大小
TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
echo ""
echo "✅ 构建完成！总大小: ${TOTAL_SIZE}"
echo "产物位置: ${OUTPUT_DIR}"
echo ""
echo "集成指南："
echo "1. 将 .so 文件复制到 Android 项目："
echo "   cp target/android/*/libloci.so app/src/main/jniLibs/"
echo ""
echo "2. 在 Java 代码中加载："
echo "   System.loadLibrary(\"loci\");"
echo ""

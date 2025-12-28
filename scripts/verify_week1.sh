#!/bin/bash
# Loci Phase 3 Week 1: 快速验证脚本
# 验证所有移动端构建配置是否正确

set -e

echo "╔══════════════════════════════════════════════╗"
echo "║   Loci Phase 3 Week 1 验证脚本              ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. 检查基础编译
echo "✓ 检查基础编译..."
cargo check --lib
echo ""

# 2. 检查 Android targets 安装情况
echo "✓ 检查 Android targets..."
rustup target list | grep -E "(aarch64-linux-android|armv7-linux-androideabi|x86_64-linux-android|i686-linux-android)" || echo "  需要安装 Android targets"
echo ""

# 3. 检查 iOS targets 安装情况（仅 macOS）
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✓ 检查 iOS targets..."
    rustup target list | grep -E "(aarch64-apple-ios|x86_64-apple-ios|aarch64-apple-ios-sim)" || echo "  需要安装 iOS targets"
    echo ""
fi

# 4. 检查 NDK 环境变量（Android）
if [ ! -z "${ANDROID_NDK_ROOT}" ] || [ ! -z "${NDK_HOME}" ]; then
    echo "✓ Android NDK: ${ANDROID_NDK_ROOT:-$NDK_HOME}"
else
    echo "⚠ 警告: 未设置 ANDROID_NDK_ROOT 或 NDK_HOME"
fi
echo ""

# 5. 检查 Xcode（仅 macOS）
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v xcrun &> /dev/null; then
        XCODE_VERSION=$(xcrun xcodebuild -version 2>/dev/null | head -1 || echo "未知")
        echo "✓ Xcode: $XCODE_VERSION"
    else
        echo "⚠ 警告: 未安装 Xcode Command Line Tools"
    fi
    echo ""
fi

# 6. 检查构建脚本权限
echo "✓ 检查构建脚本..."
if [ -x "scripts/build_android.sh" ]; then
    echo "  build_android.sh: 可执行"
else
    echo "  build_android.sh: 不可执行（运行 chmod +x scripts/build_android.sh）"
fi

if [ -x "scripts/build_ios.sh" ]; then
    echo "  build_ios.sh: 可执行"
else
    echo "  build_ios.sh: 不可执行（运行 chmod +x scripts/build_ios.sh）"
fi
echo ""

# 7. 检查核心文件完整性
echo "✓ 检查核心文件..."
FILES=(
    "src/mobile_ffi.rs"
    "build.rs"
    "scripts/build_android.sh"
    "scripts/build_ios.sh"
    "examples/flutter_demo/lib/loci_ffi.dart"
    "examples/flutter_demo/lib/main.dart"
    "PHASE3_WEEK1_DELIVERY.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (缺失)"
    fi
done
echo ""

# 8. 项目结构统计
echo "✓ 项目统计..."
echo "  Rust 源文件: $(find src -name '*.rs' | wc -l)"
echo "  构建脚本: $(find scripts -name '*.sh' 2>/dev/null | wc -l)"
echo "  示例代码: $(find examples -name '*.dart' 2>/dev/null | wc -l)"
echo ""

# 9. 总结
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "验证完成！"
echo ""
echo "下一步："
echo "1. Android 构建: ./scripts/build_android.sh"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "2. iOS 构建: ./scripts/build_ios.sh"
fi
echo ""

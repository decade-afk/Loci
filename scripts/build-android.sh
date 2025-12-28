#!/bin/bash
# Loci Phase 3: Android NDK 编译脚本
# 支持多架构编译并生成 AAR 库

set -e

# ==================== 配置 ====================

# 检查 NDK 环境变量
if [ -z "$ANDROID_NDK_ROOT" ] && [ -z "$NDK_HOME" ]; then
    echo "错误: 请设置 ANDROID_NDK_ROOT 或 NDK_HOME 环境变量"
    echo "示例: export ANDROID_NDK_ROOT=/path/to/android-ndk-r26"
    exit 1
fi

NDK_ROOT="${ANDROID_NDK_ROOT:-$NDK_HOME}"
echo "使用 Android NDK: $NDK_ROOT"

# API Level（最低支持 Android 7.0）
API_LEVEL="${ANDROID_API_LEVEL:-24}"
echo "目标 API Level: $API_LEVEL"

# 输出目录
OUTPUT_DIR="target/android"
mkdir -p "$OUTPUT_DIR"

# ==================== 目标架构列表 ====================

# 支持的 Android ABI
TARGETS=(
    "aarch64-linux-android"   # ARM64 (主流)
    "armv7-linux-androideabi" # ARMv7 (兼容旧设备)
    "x86_64-linux-android"    # x86_64 (模拟器)
    "i686-linux-android"      # x86 (兼容旧模拟器)
)

# ==================== 安装 Rust 目标 ====================

echo "安装 Rust 交叉编译目标..."
for target in "${TARGETS[@]}"; do
    rustup target add "$target"
done

# ==================== 编译循环 ====================

echo "开始编译 Android 库..."

for target in "${TARGETS[@]}"; do
    echo ""
    echo "========================================="
    echo "编译目标: $target"
    echo "========================================="

    # 设置 linker（NDK r26+ 使用 llvm）
    case "$target" in
        "aarch64-linux-android")
            ABI="arm64-v8a"
            ;;
        "armv7-linux-androideabi")
            ABI="armeabi-v7a"
            ;;
        "x86_64-linux-android")
            ABI="x86_64"
            ;;
        "i686-linux-android")
            ABI="x86"
            ;;
    esac

    # 编译为 C 动态库（cdylib）
    ANDROID_NDK_ROOT="$NDK_ROOT" \
    ANDROID_API_LEVEL="$API_LEVEL" \
    cargo build \
        --target "$target" \
        --release \
        --lib

    # 复制产物到输出目录
    mkdir -p "$OUTPUT_DIR/jniLibs/$ABI"
    cp "target/$target/release/libloci.so" "$OUTPUT_DIR/jniLibs/$ABI/"

    echo "✓ $ABI 编译完成"
done

# ==================== 生成 AAR 包（可选）====================

echo ""
echo "========================================="
echo "生成 Android AAR 包"
echo "========================================="

# 创建 AAR 目录结构
AAR_DIR="$OUTPUT_DIR/loci-aar"
rm -rf "$AAR_DIR"
mkdir -p "$AAR_DIR/jni"

# 复制 .so 文件
cp -r "$OUTPUT_DIR/jniLibs" "$AAR_DIR/jni/"

# 生成 AndroidManifest.xml
cat > "$AAR_DIR/AndroidManifest.xml" << 'EOF'
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.loci.native">
    <uses-sdk
        android:minSdkVersion="24"
        android:targetSdkVersion="34" />
</manifest>
EOF

# 生成 classes.jar（占位符，实际 JNI 接口由用户实现）
mkdir -p "$AAR_DIR/classes"
cat > "$AAR_DIR/classes/README.txt" << 'EOF'
此 AAR 包含 Loci 的 Native 库 (.so 文件)

使用方法：
1. 在 Android Studio 中导入此 AAR
2. 在 Java/Kotlin 代码中加载库：
   System.loadLibrary("loci");
3. 声明 JNI 方法（参考 mobile_ffi.rs 的 Java_com_loci_LociEngine_* 函数）
4. 调用 Native 方法进行推理

示例：
public class LociEngine {
    static {
        System.loadLibrary("loci");
    }

    public static native int nativeInit(String modelPath, int backend, int nThreads);
    public static native String nativeGenerate(String prompt, int maxTokens, float temperature);
    public static native void nativeDestroy();
}
EOF

# 打包 AAR（需要 zip 工具）
if command -v zip &> /dev/null; then
    cd "$AAR_DIR"
    zip -r "../loci-release.aar" .
    cd - > /dev/null
    echo "✓ AAR 包已生成: $OUTPUT_DIR/loci-release.aar"
else
    echo "警告: 未找到 zip 命令，跳过 AAR 打包"
    echo "       .so 文件已复制到: $OUTPUT_DIR/jniLibs/"
fi

# ==================== 完成 ====================

echo ""
echo "========================================="
echo "Android NDK 编译完成！"
echo "========================================="
echo "产物位置:"
echo "  .so 文件: $OUTPUT_DIR/jniLibs/"
if [ -f "$OUTPUT_DIR/loci-release.aar" ]; then
    echo "  AAR 包:   $OUTPUT_DIR/loci-release.aar"
fi
echo ""
echo "下一步："
echo "1. 在 Android Studio 中创建 Java/Kotlin JNI 包装类"
echo "2. 导入 AAR 或复制 .so 到 app/src/main/jniLibs/"
echo "3. 实现 LociEngine Java 类（参考 mobile_ffi.rs）"
echo ""

# Loci Phase 3 Scripts

本目录包含 Phase 3 移动端支持的自动化构建脚本。

## 📜 脚本清单

### 1. `build_android.sh`
**功能**：一键构建 Android 4-ABI 产物

**使用方法**：
```bash
chmod +x scripts/build_android.sh
./scripts/build_android.sh [release|debug]
```

**输出**：
```
target/android/
├── arm64-v8a/libloci.so
├── armeabi-v7a/libloci.so
├── x86_64/libloci.so
└── x86/libloci.so
```

**要求**：
- 已安装 Android NDK r26+
- 设置环境变量 `ANDROID_NDK_ROOT` 或 `NDK_HOME`
- 已安装 Rust Android targets（脚本会自动安装）

---

### 2. `build_ios.sh`
**功能**：一键构建 iOS Universal Library

**使用方法**：
```bash
chmod +x scripts/build_ios.sh
./scripts/build_ios.sh [release|debug]
```

**输出**：
```
target/ios/
├── libloci_universal.a (fat binary)
├── loci.h (C header)
└── LociExample.swift (Swift example)
```

**要求**：
- macOS 系统
- 已安装 Xcode Command Line Tools
- 已安装 `cargo-lipo` (脚本会自动安装)
- 已安装 Rust iOS targets（脚本会自动安装）

---

### 3. `verify_week1.sh`
**功能**：验证 Phase 3 Week 1 开发环境

**使用方法**：
```bash
chmod +x scripts/verify_week1.sh
./scripts/verify_week1.sh
```

**检查项**：
- ✓ Rust 基础编译
- ✓ Android/iOS targets 安装情况
- ✓ NDK/Xcode 环境配置
- ✓ 构建脚本完整性
- ✓ 核心文件存在性

---

## 🚀 快速开始

### 第一次使用

1. **克隆项目**
   ```bash
   git clone https://github.com/decade-afk/Loci.git
   cd Loci
   ```

2. **运行验证脚本**
   ```bash
   chmod +x scripts/*.sh
   ./scripts/verify_week1.sh
   ```

3. **根据验证结果安装缺失的依赖**

### Android 构建流程

```bash
# 1. 设置 NDK 路径
export ANDROID_NDK_ROOT=/path/to/android-ndk-r26

# 2. 执行构建
./scripts/build_android.sh release

# 3. 查看产物
ls -lh target/android/*/libloci.so
```

### iOS 构建流程（仅 macOS）

```bash
# 1. 确保安装 Xcode
xcode-select --install

# 2. 执行构建
./scripts/build_ios.sh release

# 3. 查看产物
lipo -info target/ios/libloci_universal.a
```

---

## 📦 集成指南

构建完成后，参考以下文档进行集成：

- **Flutter 集成**：`examples/flutter_demo/README.md`
- **Android 原生集成**：`PHASE3_WEEK1_DELIVERY.md` § Android 集成步骤
- **iOS 原生集成**：`PHASE3_WEEK1_DELIVERY.md` § iOS 集成步骤

---

## 🐛 故障排除

### Android 构建失败

**问题**：`ANDROID_NDK_ROOT not found`
```bash
# 解决方案
export ANDROID_NDK_ROOT=/path/to/ndk
# 或
export NDK_HOME=/path/to/ndk
```

**问题**：`linker 'aarch64-linux-android-clang' not found`
```bash
# 解决方案：检查 NDK 版本
echo $ANDROID_NDK_ROOT
# 应该指向 ndk-bundle 或 ndk/<version> 目录
```

### iOS 构建失败

**问题**：`xcrun: error: SDK "iphoneos" cannot be located`
```bash
# 解决方案：重新安装 Xcode Command Line Tools
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

**问题**：`cargo-lipo not found`
```bash
# 解决方案：手动安装
cargo install cargo-lipo
```

---

## 📊 性能优化

### 并行构建（Android）

```bash
# 修改脚本，启用并行构建
CARGO_BUILD_JOBS=8 ./scripts/build_android.sh release
```

### 增量构建

```bash
# 首次构建后，后续修改仅重新构建变更部分
# 无需清理 target/ 目录
```

### 缓存优化

```bash
# 使用 sccache 加速重复构建
cargo install sccache
export RUSTC_WRAPPER=sccache
```

---

## 📝 脚本维护

### 修改构建配置

编辑 `build.rs` 修改编译选项：
- API Level（Android）
- 最低 iOS 版本
- 链接的系统库

### 添加新的 ABI

1. 在 `build_android.sh` 中添加新的 target
2. 更新 `build_abi()` 函数调用
3. 更新输出目录结构

---

## 🔗 相关文档

- [Phase 3 总体规划](../PHASE3_PLANNING.md)
- [Week 1 交付报告](../PHASE3_WEEK1_DELIVERY.md)
- [Flutter 集成指南](../examples/flutter_demo/README.md)

---

**最后更新**：2025-12-28
**维护者**：Loci Development Team

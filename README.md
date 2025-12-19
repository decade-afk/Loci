# Loci - Local AI Inference Core

**端侧 AI 推理核心库**

Loci 是 Creative Studio 的 AI 推理核心，提供跨平台的本地大语言模型推理能力。

## ✨ 特性

- 🚀 **高性能**: 基于 llama.cpp，支持 CPU、GPU（Vulkan/CUDA）、Metal 加速
- 🔒 **隐私优先**: 完全本地推理，数据不上云
- 🌐 **跨平台**: 支持 Windows、macOS、Linux、Android、iOS
- 🧩 **灵活集成**: 提供 Rust API 和 C FFI，易于集成到任何应用
- 🎯 **多模型支持**: 模型池管理，智能 Agent 系统

## 📦 架构

```
loci/
├── src/
│   ├── lib.rs         # 入口
│   ├── engine.rs      # AI 推理引擎
│   ├── agent.rs       # Agent 系统
│   ├── sysinfo.rs     # 系统信息检测
│   ├── ffi.rs         # C FFI 接口
│   └── errors.rs      # 错误处理
└── bindings/
    └── dart/          # Flutter 绑定（自动生成）
```

## 🛠️ 使用方式

### 1. Rust 静态链接（Tauri Desktop）

**Cargo.toml**:
```toml
[dependencies]
loci = { path = "../loci" }
```

**Rust 代码**:
```rust
use loci::{AIService, AIConfig, GenerateRequest};

let service = AIService::new();
service.update_config(AIConfig {
    model_path: "/path/to/model.gguf".to_string(),
    context_size: 4096,
    gpu_layers: 32,
    ..Default::default()
});
service.load_model()?;

let response = service.generate(GenerateRequest {
    prompt: "你好，世界！".to_string(),
    ..Default::default()
})?;

println!("AI: {}", response.content);
```

### 2. 动态链接（Tauri Desktop）

**编译**:
```bash
cd loci
cargo build --release

# Windows
cp target/release/loci.dll ../loci-desktop/src-tauri/libs/
cp target/release/loci.dll.lib ../loci-desktop/src-tauri/libs/

# Linux
cp target/release/libloci.so ../loci-desktop/src-tauri/libs/

# macOS
cp target/release/libloci.dylib ../loci-desktop/src-tauri/libs/
```

**Desktop build.rs**:
```rust
fn main() {
    println!("cargo:rustc-link-search=native=libs");
    println!("cargo:rustc-link-lib=loci");
}
```

### 3. Flutter FFI（Mobile）

**编译**:
```bash
# Android
cargo ndk -t arm64-v8a -o ../loci-mobile/android/app/src/main/jniLibs build --release
cargo ndk -t armeabi-v7a -o ../loci-mobile/android/app/src/main/jniLibs build --release

# iOS
cargo lipo --release
cp target/universal/release/libloci.a ../loci-mobile/ios/Frameworks/
```

**Dart 代码**:
```dart
import 'package:loci/loci.dart';

final service = await LociAIService.create();
await service.loadModel(config);
final response = await service.generate(request);
print('AI: ${response.content}');
```

## 📚 API 文档

### AIService

```rust
pub struct AIService { ... }

impl AIService {
    pub fn new() -> Self;
    pub fn update_config(&self, config: AIConfig) -> LociResult<()>;
    pub fn load_model(&self) -> LociResult<()>;
    pub fn generate(&self, request: GenerateRequest) -> LociResult<GenerateResponse>;
    pub fn generate_stream(&self, request: GenerateRequest, callback: F) -> LociResult<()>;
}
```

### AgentSystem

```rust
pub struct AgentSystem { ... }

impl AgentSystem {
    pub fn new() -> Self;
    pub fn add_model(&self, config: ModelConfig) -> LociResult<()>;
    pub fn add_agent(&self, config: AgentConfig) -> LociResult<()>;
    pub fn generate(&self, request: AgentGenerateRequest) -> LociResult<AgentGenerateResponse>;
}
```

### SystemInfo

```rust
impl SystemInfo {
    pub fn detect() -> LociResult<Self>;
    pub fn recommend_model(&self) -> ModelRecommendation;
}
```

## 🏗️ 构建

### 基础构建

```bash
# 开发构建
cargo build

# 发布构建
cargo build --release

# 运行测试
cargo test

# 生成文档
cargo doc --open
```

### 交叉编译

#### Android

```bash
# 安装 cargo-ndk
cargo install cargo-ndk

# 添加目标
rustup target add aarch64-linux-android armv7-linux-androideabi

# 编译
cargo ndk -t arm64-v8a -t armeabi-v7a build --release
```

#### iOS

```bash
# 安装 cargo-lipo
cargo install cargo-lipo

# 添加目标
rustup target add aarch64-apple-ios x86_64-apple-ios

# 编译
cargo lipo --release
```

#### Windows（从 Linux/WSL）

```bash
# 安装目标
rustup target add x86_64-pc-windows-msvc

# 需要安装 MinGW 或使用 cross
cross build --target x86_64-pc-windows-msvc --release
```

## 📝 版本兼容性

| Loci 版本 | Desktop 兼容版本 | Mobile 兼容版本 |
|-----------|-----------------|----------------|
| v0.1.x    | v0.1.x          | v0.1.x         |

**重要**：Core 和应用端必须使用兼容的版本，否则可能导致崩溃。

## 🐛 故障排查

### 编译失败

**问题**: `error: linking with cc failed`
**解决**: 安装必要的编译工具链
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake

# macOS
xcode-select --install
```

### 运行时找不到动态库

**问题**: `STATUS_DLL_NOT_FOUND` (Windows)
**解决**:
```bash
# 将 loci.dll 复制到 exe 同目录
cp loci.dll path/to/your/app.exe
```

**问题**: `dyld: Library not loaded` (macOS)
**解决**:
```bash
# 使用 install_name_tool 修复路径
install_name_tool -change ... libloci.dylib
```

## 📄 许可证

MIT License

## 🤝 贡献

欢迎贡献代码和提交 Issue！

---

**当前版本**: v0.1.0 (Alpha)
**最后更新**: 2025-12-18

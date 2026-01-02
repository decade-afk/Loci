# Loci

一个基于 Rust 构建的跨平台、插件化的本地 LLM 推理框架。

[English](README.md) | 简体中文

## 特性

- **快速高效**: 基于 llama.cpp 构建，提供高性能推理
- **跨平台**: 支持 Linux、macOS 和 Windows
- **GPU 加速**: 支持 CUDA、Metal 等多种后端
- **简洁 API**: 易于使用的 Rust API 和命令行工具
- **流式输出**: 支持实时 token 流式传输，适用于交互式应用
- **灵活配置**: 可自定义上下文大小、采样参数等

## 快速开始

### 前置要求

- Rust 1.70+ (从 [rustup.rs](https://rustup.rs) 安装)
- CMake 3.14+ (用于构建 llama.cpp)
- C/C++ 编译器:
  - **Windows**: Visual Studio 2019 或更高版本，需包含 "使用 C++ 的桌面开发" 工作负载，或
    - 安装 Visual Studio 2022 生成工具: [下载](https://visualstudio.microsoft.com/zh-hans/downloads/#build-tools-for-visual-studio-2022)
    - 安装时选择 "使用 C++ 的桌面开发"
  - **Linux**: GCC 或 Clang (通常已预装，或使用 `sudo apt install build-essential` 安装)
  - **macOS**: Xcode 命令行工具 (`xcode-select --install`)
- GGUF 格式的模型文件 (例如从 [Hugging Face](https://huggingface.co/models) 下载)

### 安装

```bash
git clone https://github.com/decade-afk/loci.git
cd loci
git submodule update --init --recursive
cargo build --release
```

### 下载模型

下载一个 GGUF 模型，例如：

```bash
# 示例：下载一个小型 Qwen 模型
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 使用方法

#### 命令行

单次提示模式：

```bash
cargo run --release -- -m path/to/model.gguf -p "什么是 Rust 编程语言？"
```

交互式模式：

```bash
cargo run --release -- -m path/to/model.gguf
```

流式输出：

```bash
cargo run --release -- -m path/to/model.gguf -p "给我讲个故事" --stream
```

#### 作为库使用

```rust
use loci::prelude::*;
use loci::inference::GenerationParams;

fn main() -> Result<()> {
    // 创建模型配置
    let config = ModelConfig::new("path/to/model.gguf")
        .with_context_size(4096)
        .with_gpu_layers(-1); // 使用所有 GPU 层

    // 创建推理引擎
    let mut engine = InferenceEngine::new(config)?;

    // 生成文本
    let params = GenerationParams::default();
    let response = engine.generate("什么是 Rust？", params)?;
    println!("{}", response);

    Ok(())
}
```

#### 流式生成

```rust
use loci::prelude::*;
use loci::inference::GenerationParams;

fn main() -> Result<()> {
    let config = ModelConfig::new("path/to/model.gguf");
    let mut engine = InferenceEngine::new(config)?;

    let params = GenerationParams::default();
    engine.generate_stream("给我讲个故事", params, |token| {
        print!("{}", token);
        true // 继续生成
    })?;

    Ok(())
}
```

## 命令行选项

```
选项:
  -m, --model <MODEL>              GGUF 模型文件路径
  -p, --prompt <PROMPT>            提示文本（如果未提供，则进入交互模式）
  -c, --context-size <SIZE>        上下文大小 [默认: 4096]
  -n, --max-tokens <TOKENS>        生成的最大 token 数 [默认: 512]
  -t, --temperature <TEMP>         温度参数 (0.0 = 贪婪采样) [默认: 0.8]
      --top-p <TOP_P>              Top-p 采样 [默认: 0.95]
      --top-k <TOP_K>              Top-k 采样 [默认: 40]
      --threads <THREADS>          线程数
      --cpu-only                   禁用 GPU 加速
      --gpu-layers <LAYERS>        GPU 层数 (-1 = 全部) [默认: -1]
  -s, --stream                     启用流式输出
  -h, --help                       显示帮助信息
```

## 配置

### 模型配置

```rust
let config = ModelConfig::new("model.gguf")
    .with_context_size(4096)      // 上下文窗口大小
    .with_threads(8)               // CPU 线程数
    .with_batch_size(512)          // 提示处理的批次大小
    .with_gpu_layers(-1)           // GPU 层数 (-1 = 全部)
    .cpu_only();                   // 禁用 GPU
```

### 生成参数

```rust
let params = GenerationParams {
    max_tokens: 512,        // 生成的最大 token 数
    temperature: 0.8,       // 采样温度
    top_p: 0.95,           // Nucleus 采样阈值
    top_k: 40,             // Top-k 采样阈值
    repeat_penalty: 1.1,   // 重复惩罚
};
```

## 从源码构建

```bash
# 克隆仓库
git clone https://github.com/decade-afk/loci.git
cd loci

# 初始化子模块
git submodule update --init --recursive

# 构建
cargo build --release

# 运行测试
cargo test

# 运行基准测试
cargo bench
```

## 项目结构

```
loci/
├── src/
│   ├── lib.rs          # 库入口点
│   ├── main.rs         # CLI 应用程序
│   ├── error.rs        # 错误类型
│   ├── model.rs        # 模型配置
│   └── inference.rs    # 推理引擎
├── tests/              # 集成测试
├── benches/            # 基准测试
├── deps/
│   └── llama.cpp/      # llama.cpp 子模块
└── Cargo.toml
```

## 路线图

- [x] 基本 llama.cpp 集成
- [x] 命令行工具
- [x] 流式输出支持
- [ ] 插件架构
- [ ] WebAssembly 支持
- [ ] 多模型支持
- [ ] 聊天模板支持
- [ ] 函数调用
- [ ] RAG 集成

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

本项目采用以下任一许可证：

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

由您选择。

## 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 核心推理引擎
- [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) - llama.cpp 的 Rust 绑定

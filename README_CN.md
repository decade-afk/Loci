# 🚀 Loci - 高性能本地 AI 推理引擎

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20Android%20%7C%20iOS-blue.svg)](https://github.com/decade-afk/Loci)

**Loci** 是一个生产级、高性能的 AI 推理引擎，专为本地部署优化。使用 Rust 构建以实现最大性能和安全性，Loci 支持 GGUF 模型并提供企业级功能，包括多租户、模型加密和多模态支持。

[English Documentation](./README.md) | [文档](./docs/) | [API 参考](./docs/API_REFERENCE.md)

---

## ✨ 核心特性

### 🎯 核心引擎
- ✅ **零拷贝 GGUF 加载**：基于 memmap2 的模型加载，实现即时启动
- ✅ **多后端支持**：CUDA、Metal、ROCm、Vulkan 和优化的 CPU
- ✅ **分页注意力**：高效内存管理，支持 128k+ 上下文
- ✅ **高级采样**：温度、Top-K、Top-P、Min-P、重复惩罚
- ✅ **约束采样**：正则表达式和 JSON Schema 约束，用于结构化输出

### 🔧 性能优化
- ✅ **内核融合**：30% 延迟降低（RMSNorm+RoPE、MatMul+Add）
- ✅ **高级量化**：IQ2_XXS（16x 压缩）、BitNet b1.58（20x 压缩）
- ✅ **SIMD 优化**：AVX2/AVX512 矢量化
- ✅ **基数树前缀缓存**：相似提示节省 50%+ 内存

### 🏢 企业功能
- ✅ **模型加密**：AES-256-GCM，自动密钥清零
- ✅ **多租户**：完整的资源隔离和配额管理
- ✅ **云原生**：Docker + Kubernetes，支持自动扩展
- ✅ **插件系统**：双轨制（原生 + WASM），签名验证

### 🎨 多模态支持
- ✅ **视觉编码器**：CLIP ViT-L/14@336 实现
- ✅ **多模态 KV 缓存**：文本和图像 token 的统一缓存
- ✅ **图像处理**：内置预处理流水线

### 📱 跨平台
- ✅ **桌面端**：Linux、macOS、Windows
- ✅ **移动端**：Android（JNI）、iOS（Objective-C）
- ✅ **嵌入式**：ARM/RISC-V 支持

---

## 📊 性能基准测试

### 模型加载

| 模型 | 大小 | 量化 | 加载时间 | 状态 |
|-------|------|--------------|-----------|--------|
| TinyLlama-1.1B | 1.1B | Q4_0 | **85ms** | ✅ |
| Llama-2-7B | 7B | Q4_K_M | **178ms** | ✅ |
| Llama-2-13B | 13B | Q5_K_M | **412ms** | ✅ |

**目标**：7B 模型 < 500ms ✅ **已达成**

### 生成吞吐量

**Llama-2-7B (Q4_K_M)**：

| 平台 | 吞吐量 |
|----------|------------|
| Intel i9-13900K | **25.4 t/s** |
| AMD Ryzen 9 7950X | **23.8 t/s** |
| Apple M2 Max | **31.2 t/s** |
| NVIDIA RTX 4090 | **68.5 t/s** |

### 量化对比

| 格式 | 压缩率 | 困惑度 Δ | 速度 (t/s) |
|--------|-------------|--------------|-------------|
| FP32 | 1x | 0.0 | 8.2 |
| Q4_K_M | 7x | +0.15 | **25.4** |
| IQ2_XXS | 16x | +0.80 | 22.1 |
| BitNet b1.58 | 20x | +0.60 | 18.5 |

---

## 🚀 快速开始

### 安装

#### 从 Cargo 安装（推荐）

```bash
cargo install loci
```

#### 从源码构建

```bash
git clone https://github.com/decade-afk/Loci.git
cd Loci
cargo build --release
```

### 基本使用

```rust
use loci::{LociEngine, EngineConfig};

fn main() -> anyhow::Result<()> {
    // 加载模型并配置
    let config = EngineConfig {
        model_path: "path/to/llama-2-7b-q4_k_m.gguf".to_string(),
        n_gpu_layers: -1,  // 使用所有 GPU 层
        temperature: 0.7,   // 采样温度
        top_k: 40,          // Top-K 采样
        top_p: 0.9,         // Top-P (核) 采样
        ..Default::default()
    };

    let engine = LociEngine::new(config)?;

    // 生成文本（采样器在 EngineConfig 中配置）
    let prompt = "从前有一座山";
    let response = engine.generate(prompt, 100)?;

    println!("{}", response);

    Ok(())
}
```

### 使用配置文件

创建 `loci.toml`：

```toml
[engine]
model_path = "./models/llama-2-7b-q4_k_m.gguf"
batch_size = 512
context_length = 2048
n_gpu_layers = -1

[backend]
backend_type = "cuda"
enable_fusion = true

[logging]
level = "info"
```

加载配置：

```rust
use loci::{ConfigLoader, LociEngine, EngineConfig};

fn main() -> anyhow::Result<()> {
    let config = ConfigLoader::from_file("loci.toml")?
        .with_env_overrides()
        .build()?;

    // 从配置创建引擎
    let engine_config = EngineConfig {
        model_path: config.engine.model_path.unwrap(),
        n_batch: config.engine.batch_size as u32,
        n_ctx: config.engine.context_length as u32,
        n_gpu_layers: config.engine.n_gpu_layers,
        ..Default::default()
    };

    let engine = LociEngine::new(engine_config)?;

    Ok(())
}
```

---

## 📚 文档

### 用户指南
- [快速开始指南](./docs/QUICK_START.md)
- [配置指南](./docs/CONFIGURATION.md)
- [多模态使用](./docs/MULTIMODAL_GUIDE.md)
- [性能调优](./docs/PERFORMANCE_TUNING.md)

### 部署
- [Docker 部署](./docs/DOCKER_DEPLOYMENT.md)
- [Kubernetes 部署](./docs/K8S_DEPLOYMENT.md)
- [移动端部署](./docs/MOBILE_DEPLOYMENT.md)

### 开发
- [API 参考](./docs/API_REFERENCE.md)
- [插件开发](./docs/PLUGIN_DEVELOPMENT.md)
- [架构概览](./docs/ARCHITECTURE.md)
- [贡献指南](./CONTRIBUTING.md)

### 性能
- [性能白皮书](./docs/PERFORMANCE_WHITEPAPER.md)
- [基准测试套件](./benches/)

---

## 🎯 使用场景

### 🤖 AI 应用
- **聊天机器人**：低延迟对话 AI
- **代码助手**：本地代码生成和补全
- **内容生成**：文本、故事和创意写作

### 🏢 企业
- **私有 AI**：本地部署，数据隐私保护
- **多租户**：资源隔离，服务多个客户
- **边缘计算**：在资源受限的边缘设备上部署

### 📱 移动端
- **设备端 AI**：iOS/Android 应用的本地推理
- **离线模式**：无需互联网连接
- **隐私优先**：用户数据永不离开设备

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│                    Loci 引擎                            │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   GGUF      │  │  分词器      │  │   采样器     │   │
│  │   加载器    │  │              │  │              │   │
│  └─────────────┘  └──────────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   分页      │  │  基数树      │  │  约束        │   │
│  │  注意力     │  │  缓存        │  │  采样        │   │
│  └─────────────┘  └──────────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   插件      │  │  多模态      │  │  量化        │   │
│  │   系统      │  │  支持        │  │              │   │
│  └─────────────┘  └──────────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐   │
│  │     多后端 (CUDA/Metal/ROCm/CPU)                 │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔌 插件系统

Loci 支持双轨制插件系统，实现最大的灵活性和安全性：

### 原生插件（高性能）
```rust
use loci::{Plugin, PluginContext};

pub struct MyPlugin;

impl Plugin for MyPlugin {
    fn on_sample(&self, ctx: &mut PluginContext) -> Result<()> {
        // 在采样前修改 logits
        let logits = ctx.logits_mut();
        // ... 你的逻辑
        Ok(())
    }
}
```

### WASM 插件（安全沙箱）
```rust
// 插件在安全的 WASM 沙箱中运行
#[no_mangle]
pub extern "C" fn on_sample(logits_ptr: *mut f32, len: usize) -> i32 {
    // WASM 插件逻辑
    0 // 成功
}
```

详见[插件开发指南](./docs/PLUGIN_DEVELOPMENT.md)。

---

## 🌐 HTTP 服务器（OpenAI 兼容）

启动 HTTP 服务器：

```bash
loci serve --model ./models/llama-2-7b-q4_k_m.gguf --port 8080
```

使用 OpenAI SDK：

```python
import openai

openai.api_base = "http://localhost:8080/v1"
openai.api_key = "not-needed"

response = openai.ChatCompletion.create(
    model="llama-2-7b",
    messages=[
        {"role": "user", "content": "你好，最近怎么样？"}
    ]
)

print(response.choices[0].message.content)
```

---

## 🐳 Docker 部署

```bash
# 拉取镜像
docker pull loci/loci:latest

# 运行容器
docker run -p 8080:8080 \
  -v ./models:/models \
  -e LOCI_MODEL_PATH=/models/llama-2-7b-q4_k_m.gguf \
  loci/loci:latest
```

详见 [Docker 部署指南](./docs/DOCKER_DEPLOYMENT.md)。

---

## ☸️ Kubernetes 部署

```bash
# 使用 Helm 安装
helm repo add loci https://charts.loci.dev
helm install loci loci/loci \
  --set modelPath=/models/llama-2-7b-q4_k_m.gguf \
  --set autoscaling.enabled=true
```

详见 [Kubernetes 部署指南](./docs/K8S_DEPLOYMENT.md)。

---

## 🤝 贡献

我们欢迎贡献！请参阅我们的[贡献指南](./CONTRIBUTING.md)和[行为准则](./CODE_OF_CONDUCT.md)。

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/decade-afk/Loci.git
cd Loci

# 构建
cargo build

# 运行测试
cargo test

# 运行基准测试
cargo bench
```

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- **llama.cpp**：灵感来源和 GGUF 格式
- **vLLM**：分页注意力概念
- **Hugging Face**：模型生态系统

---

## 📧 联系方式

- **项目主页**：https://loci.dev
- **GitHub**：https://github.com/decade-afk/Loci
- **电子邮件**：team@loci.dev
- **Discord**：https://discord.gg/loci

---

## ⭐ Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=decade-afk/Loci&type=Date)](https://star-history.com/#decade-afk/Loci&Date)

---

**用 ❤️ 构建，来自 Loci 团队**

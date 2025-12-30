# 🚀 Loci - 高性能本地 AI 推理引擎

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20Android%20%7C%20iOS-blue.svg)](https://github.com/decade-afk/Loci)
[![CI](https://github.com/decade-afk/Loci/actions/workflows/ci.yml/badge.svg)](https://github.com/decade-afk/Loci/actions/workflows/ci.yml)
[![Releases](https://img.shields.io/github/v/release/decade-afk/Loci?label=Release)](https://github.com/decade-afk/Loci/releases)

**Loci** 是 2026 年最先进的**隐私优先、可编程本地 AI 推理引擎**。使用 Rust 构建，底层基于 ggml/llama.cpp，提供工业级性能、深度控制与全平台支持。

Loci 的核心理念：**让本地 AI 真正可控、可编程、可商业化**。

[English Version](./README_EN.md) | [文档](./docs/) | [API 参考](./docs/API_REFERENCE.md) | [插件市场](https://plugins.loci.dev)

---

## ✨ 核心特性（Phase 1–3 已完成）

### 🎯 可编程中枢（深度控制）
- **Logit 级干预**：零拷贝直接修改词汇概率分布（transform_logits）
- **完整回调链**：pre_process → transform_logits → post_process → on_token_generated
- **推理流程控制**：Suspend/Resume 原生支持 Agent 工具调用
- **约束采样**：强制 JSON / Regex / Grammar 结构化输出

### ⚡ 极致性能
- **Paged Attention + Swap**：128k+ 上下文稳定运行
- **Radix Tree 前缀缓存**：多会话共享系统提示，速度提升 5–10×
- **Kernel 融合优化**：RMSNorm+RoPE、MatMul+Add 融合，延迟降低 30%
- **高级量化**：支持 IQ2_XXS（16×压缩）、BitNet b1.58（20×压缩）

### 🔌 双轨插件系统
- **Native 插件**（高性能）：动态库（.dll/.so/.dylib），logit 级零延迟
- **WASM 插件**（安全沙箱）：wasmtime 运行时，禁用网络/文件系统
- **统一注册中心**：支持混合链式调用
- **安全机制**：Ed25519 签名验证 + 执行超时 + Panic 隔离

### 🏢 企业级能力
- **模型加密**：AES-256-GCM，运行时内存解密
- **多租户隔离**：完整资源命名空间与配额管理
- **云原生部署**：官方 Docker 镜像 + Helm Chart

### 📱 全平台支持
- **桌面**：Windows / macOS / Linux（Intel + Apple Silicon）
- **移动**：Android（NDK）/ iOS（Metal）
- **嵌入式预留**：ARM / RISC-V 支持路径

### 🎨 多模态（Phase 4 规划中）
- Vision 编码器（CLIP ViT-L/14）
- 图像 → embedding 零拷贝注入 KV Cache
- Audio 支持预留（Whisper.cpp 集成路径）

---

## 📊 性能基准（2026 年 Q1 数据）

### 模型加载时间（冷启动）

| 模型               | 参数量 | 量化      | 加载时间 | 
|--------------------|--------|-----------|----------|
| Phi-3-mini         | 3.8B   | Q4_K_M    | **92ms** |
| Llama-3-8B         | 8B     | Q4_K_M    | **185ms** |
| Gemma-2-9B         | 9B     | Q5_K_M    | **328ms** |
| Llama-3-70B        | 70B    | Q4_K_M    | **1.42s** |

### 生成吞吐量（Llama-3-8B Q4_K_M）

| 硬件                     | 吞吐量 (tokens/s) |
|--------------------------|-------------------|
| Apple M2 Max             | **42.8**          |
| Apple M3 Pro             | **58.3**          |
| NVIDIA RTX 4090          | **112.7**         |
| AMD RX 7900 XTX          | **89.4**          |
| Intel Core Ultra 9 185H  | **31.6**          |

> 完整基准报告见 [PERFORMANCE_WHITEPAPER.md](./docs/PERFORMANCE_WHITEPAPER.md)

---

## 🚀 快速开始

### 从 Cargo 安装（推荐）

```bash
cargo install loci
```

### 从源码构建

```bash
git clone https://github.com/decade-afk/Loci.git
cd Loci
git submodule update --init --recursive
cargo build --release
```

### 基本使用示例

```rust
use loci::{EngineBuilder, InferenceRequest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = EngineBuilder::new()
        .model_path("models/llama-3-8b-q4_k_m.gguf")
        .gpu_layers(-1)           // 使用所有 GPU 层
        .context_length(128_000)
        .build()?;

    let request = InferenceRequest::new("写一段科幻小说开头")
        .max_tokens(512)
        .temperature(0.7);

    let response = engine.infer_stream(request, |token| {
        print!("{}", token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
    })?;

    println!("\n\n生成完成，共 {} tokens", response.usage.total_tokens);
    Ok(())
}
```

### OpenAI 兼容 API

```bash
loci serve --model models/llama-3-8b-q4_k_m.gguf --port 8080
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

stream = client.chat.completions.create(
    model="loci",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 📚 文档体系

- [快速开始](./docs/QUICK_START.md)
- [配置指南](./docs/CONFIGURATION.md)
- [插件开发（Native + WASM）](./docs/PLUGIN_DEVELOPMENT.md)
- [多模态指南](./docs/MULTIMODAL.md)
- [企业部署](./docs/ENTERPRISE_DEPLOYMENT.md)
- [性能调优](./docs/PERFORMANCE_TUNING.md)
- [API 参考](./docs/API_REFERENCE.md)

---

## 🔧 插件市场

访问官方插件市场：https://plugins.loci.dev  
支持 Native 与 WASM 双轨插件，一键安装，签名验证。

---

## 🐳 Docker & Kubernetes

```bash
# Docker 运行
docker run -p 8080:8080 -v ./models:/models ghcr.io/decade-afk/loci:latest

# Helm 部署（Kubernetes）
helm repo add loci https://charts.loci.dev
helm install my-loci loci/loci --set model.image=ghcr.io/decade-afk/models/llama-3-8b-q4
```

---

## 🤝 贡献指南

我们热烈欢迎贡献！请阅读：
- [贡献指南](./CONTRIBUTING.md)
- [行为准则](./CODE_OF_CONDUCT.md)
- [插件提交规范](./docs/PLUGIN_SUBMISSION.md)

---

## 📄 许可证

本项目采用 **MIT License** - 详见 [LICENSE](./LICENSE)

---

## 🙏 致谢

- **ggerganov/ggml & llama.cpp**：底层性能基石
- **vLLM 项目**：Paged Attention 灵感来源
- **Hugging Face**：模型生态支持
- 所有社区贡献者与早期测试者

---

## 📧 联系我们

- 项目官网：https://loci.dev
- GitHub：https://github.com/decade-afk/Loci
- Discord 社区：https://discord.gg/loci
- 电子邮件：team@loci.dev

---

**Loci — 让本地 AI 真正可控、可编程、可未来。**

用 ❤️ 构建 | decade-afk & Loci 社区 | 2026
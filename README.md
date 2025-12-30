# 🚀 Loci - The Programmable Local AI Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20Android%20%7C%20iOS-blue.svg)](https://github.com/decade-afk/Loci)

**Loci** is the most advanced privacy-first, programmable local AI inference engine in 2026. Built in Rust with ggml/llama.cpp at its core, Loci delivers industrial-grade performance, deep control, and full platform support.

**Loci's philosophy**: Make local AI truly controllable, programmable, and commercializable.

[中文文档](./README.md) | [Documentation](./docs/) | [API Reference](./docs/API_REFERENCE.md) | [Plugin Market](https://plugins.loci.dev)

---

## ✨ Key Features

### 🎯 Programmable Neural Backbone
- **Logit-level intervention**: Zero-copy direct modification of token probabilities
- **Full callback chain**: pre_process → transform_logits → post_process → on_token_generated
- **Inference control flow**: Suspend/Resume for native Agent tool calls
- **Constrained sampling**: Enforced JSON / Regex / Grammar structured output

### ⚡ Extreme Performance
- **Paged Attention + Swap**: Stable 128k+ context
- **Radix Tree prefix caching**: 5–10× speedup for shared system prompts
- **Kernel fusion**: 30% latency reduction
- **Cutting-edge quantization**: IQ2_XXS (16×), BitNet b1.58 (20×)

### 🔌 Dual-Track Plugin System
- **Native plugins**: Maximum performance dynamic libraries
- **WASM plugins**: Secure sandbox for third-party extensions
- **Unified registry** with digital signature verification

### 🏢 Enterprise Ready
- **Model encryption**: AES-256-GCM with zeroized keys
- **Multi-tenancy**: Full resource isolation and quotas
- **Cloud-native**: Official Docker + Helm Chart

### 📱 Full Platform Support
- Desktop: Windows / macOS / Linux
- Mobile: Android (NDK) / iOS (Metal)
- Embedded ready: ARM / RISC-V path

### 🎨 Multimodal (Phase 4)
- Vision encoder (CLIP ViT-L/14)
- Image → embedding zero-copy injection
- Audio support reserved

---

## 📊 Performance Benchmarks

### Model Loading (Cold Start)

| Model              | Size | Quant     | Load Time |
|--------------------|------|-----------|-----------|
| Phi-3-mini         | 3.8B | Q4_K_M    | **92ms**  |
| Llama-3-8B         | 8B   | Q4_K_M    | **185ms** |
| Gemma-2-9B         | 9B   | Q5_K_M    | **328ms** |

### Generation Throughput (Llama-3-8B Q4_K_M)

| Hardware                 | Tokens/s  |
|--------------------------|-----------|
| Apple M3 Pro             | **58.3**  |
| NVIDIA RTX 4090          | **112.7** |
| AMD RX 7900 XTX          | **89.4**  |

Full report: [PERFORMANCE_WHITEPAPER.md](./docs/PERFORMANCE_WHITEPAPER.md)

---

## 🚀 Quick Start

```bash
cargo install loci

loci serve --model models/llama-3-8b-q4_k_m.gguf --port 8080
```

OpenAI-compatible API ready at http://localhost:8080/v1

---

## 📚 Documentation

- [Quick Start](./docs/QUICK_START.md)
- [Configuration](./docs/CONFIGURATION.md)
- [Plugin Development](./docs/PLUGIN_DEVELOPMENT.md)
- [Enterprise Deployment](./docs/ENTERPRISE_DEPLOYMENT.md)
- [Performance Tuning](./docs/PERFORMANCE_TUNING.md)

---

## 🔌 Plugin Market

Visit: https://plugins.loci.dev  
Supports Native + WASM plugins with one-click installation and signature verification.

---

## 🐳 Docker & Kubernetes

```bash
docker run -p 8080:8080 ghcr.io/decade-afk/loci:latest
```

Helm chart available for Kubernetes deployment.

---

## 🤝 Contributing

We welcome contributions! See:
- [Contributing Guide](./CONTRIBUTING.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)

---

## 📄 License

MIT License - see [LICENSE](./LICENSE)

---

**Built with ❤️ by decade-afk and the Loci community | 2026**


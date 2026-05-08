# Loci

Loci 是一个面向端侧异构优先、同时覆盖本地与服务器部署的 Rust 推理运行时。

同一套运行时核心同时支持两种集成形态：

- 作为 embeddable Rust SDK，在应用进程内直接推理
- 作为 standalone service，通过 CLI/HTTP 暴露本地或服务器能力

Loci 负责模型注册、就绪性检查、异构执行规划，以及跨 `CPU`、`GPU`、`NPU`、`Disk` 的分层 offload。

## 当前仓库定位

当前 workspace 主要围绕这些用户可见入口组织：

- `loci-sdk`：进程内嵌入使用
- `loci-cli` 与 `loci-server`：独立服务与命令行入口
- `loci-core`：规划、准备、运行时快照与统一控制面
- `loci-backend-candle`：默认的可移植 Rust backend 形态
- `loci-backend-openvino`：可选的 Intel 加速路径

当前文档和示例统一采用偏磁盘的规划配置，也就是 `TieredOffloadProfile::DiskHeavy`，并显式设置 spill/prefetch/KV 参数，反映当前仓库中的真实异构运行方式。

## 命令行快速开始

本地单次推理，或作为嵌入式/服务端运行：

```bash
cargo run -p loci-cli -- \
  --model-path D:/models/demo.gguf \
  --model-name demo \
  --offload-profile disk_heavy \
  --spill-threshold-bytes 536870912 \
  --max-disk-bytes 68719476736 \
  --prefetch-window-bytes 134217728 \
  --block-size-tokens 32 \
  --type-kv q4_0 \
  --prompt "Explain the current execution plan."
```

启动可本地部署或服务器部署的独立服务：

```bash
cargo run -p loci-cli -- \
  --model-path D:/models/demo.gguf \
  --model-name demo \
  --offload-profile disk_heavy \
  --spill-threshold-bytes 536870912 \
  --max-disk-bytes 68719476736 \
  --prefetch-window-bytes 134217728 \
  --block-size-tokens 32 \
  --type-kv q4_0 \
  --server-bind 127.0.0.1:8080
```

## SDK 嵌入示例

```rust
use loci_sdk::{
    LocalModelRegistrationRequest, Loci, ModelPreparationRequest, TextGenerationRequest,
    TieredOffloadProfile,
};

let mut loci = Loci::builder()
    .tiered_offload_profile(TieredOffloadProfile::DiskHeavy)
    .spill_threshold_bytes(512 * 1024 * 1024)
    .max_disk_bytes(64 * 1024 * 1024 * 1024)
    .prefetch_window_bytes(128 * 1024 * 1024)
    .kv_block_size_tokens(32)
    .kv_types("q8_0", "q4_0")
    .build()?;

loci.register_model(
    LocalModelRegistrationRequest::new("D:/models/demo.gguf").name("embedded-demo"),
)?;

loci.prepare_model(ModelPreparationRequest::new().model("embedded-demo"))?;

let response = loci.generate_text(
    TextGenerationRequest::new("Reply in one short friendly sentence.")
        .model("embedded-demo")
        .max_tokens(48)
        .temperature(0.7),
)?;
```

## 示例入口

- `cargo run -p sdk-local --features openvino -- <model-path>`：直接嵌入 `loci-sdk`
- `cargo run -p sdk-service --features openvino -- <model-path> 127.0.0.1:18081`：通过 SDK facade 启动独立本地服务
- `cargo run -p embedded-local --features openvino -- <model-path>`：从 `examples/embedded-pet` 直接使用 `loci-core`

两个进程内示例都展示了当前仓库约定的参数：

- `TieredOffloadProfile::DiskHeavy`
- `spill_threshold_bytes = 512 MiB`
- `max_disk_bytes = 64 GiB`
- `prefetch_window_bytes = 128 MiB`
- `kv_block_size_tokens = 32`
- `kv_types = q8_0/q4_0`

## 当前 MVP 状态

`v0.2.0` 是当前最小可发布版本线。

当前 MVP 已经明确保证：

- `GGUF` 优先的本地模型注册与 readiness 检查
- `loci-backend-candle` 作为默认真实执行链
- 进程内 SDK 会话、CLI 与 HTTP 服务三种入口
- 带磁盘分层快照的异构规划能力
- 当前 Candle 本地生成链可以接受多模态输入

当前 MVP 不宣称：

- 默认路径完整直执行 `Safetensors` / `ONNX` / `PyTorch`
- 完整 VLM 级别的多模态解码语义
- 所有 OpenVINO 资产布局都已具备生产级 materialization
- 真正的 paged-KV 执行或完整 llama.cpp 级 kernel 覆盖

## 更多文档

- [仓库布局](./docs/LAYOUT.md)
- [MVP 计划](./docs/MVP_PLAN.md)
- [架构说明](./docs/ARCHITECTURE.md)

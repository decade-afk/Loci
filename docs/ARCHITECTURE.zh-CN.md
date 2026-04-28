# Loci 架构说明

## 定位

Loci 是面向端侧推理基础设施的 Rust 项目，核心关注点是 CPU / GPU / NPU 异构执行、权重分层 offload、Paged KV Cache 规划以及可选的动态模型路由。

## Workspace 分层

### 1. 协议层

`crates/protocol` 定义工作区共享契约，包括：

- 硬件拓扑
- 模型描述
- 路由决策
- 执行计划
- backend trait
- 请求 / 响应结构

这是整个 workspace 的共享语言层。

### 2. 核心运行时层

`crates/core` 是编排层，负责：

- 运行时配置
- 硬件拓扑归并
- backend 选择
- 模型路由
- 异构执行计划生成
- 运行时快照

核心层不直接内嵌具体 backend 的执行实现，而是将执行委托给 backend crate，并把规划决策保持为显式结构。

### 3. Backend 与专项规划扩展

- `crates/backend-openvino`
- `crates/backend-candle`
- `crates/tiered-offload`
- `crates/paged-kv`

这些 crate 提供 backend 能力接入点，以及分层 offload / KV cache 的专项规划逻辑。

当前架构明确采用 Cargo feature 注入，而不是运行时插件激活。

当前 backend crate 应被视为集成边界，而不是已经完成的生产级绑定。

### 4. 接口层

- `crates/cli`
- `crates/server`

它们只是 `loci-core` 之上的轻入口。

## 执行模型

运行时主流程如下：

1. 发现可用 backend 能力。
2. 将 backend 上报的设备信息归并为统一硬件拓扑。
3. 直接选模型，或者通过可选路由选择模型。
4. 构建异构执行计划：
   - 吞吐优先的 prefill
   - 功耗优先的 decode
   - KV cache 放置
   - 可选的冷权重磁盘 spill
5. 通过选定 backend 执行请求。

## Feature 模型

`loci-core` 当前暴露这些 feature：

- `openvino`
- `candle`
- `tiered-offload`
- `paged-kv`
- `power-aware`
- `dynamic-routing`

默认 feature：

```toml
default = ["openvino", "power-aware", "tiered-offload", "paged-kv"]
```

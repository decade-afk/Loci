# Loci

Loci 是一个面向端侧异构执行、分层 offload、Paged KV Cache 与可选动态模型路由的 Rust 推理基础设施项目。

Loci 当前由一组清晰的 workspace crate 组成：

- `loci-protocol` 负责共享契约
- `loci-core` 负责编排、路由与规划
- `loci-backend-openvino` 与 `loci-backend-candle` 负责 backend 集成边界
- `loci-tiered-offload` 与 `loci-paged-kv` 负责专项规划
- `loci-cli` 与 `loci-server` 提供本地运行入口

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

基础用法：

```bash
cargo run -p loci-cli -- --model-path D:/models/demo.gguf --model-name demo --model-memory-bytes 2147483648
```

```bash
cargo run -p loci-cli -- \
  --model-path D:/models/demo.gguf \
  --model-name demo \
  --model-memory-bytes 2147483648 \
  --prompt "Explain the current execution plan."
```

```bash
cargo run -p loci-cli -- \
  --model-path D:/models/demo.gguf \
  --model-name demo \
  --model-memory-bytes 2147483648 \
  --server-bind 127.0.0.1:8080
```

HTTP 路由：

- `GET /health`
- `GET /v1/runtime`
- `GET /v1/models`
- `POST /v1/models/register`
- `POST /v1/models/unregister`
- `POST /v1/plan`
- `POST /v1/inference`
- `POST /v1/completions`
- `POST /v1/chat/completions`

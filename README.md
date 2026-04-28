# Loci

Loci is a Rust inference infrastructure project for end-side heterogeneous execution, tiered offload, paged KV cache management, and optional dynamic model routing.

Loci is organized around a small set of workspace crates:

- `loci-protocol` for shared contracts
- `loci-core` for orchestration, routing, and planning
- `loci-backend-openvino` and `loci-backend-candle` for backend integration boundaries
- `loci-tiered-offload` and `loci-paged-kv` for specialized planning
- `loci-cli` and `loci-server` for local runtime surfaces

`loci-core` exposes these feature switches:

- `openvino`
- `candle`
- `tiered-offload`
- `paged-kv`
- `power-aware`
- `dynamic-routing`

Default features:

```toml
default = ["openvino", "power-aware", "tiered-offload", "paged-kv"]
```

Basic usage:

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

HTTP routes:

- `GET /health`
- `GET /v1/runtime`
- `GET /v1/models`
- `POST /v1/models/register`
- `POST /v1/models/unregister`
- `POST /v1/plan`
- `POST /v1/inference`
- `POST /v1/completions`
- `POST /v1/chat/completions`

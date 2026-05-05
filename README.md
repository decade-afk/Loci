# Loci

Loci is a Rust heterogeneous inference runtime for teams that need one runtime model across edge devices, local hosts, and server deployments.

It is designed to be used in two ways from the same runtime core:

- as an embeddable Rust SDK for in-process inference
- as a standalone service for process-isolated local or server-side integration

The runtime plans heterogeneous execution across `CPU`, `GPU`, `NPU`, and `Disk`, prepares model assets, and exposes the same model/runtime surface through SDK, CLI, and HTTP entrypoints.

## Why Loci

AI runtime teams usually have to solve the same problems repeatedly:

- model assets come from different ecosystems and packaging styles
- hardware availability changes across desktops, laptops, mobile-class devices, edge hosts, and servers
- memory pressure forces weights, KV cache, or activations to spill or stage from disk
- some products want direct embedding while others need a service boundary

Loci keeps that logic in one runtime instead of pushing it into each application.

## Current Runtime Shape

Today the repository is organized around:

- `loci-sdk` for embeddable in-process use
- `loci-server` and `loci-cli` for standalone service and command-line use
- `loci-core` for planning, readiness, model preparation, and runtime snapshots
- `loci-backend-candle` as the default portable Rust backend shape
- optional `loci-backend-openvino` acceleration for Intel-oriented deployments

The current examples and docs use the planner's disk-oriented path explicitly: `TieredOffloadProfile::DiskHeavy` with tuned spill and prefetch settings for memory-constrained heterogeneous execution.

## Quick Start

Run a one-shot inference request from the CLI:

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

Run the same runtime as a standalone local or server-facing service:

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

Embed Loci directly in a Rust application:

```rust
use loci_sdk::loci_core::TieredOffloadProfile;
use loci_sdk::{
    LocalModelRegistrationRequest, Loci, ModelPreparationRequest, TextGenerationRequest,
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

## Examples

Use the current example crates when you want the exact repo-supported shapes:

- `cargo run -p sdk-local --features openvino -- <model-path>` embeds `loci-sdk` directly and prints the tiered-offload runtime snapshot.
- `cargo run -p sdk-service --features openvino -- <model-path> 127.0.0.1:18081` starts the standalone local service from the SDK facade.
- `cargo run -p embedded-local --features openvino -- <model-path>` uses `loci-core` directly from [`examples/embedded-pet`](./examples/embedded-pet).

The two in-process examples are intentionally configured with:

- `TieredOffloadProfile::DiskHeavy`
- `spill_threshold_bytes = 512 MiB`
- `max_disk_bytes = 64 GiB`
- `prefetch_window_bytes = 128 MiB`
- `kv_block_size_tokens = 32`
- `kv_types = q8_0/q4_0`

## Design Direction

Loci keeps the control plane backend-agnostic:

- `GGUF` is the practical deployment-first model format for the current runtime
- `Candle` is the default portable Rust execution path
- `OpenVINO` is an optional acceleration path when that backend is enabled
- tiered offload and paged-KV configuration stay visible in runtime snapshots and service APIs

The intent is one runtime model that can be embedded as a library or deployed as a local or server-side service without changing how applications reason about models, readiness, or planner state.

## Learn More

- [Design](./design.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Repository Layout](./docs/LAYOUT.md)
- [MVP Plan](./docs/MVP_PLAN.md)
- [Backend Authoring](./docs/BACKEND_AUTHORING.md)
- [Intel OpenVINO Path](./docs/backends/INTEL_OPENVINO.md)
- [Contributing](./CONTRIBUTING.md)

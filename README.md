# Loci

Loci is a lightweight, plugin-oriented local LLM infra runtime built in Rust.

This repository is intentionally scoped as infra only:

- model loading
- inference runtime
- hardware backend selection
- plugin discovery and activation
- plugin runtime materialization for declared native or wasm artifacts
- embeddable Rust and C interfaces
- optional local HTTP server surface

What is intentionally out of scope:

- desktop UI
- pet / companion product logic
- workflow shells for end-user apps

## Workspace

```text
loci/
|-- crates/
|   |-- cli/          # local runtime launcher
|   |-- core/         # plugin registry, inference engine, llama.cpp binding
|   |-- ffi/          # C ABI for embedding
|   |-- plugin-api/   # stable plugin manifest types
|   `-- server/       # minimal local HTTP server surface
|-- deps/llama.cpp/   # pinned upstream backend source
|-- docs/
|-- include/
`-- plugins/          # sample manifests
```

## Architecture

Loci follows the three-layer design agreed for the infra track:

1. `loci-core`
   Unified inference API, load/switch pipeline, runtime snapshot, plugin registry.
2. Plugin layer
   The current mainline only stabilizes `model_loader` and `hardware_backend`.
   Declared plugin runtimes can currently be materialized as native libraries or validated as wasm artifacts during activation, but there is not yet a stable symbol ABI.
   `kv_cache`, `distributed`, `multimodal`, and `agent` stay as roadmap topics until they have real core contracts.
3. Interface layer
   Rust crate API, C ABI, and a local HTTP server surface for sidecar mode.

The current implementation keeps `llama.cpp` as the primary backend and uses plugin activation to gate hardware-specific behavior instead of baking application logic into the runtime.

The current C ABI stays narrow and runtime-oriented:

- create and free an engine
- load and unload a model
- raw text generation or structured inference JSON
- runtime snapshot and backend capability queries

## Quick Start

Build the workspace:

```bash
cargo build
```

Run the CLI and print the runtime snapshot:

```bash
cargo run -p loci-cli
```

Load plugins, model, and start the local server:

```bash
cargo run -p loci-cli -- \
  --plugin-dir plugins \
  --backend mock \
  --model D:/models/demo.gguf \
  --server-bind 127.0.0.1:8080
```

Current `loci-server` routes stay inference-focused:

- `GET /health`
- `GET /v1/runtime`
- `POST /v1/model/load`
- `POST /v1/model/unload`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

OpenAI-compatible routes currently support single-active-model text inference only. Streaming, tools, assistants, and workflow orchestration are intentionally out of scope.

## Current Boundary Decisions

- `Loci` remains pure infra and does not absorb `PetCompanion` product concerns.
- UI-host and workflow-governance experiments from `Loci-refactor` are not part of the new mainline build.
- Legacy plugin compatibility crates remain in the tree as historical material, but they are no longer workspace members or active dependencies.

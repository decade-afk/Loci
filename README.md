# Loci

Loci is a plugin-governed local AI runtime and management plane built as a Rust workspace.

The refactored repository treats the new workspace architecture as authoritative:

- `crates/core`: runtime kernel, model loading, governance seams, plugin inventory, management service
- `crates/cli`: `loci` binary for plugin discovery and management HTTP serving
- `crates/plugin-api`: stable manifest and capability types shared by plugins and host code
- `crates/ffi`: native integration crate reserved for future ABI hardening
- `crates/legacy-plugin-api` and `crates/legacy-plugin-compat`: bounded compatibility island for legacy text plugins

Loci is not positioned as a chat shell. It is the runtime substrate that host products can embed behind desktop apps, IDE assistants, local agent systems, and enterprise automation products.

## Architecture Direction

The new architecture is organized around governable seams instead of hardcoded feature paths. Core behavior can be rewritten by plugins at the component boundary level.

Current core rewriter seams:

- `inference`
- `model`
- `hardware`
- `workflow`
- `event_bus`
- `plugin_manager`
- `ui_host`

Plugin bundles are manifest-first. Native manifests are the mainline path. Legacy plugin support remains available only through explicit compatibility bridges.

## Workspace Layout

```text
loci/
|-- crates/
|   |-- cli/
|   |-- core/
|   |-- ffi/
|   |-- plugin-api/
|   |-- legacy-plugin-api/
|   `-- legacy-plugin-compat/
|-- plugins/                 # example plugin bundle manifests for the new architecture
|-- deps/llama.cpp/          # pinned submodule used by the optional `llama` feature
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- MANAGEMENT_API.md
|   |-- PRODUCT_STRATEGY_2026.md
|   `-- architecture/
|-- scripts/
|   `-- full_test.ps1
`-- wasm-plugin-sdk/
```

## Quick Start

Clone with the `llama.cpp` submodule:

```bash
git clone https://github.com/decade-afk/loci.git
cd loci
git submodule update --init --recursive
```

Build the CLI with real `llama.cpp` backend support:

```bash
cargo build -p loci-cli --release --features llama
```

Start the management server with bundled example manifests:

```bash
cargo run -p loci-cli --features llama -- \
  --plugin-dir plugins \
  --management-bind 127.0.0.1:8080
```

Basic checks:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/runtime
curl http://127.0.0.1:8080/v1/core/rewriters/inventory
```

Load a model through the control plane:

```bash
curl http://127.0.0.1:8080/v1/model/load \
  -H "Content-Type: application/json" \
  -d "{\"backend_name\":\"llama.cpp\",\"config\":{\"model_path\":\"D:/models/qwen.gguf\"}}"
```

Run generation:

```bash
curl http://127.0.0.1:8080/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"hello from loci\"}"
```

## Plugin Model

Loci now prefers manifest bundles over ad hoc runtime loading contracts.

- Load a whole plugin directory with `--plugin-dir <path>`
- Load a bundle or directory over HTTP with `POST /v1/plugins/load`
- Activate rewriter ownership with `POST /v1/core/rewriters/activate`
- Activate legacy text compatibility only when required with `POST /v1/legacy-text/activate`

Example manifests live in:

- `plugins/example-inference`
- `plugins/example-infra`
- `plugins/example-agent`

More detail is in [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md).

## Real Build And Test Commands

Workspace validation:

```bash
cargo test -q
```

`llama.cpp` integration validation:

```bash
cargo test -q -p loci-core --features llama
cargo test -q -p loci-cli --features llama
```

Windows helper script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/full_test.ps1
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Management API](docs/MANAGEMENT_API.md)
- [Plugin Guide](PLUGIN_GUIDE.md)
- [Architecture ADRs](docs/architecture/README.md)
- [Product Strategy 2026](docs/PRODUCT_STRATEGY_2026.md)
- [Build Guide](BUILD.md)

## Current Status

The workspace migration is authoritative. Old root-level monolith sources, old example programs, and old serve/generate-era docs are no longer the source of truth.

What is production-oriented today:

- plugin-governed runtime core
- management HTTP control plane
- runtime snapshot and plugin inventory
- rewriter activation for inference, model, hardware, workflow, event bus, plugin manager, and UI host
- real model load governance and `llama.cpp` optional backend integration
- bounded legacy text plugin compatibility

What is intentionally bounded:

- legacy plugin bridging stays isolated in compat crates
- native FFI surface is not yet a stabilized public C ABI

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

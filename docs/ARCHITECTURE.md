# Loci Architecture

## Positioning

Loci is an end-side inference infrastructure project focused on heterogeneous CPU / GPU / NPU execution, tiered weight offload, paged KV cache planning, and optional dynamic model routing.

## Workspace Layers

### 1. Protocol Layer

`crates/protocol` defines the shared contracts for:

- hardware topology
- model descriptors
- routing decisions
- execution plans
- backend traits
- request / response payloads

This crate is the shared language of the workspace.

### 2. Core Runtime Layer

`crates/core` is the orchestration layer. It owns:

- runtime configuration
- hardware topology merge
- backend selection
- routing
- heterogenous execution planning
- runtime snapshot generation

The core does not embed backend-specific execution logic directly. It delegates execution through backend crates and keeps planning decisions explicit.

### 3. Backend and Planning Extensions

- `crates/backend-openvino`
- `crates/backend-candle`
- `crates/tiered-offload`
- `crates/paged-kv`

These crates provide the concrete integration points for backend capabilities and specialized planning logic.

The architecture is intentionally feature-gated instead of plugin-driven. Backends are injected through Cargo features rather than runtime plugin activation.

The current backend crates should be read as integration boundaries, not as finished production bindings.

### 4. Interface Layer

- `crates/cli`
- `crates/server`

These crates are thin entry points over `loci-core`.

## Execution Model

The runtime follows this high-level flow:

1. Discover the available backend capabilities.
2. Merge backend-reported devices into a single hardware topology.
3. Select a model directly or through optional routing.
4. Build a heterogeneous execution plan:
   - throughput-biased prefill
   - power-biased decode
   - KV cache placement
   - optional disk spill for cold weights
5. Dispatch the request through the chosen backend.

## Feature Model

`loci-core` currently exposes these feature switches:

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

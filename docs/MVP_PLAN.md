# Loci MVP Plan

## Goal

Deliver a real, testable MVP of Loci as a local inference runtime that can ship either as an embeddable SDK or as a standalone local service.

The MVP is not "all model formats, all backends, all kernels".

The MVP is:

- one control-plane runtime that can register and inspect models
- one real execution backend path that can plan, prepare, and execute
- one stable runtime snapshot plus SDK/HTTP/CLI integration surfaces
- one honest fallback path for the future pure Rust direction

## MVP Scope

### Included

- model registration and alias resolution
- model asset inspection and readiness reporting
- backend selection based on readiness, not file extension alone
- heterogeneous execution planning across `CPU/GPU/NPU/Disk`
- disk-heavy tiered-offload planning and runtime snapshot exposure
- paged-KV configuration that can be surfaced through SDK, CLI, and service layers
- `Candle` as the default generic execution path
- `OpenVINO` as an optional real execution backend
- static kernel catalog reporting
- embeddable SDK entrypoints plus CLI and HTTP endpoints for inspect, plan, prewarm, infer, and runtime snapshot

### Excluded

- real paged-KV execution
- automated GGUF/Safetensors/ONNX conversion pipelines
- real imported upstream kernels from `llama.cpp`, `candle`, or others
- per-layer affinity writeback into OpenVINO graphs
- multi-model routing as a required MVP capability
- Android / Hexagon / Qualcomm / Rockchip runtime integration

## MVP Architecture Position

For the MVP:

- `Candle` is the default generic path
- `OpenVINO` is an optional real execution path
- readiness and planning must be honest about which backend is actually ready

This means documentation and runtime behavior must both reflect:

- long-term direction: `Candle` + `GGUF` + portable kernels
- MVP reality: `OpenVINO` can execute today when enabled, while `Candle` defines the generic architecture
- current operating profile: examples and docs should show the disk-heavy planner path instead of pretending local memory is always sufficient

## MVP Deliverables

### 1. Runtime Core

The runtime must support:

- build engine from config + models
- inspect one model or all models
- compute an execution plan
- prewarm a model/backend session
- run one inference request
- expose a serializable runtime snapshot
- expose planner-facing spill and KV configuration in that snapshot

### 2. Backend Selection

The backend selector must:

- prefer a ready backend over a merely preferred backend
- fall back from a non-ready preferred backend to a ready backend
- exclude non-multimodal backends for image requests
- use readiness inspection as the source of truth

### 3. OpenVINO Execution Path

The OpenVINO path must:

- accept ready IR / GenAI-exported assets
- reject raw checkpoints with clear adaptation/export errors
- prepare reusable backend sessions
- execute text requests or return a deterministic fallback when runtime bootstrap is unavailable
- support multimodal only for architectures and assets that are actually executable

### 4. Candle Generic Path

The Candle path must:

- compile and expose topology, asset capabilities, lowering capabilities, and kernel catalog
- remain the default generic backend shape
- never block the control plane even when OpenVINO is disabled

### 5. Kernel Catalog

The MVP kernel catalog must:

- be exposed in the runtime snapshot
- distinguish `planned`, `stubbed`, and `integrated`
- preserve upstream origin metadata
- remain declarative only; no fake dispatch logic

### 6. Integration Shapes

The MVP must ship one runtime in two supported forms:

- embeddable through `loci-sdk` for in-process callers
- exposable as a standalone local service through `loci-server`/`loci-cli`

Those shapes should describe the same runtime state, model readiness, and planner configuration.

## Acceptance Criteria

The MVP is complete when all of the following are true:

1. `cargo test -q -p loci-protocol` passes.
2. `cargo test -q -p loci-backend-candle` passes.
3. `cargo test -q -p loci-backend-openvino` passes when the feature is enabled.
4. `cargo test -q -p loci-core` passes.
5. CLI can:
   - print runtime snapshot
   - inspect models
   - prewarm a model
   - run a prompt
6. SDK can:
   - register a local model
   - prepare it with disk-heavy tiered offload enabled
   - expose the same runtime snapshot fields as the CLI/service layers
7. Runtime snapshot includes:
   - backend descriptors
   - asset capability summaries
   - lowering capability summaries
   - kernel catalogs
   - model diagnostics
   - tiered-offload and KV configuration
8. Preferred backend fallback works:
   - if preferred backend is not ready, a ready backend is chosen instead
9. Multimodal exclusion works:
   - image requests do not select a text-only backend
10. Default `candle` build compiles without requiring `openvino`

## Implementation Plan

### Phase A: Correctness

- fix backend selection semantics in `loci-core`
- align readiness reporting with test expectations
- fix OpenVINO validation error wording drift

### Phase B: MVP Surfaces

- keep `runtime_snapshot()` stable and complete
- ensure SDK, CLI, and server routes expose MVP state correctly
- ensure kernel catalog appears in snapshot outputs
- keep user-facing examples aligned with the documented disk-heavy planner profile

### Phase C: Verification

- run crate-level tests
- fix broken assertions or logic regressions
- confirm the documented MVP matches runtime reality

## Post-MVP

After MVP, the next practical steps are:

- planner-driven kernel selection from `KernelRegistry`
- first real imported kernel path: quantized matmul or decode attention
- real Candle execution for one narrow architecture path
- GGUF-first loader/executor path

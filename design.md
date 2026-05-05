# Loci Design

## What Loci Is

Loci is a lightweight, extensible heterogeneous inference foundation built in Rust for edge-first applications and deployers that also need a consistent server path.

Its job is to make model execution usable across real edge and deployment environments: laptops, desktops, compact PCs, embedded systems, future mobile targets, and server hosts. Instead of treating inference as a single backend problem, Loci treats it as a placement and runtime orchestration problem across heterogeneous resources.

That is why the project is organized around a small control-plane core, portable execution defaults, and optional vendor accelerators.

## What Loci Is For

Loci is designed for teams that need one of two integration styles:

- direct in-process inference through a Rust SDK
- a standalone AI service that can be consumed by host applications in local or server deployments

Those two modes share the same runtime concepts:

- model registration
- readiness inspection
- backend selection
- heterogeneous planning
- execution and streaming output

## Core Design Goals

### 1. Portable by default

Loci uses `Candle` as the default Rust-native execution path so the project can stay portable across platforms without making a vendor runtime the center of the architecture.

### 2. Practical edge model support

Loci is `GGUF`-first because local deployment depends on formats that are compact, inspectable, and well-suited for quantized edge workloads.

Other model sources such as `ONNX`, `Safetensors`, or upstream checkpoint formats matter, but they should enter the system through explicit adapters and lowering workflows rather than defining the entire architecture.

### 3. Heterogeneous planning as the differentiator

The distinctive value of Loci is not just loading a model. It is deciding where that model should run and what should happen when resources are constrained.

The planner is responsible for reasoning about:

- `CPU`, `GPU`, `NPU`, and `Disk`
- memory pressure and spill behavior
- runtime capability differences between backends
- execution readiness of model assets
- later, thermal and power-aware placement

### 4. Explicit backend boundaries

Backends should be replaceable without rewriting the core runtime.

The core owns planning, model lifecycle, and integration contracts. Backends own execution details, low-level lowering behavior, and backend-local optimizations.

### 5. Import-first kernel strategy

Loci does not try to invent every operator from scratch.

It ports the high-value pieces that matter most for local inference, especially from projects such as `llama.cpp`, and integrates them through explicit provenance, validation, and benchmark gates.

## System Shape

```text
loci/
├── crates/protocol
├── crates/core
├── crates/sdk
├── crates/gguf
├── crates/kernels-llama
├── crates/backend-candle
├── crates/backend-openvino
├── crates/tiered-offload
├── crates/paged-kv
├── crates/cli
└── crates/server
```

## Architectural Layers

### Core Layer

`crates/core` defines the runtime model for:

- model registration and aliasing
- readiness inspection
- backend capability selection
- runtime snapshots
- planner-visible execution topology
- embedded and service integration flows

This layer must remain backend-agnostic.

### Format Layer

`crates/gguf` handles the format boundary most important for the first version of Loci:

- GGUF metadata inspection
- architecture normalization
- tensor table summary
- loader-facing structure needed by execution backends

This keeps file-format knowledge separate from planner and service code.

### Kernel Layer

`crates/kernels-llama` is the landing zone for curated kernel ports.

This layer exists so operator work is:

- explicit
- testable
- benchmarkable
- decoupled from a single backend implementation file

### Backend Layer

`crates/backend-candle` is the default execution direction for Loci. It is the portable Rust path that keeps the project broadly deployable.

`crates/backend-openvino` is an optional acceleration backend for Intel platforms. It exists as an optimization path, not as the architectural center of the project.

### Runtime Helpers

`crates/tiered-offload` and `crates/paged-kv` support the memory side of local inference:

- residency tracking
- spill and prefetch
- paged KV planning
- future cache reuse strategies

## Execution Model

Loci treats execution as a pipeline:

1. register or discover a model asset
2. inspect format and readiness
3. merge hardware and backend topology
4. build a heterogeneous execution plan
5. prepare backend state and optional offload state
6. run inference through the selected backend
7. surface output through SDK or service APIs

This same model should work whether an application embeds Loci directly or starts it as a local runtime service.

## Backend Strategy

Loci uses feature-gated composition instead of runtime plugins.

That keeps:

- binaries predictable
- startup simple
- capability boundaries explicit
- cross-platform builds manageable

The default direction is:

- `candle` for portable Rust execution
- `gguf` for practical edge ingestion
- `kernels-llama` for curated operator ports
- `openvino` as an optional Intel-specific feature

## Planner Strategy

The planner is the product-level advantage of Loci.

Its responsibility is to make local inference usable under real device constraints. In the early design, that means rule-based and inspectable decisions:

- prefer the highest-value accelerator that is actually ready
- fall back cleanly when a backend cannot execute the asset
- spill to disk when memory pressure requires it
- keep placement and readiness visible to the application

Over time, the planner can become more sophisticated, but its contract should stay understandable and debuggable.

## Model Scope

Loci is broad in intent but disciplined in scope.

The primary path is:

- mainstream decoder models
- `GGUF` packaging
- `Llama`, `Mistral`, and `Qwen`-class families

That gives the project a useful center while leaving room for later architecture registry growth.

## Integration Model

Loci should feel natural in both of these shapes:

- as an SDK linked directly into an application process
- as a local service that exposes stable inference APIs

That matters because some products want zero network hops and others want isolation, supervision, or multi-client access. Loci should support both without splitting into two runtimes.

## Design Standard

Every major addition to Loci should satisfy the same standard:

- it strengthens local deployment rather than abstract elegance alone
- it preserves a small, backend-agnostic core
- it keeps provenance explicit when importing external ideas or kernels
- it improves integration quality for SDK and service users

That is the bar for the project.

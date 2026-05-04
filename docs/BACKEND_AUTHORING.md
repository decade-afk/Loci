# Backend Authoring

## Goal

Loci backends are not thin wrappers around random runtimes. A backend is the execution layer that receives a core plan and lowers it into a chip- or runtime-specific path.

If you want to add a new backend, keep the split strict:

- `loci-core` decides what should happen
- the backend decides how that plan maps onto a concrete runtime

## What A Backend Must Provide

At minimum, a backend must implement the `Backend` trait from `crates/protocol`.

That includes:

- `descriptor()`
- `asset_capabilities()`
- `lowering_capabilities()`
- `kernel_catalog()`
- `discover_topology()`
- `supports_model()`
- `prepare()`
- `execute()`

## What `descriptor()` Means

`BackendDescriptor` is the coarse capability contract.

Use it to describe:

- available accelerator families
- disk-tier support
- paged-KV support
- multimodal support

Do not use it to overstate execution maturity. If the backend is still partial, keep that truth in `lowering_capabilities()` and readiness diagnostics.

## What `asset_capabilities()` Means

`BackendAssetCapabilities` is the execution-asset contract between the core and a backend.

Use it to declare:

- which asset layouts are directly executable right now
- which layouts are ingestible but still require lowering or conversion
- which execution artifact family the backend prefers
- whether lowering is mandatory before real execution

This contract is what `loci-core` now uses for:

- readiness inspection
- runtime snapshot reporting
- backend selection when several partial backends are available

If this method is left at the default implementation, the core treats the backend ingestion boundary as undefined.

## What `lowering_capabilities()` Means

`BackendLoweringCapabilities` is the low-level integration contract for future chip work.

Use it to state:

- the lowest lowering granularity your backend can currently accept
- whether real execution exists
- whether graph partitioning is real
- whether layer affinity is real
- whether custom operators are supported
- which operator classes are meaningful for that backend

This is the first place new chip contributors should document what their backend can truly do.

When the planner builds an `ExecutionPlan`, the backend now also receives a `BackendLoweringPlan` that can contain:

- coarse `subgraphs`
- grouped `partitions`
- normalized `operators`

Backends do not need to consume all three at once. A practical progression is:

1. validate `partitions`
2. consume affinity/device hints from `partitions`
3. later lower `operators` into per-layer or per-kernel execution hooks

## What `kernel_catalog()` Means

`BackendKernelCatalog` is the backend's explicit declaration of low-level operator coverage.

Use it to describe:

- which kernels are real today
- which kernels are only planned or stubbed
- whether a kernel is portable Rust, vendor-runtime-backed, IR-mediated, or bridged from an external component
- where the implementation came from
- which targets, formats, and architecture families it is meant to serve

This exists for one specific reason: Loci intends to import and adapt strong kernels from external projects without hiding that work inside undocumented backend-private code.

If a backend ports or adapts an upstream operator, keep the origin explicit.

Examples:

- `llama.cpp`-inspired GGUF quantized matmul
- `candle` attention or normalization paths
- vendor runtime kernels exposed indirectly through OpenVINO/QNN/RKNN

Do not claim `Integrated` or `Validated` maturity unless real execution exists in the repository.

## Backend Boundaries

### Keep In Core

- backend selection
- routing
- model readiness inspection
- generic heterogeneous plan construction
- cross-backend runtime snapshots

### Keep In The Backend

- vendor runtime setup
- model compilation/loading
- graph/subgraph lowering
- affinity mapping
- low-level operator handling
- upstream kernel port integration
- backend-local telemetry extraction

## Recommended Development Order

1. Implement `descriptor()` truthfully.
2. Implement `lowering_capabilities()` truthfully.
3. Implement `discover_topology()`.
4. Implement `asset_capabilities()` truthfully.
5. Implement `kernel_catalog()` truthfully, including maturity and provenance.
6. Implement `supports_model()` conservatively as a backend-local safety check, not as the primary source of readiness truth.
7. Implement `prepare()` with clear validation failures.
8. Implement `execute()` only after `prepare()` is real.

## Readiness Rules

A backend should not claim to be ready just because:

- it recognizes a file extension
- it can build a fake session
- it can return synthetic telemetry

Real readiness means the backend can drive a real execution runtime for the declared asset layout.

## Current Reference Backends

### `backend-openvino`

This is the current reference implementation for:

- Intel CPU/GPU/NPU-oriented execution
- OpenVINO runtime integration
- OpenVINO GenAI multimodal execution

### `backend-candle`

This is the current reference shape for:

- a pure Rust fallback path
- future tensor-level placement

It is not yet a reference for real execution completeness.

## Adding A New Chip Backend

Examples of future backend families:

- `backend-qnn`
- `backend-rknn`
- `backend-onnxruntime`
- `backend-tract`

Before opening a large implementation PR, contributors should document:

- target runtime family
- supported devices
- supported model layouts
- lowering granularity
- kernel sourcing strategy
- whether graph partitioning is real
- whether custom operators are required

That document should live in `docs/backends/`.

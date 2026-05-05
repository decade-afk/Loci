# Loci Progress Report

Date: 2026-04-30  
Scope: current workspace state plus recent real runtime validation

## Executive Summary

Loci is now a real Rust heterogeneous inference control plane with a real OpenVINO execution path, a partial tiered-offload runtime, multimodal request plumbing, and an explicit model-readiness inspection layer.

It is not yet a complete production-grade heterogeneous runtime for arbitrary edge models. The main remaining gaps are:

- no planner-to-subgraph affinity lowering for real backend partitioning
- no low-level chip operator ABI for future backend families
- no real Candle execution path yet
- no full paged-KV runtime yet
- no automated export/conversion workflow for raw model assets

## What Is Real Today

### 1. Core Orchestration

Implemented in `loci-core`:

- backend-agnostic planning
- CPU/GPU/NPU/Disk placement plans
- model registry, aliases, residency tracking, keep-alive eviction
- resident-memory budget enforcement
- optional dynamic routing configuration
- runtime snapshots

Current assessment:

- this is the strongest part of the codebase
- the architecture matches the `tmp/design.md` direction well

### 2. OpenVINO Backend

Implemented in `loci-backend-openvino`:

- real `openvino-rs` runtime discovery
- real `openvino-genai` text path via `LlmPipeline`
- real `openvino-genai` multimodal path via `VlmPipeline`
- image input loading from `path`, `file://`, `data:` and base64 payloads
- real fallback reporting when the runtime path cannot execute
- validation of expected OpenVINO GenAI export layout

Current assessment:

- this is no longer a mock backend
- real runtime integration is present
- the missing part is fine-grained backend partition control, not basic integration

### 3. Tiered Offload

Implemented in `loci-tiered-offload`:

- spill planning
- profile and policy derivation
- mmap-backed spill artifacts
- prefetch runtime
- prepare/evict integration with the core engine

Current assessment:

- this is beyond policy-only scaffolding
- the remaining gap is a richer reload/state-machine and stronger I/O-aware scheduling

### 4. Multimodal Plumbing

Implemented end to end:

- `SessionRequest.images`
- CLI `--image`
- HTTP image request parsing
- OpenAI-style chat image inputs
- backend filtering that skips non-multimodal backends for image requests
- architecture-based multimodal detection, including `MiniCPM-V`

Current assessment:

- request ingestion is real
- successful execution still depends on backend-ready exported assets

### 5. Model Readiness Inspection

Newly implemented:

- asset layout detection
- per-backend readiness reporting
- conversion/export requirement reporting
- `ready_for_inference` and `recommended_backend`
- runtime snapshot diagnostics
- CLI inspection output
- server inspection routes

This is important because Loci now distinguishes:

- format compatibility
- asset readiness
- backend implementation readiness

That distinction was previously missing and is required for “arbitrary model support”.

## What Is Still Partial

### 1. Candle Backend

Current state:

- accepts the right high-level formats
- rejects unsupported NPU and multimodal cases clearly
- estimates residency correctly enough for planning

Still missing:

- real Candle tensor execution
- real GGUF loading and execution
- real SafeTensors execution
- real device placement runtime

Assessment:

- Candle is still a control-plane fallback, not a real inference backend yet

### 2. Paged KV

Current state:

- planning-level KV strategy exists
- page/block/cache sizing exists
- sharing heuristics exist

Still missing:

- real paged attention runtime
- real page allocation/reclamation
- real prefix cache store
- real cross-model cache reuse

### 3. Structured Output and Tool Calling

Current state:

- request-level flags exist
- transport and serving layers pass them through

Still missing:

- grammar/schema-constrained decoding
- tool orchestration semantics
- backend-enforced structured generation

## What Is Missing For Future Chip Backends

These are the main architectural gaps found during the deeper audit of `tmp/references`:

- no subgraph/layer affinity ABI
- no operator-lowering boundary for backend families
- no low-level chip operator registry
- no partition callback interface for backend-specialized runtimes

That means the current Loci implementation is:

- strong at unified orchestration
- partially real at backend execution
- not yet a low-level multi-chip operator infrastructure

## Real Validation Performed

### OpenVINO Runtime Validation

Validated on this machine:

- real OpenVINO GenAI runtime loads correctly when the local runtime environment is configured
- real visible devices are `CPU` and `GPU`
- no usable `NPU` is available on this machine

### MiniCPM-V-4_5 Validation

Validated against:

- `vendor/models/MiniCPM-V-4_5-meta`

Observed:

- Loci generates a real hetero plan
- Loci classifies the asset as a raw Transformers checkpoint
- Loci now reports that this model requires an OpenVINO GenAI multimodal export before real execution

This confirms:

- the current blocker is model asset readiness, not basic OpenVINO integration

## New User-Facing Introspection Added

Available now:

- runtime snapshot includes model diagnostics
- CLI:
  - `cargo run -p loci-cli -- --model-path <path> --model-name <name> --architecture <arch> --inspect-models`
- server:
  - `GET /v1/models/inspect`
  - `POST /v1/models/inspect`

Example current diagnostic outcome for the raw MiniCPM-V directory:

- `asset_layout = transformers_checkpoint`
- `ready_for_inference = false`
- `recommended_backend = null`
- OpenVINO readiness:
  - real execution path exists
  - conversion/export is required first

## Recommended Next Work

1. Lower planner placements into real backend-specific subgraph/affinity control for OpenVINO.
2. Define the low-level backend ABI needed for future QNN/RKNN/other chip families.
3. Expand model asset workflow with export/conversion helpers and stronger validation.
4. Turn paged-KV into a real cache runtime.
5. Implement a real Candle execution path for GGUF/SafeTensors fallback.

## Verification Status

Passed:

- `cargo fmt`
- `cargo check -q`
- `cargo test -q -p loci-protocol`
- `cargo test -q -p loci-server`
- `cargo test -q -p loci-backend-openvino -p loci-backend-candle`
- targeted `loci-core` tests for:
  - model inspection
  - readiness-aware backend selection
  - runtime snapshot diagnostics

Known environment-specific failure:

- full `cargo test -q -p loci-core` can still fail on `prepare_materializes_tiered_offload_runtime_for_disk_backed_models`
- root cause on this machine is insufficient disk space: `os error 112`
- this is not currently treated as a logic regression in the new readiness work

## Bottom Line

Loci now has:

- a real OpenVINO execution path
- a real heterogeneous control plane
- real multimodal request plumbing
- a real model-readiness inspection layer

Loci still lacks:

- low-level chip operator support
- backend-level subgraph lowering
- real Candle runtime execution
- real paged-KV execution

That is the accurate state of the project as of 2026-04-30.

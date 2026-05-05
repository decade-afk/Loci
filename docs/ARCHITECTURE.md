# Loci Architecture

Loci is organized as an edge-first heterogeneous inference platform with a small core and explicit execution boundaries.

The architecture is designed for one purpose: let applications run models through the same runtime whether they need an embedded SDK, a standalone service, or a backend mix that changes across edge devices and servers.

## Architectural Overview

Loci separates inference into four concerns:

- control-plane orchestration
- model and format understanding
- execution backends
- memory and residency helpers

This separation keeps the project portable while still allowing specialized acceleration paths where they make sense.

## 1. Control Plane

The control plane lives in `crates/core`.

It is responsible for:

- model registration and alias management
- backend topology discovery
- readiness inspection
- heterogeneous plan construction
- runtime snapshots
- service and embedded runtime coordination

The control plane does not own backend-specific kernels or vendor runtime assumptions. It only reasons about capabilities, readiness, and placement.

## 2. Model and Format Layer

The format layer starts with `crates/gguf`.

Its role is to turn model assets into structured runtime information:

- model identity
- architecture family
- context length
- tensor table structure
- basic readiness metadata for execution backends

This keeps file parsing separate from planning and allows backends to consume normalized model facts instead of raw asset logic.

## 3. Execution Backends

Execution backends are isolated crates that implement the runtime contract in different ways.

### `backend-candle`

`crates/backend-candle` is the default backend direction for Loci.

It exists to provide:

- a pure Rust execution path
- a portable integration surface across host platforms
- a natural landing zone for curated operator ports
- the default composition for applications that want to embed Loci without vendor lock-in

### `backend-openvino`

`crates/backend-openvino` is an optional backend for Intel-oriented acceleration.

It exists to provide:

- optimized execution on supported Intel devices
- access to Intel heterogeneous placement across accelerator classes
- a production-oriented optional path without redefining the architecture of the rest of the project

## 4. Kernel Layer

Low-level operator work lives in `crates/kernels-llama`.

This layer is where Loci ports and validates selected high-value kernels inspired by strong upstream projects such as `llama.cpp`.

The point of a separate kernel layer is to make operator work:

- explicit
- benchmarkable
- provenance-aware
- reusable across backend integration work

The kernel layer is not a dumping ground for backend-private code. It is a curated operator boundary.

## 5. Memory and Residency Helpers

Heterogeneous inference is often constrained by memory before it is constrained by API shape.

That is why Loci isolates spill and cache support into dedicated crates:

- `crates/tiered-offload`
- `crates/paged-kv`

These crates support the planner and backends with:

- residency tracking
- disk spill and prefetch
- paged KV structures
- future reuse and prefix-cache policies

## Planning Model

The planner is the piece that ties the architecture together.

It uses the model layer, backend layer, and runtime topology to decide:

- which backend can actually execute the asset
- which compute targets should be preferred
- when to fall back
- when to spill or preserve memory
- how to describe the decision back to the application

The planner should remain inspectable. Applications need to understand why a model ran on a given path, not just that it ran.

## Execution Flow

The normal runtime flow is:

1. a model is registered through the SDK or service
2. the format layer inspects the asset
3. the control plane merges hardware and backend capabilities
4. the planner selects a usable execution path
5. backend state is prepared
6. inference runs through the selected backend
7. output is returned through an embedded API or local service surface

This flow is shared across all integration styles so the runtime behaves consistently.

## Service and SDK Surfaces

Loci is intentionally dual-surface.

Applications can:

- link `crates/sdk` and call the runtime directly
- run `crates/server` and consume a standalone service for local or server deployment

These are not separate products. They are two interfaces over the same runtime model.

## Extension Model

Loci uses Cargo features instead of runtime plugins.

That keeps the architecture explicit:

- portable defaults stay lightweight
- vendor features stay opt-in
- platform-specific backends do not leak into the core

This is especially important for local inference, where binary size, startup behavior, and dependency control matter.

## Architectural Direction

The long-term architectural center of Loci is:

- `GGUF`-first local model ingestion
- `Candle` as the default portable backend path
- curated kernel imports and rewrites for important operator hotspots
- planner-driven heterogeneous execution
- one runtime that can act as both SDK and service

That is the model the rest of the repository should reinforce.

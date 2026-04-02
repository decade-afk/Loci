# Loci Architecture

Last updated: 2026-04-02

## Positioning

Loci is a plugin-governed local AI runtime and management plane for host products.

It sits between low-level backend bindings and end-user UX shells:

- below product-specific chat or agent UIs
- above raw inference backend integration
- focused on governable runtime seams, pluginized upgrades, and host-controlled operations

## Workspace Structure

The repository is a Rust workspace instead of a root monolith.

| Crate | Responsibility |
|---|---|
| `crates/core` | runtime kernel, model loading, plugin inventory, governance seams, management service |
| `crates/cli` | `loci` binary and management HTTP serving |
| `crates/plugin-api` | plugin manifest schema and shared capability types |
| `crates/ffi` | stable public C ABI over the current runtime and management surface |
| `crates/legacy-plugin-api` | legacy plugin contract types |
| `crates/legacy-plugin-compat` | bounded bridge for legacy text plugins |

## Architectural Thesis

The primary rule is that governable behavior should exist behind explicit seams rather than hardcoded branches.

The current rewriter seams are defined by `CoreComponent`:

- `Inference`
- `Model`
- `Hardware`
- `Workflow`
- `EventBus`
- `PluginManager`
- `UiHost`

Each seam can be:

- declared by a plugin manifest
- inspected in runtime inventory
- activated by the host through the management plane

## Layer Model

```text
Host product / automation system / IDE assistant
    |
    v
CLI management surface (`crates/cli`)
    |
    v
Management service (`crates/core::management`)
    |
    +--> runtime snapshot / plugin inventory / activation
    +--> model load governance
    +--> workflow, command, event, ui routing
    |
    v
Inference engine (`crates/core::engine`)
    |
    +--> backend registry
    +--> core registry traits
    +--> plugin manager
    +--> legacy compatibility materialization
    |
    v
Optional `llama.cpp` backend (`crates/core --features llama`)
```

## Core Registry

The core runtime is intentionally decomposed behind traits:

- `ModelRepository`
- `WorkflowEngine`
- `EventBus`
- `HardwareAbstraction`
- `UiHost`
- `PluginManager`
- `CoreRegistry`

This keeps the engine open for plugin-driven or host-driven replacement without turning the runtime into one monolithic service object.

## Plugin Model

Plugins are manifest-first.

Important manifest concepts:

- target tracks: `ai_infra`, `ai_agent`
- contribution points: model providers, accelerators, inference hooks, events, workflows, custom nodes, commands, UI contributions
- core rewriters: seam-level governance claims
- runtime artifacts: library path, wasm path, sampling profile
- bootstrap activation: optional auto-activation for declared seams
- compatibility metadata: legacy bridge information

Example manifests live under `plugins/`.

## Management Plane

The management plane is intentionally narrow and operational:

- runtime discovery
- backend discovery
- plugin status and detailed inspection
- core rewriter inventory and activation
- model load requests
- text generation requests
- workflow execution
- UI surface presentation
- command routing
- event publishing
- legacy text plugin activation

See `docs/MANAGEMENT_API.md` for the route surface.

## Compatibility Strategy

Compatibility is bounded.

- New architecture is the source of truth.
- Legacy text plugin support remains available only through `legacy-plugin-api` and `legacy-plugin-compat`.
- Old root-level monolith modules, old CLI routes, and old example programs are not part of the maintained architecture.

This keeps migration debt contained instead of leaking old contracts back into the mainline runtime.

## Runtime Backends

`loci-core` can be built without native backend bindings for architectural work and governance validation.

For real local model execution, enable:

```bash
cargo build -p loci-core --features llama
```

The `llama.cpp` integration is wired through the `deps/llama.cpp` submodule and the crate-local build script in `crates/core/build.rs`.

## Maintenance Rules

- Workspace crates are authoritative.
- New runtime capabilities should land behind a seam or registry when they are governable.
- Documentation should describe only maintained entry points.
- Legacy compatibility must stay isolated and explicitly activated.

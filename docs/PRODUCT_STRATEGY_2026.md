# Loci Product Strategy 2026

Last updated: 2026-03-14

## Positioning

Loci should be positioned as:

- an embeddable AI inference engine
- a plugin-upgradeable local runtime
- a host-controlled control plane for models, tools, sessions, and policies

It should not be positioned as a direct competitor to chat-first end-user shells by copying their UI. Its competitive angle is that product teams can build their own shells, assistants, IDE copilots, and local automation systems on top of it.

## Strategic Thesis

There is room between low-level inference libraries and end-user chat shells.

Loci occupies that middle layer:

- lower than assistant UX
- higher than raw backend bindings
- opinionated about runtime control, but not about one specific product UX

## Target Integrators

- Tauri/Electron desktop products
- IDE assistants
- local-first copilots
- developer tools with embedded reasoning
- assistant hosts such as `localhand`

## Capability Pillars

1. Inference substrate
2. Plugin upgradeability
3. Model asset lifecycle and governance
4. Tool and MCP orchestration
5. Session and state control
6. Resource-aware placement and large-model operation

## Must-Win Features

### Near term

- Stable model inventory and pull governance
- Stable plugin and policy registries
- Better architecture and integration documentation
- Structured eventing and auditability

### Mid term

- Stronger large-model tiering across VRAM, RAM, and disk
- Typed IPC/gRPC control plane
- Out-of-process execution workers
- First-class multimodal serving path

### Longer term

- C-stable plugin ABI families
- Signed plugin distribution and trust chain
- More complete automation governance for high-privilege assistants

## Roadmap

### Phase A: Trusted Core

- harden control-plane governance
- harden model lifecycle
- harden plugin contract validation
- improve docs and architectural clarity

### Phase B: Deployable Runtime

- typed IPC/gRPC
- event bus
- structured telemetry
- packaging for embedded products

### Phase C: Large-Model Runtime

- memory-tier orchestration
- out-of-process workers
- scheduler and isolation controls

### Phase D: Assistant Substrate

- multimodal mainline path
- OS/browser/CLI action policies
- auditable tool governance for products like `localhand`

## Success Criteria

- Integrators can embed Loci without forking core runtime code.
- Runtime upgrades happen by plugin loading or controlled service rollout.
- Model lifecycle is governed, not ad hoc.
- Higher-level products can inherit Loci without turning Loci into UI code.

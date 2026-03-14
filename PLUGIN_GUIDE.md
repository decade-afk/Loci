# Loci Plugin Guide

Last updated: 2026-03-14

This guide is the short practical entry point for plugin development in the current repository.

## Plugin Families

Loci currently supports several extension families:

- text runtime plugins
- tool plugins
- execution policy plugins
- management auth policy plugins
- serve dispatch policy plugins
- model pull policy plugins
- model pull verifier plugins
- backend kernel plugins
- image kernel plugins
- WASM text plugins

## Core Principles

- Plugins extend the runtime, not just prompt formatting.
- Dynamic plugins should ship a sidecar manifest when possible.
- Plugin names should be stable because registries, activation, and host configs depend on them.
- Host software should treat plugin activation as an operational change, not just a code-load event.

## Manifest Contract

Dynamic plugins may ship a sidecar manifest such as:

- `my_plugin.loci-plugin.json`
- `my_plugin.loci-plugin.toml`

The manifest is validated against:

- plugin kind
- ABI version
- optional host version bounds

Relevant implementation:

- `src/plugin_contract.rs`

## Main Runtime Text Plugins

Text plugins participate in the generation hook chain.

Relevant modules:

- `src/plugin.rs`
- `src/plugin_registry.rs`
- `src/wasm_plugin.rs`

Constructor pattern:

- preferred: `create_plugin_v1()`
- legacy fallback: `create_plugin()`

Reference examples:

- `examples/dynamic_plugin_example/`
- `examples/openclaw_adapter_plugin/`

## Tool Plugins

Tool plugins register callable functions into the function-calling subsystem.

Relevant module:

- `src/tool_plugin.rs`

Constructor pattern:

- preferred: `create_tool_plugin_v1()`
- legacy fallback: `create_tool_plugin()`

Reference example:

- `examples/browser_tool_plugin/`

## Policy Plugins

Policy plugins let the host upgrade runtime governance without rewriting core engine code.

Families:

- execution policy
- management auth policy
- serve dispatch policy
- model pull policy
- model pull verifier

Relevant modules:

- `src/execution_policy_plugin.rs`
- `src/management_auth.rs`
- `src/serve_dispatch.rs`
- `src/model_pull_policy.rs`
- `src/model_pull_verifier.rs`

Reference examples:

- `examples/execution_policy_plugin/`
- `examples/management_auth_plugin/`
- `examples/serve_dispatch_plugin/`

## Backend and Image Kernel Plugins

These plugins extend execution kernels rather than prompt hooks.

Relevant modules:

- `src/backends/dynamic.rs`
- `src/image_kernel.rs`

Reference examples:

- `examples/backend_kernel_plugin/`
- `examples/image_kernel_plugin/`

## Operational Advice

- Prefer sidecar manifests for all dynamic plugins.
- Keep plugin constructor names stable.
- Use registry-backed activation in production rather than hard-coding plugin load order.
- Treat policy plugin changes as governance changes and audit them.

## Known Limitation

Current dynamic plugin families still rely on opaque Rust trait-object payloads. This is safer than exposing raw trait pointers directly, but it is not yet a fully C-stable plugin ABI.

For that reason:

- keep host and plugin toolchains aligned
- pin target triples consistently
- treat dynamic plugin compatibility as part of release engineering

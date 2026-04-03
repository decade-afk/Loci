# Loci Plugin Guide

This guide documents the plugin model used by the refactored workspace.

## Design Rules

- Plugin manifests are the source of truth.
- Runtime governance is activated at seam boundaries, not through hardcoded branches.
- Legacy plugins may be bridged, but only inside the compatibility island.
- Host products should treat plugin activation as an operational configuration change.

## Main Crates

- `crates/plugin-api`: manifest schema and shared enums
- `crates/core`: plugin discovery, registration, activation, runtime snapshot
- `crates/legacy-plugin-api`: old plugin contract types
- `crates/legacy-plugin-compat`: bounded bridge for old text plugins

## Tracks

Plugins can target one or both platform tracks:

- `ai_infra`
- `ai_agent`

If `target_tracks` is omitted, the manifest is treated as available to both tracks.

## Core Rewriter Seams

Plugins can declare ownership of the following core components:

- `inference`
- `model`
- `hardware`
- `workflow`
- `event_bus`
- `plugin_manager`
- `ui_host`

Declaring a rewriter does not activate it by itself unless the plugin bootstrap explicitly requests activation or the host activates it through the management API.

## Manifest Shape

Example:

```toml
name = "example-inference"
version = "0.1.0"
api_version = "1.0"
target_tracks = ["ai_infra"]

[contributes]
inference_hooks = ["sampling-profile"]
commands = ["inference:activate"]

[core_rewriters]
inference = true

[runtime]
sampling_profile = "sampling-hook.toml"
```

Relevant sections:

- root fields: identity and compatibility
- `[contributes]`: declared capabilities surfaced in runtime inventory
- `[core_rewriters]`: which core seams the plugin can govern
- `[runtime]`: artifacts such as dynamic library paths, wasm paths, or sampling profiles
- `[bootstrap]`: optional auto-activation list
- `[compatibility]`: legacy bridge information

## Example Bundles

The repository ships manifest-first examples in:

- `plugins/example-inference`
- `plugins/example-infra`
- `plugins/example-agent`
- `plugins/example-ui-shell`

These examples are intentionally simple and aligned with the current workspace architecture.

## Loading Plugins

At CLI startup:

```bash
cargo run -p loci-cli -- --plugin-dir plugins
```

Over management HTTP:

```bash
curl http://127.0.0.1:8080/v1/plugins/load \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"plugins\",\"source_kind\":\"directory\"}"
```

Load request shapes:

- `source_kind = "bundle_file"` for a specific manifest file
- `source_kind = "directory"` for recursive directory discovery

## Activating Governance

Activate a declared rewriter:

```bash
curl http://127.0.0.1:8080/v1/core/rewriters/activate \
  -H "Content-Type: application/json" \
  -d "{\"component\":\"workflow\",\"plugin_name\":\"example-agent\"}"
```

Activate the example UI host:

```bash
curl http://127.0.0.1:8080/v1/core/rewriters/activate \
  -H "Content-Type: application/json" \
  -d "{\"component\":\"ui_host\",\"plugin_name\":\"example-ui-shell\"}"
curl http://127.0.0.1:8080/v1/ui
```

Inspect runtime status:

```bash
curl http://127.0.0.1:8080/v1/core/rewriters
curl http://127.0.0.1:8080/v1/core/rewriters/inventory
curl http://127.0.0.1:8080/v1/plugins
curl http://127.0.0.1:8080/v1/plugins/example-agent
```

## Legacy Compatibility

Legacy text plugins are still supported through compatibility metadata and explicit activation:

- `crates/legacy-plugin-api`
- `crates/legacy-plugin-compat`
- management routes:
  - `POST /v1/legacy-text/activate`
  - `POST /v1/legacy-text/deactivate`

This is intentionally a compatibility island. New plugins should target the manifest-first workspace contracts.

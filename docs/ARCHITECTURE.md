# Loci Architecture

## Positioning

Loci is the infra runtime below products such as `PetCompanion`. It owns local inference concerns and avoids application-layer ownership.

## Layering

### 1. Core Layer

`crates/core` contains:

- backend registry
- model load configuration
- inference pipeline
- plugin manifest loading
- runtime snapshot and active plugin state

The core does not know about product UI, desktop windows, or end-user workflow shells.

### 2. Plugin Layer

Plugins are declared through `manifest.toml`.

The current stable mainline only recognizes:

- `model_loader`
- `hardware_backend`

Other plugin categories remain roadmap items and are intentionally not part of the current stable host contract. `llama.cpp` remains the default built-in backend rather than being treated as product logic.

### 3. Interface Layer

Loci exposes three embedding surfaces:

- Rust crate API via `loci-core`
- C ABI via `loci-ffi`
- local sidecar HTTP surface via `loci-server`

## Runtime Flow

1. Discover plugin manifests from a directory or explicit path.
2. Register or activate plugins by kind.
3. Load a model through the selected backend.
4. Build effective inference params from the pipeline defaults and request overrides.
5. Run generation and return runtime-aware output metadata.

## Boundary Rules

- No desktop shell logic in `loci-core`.
- No Tauri or product animation concerns in the workspace mainline.
- No requirement for `PetCompanion`-specific abstractions in the plugin API.
- Hardware-specific policy belongs to hardware backend plugins or backend config, not app code.

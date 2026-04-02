# Loci FFI

`crates/ffi` is the stable public C ABI for the current Loci workspace.

Public header:

- `include/loci.h`

Primary library outputs:

- Windows: `loci.dll`, `loci.lib`
- Linux: `libloci.so`, `libloci.a`
- macOS: `libloci.dylib`, `libloci.a`

## Scope

The stable ABI is aligned with the new architecture rather than the removed root monolith.

It exposes:

- engine lifecycle
- model loading through the new runtime configuration path
- text generation
- runtime snapshot and backend inventory as JSON
- plugin inventory, plugin detail, and core rewriter inventory as JSON
- plugin directory / bundle loading
- core rewriter activation
- legacy text plugin activation / deactivation

## Design Rules

- Opaque runtime state is represented by `LociEngine`
- complex control-plane structures are returned as UTF-8 JSON strings
- generated text is returned as owned UTF-8 strings
- callers must free owned strings with `loci_free_string`
- the latest thread-local error can be read with `loci_get_last_error`

## Main Entry Points

- `loci_engine_new`
- `loci_engine_new_with_model`
- `loci_engine_load_model_json`
- `loci_generate`
- `loci_generate_with_len`
- `loci_generate_with_options`
- `loci_engine_runtime_snapshot_json`
- `loci_engine_backend_capabilities_json`
- `loci_engine_plugin_statuses_json`
- `loci_engine_plugin_detail_json`
- `loci_engine_core_rewriter_inventory_json`
- `loci_engine_load_plugin_bundle_json`
- `loci_engine_load_plugin_dir_json`
- `loci_engine_activate_core_rewriter_json`
- `loci_engine_activate_legacy_text_plugin_json`

## Build

```bash
cargo build -p loci-ffi --release
```

With `llama.cpp` support:

```bash
cargo build -p loci-ffi --release --features llama
```

## Test

```bash
cargo test -q -p loci-ffi
```

With `llama.cpp` enabled:

```bash
cargo test -q -p loci-ffi --features llama
```

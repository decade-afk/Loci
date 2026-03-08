# Loci API Reference (v1)

Applies to: `loci v0.1.x` in this repository.

This document is for external integrators (desktop apps, web services, code tools) that need to call Loci.

## Integration Surfaces

- Rust crate embedding (`loci` crate)
- C ABI (`include/loci.h`)
- REST API (`loci serve`)
- Dynamic backend kernel plugin (`--backend-lib`)
- Dynamic image kernel plugin (`loci image --kernel-plugin`)

Template package:

- `examples/integration/templates/` (Python/Go/Java/Tauri)

## REST API

Start service:

```bash
target/release/loci.exe serve \
  --model D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --max-prompt-bytes 65536 \
  --cpu-only
```

Base URL: `http://127.0.0.1:8080`

### Health

- `GET /health`
- `GET /v1/health`

Response:

```json
{"status":"ok"}
```

### Runtime Model Info

- `GET /info`
- `GET /v1/info`

Response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "n_vocab": 151936,
  "n_ctx_train": 32768,
  "n_embd": 1024
}
```

### Generate

- `POST /generate`
- `POST /v1/generate`
- Content type: `application/json`

Request body:

```json
{
  "prompt": "hello",
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.95,
  "min_p": 0.0,
  "top_k": 40,
  "repetition_penalty": 1.1
}
```

Success response:

```json
{
  "response": "..."
}
```

Error response:

```json
{
  "error": "..."
}
```

### HTTP Status Semantics

- `400`: malformed request or invalid JSON
- `413`: payload too large (`Content-Length > 8 MiB`)
- `404`: endpoint not found
- `500`: inference/internal error

OpenAPI file:

- `docs/openapi/loci-rest-v1.yaml`

### Prompt Safety Limit (CLI/REST path)

- CLI commands that build an inference engine support `--max-prompt-bytes <N>`:
  - `loci generate ...`
  - `loci serve ...`
  - `loci agent ...`
  - legacy mode (`loci --model ...`)
- Minimum value is `1024` bytes.
- If `--max-prompt-bytes` is provided, it overrides `LOCI_MAX_PROMPT_BYTES` for that process.
- If `--max-prompt-bytes` is not provided:
  - valid `LOCI_MAX_PROMPT_BYTES` is used;
  - invalid/too-small env value falls back to internal default.
- Internal default is about `24 KiB` UTF-8 bytes.

## C ABI (`include/loci.h`)

Build artifacts:

- Windows: `target/release/loci.dll`
- Linux: `target/release/libloci.so`
- macOS: `target/release/libloci.dylib`
- Header: `include/loci.h`

### Engine Lifecycle

- `loci_engine_new(model_path, n_ctx, n_gpu_layers)`
- `loci_engine_new_auto(model_path, n_ctx)`
- `loci_engine_new_with_device(model_path, n_ctx, device_id, n_gpu_layers)`
- `loci_engine_free(engine)`
- `loci_engine_free_safe(&engine_ptr)` (recommended for host-side safety)

### Generation

- `loci_generate(engine, prompt, max_tokens, temperature)`
- `loci_generate_with_len(engine, prompt, prompt_len, max_tokens, temperature)` (preferred for FFI safety)
- `loci_generate_wait(engine, prompt, max_tokens, temperature, wait_timeout_ms)`
- `loci_generate_wait_with_len(engine, prompt, prompt_len, max_tokens, temperature, wait_timeout_ms)`
- `loci_generate_stream(engine, prompt, max_tokens, temperature, callback, user_data)`
- `loci_generate_stream_with_len(engine, prompt, prompt_len, max_tokens, temperature, callback, user_data)`
- `loci_generate_stream_wait(engine, prompt, max_tokens, temperature, callback, user_data, wait_timeout_ms)`
- `loci_generate_stream_wait_with_len(engine, prompt, prompt_len, max_tokens, temperature, callback, user_data, wait_timeout_ms)`
- `loci_free_string(str_ptr)`
- Native safety guard defaults to ~24 KiB UTF-8 bytes, configurable via `LOCI_MAX_PROMPT_BYTES` (minimum `1024`).
- Long prompts are tokenized internally in UTF-8-safe chunks to reduce tokenizer stack-overflow risk on some platforms.
- `*_with_len` APIs accept explicit byte length and avoid C-string scan/termination ambiguity.
- `*_with_len` tokenizer path is binary-safe for UTF-8 payloads (interior NUL bytes are allowed).

FFI recommendation:

- Prefer `*_with_len` APIs in production.
- For `*_wait` variants, set `wait_timeout_ms` according to worst-case single-call latency under your hardware/load profile.
- If your host process also uses CLI paths, keep `--max-prompt-bytes` and `LOCI_MAX_PROMPT_BYTES` aligned to avoid mismatched limits.

### Model Info

- `loci_get_vocab_size(engine)`
- `loci_get_context_size(engine)`

### Plugin Registry

- `loci_registry_new()` / `loci_registry_free(registry)`
- `loci_registry_load_plugin(registry, path)` (`.wasm` path uses sandbox loader)
- `loci_registry_unload_plugin(registry, name)` (dynamic/wasm only)
- `loci_registry_reload_plugin(registry, name)` (dynamic/wasm only)
- `loci_registry_enable_plugin(registry, name)`
- `loci_registry_disable_plugin(registry, name)`
- `loci_registry_count(registry)`
- `loci_registry_list_json(registry)` (returns JSON, free with `loci_free_string`)
- `loci_registry_save(registry, path)`
- `loci_registry_load(registry, path)`

Hot-swap behavior:

- Dynamic and WASM plugins support runtime unload/reload.
- Static plugins are not unloadable/reloadable at runtime (use enable/disable).
- `loci_registry_list_json` returns fields: `name`, `version`, `enabled`,
  `plugin_type`, `source`, `hot_reloadable`.
- Example payload:

```json
[
  {
    "name": "openclaw_adapter",
    "version": "1.0.0",
    "enabled": true,
    "plugin_type": "dynamic",
    "source": "examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll",
    "hot_reloadable": true
  }
]
```

### Device Selection

- `loci_device_selector_new()` / `loci_device_selector_free(selector)`
- `loci_get_device_count(selector)`
- `loci_get_device_info(selector, index, out_info)`
- `loci_auto_select_device(selector)`
- `loci_recommend_device_for_model(selector, model_size_gb)`
- `loci_has_backend(selector, device_type)`
- `loci_has_gpu_support()`

### Version and Error

- `loci_version()`
- `loci_get_last_error()`

### C ABI Concurrency Semantics

- Calls on the same `LociEngine*` are serialized internally.
- If another call is in progress, APIs return an error and `loci_get_last_error()` reports:
  `engine is busy (another inference call is in progress)`.
- For queue-like behavior, use `*_wait` variants with timeout:
  - `loci_generate_wait(...)`
  - `loci_generate_stream_wait(...)`
- `loci_engine_free(...)` / `loci_engine_free_safe(...)` now wait for in-flight
  calls (with internal timeout) before destroying engine memory.
- Stream callback should not re-enter generation APIs on the same engine handle.
  Avoid calling free APIs reentrantly from the same stream callback.

Stress validation script:

```bash
python scripts/stress_ffi.py \
  --dll target/release/loci.dll \
  --model D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf \
  --threads 4 \
  --iters 20 \
  --use-wait \
  --wait-timeout-ms 1000
```

## Rust Embedding

```rust
use loci::inference::{GenerationParams, InferenceEngine};
use loci::model::ModelConfig;

fn main() -> loci::Result<()> {
    let config = ModelConfig::new("D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf")
        .with_context_size(4096)
        .with_gpu_layers(0);
    let mut engine = InferenceEngine::new(config)?;

    let out = engine.generate("Hello", GenerationParams::default())?;
    println!("{out}");
    Ok(())
}
```

Dynamic backend kernel from Rust builder:

```rust
let mut engine = InferenceEngine::builder()
    .model_path("model.gguf")
    .load_dynamic_backend("plugin.llama.cpp", "backend_kernel_plugin.dll")
    .backend("plugin.llama.cpp")
    .build()?;
```

## Plugin Kernel ABI

### Text backend kernel plugin

Export at least one constructor:

- `create_backend_v1` (preferred, opaque ABI)
- `create_backend` (legacy fallback)

Reference: `examples/backend_kernel_plugin/src/lib.rs`

### Image kernel plugin

Export at least one constructor:

- `create_image_plugin_v1` (preferred, opaque ABI)
- `create_image_plugin` (legacy fallback)

Reference: `examples/image_kernel_plugin/src/lib.rs`

OpenClaw-style text agent adapter template:

- `examples/openclaw_adapter_plugin/`
- Dynamic text plugin with JSON tool-call envelope normalization (`tool_call` / `final`)
- Designed for host-side tool execution loops

## Compatibility Notes

- Dynamic plugin/kernel ABI uses opaque trait-object payloads. Keep host and plugin on compatible Rust toolchain + target.
- For long-term stable third-party integration, prefer REST or C ABI.
- Pin Loci version and model version in production.

## Related Docs

- `docs/INTEGRATION_GUIDE.md`
- `docs/openapi/loci-rest-v1.yaml`
- `examples/integration/templates/README.md`

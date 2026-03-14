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

Runtime-discoverable OpenAPI document:

- `GET /openapi.yaml`
- `GET /openapi.json`
- `GET /v1/openapi.yaml`
- `GET /v1/openapi.json`

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
  "engine": "loci",
  "positioning": "embeddable-ai-inference-engine",
  "runtime_model": "loci/llama.cpp:llama",
  "integrations": {
    "rest": true,
    "c_api": true,
    "plugin_upgradeable": true,
    "openapi_spec_path": "/openapi.yaml",
    "openapi_spec_json_path": "/openapi.json"
  },
  "n_vocab": 151936,
  "n_ctx_train": 32768,
  "n_embd": 1024,
  "n_layer": 24,
  "architecture": "llama"
}
```

Notes:

- `GET /info` is the primary runtime self-description endpoint for external integrators.
- It includes backend capabilities, plugin/tool counts, and compatibility/integration flags in addition to the legacy model fields shown above.
- `integrations.openapi_spec_path` points at the embedded YAML OpenAPI document exposed by the running service.
- `integrations.openapi_spec_json_path` points at the JSON form of the same embedded spec.

### Runtime Metrics

- `GET /metrics`
- `GET /v1/metrics`

Response:

```json
{
  "status": "ok",
  "started_at_unix_ms": 1741300000000,
  "uptime_ms": 3210,
  "total_requests": 12,
  "total_client_errors": 1,
  "total_server_errors": 0,
  "average_latency_ms": 14.5,
  "endpoint_hits": {
    "health": 2,
    "info": 1,
    "generate": 4
  }
}
```

Notes:

- Metrics are runtime-local and reset when the serve process restarts.
- `endpoint_hits` uses logical endpoint names, not raw URL paths.

### OpenAPI Discovery

- `GET /openapi.yaml`
- `GET /openapi.json`
- `GET /v1/openapi.yaml`
- `GET /v1/openapi.json`

Notes:

- The response body is the embedded `docs/openapi/loci-rest-v1.yaml` served directly by the running Loci process.
- The JSON route is generated from the embedded YAML spec by the running Loci process.
- These are the preferred machine-readable discovery surfaces for code generators, desktop hosts, browsers, and agent frameworks.

### Runtime Events / Audit Stream

Routes:

- `GET /events`
- `GET /events/stream`
- versioned aliases under `/v1/...`

Notes:

- `GET /events?limit=100` returns the most recent structured runtime events recorded by the in-process event bus.
- `GET /events/stream?replay=100&follow=true` returns `application/x-ndjson`; it can replay a recent window first and then continue streaming live events.
- Events currently cover high-value control-plane mutations such as tool invocation, tool plugin lifecycle, session lifecycle, model asset registration/import/removal, background pull submission/cancellation, and model pull policy/verifier changes.
- Each event includes a monotonic sequence number, timestamp, category, action, outcome, HTTP metadata, subject, and optional JSON details.
- This stream is intended for hosts that need auditability, UI activity feeds, external logging bridges, or runtime supervision around Loci.

Example event:

```json
{
  "sequence": 12,
  "at_unix_ms": 1763126400123,
  "category": "session",
  "action": "sessions.generate",
  "outcome": "success",
  "endpoint": "sessions",
  "method": "POST",
  "path": "/sessions/7/generate",
  "status_code": 200,
  "subject": "7",
  "details": {
    "persisted": true,
    "state": "Ready"
  }
}
```

### Model Placement Planning

- `POST /models/plan`
- `POST /v1/models/plan`

Request body:

```json
{
  "model": "D:/models/qwen.gguf",
  "model_id": null,
  "context_size": 8192
}
```

Notes:

- `model` and `model_id` are optional; if both are omitted on the serve runtime, Loci plans against the currently served model.
- The response includes detected devices, the GGUF-aware estimate, and the selected placement plan.

### Model Asset Control Plane

Routes:

- `GET /models/assets`
- `POST /models/assets`
- `POST /models/assets/pull`
- `GET /models/assets/pulls`
- `POST /models/assets/pulls`
- `GET /models/assets/pulls/{job_id}`
- `POST /models/assets/pulls/{job_id}/cancel`
- `GET /models/assets/pulls/{job_id}/events`
- `GET /models/assets/{id}`
- `DELETE /models/assets/{id}?delete_file=true|false`
- versioned aliases under `/v1/...`

Register an existing local model file:

```json
{
  "path": "D:/models/qwen.gguf",
  "id": "qwen-local",
  "name": "Qwen Local",
  "tags": ["reasoning", "gguf"]
}
```

Import/copy a model into the managed store:

```json
{
  "source": "D:/downloads/qwen.gguf",
  "mirrors": [],
  "id": "qwen-managed",
  "name": "Qwen Managed",
  "sha256": null,
  "stream": false,
  "no_resume": false,
  "tags": ["managed"]
}
```

Notes:

- `POST /models/assets` registers an existing file without copying it.
- `POST /models/assets/pull` imports from a local path or HTTP(S) source into the managed model store.
- `POST /models/assets/pull` supports `stream=true` in the request body or query string and returns `application/x-ndjson` progress events followed by a final completion event.
- `POST /models/assets/pulls` starts the same import flow as a background control-plane job and returns `202 Accepted` with a job snapshot.
- `GET /models/assets/pulls/{job_id}/events` returns a snapshot event first, then live NDJSON events until the job reaches `completed`, `failed`, or `cancelled`.
- `POST /models/assets/pulls/{job_id}/cancel` requests cancellation for an in-flight background job and returns `409 Conflict` if the job has already finished.
- Both synchronous and background pulls are evaluated against the currently active model pull policy before bytes are imported into the managed store.
- After download and checksum validation, the currently active model pull verifier can still reject the asset before it is persisted into the managed store.
- `DELETE /models/assets/{id}` removes the model record; set `delete_file=true` to also remove the referenced file from disk.
- These routes are part of the control plane and are intended for desktop hosts, service wrappers, and orchestration layers that manage model inventories around Loci.

Example NDJSON stream:

```json
{"kind":"progress","progress":{"phase":"Fetching","source":"D:/downloads/qwen.gguf","model_id":"qwen-managed","destination":"D:/loci/models/blobs/qwen-managed.gguf","bytes_transferred":1048576,"total_bytes":734003200,"resumed_bytes":0,"done":false,"message":"importing local model bytes","error":null}}
{"kind":"progress","progress":{"phase":"Verifying","source":"D:/downloads/qwen.gguf","model_id":"qwen-managed","destination":"D:/loci/models/blobs/qwen-managed.gguf","bytes_transferred":734003200,"total_bytes":734003200,"resumed_bytes":0,"done":false,"message":"verifying checksums","error":null}}
{"kind":"complete","model":{"id":"qwen-managed","name":"qwen-managed","path":"D:/loci/models/blobs/qwen-managed.gguf","source":"D:/downloads/qwen.gguf","size_bytes":734003200,"checksum_xxh64":"...","checksum_sha256":"...","created_at_unix_ms":1730937600000,"tags":["managed"],"managed":true}}
```

Example background job snapshot:

```json
{
  "job_id": "pull-1730937600000-1",
  "state": "running",
  "created_at_unix_ms": 1730937600000,
  "started_at_unix_ms": 1730937600050,
  "finished_at_unix_ms": null,
  "policy_name": "checksum-required-remote.model.pull",
  "verifier_name": "sidecar-sha256.model.verify",
  "request": {
    "source": "https://example.com/qwen.gguf",
    "mirrors": [],
    "id": "qwen-managed",
    "name": null,
    "sha256": null,
    "resume": true,
    "tags": ["managed"]
  },
  "progress": {
    "phase": "Fetching",
    "source": "https://example.com/qwen.gguf",
    "model_id": "qwen-managed",
    "destination": "D:/loci/models/blobs/qwen-managed.gguf",
    "bytes_transferred": 1048576,
    "total_bytes": 734003200,
    "resumed_bytes": 0,
    "done": false,
    "message": "downloading model bytes",
    "error": null
  },
  "model": null,
  "error": null
}
```

### Session Control Plane

Routes:

- `GET /sessions`
- `POST /sessions`
- `GET /sessions/{id}`
- `POST /sessions/{id}/generate`
- `POST /sessions/{id}/suspend`
- `POST /sessions/{id}/resume`
- `POST /sessions/{id}/save`
- `POST /sessions/{id}/restore`
- `POST /sessions/{id}/clear`
- `POST /sessions/{id}/destroy`
- `DELETE /sessions/{id}/destroy`
- versioned aliases under `/v1/...`

Create request:

```json
{
  "model": "D:/models/qwen.gguf",
  "model_id": null,
  "context_size": 4096,
  "save": true
}
```

Generate request:

```json
{
  "prompt": "Continue the previous task",
  "max_tokens": 128,
  "save": true
}
```

Notes:

- `GET /sessions` returns both active in-memory sessions and persisted session ids.
- `GET /sessions/{id}` returns session summary plus conversation records.
- `suspend`, `resume`, `save`, `restore`, `clear`, and `destroy` are intended for host-side orchestration and resumable agent workflows.

### Tool Registry and Invocation

Routes:

- `GET /tools`
- `GET /tools/{name}`
- `POST /tools/invoke`
- `GET /tools/plugins`
- `GET /tools/plugins/{name}`
- `POST /tools/plugins/load`
- `POST /tools/plugins/{name}/reload`
- `POST /tools/plugins/{name}/unload`
- versioned aliases under `/v1/...`

Invoke request:

```json
{
  "name": "browser_open_session",
  "arguments": {
    "url": "https://example.com"
  }
}
```

Plugin load request:

```json
{
  "path": "plugins/browser_tool_plugin.dll",
  "activate": false
}
```

Notes:

- `POST /tools/invoke` returns `{ "tool": "...", "ok": bool, "result": ..., "error": ... }`.
- Tool plugin registry responses include plugin name, version, source path, dynamic flag, and exported function names.
- Runtime tool plugins remain subject to Loci's plugin ABI and manifest validation rules.

### Policy Registries

Routes:

- `GET /dispatch-policies`
- `GET /dispatch-policies/{name}`
- `POST /dispatch-policies/load`
- `POST /dispatch-policies/{name}/activate`
- `POST /dispatch-policies/{name}/reload`
- `POST /dispatch-policies/{name}/unload`
- `GET /execution-policies`
- `GET /execution-policies/{name}`
- `POST /execution-policies/load`
- `POST /execution-policies/{name}/activate`
- `POST /execution-policies/{name}/reload`
- `POST /execution-policies/{name}/unload`
- `GET /auth-policies`
- `GET /auth-policies/{name}`
- `POST /auth-policies/load`
- `POST /auth-policies/{name}/activate`
- `POST /auth-policies/{name}/reload`
- `POST /auth-policies/{name}/unload`
- `GET /model-pull-policies`
- `GET /model-pull-policies/{name}`
- `POST /model-pull-policies/load`
- `POST /model-pull-policies/{name}/activate`
- `POST /model-pull-policies/{name}/reload`
- `POST /model-pull-policies/{name}/unload`
- `GET /model-pull-verifiers`
- `GET /model-pull-verifiers/{name}`
- `POST /model-pull-verifiers/load`
- `POST /model-pull-verifiers/{name}/activate`
- `POST /model-pull-verifiers/{name}/reload`
- `POST /model-pull-verifiers/{name}/unload`
- versioned aliases under `/v1/...`

Common load request:

```json
{
  "path": "plugins/policy_plugin.dll",
  "activate": true
}
```

Notes:

- Collection responses report the currently active policy and the registered policy descriptors.
- `reload` and `unload` reject active policies with `409 Conflict`.
- Management auth policy activation can also fail with `409 Conflict` if the candidate policy would deny the current request and lock out management access.
- Builtin model pull policies currently include `allow-all.model.pull`, `local-only.model.pull`, and `checksum-required-remote.model.pull`.
- Model pull verifiers are the post-download governance layer; builtin verifiers currently include `allow-all.model.verify` and `sidecar-sha256.model.verify`.

### Management Auth Behavior

When control-plane auth is enabled for a route, Loci returns:

```json
{
  "error": "request denied by policy",
  "policy": "loopback-only.management.auth"
}
```

Notes:

- Public runtime discovery paths such as `/health`, `/info`, `/metrics`, `/openapi.yaml`, and `/openapi.json` remain outside the default `control-plane` auth scope.
- Control-plane routes such as `/models/plan`, `/sessions`, and policy-management endpoints may require auth depending on the active management auth policy.

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

### OpenAI-Compatible Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Notes:

- `POST /v1/chat/completions` supports standard JSON responses and `stream=true` SSE responses.
- Streaming uses native token callbacks when the active backend supports them; otherwise Loci falls back to buffered compatibility chunks.
- `POST /v1/embeddings` accepts either a single string or a string array in `input`.
- These routes reuse the active native Loci engine instead of launching a separate compatibility adapter.

### Ollama-Compatible Endpoints

- `GET /api/tags`
- `POST /api/generate`

Notes:

- `POST /api/generate` supports standard JSON responses and `stream=true` NDJSON-style chunk responses.
- Streaming uses native token callbacks when the active backend supports them; otherwise Loci falls back to buffered compatibility chunks.
- `options.num_predict`, `options.temperature`, and `options.top_p` are mapped onto Loci sampling parameters.

### HTTP Status Semantics

- `400`: malformed request or invalid JSON
- `401`: management auth policy rejected the request
- `413`: payload too large (`Content-Length > 8 MiB`)
- `404`: endpoint not found
- `405`: method not allowed for a valid endpoint family
- `409`: state conflict (for example reloading/unloading an active policy)
- `500`: inference/internal error

OpenAPI file:

- `docs/openapi/loci-rest-v1.yaml`
- live endpoint: `GET /openapi.yaml`
- live JSON endpoint: `GET /openapi.json`

### Prompt Safety Limit (CLI/REST path)

- CLI commands that build an inference engine support `--max-prompt-bytes <N>`:
  - `loci generate ...`
  - `loci serve ...`
  - `loci agent ...`
  - legacy mode (`loci --model ...`)
- CLI and builder also support automatic placement with `--auto-resource-plan` /
  `InferenceEngine::builder().with_auto_resource_plan(true)`.
- Offline planning is available with `loci model plan --model <PATH> [--context-length N] [--json]`.
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

### Plugin Contract Manifest

Dynamic plugins may ship a sidecar manifest:

- `plugin_name.loci-plugin.json`
- `plugin_name.loci-plugin.toml`

Supported validation fields:

- `name`
- `version`
- `kind`
- `abi_version`
- `min_host_version`
- `max_host_version`
- `capabilities`

Relevant `kind` values for policy-style runtime upgrades include `execution_policy`, `management_auth_policy`, `serve_dispatch_policy`, `model_pull_policy`, and `model_pull_verifier`.

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

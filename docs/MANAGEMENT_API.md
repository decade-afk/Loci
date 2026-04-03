# Loci Management API

This document describes the management HTTP surface exposed by `crates/cli`.

## Starting The Server

```bash
cargo run -p loci-cli --features llama -- \
  --plugin-dir plugins \
  --management-bind 127.0.0.1:8080
```

Base URL example: `http://127.0.0.1:8080`

## Discovery Routes

### `GET /health`

Response:

```json
{"status":"ok"}
```

### `GET /v1/runtime`

Returns the runtime snapshot:

- loaded plugin count and names
- backend capability inventory
- active backend and model
- active inference rewriter
- configured core rewriters
- compatibility diagnostics
- plugin runtime status list

### `GET /v1/backends`

Returns backend capability metadata known to the engine.

### `GET /v1/plugins`

Returns runtime status for all loaded plugins.

### `GET /v1/plugins/{plugin_name}`

Returns detailed plugin runtime information:

- declared and active core rewriters
- contribution points
- runtime artifacts
- UI contribution inventory
- compatibility metadata

### `GET /v1/core/rewriters`

Returns the currently activated seam ownership set.

### `GET /v1/core/rewriters/inventory`

Returns all core seams with:

- current active plugin, if any
- available plugins that declare the seam

### `GET /v1/workflows`

Returns workflow inventory and active workflow rewriter.

### `GET /v1/ui`

Returns UI contribution inventory and active UI host rewriter.

### `GET /v1/events`

Returns event inventory and recent recorded events.

### `GET /v1/commands`

Returns command inventory and active plugin manager rewriter.

## Mutation Routes

### `POST /v1/plugins/load`

Request:

```json
{
  "path": "plugins",
  "source_kind": "directory"
}
```

`source_kind` values:

- `bundle_file`
- `directory`

### `POST /v1/core/rewriters/activate`

Request:

```json
{
  "component": "workflow",
  "plugin_name": "example-agent"
}
```

`component` values:

- `inference`
- `model`
- `hardware`
- `workflow`
- `event_bus`
- `plugin_manager`
- `ui_host`

### `POST /v1/core/inference/activate`

Shortcut for inference rewriter activation.

Request:

```json
{
  "plugin_name": "example-inference"
}
```

### `POST /v1/model/load`

Request:

```json
{
  "backend_name": "llama.cpp",
  "config": {
    "model_path": "D:/models/qwen.gguf",
    "n_ctx": 4096,
    "use_gpu": true,
    "n_gpu_layers": -1,
    "use_mmap": true,
    "split_mode": "layer",
    "load_strategy": { "kind": "strict" }
  }
}
```

Important config fields:

- `model_path`
- `n_ctx`
- `n_threads`
- `n_batch`
- `use_gpu`
- `n_gpu_layers`
- `use_mmap`
- `use_mlock`
- `kv_offload`
- `op_offload`
- `split_mode`
- `main_gpu`
- `tensor_split`
- `load_strategy`

`load_strategy.kind` values:

- `strict`
- `auto_reduce_gpu_layers`

### `POST /v1/inference/generate`

Request:

```json
{
  "prompt": "hello from loci",
  "params": {
    "max_tokens": 128,
    "temperature": 0.8
  }
}
```

### `POST /v1/workflows/run`

Request:

```json
{
  "workflow": "chat-workflow"
}
```

### `POST /v1/ui/present`

Request:

```json
{
  "surface_kind": "panel",
  "surface": "status"
}
```

`surface_kind` values:

- `panel`
- `window`
- `widget`

### `POST /v1/commands/run`

Request:

```json
{
  "command": "agent:start"
}
```

### `POST /v1/events/publish`

Request:

```json
{
  "event": "runtime.booted"
}
```

### `POST /v1/legacy-text/activate`

Request:

```json
{
  "plugin_name": "legacy-text-plugin"
}
```

### `POST /v1/legacy-text/deactivate`

Request:

```json
{
  "plugin_name": "legacy-text-plugin"
}
```

## Error Model

- malformed JSON or missing required fields return `400`
- missing plugin detail returns `404`
- runtime or serialization failures return `500`

The current server is intentionally minimal and uses JSON bodies with a lightweight HTTP implementation. It is designed as an operational control plane, not as a public compatibility facade.

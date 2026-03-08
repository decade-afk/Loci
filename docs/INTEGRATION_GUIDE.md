# Loci Integration Guide

Date: 2026-03-01  
Scope: current repository implementation

This guide documents direct integration paths that are available now.

API reference:
- `docs/API_REFERENCE.md`
- `docs/openapi/loci-rest-v1.yaml`

Template package:
- `examples/integration/templates/` (Python/Go/Java/Tauri)

## 1. Integration Surfaces

- Rust crate embedding (highest control).
- C ABI embedding (C/C++ and FFI for other languages).
- Python via `ctypes` (through C ABI).
- Go via `cgo` (through C ABI).
- HTTP integration via `loci serve` REST endpoints.
- Dynamic backend plugin as inference kernel (`--backend-lib`).
- Dynamic image kernel plugin (`loci image --kernel-plugin`).

Current limitation:
- gRPC server is not wired in the current `loci serve` CLI path.

## 2. Build Artifacts

Build release artifacts first:

```bash
cargo build --release --lib --bin loci
```

Common outputs:
- Windows: `target/release/loci.dll` and `target/release/loci.exe`
- Linux: `target/release/libloci.so`
- macOS: `target/release/libloci.dylib`
- Header: `include/loci.h`

## 3. Rust Embedding

Example:

```rust
use loci::inference::{GenerationParams, InferenceEngine};
use loci::model::ModelConfig;

fn main() -> loci::Result<()> {
    let config = ModelConfig::new("D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf")
        .with_context_size(512)
        .with_gpu_layers(0);
    let mut engine = InferenceEngine::new(config)?;

    let params = GenerationParams { max_tokens: 64, ..Default::default() };
    let out = engine.generate("Hello from Rust", params)?;
    println!("{out}");
    Ok(())
}
```

## 4. C ABI Embedding

Reference example: `examples/integration/c/basic_inference.c`

Core C API functions:
- `loci_engine_new`
- `loci_generate` / `loci_generate_with_len` (preferred)
- `loci_generate_stream` / `loci_generate_stream_with_len` (preferred)
- `loci_generate_wait` / `loci_generate_wait_with_len`
- `loci_generate_stream_wait` / `loci_generate_stream_wait_with_len`
- `loci_free_string`
- `loci_engine_free`

C ABI prompt guard notes:

- Prompt byte safety limit is controlled by `LOCI_MAX_PROMPT_BYTES` (minimum `1024`, default `~24 KiB`).
- `*_with_len` APIs support UTF-8 payload with interior `NUL` bytes and avoid C-string termination ambiguity.

Windows (MinGW example):

```bash
gcc examples/integration/c/basic_inference.c -Iinclude -Ltarget/release -lloci -o loci_c_demo.exe
```

Linux:

```bash
gcc examples/integration/c/basic_inference.c -Iinclude -Ltarget/release -lloci -ldl -lm -lpthread -o loci_c_demo
```

## 5. Python (`ctypes`)

Reference example: `examples/integration/python/ctypes_inference.py`

Run:

```bash
python examples/integration/python/ctypes_inference.py
```

## 6. Go (`cgo`)

Reference example: `examples/integration/go/main.go`

Run (from repository root):

```bash
go run ./examples/integration/go
```

REST template:

```bash
cd examples/integration/templates/go-rest
go run .
```

## 6.1 Java (REST)

Reference template: `examples/integration/templates/java-rest`

```bash
cd examples/integration/templates/java-rest
mvn exec:java
```

## 6.2 Tauri (Rust command + REST)

Reference template: `examples/integration/templates/tauri-rest`

- Copy `src-tauri/src/{main.rs,lib.rs,loci.rs}` into your Tauri project.
- Copy `src/loci.ts` for frontend invocation wrapper.
- Start Loci service first, then invoke `loci_health/loci_info/loci_generate`.

## 7. HTTP REST Integration

Start server:

```bash
target/release/loci.exe serve \
  --model D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --max-prompt-bytes 65536 \
  --cpu-only --gpu-layers 0 --context-length 512 --threads 1
```

Health:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/health
```

Info:

```bash
curl http://127.0.0.1:8080/info
curl http://127.0.0.1:8080/v1/info
```

Generate:

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Hello\",\"max_tokens\":32}"
```

Versioned path:

```bash
curl -X POST http://127.0.0.1:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Hello\",\"max_tokens\":32}"
```

Prompt size guard notes (REST/CLI):

- Use `--max-prompt-bytes <N>` on `generate/serve/agent` (and legacy mode) to set prompt-byte safety limit.
- Minimum accepted value is `1024`.
- If omitted, `LOCI_MAX_PROMPT_BYTES` is used when valid; otherwise default (`~24 KiB`) applies.
- If both are set, CLI `--max-prompt-bytes` takes priority.

## 8. Error Contracts (REST)

Current parser behavior:
- malformed request / invalid JSON: `400 Bad Request`
- oversized request body (`Content-Length` > 8 MiB): `413 Payload Too Large`
- unknown endpoint: `404 Not Found`
- internal generation failure: `500 Internal Server Error`

## 9. Plugin Integration Notes

- Runtime loading in CLI:
  - `--plugin path/to/plugin.wasm`
  - `--plugin path/to/plugin.dll` (or `.so`/`.dylib`)
- Registry management:
  - `loci plugin load <path>`
  - `loci plugin list`
  - `loci plugin info <name>`
  - `loci plugin reload <name>`
  - `loci plugin unload <name>`
  - `loci plugin enable <name>`
  - `loci plugin disable <name>`

Hot-swap notes:

- Dynamic (`.dll/.so/.dylib`) and WASM (`.wasm`) plugins support reload/unload.
- Static plugins are not hot-reloadable and should be toggled with enable/disable.

Important ABI note:
- Dynamic plugin constructor now uses an opaque two-pointer payload (`DynamicPluginOpaque`) in `src/plugin_registry.rs`.
- This is safer than exporting `*mut dyn Plugin` directly, but it is still Rust toolchain/target sensitive and should be treated as same-toolchain compatible.

## 10. Dynamic Backend Plugin As Inference Kernel

Reference example: `examples/backend_kernel_plugin`

Build:

```bash
cargo build --release --manifest-path examples/backend_kernel_plugin/Cargo.toml
```

Run with plugin backend:

```bash
target/release/loci.exe generate \
  --model D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf \
  --backend-lib examples/backend_kernel_plugin/target/release/backend_kernel_plugin.dll \
  --backend plugin.llama.cpp \
  --prompt "hello from plugin backend" \
  --cpu-only
```

Notes:
- If `--backend-lib` is provided and `--backend` is omitted (or `llama.cpp`), CLI auto-selects `dynamic.plugin`.
- You can force a custom registration name using `--backend-register-name`.

## 11. Dynamic Image Kernel Plugin (Text-to-Image)

Reference example: `examples/image_kernel_plugin`

Build:

```bash
cargo build --release --manifest-path examples/image_kernel_plugin/Cargo.toml
```

Run:

```bash
target/release/loci.exe image \
  --prompt "a cute robot reading a book" \
  --model-id hf-internal-testing/tiny-stable-diffusion-pipe \
  --kernel-plugin examples/image_kernel_plugin/target/release/image_kernel_plugin.dll \
  --output outputs/t2i.png \
  --steps 4 \
  --guidance-scale 0
```

Offline fallback mode:

```bash
set LOCI_T2I_FALLBACK=1
```

- When enabled, if model download/load fails, script-based kernel returns a placeholder PNG so pipeline wiring can still be validated.

## 12. OpenClaw-Style Agent Adapter Plugin

Reference example: `examples/openclaw_adapter_plugin`

Build:

```bash
cargo build --release --manifest-path examples/openclaw_adapter_plugin/Cargo.toml
```

Load + inspect:

```bash
set LOCI_OPENCLAW_TOOLS_PATH=examples/openclaw_adapter_plugin/tools.example.json
target/release/loci.exe plugin load examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll
target/release/loci.exe plugin info openclaw_adapter
```

Hot swap:

```bash
target/release/loci.exe plugin reload openclaw_adapter
target/release/loci.exe plugin unload openclaw_adapter
```

This plugin enforces a JSON tool protocol (`tool_call` / `final`) and is designed for host-side tool executors.

Smoke test (hot-swap + optional model runtime):

```bash
powershell -ExecutionPolicy Bypass -File scripts/plugin_hot_swap_smoke.ps1 \
  -LociExe target/release/loci.exe \
  -OpenClawPlugin examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll \
  -ModelPath D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf
```

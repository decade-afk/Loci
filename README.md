# Loci
Loci is an embeddable AI inference engine and control plane built in Rust.

It is designed for teams that need to integrate local model execution into their own software, not for shipping yet another end-user chat shell. Loci focuses on runtime execution, host integration, plugin-based upgrades, model asset governance, and tool/session control surfaces that can sit behind desktop apps, IDE copilots, local automation products, and custom agents.

## Plugin System - Highly Extensible

Loci features a **highly plugin-capable architecture** that allows you to extend functionality without modifying the core engine:

### Three Plugin Types
- **Static Plugins**: Compiled into the binary for maximum performance
- **Dynamic Plugins**: Load/unload at runtime via shared libraries
- **WASM Plugins**: Sandboxed execution for security and cross-language support

### Plugin Contract Manifest
- Dynamic plugins can ship a sidecar manifest such as `my_plugin.loci-plugin.json`
- Manifest validation checks plugin kind, ABI version, and optional host-version bounds before load
- Runtime plugin identity is cross-checked against the loaded implementation when version metadata is available
- Policy-oriented plugin kinds include execution, management-auth, serve-dispatch, model-pull policy, and model-pull verifier plugins

### Plugin Hooks
- `pre_generate`: Modify prompts before inference
- `transform_logits`: Advanced token-level control
- `on_token`: Real-time streaming processing
- `post_generate`: Format and filter responses

### Example Plugins
```rust
use loci::prelude::*;
use loci::examples::plugins::*;

// Filter offensive language
let profanity = ProfanityFilterPlugin::new("filter");
engine.plugin_manager_mut().register(profanity)?;

// Format output as JSON
let json = JsonFormatterPlugin::new("json");
engine.plugin_manager_mut().register(json)?;

// Auto-translate
let translator = TranslationPlugin::english_to_chinese("trans");
engine.plugin_manager_mut().register(translator)?;

// Explain code
let explainer = CodeExplainerPlugin::detailed("explainer");
engine.plugin_manager_mut().register(explainer)?;
```

See [PLUGIN_GUIDE.md](PLUGIN_GUIDE.md) for plugin family overview and development entry points.
API docs for integrators:
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- [docs/openapi/loci-rest-v1.yaml](docs/openapi/loci-rest-v1.yaml)
- [examples/integration/templates/README.md](examples/integration/templates/README.md)

## Features

### Core Capabilities
- **Fast & Efficient**: Built on native llama.cpp for high-performance inference
- **Multi-Platform Support**:
  - Desktop: Windows (MSVC/MinGW), Linux (x86_64/ARM64), macOS (Intel/Apple Silicon)
  - Mobile: iOS (Device/Simulator), Android (ARM64/ARMv7/x86_64/x86)
- **Multiple Distribution Formats**:
  - Standalone CLI executable
  - Static library (`.a`/`.lib`) - 25 MB
  - Dynamic library (`.dll`/`.so`/`.dylib`) - 7.6 MB
  - C API for language interop (engine lifecycle, generation, streaming, device/plugin helpers)
- **GPU Acceleration**: Supports CUDA, Metal, and other backends
- **Streaming Support**: Real-time token streaming for interactive applications
- **Flexible Configuration**: Customizable context size, sampling parameters, and more

### Plugin System (Phase 1.1)
- **Hot-Swappable Plugins**: Load/unload/reload plugins at runtime
- **Static & Dynamic**: Support both compiled-in and shared library plugins
- **Persistent Configuration**: Save/load plugin state via TOML files
- **Centralized Registry**: Unified management for all plugins
- **Text Processing Hooks**: pre_generate, post_generate, on_token
- **Third-Party Integration**: Easy plugin development API
- **Manifest Validation**: Optional sidecar contract for safer plugin upgrades

### Compatibility APIs
- **OpenAI-compatible routes**: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/embeddings`
- **Ollama-compatible routes**: `GET /api/tags`, `POST /api/generate`
- **Same core runtime**: Compatibility routes reuse the same engine, plugin hooks, auth, and metrics path as native REST endpoints
- **Streaming compatibility**: `stream=true` is supported for OpenAI chat and Ollama generate; Loci uses native token streaming when available and falls back to buffered chunk streaming when required for compatibility
- **Control-plane introspection**: `GET /info` exposes engine/runtime capabilities, `GET /metrics` exposes runtime request metrics, and `POST /models/plan` exposes remote placement planning for integrators
- **Runtime audit stream**: `GET /events` and `GET /events/stream` expose a structured control-plane event spine for hosts that need supervision, logging, or activity feeds
- **Model inventory control plane**: `GET/POST/DELETE /models/assets...` lets host software manage registered and imported model assets over REST
- **Model source governance**: `GET/POST /model-pull-policies...` lets hosts activate builtin or plugin-provided policies for model source admission and checksum enforcement
- **Model trust verification governance**: `GET/POST /model-pull-verifiers...` lets hosts activate builtin or plugin-provided post-download verifiers for sidecar, signature, or certificate-style checks before import is committed
- **Live OpenAPI discovery**: `GET /openapi.yaml` and `GET /openapi.json` serve the embedded machine-readable API spec from the running process

### Language Integration
- **Rust API**: Type-safe, zero-cost abstractions
- **C/C++ API**: Full FFI support with header files
- **Python**: ctypes integration examples
- **Node.js**: ffi-napi support
- **Multi-Language**: Any language with FFI support

## Quick Start

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- CMake 3.14+ (for building llama.cpp)
- C/C++ Compiler:
  - **Windows**: Visual Studio 2019 or later with "Desktop development with C++" workload, OR
    - Install Build Tools for Visual Studio 2022: [Download](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
    - Select "Desktop development with C++" during installation
  - **Linux**: GCC or Clang (usually pre-installed, or install with `sudo apt install build-essential`)
  - **macOS**: Xcode Command Line Tools (`xcode-select --install`)
- A GGUF format model file (e.g., from [Hugging Face](https://huggingface.co/models))

### Installation

```bash
git clone https://github.com/decade-afk/loci.git
cd loci
git submodule update --init --recursive
cargo build --release
```

### Download a Model

Download a GGUF model, for example:

```bash
# Example: Download a small Qwen model
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Usage

#### Command Line

Generate (single prompt):

```bash
cargo run --release -- generate \
  --model path/to/model.gguf \
  --auto-resource-plan \
  --prompt "What is Rust programming language?"
```

Generate (interactive):

```bash
cargo run --release -- generate --model path/to/model.gguf
```

Generate (streaming):

```bash
cargo run --release -- generate \
  --model path/to/model.gguf \
  --prompt "Tell me a story" \
  --stream
```

Agent mode:

```bash
cargo run --release -- agent \
  --model path/to/model.gguf \
  --auto-resource-plan \
  --tool calculator \
  --prompt "Use the calculator tool to add 144 and 256."
```

OpenAI-compatible chat:

```bash
curl http://127.0.0.1:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"loci-local\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"
```

Ollama-compatible generate:

```bash
curl http://127.0.0.1:8080/api/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"loci-local\",\"prompt\":\"hello\",\"stream\":false}"
```

Control-plane model planning:

```bash
curl http://127.0.0.1:8080/models/plan ^
  -H "Content-Type: application/json" ^
  -d "{\"context_size\":8192}"
```

Register an existing model asset in the managed inventory:

```bash
curl http://127.0.0.1:8080/models/assets ^
  -H "Content-Type: application/json" ^
  -d "{\"path\":\"D:/models/qwen.gguf\",\"id\":\"qwen-local\",\"tags\":[\"reasoning\"]}"
```

Stream model import progress as NDJSON:

```bash
curl http://127.0.0.1:8080/models/assets/pull?stream=true ^
  -H "Content-Type: application/json" ^
  -d "{\"source\":\"D:/downloads/qwen.gguf\",\"id\":\"qwen-managed\"}"
```

The NDJSON stream emits `progress`, `complete`, and `error` events so hosts can surface asset import state in real time.

Start the same import as a background control-plane job:

```bash
curl http://127.0.0.1:8080/models/assets/pulls ^
  -H "Content-Type: application/json" ^
  -d "{\"source\":\"https://example.com/qwen.gguf\",\"id\":\"qwen-managed\"}"
```

Subscribe to one background pull job:

```bash
curl http://127.0.0.1:8080/models/assets/pulls/pull-1730937600000-1/events
```

Activate a stricter remote model pull policy:

```bash
curl http://127.0.0.1:8080/model-pull-policies/checksum-required-remote.model.pull/activate -X POST
```

Activate a post-download model pull verifier:

```bash
curl http://127.0.0.1:8080/model-pull-verifiers/sidecar-sha256.model.verify/activate -X POST
```

Live OpenAPI spec:

```bash
curl http://127.0.0.1:8080/openapi.yaml
```

```bash
curl http://127.0.0.1:8080/openapi.json
```

Read the recent runtime audit buffer:

```bash
curl http://127.0.0.1:8080/events?limit=20
```

Follow the live runtime audit stream:

```bash
curl http://127.0.0.1:8080/events/stream?replay=20
```

Image generation (plugin kernel):

```bash
cargo run --release -- image \
  --prompt "a cute robot reading a book" \
  --model-id hf-internal-testing/tiny-stable-diffusion-pipe \
  --kernel-plugin examples/image_kernel_plugin/target/release/image_kernel_plugin.dll \
  --output outputs/t2i.png \
  --steps 4 --guidance-scale 0
```

Dynamic backend as inference kernel:

```bash
# 1) Build backend kernel plugin (cdylib)
cargo build --release --manifest-path examples/backend_kernel_plugin/Cargo.toml

# 2) Run Loci with the plugin backend as the active inference kernel
cargo run --release -- generate \
  --model path/to/model.gguf \
  --backend-lib examples/backend_kernel_plugin/target/release/backend_kernel_plugin.dll \
  --backend plugin.llama.cpp \
  --prompt "Hello from plugin backend"
```

Serve mode (REST):

```bash
cargo run --release -- serve \
  --model path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --max-prompt-bytes 65536
```

Plugin registry management:

```bash
cargo run --release -- plugin load path/to/plugin.wasm
cargo run --release -- plugin list
cargo run --release -- plugin info your_plugin_name
cargo run --release -- plugin reload your_plugin_name
cargo run --release -- plugin unload your_plugin_name
cargo run --release -- plugin enable your_plugin_name
cargo run --release -- plugin disable your_plugin_name
```

OpenClaw-style agent adapter plugin (dynamic):

```bash
cargo build --release --manifest-path examples/openclaw_adapter_plugin/Cargo.toml
set LOCI_OPENCLAW_TOOLS_PATH=examples/openclaw_adapter_plugin/tools.example.json
cargo run --release -- plugin load examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll
```

Hot-swap smoke regression:

```bash
powershell -ExecutionPolicy Bypass -File scripts/plugin_hot_swap_smoke.ps1 `
  -LociExe target/release/loci.exe `
  -OpenClawPlugin examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll `
  -ModelPath D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf
```

Legacy compatibility mode (still supported):

```bash
cargo run --release -- -m path/to/model.gguf -p "Hello" --stream
```

#### As a Library

```rust
use loci::prelude::*;
use loci::inference::GenerationParams;

fn main() -> Result<()> {
    // Create model configuration
    let config = ModelConfig::new("path/to/model.gguf")
        .with_context_size(4096)
        .with_gpu_layers(-1); // Use all GPU layers

    // Create inference engine
    let mut engine = InferenceEngine::new(config)?;

    // Generate text
    let params = GenerationParams::default();
    let response = engine.generate("What is Rust?", params)?;
    println!("{}", response);

    Ok(())
}
```

#### Streaming Generation

```rust
use loci::prelude::*;
use loci::inference::GenerationParams;

fn main() -> Result<()> {
    let config = ModelConfig::new("path/to/model.gguf");
    let mut engine = InferenceEngine::new(config)?;

    let params = GenerationParams::default();
    engine.generate_stream("Tell me a story", params, |token| {
        print!("{}", token);
        true // Continue generating
    })?;

    Ok(())
}
```

## CLI Options

```
Subcommands:
  generate                          Generate text
  image                             Generate image (text-to-image)
  serve                             Start REST server
  agent                             Run agent mode
  plugin                            Manage plugin registry

Key generate/agent/serve options:
  -m, --model <MODEL>               Path to GGUF model
  -p, --prompt <PROMPT>             Prompt text
  -c, --context-length <SIZE>       Context length (alias: --context-size)
      --max-prompt-bytes <BYTES>    Prompt-byte safety limit (min: 1024; overrides LOCI_MAX_PROMPT_BYTES)
  -n, --max-tokens <TOKENS>         Maximum generated tokens
  -t, --temperature <TEMP>          Sampling temperature
      --top-p <TOP_P>               Top-p sampling
      --min-p <MIN_P>               Min-p sampling
      --top-k <TOP_K>               Top-k sampling
      --repetition-penalty <VAL>    Repetition penalty (alias: --repeat-penalty)
      --backend <BACKEND>           Backend name (default: llama.cpp)
      --backend-lib <PATH>          Register dynamic backend library before build
      --backend-register-name <N>   Registration name for --backend-lib
      --threads <THREADS>           Number of threads
      --cpu-only                    Disable GPU acceleration
      --gpu-layers <LAYERS>         GPU layers to offload (-1 = all)
      --auto-gpu-fallback          Retry with fewer GPU layers if requested placement fails
      --gpu-fallback-step <STEP>   GPU layer decrement for auto fallback
      --auto-resource-plan         Auto-plan GPU/CPU/mmap placement from model size + hardware
      --lora-path <PATH>            LoRA path argument (validated; merge support backend-dependent)
      --plugin <PATH>               Load runtime plugin (.wasm or dynamic library)
  -s, --stream                      Enable streaming output
```

Prompt safety limit behavior:
- Default limit is about `24 KiB` UTF-8 bytes.
- If `--max-prompt-bytes` is set, it takes priority.
- Otherwise, valid `LOCI_MAX_PROMPT_BYTES` is used.

Image command options:
- `--prompt` text prompt
- `--model-id` model id or local model path
- `--kernel-plugin` dynamic image kernel plugin path (`.dll/.so/.dylib`)
- `--output` output image path
- `--steps`, `--guidance-scale`, `--width`, `--height`, `--seed`, `--use-cuda`

Offline fallback for image kernel/plugin:
- Set `LOCI_T2I_FALLBACK=1` to allow placeholder output when model download/inference is unavailable.

### Windows (MinGW) Stability Notes

To avoid decode-time access violations on some Windows MinGW setups, Loci exposes a build-time CPU optimization tier:

- `LOCI_CPU_OPT=safe`: disable SIMD extensions (most conservative)
- `LOCI_CPU_OPT=sse42`: enable SSE4.2 only (default, recommended)
- `LOCI_CPU_OPT=avx`: enable AVX path
- `LOCI_CPU_OPT=avx2`: enable AVX2/FMA/F16C/BMI2 path

Example:

```bash
set LOCI_CPU_OPT=sse42
cargo build --release
```

### Runtime Backend & RAG Selection

```rust
use loci::prelude::*;
use loci::inference::InferenceEngine;

// Backend selection at build time
let mut engine = InferenceEngine::builder()
    .model_path("model.gguf")
    .backend("llama.cpp")
    .build()?;

// Hot-swappable RAG plugin registration and activation
engine.add_in_memory_rag_plugin(
    "kb",
    vec![RagDocument::new("doc-1", "Loci supports plugin-based inference.")],
    3,
    Some("Answer using retrieved context.".to_string()),
)?;
engine.activate_rag_plugin("kb")?;
```

## Configuration

### Model Configuration

```rust
let config = ModelConfig::new("model.gguf")
    .with_context_size(4096)      // Context window size
    .with_threads(8)               // Number of CPU threads
    .with_batch_size(512)          // Batch size for prompt processing
    .with_gpu_layers(-1)           // GPU layers (-1 = all)
    .with_auto_gpu_layer_fallback(8) // Retry with lower GPU residency if needed
    .cpu_only();                   // Disable GPU
```

Automatic resource planning from Rust:

```rust
let mut engine = InferenceEngine::builder()
    .model_path("model.gguf")
    .with_auto_resource_plan(true)
    .build()?;
```

Inspect placement before loading the model:

```bash
cargo run --release -- model plan --model model.gguf --context-length 8192 --json
```

### Generation Parameters

```rust
let params = GenerationParams {
    max_tokens: 512,        // Maximum tokens to generate
    temperature: 0.8,       // Sampling temperature
    top_p: 0.95,           // Nucleus sampling threshold
    top_k: 40,             // Top-k sampling threshold
    repeat_penalty: 1.1,   // Repetition penalty
};
```

## Using as a Library

### C/C++ Integration

Loci provides a C API for integration with other languages:

```c
#include "loci.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main() {
    // Create inference engine
    LociEngine* engine = loci_engine_new("model.gguf", 4096, -1);
    if (!engine) return 1;

    // Preferred path: explicit byte length API (UTF-8 payload, interior NUL-safe)
    const char* prompt = "Hello, world!";
    uint32_t prompt_len = (uint32_t)strlen(prompt);
    char* result = loci_generate_with_len(engine, prompt, prompt_len, 50, 0.8f);
    if (!result) {
        const char* err = loci_get_last_error();
        fprintf(stderr, "loci error: %s\n", err ? err : "(null)");
        loci_engine_free(engine);
        return 1;
    }
    printf("%s\n", result);

    // Cleanup
    loci_free_string(result);
    loci_engine_free(engine);
    return 0;
}
```

**Linking:**

```bash
# Static linking
gcc your_app.c -I./include -L./target/release -lloci -ldl -lm -lpthread -o your_app

# Dynamic linking
gcc your_app.c -I./include -L./target/release -lloci -Wl,-rpath,'$ORIGIN' -o your_app
```

See [BUILD.md](BUILD.md) for detailed build instructions for all platforms.

## Building from Source

```bash
# Clone with submodules
git clone --recursive https://github.com/decade-afk/loci.git
cd loci

# Build executable
cargo build --release

# Build as library (generates .a, .dll/.so/.dylib)
cargo build --release --lib

# Run tests
cargo test

# Recommended local full validation on Windows
powershell -ExecutionPolicy Bypass -File scripts/full_test.ps1

# Run benchmarks
cargo bench
```

For cross-compilation and platform-specific builds, see [BUILD.md](BUILD.md).

## Project Structure

```
loci/
|-- src/
|   |-- lib.rs          # Library entry point
|   |-- main.rs         # CLI application
|   |-- error.rs        # Error types
|   |-- model.rs        # Model configuration
|   `-- inference.rs    # Inference engine
|-- docs/               # Documentation
|-- tests/              # Integration tests
|-- benches/            # Benchmarks
|-- examples/           # Usage examples
|-- models/             # Model files directory
|-- deps/
|   `-- llama.cpp/      # llama.cpp submodule
`-- Cargo.toml
```

## Roadmap

Current snapshot (2026-03-01):

- Phase 1.0: available and validated (core local inference, CLI, C API, streaming).
- Phase 1.1: available (plugin registry, dynamic/WASM loading, persistence).
- Phase 1.5/2/3: partially implemented; many modules exist, but not all are fully wired as production end-to-end pipelines.

Detailed status with evidence:
- [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md)
- [docs/PRODUCT_STRATEGY_2026.md](docs/PRODUCT_STRATEGY_2026.md)

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System layers, diagrams, flows, and recommended next capabilities
- **[Architecture ADRs](docs/architecture/README.md)** - Decision records for positioning, adapters, and governance layering
- **[Integration Guide](docs/INTEGRATION_GUIDE.md)** - Rust/C/Python/Go/HTTP integration
- **[Phase Status](docs/PHASE_STATUS.md)** - Stage-goal audit with implementation evidence
- **[Product Strategy 2026](docs/PRODUCT_STRATEGY_2026.md)** - Positioning, milestones, and KPI roadmap
- **[Build Guide](BUILD.md)** - Platform-specific build instructions
- **[Plugin Guide](PLUGIN_GUIDE.md)** - Developing plugins for Loci

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The core inference engine
- [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) - Rust bindings for llama.cpp


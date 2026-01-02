# Loci

A cross-platform, plugin-based local LLM inference framework built in Rust.

## Features

- **Fast & Efficient**: Built on native llama.cpp for high-performance inference
- **Multi-Platform Support**:
  - Desktop: Windows (MSVC/MinGW), Linux (x86_64/ARM64), macOS (Intel/Apple Silicon)
  - Mobile: iOS (Device/Simulator), Android (ARM64/ARMv7/x86_64/x86)
- **Multiple Distribution Formats**:
  - Standalone CLI executable
  - Static library (`.a`/`.lib`)
  - Dynamic library (`.dll`/`.so`/`.dylib`)
  - C API for language interop
- **GPU Acceleration**: Supports CUDA, Metal, and other backends
- **Plugin System**: Extensible architecture for custom functionality
- **Simple API**: Easy-to-use Rust API with C bindings
- **Streaming Support**: Real-time token streaming for interactive applications
- **Flexible Configuration**: Customizable context size, sampling parameters, and more

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

Single prompt mode:

```bash
cargo run --release -- -m path/to/model.gguf -p "What is Rust programming language?"
```

Interactive mode:

```bash
cargo run --release -- -m path/to/model.gguf
```

With streaming output:

```bash
cargo run --release -- -m path/to/model.gguf -p "Tell me a story" --stream
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
Options:
  -m, --model <MODEL>              Path to the GGUF model file
  -p, --prompt <PROMPT>            Prompt text (if not provided, enters interactive mode)
  -c, --context-size <SIZE>        Context size [default: 4096]
  -n, --max-tokens <TOKENS>        Maximum tokens to generate [default: 512]
  -t, --temperature <TEMP>         Temperature (0.0 = greedy) [default: 0.8]
      --top-p <TOP_P>              Top-p sampling [default: 0.95]
      --top-k <TOP_K>              Top-k sampling [default: 40]
      --threads <THREADS>          Number of threads
      --cpu-only                   Disable GPU acceleration
      --gpu-layers <LAYERS>        GPU layers to offload (-1 = all) [default: -1]
  -s, --stream                     Enable streaming output
  -h, --help                       Print help
```

## Configuration

### Model Configuration

```rust
let config = ModelConfig::new("model.gguf")
    .with_context_size(4096)      // Context window size
    .with_threads(8)               // Number of CPU threads
    .with_batch_size(512)          // Batch size for prompt processing
    .with_gpu_layers(-1)           // GPU layers (-1 = all)
    .cpu_only();                   // Disable GPU
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

int main() {
    // Create inference engine
    LociEngine* engine = loci_engine_new("model.gguf", 4096, -1);

    // Generate text
    char* result = loci_generate(engine, "Hello, world!", 50, 0.8);
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

# Run benchmarks
cargo bench
```

For cross-compilation and platform-specific builds, see [BUILD.md](BUILD.md).

## Project Structure

```
loci/
├── src/
│   ├── lib.rs          # Library entry point
│   ├── main.rs         # CLI application
│   ├── error.rs        # Error types
│   ├── model.rs        # Model configuration
│   └── inference.rs    # Inference engine
├── tests/              # Integration tests
├── benches/            # Benchmarks
├── deps/
│   └── llama.cpp/      # llama.cpp submodule
└── Cargo.toml
```

## Roadmap

- [x] Basic llama.cpp integration
- [x] CLI tool
- [x] Streaming support
- [ ] Plugin architecture
- [ ] WebAssembly support
- [ ] Multi-model support
- [ ] Chat template support
- [ ] Function calling
- [ ] RAG integration

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

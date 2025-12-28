# Loci Quick Start Guide

Welcome to Loci! This guide will help you get started with the Loci AI inference engine in just a few minutes.

---

## Prerequisites

- **Rust**: 1.85 or higher ([Install Rust](https://rustup.rs/))
- **Operating System**: Linux, macOS, or Windows
- **Hardware**:
  - Minimum: 8GB RAM, 4-core CPU
  - Recommended: 16GB RAM, GPU (NVIDIA/AMD/Apple Silicon)

---

## Installation

### Option 1: Install from Cargo (Recommended)

```bash
cargo install loci
```

### Option 2: Build from Source

```bash
# Clone the repository
git clone https://github.com/decade-afk/Loci.git
cd Loci

# Build in release mode
cargo build --release

# The binary will be at target/release/loci
```

### Option 3: Use Docker

```bash
docker pull loci/loci:latest
```

---

## Download a Model

Loci uses GGUF format models. Download a model from Hugging Face:

```bash
# Example: TinyLlama 1.1B (Q4_0, ~600MB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Or Llama-2-7B (Q4_K_M, ~4GB)
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
```

---

## Your First Inference

### Step 1: Create a Simple Rust Program

Create a new file `main.rs`:

```rust
use loci::{LociEngine, EngineConfig};

fn main() -> anyhow::Result<()> {
    // Configure the engine (including sampling parameters)
    let config = EngineConfig {
        model_path: "tinyllama-1.1b-chat-v1.0.Q4_0.gguf".to_string(),
        n_gpu_layers: -1,    // Use all available GPU layers
        temperature: 0.7,    // Sampling temperature
        top_k: 40,           // Top-K sampling
        top_p: 0.9,          // Top-P sampling
        ..Default::default()
    };

    // Load the model
    println!("Loading model...");
    let engine = LociEngine::new(config)?;
    println!("Model loaded successfully!");

    // Generate text
    let prompt = "What is the capital of France?";
    println!("\nPrompt: {}", prompt);
    println!("\nGenerating response...\n");

    let response = engine.generate(prompt, 100)?;
    println!("Response: {}", response);

    // Show performance stats
    let stats = engine.stats();
    println!("\nPerformance:");
    println!("  Generation: {:.2} tokens/s", stats.eval_tokens_per_second());

    Ok(())
}
```

### Step 2: Add Dependencies

Create `Cargo.toml`:

```toml
[package]
name = "loci-example"
version = "0.1.0"
edition = "2021"

[dependencies]
loci = "0.1"
anyhow = "1.0"
```

### Step 3: Run

```bash
cargo run --release
```

Expected output:
```
Loading model...
Model loaded successfully!

Prompt: What is the capital of France?

Generating response...

Response: The capital of France is Paris.
```

---

## Using the CLI

Loci also provides a command-line interface:

### Basic Generation

```bash
loci generate --model tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "Hello, world!"
```

### Interactive Mode

```bash
loci interactive --model tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

### Start HTTP Server

```bash
loci serve --model tinyllama-1.1b-chat-v1.0.Q4_0.gguf --port 8080
```

Then use with curl:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

## Configuration File

Instead of hardcoding settings, use a configuration file:

### Create `loci.toml`

```toml
[engine]
model_path = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
batch_size = 512
context_length = 2048
n_gpu_layers = -1

[backend]
backend_type = "cpu"  # or "cuda", "metal", "rocm"
enable_fusion = true

[logging]
level = "info"
```

### Load Configuration in Code

```rust
use loci::{ConfigLoader, LociEngine, EngineConfig};

fn main() -> anyhow::Result<()> {
    let loci_config = ConfigLoader::from_file("loci.toml")?
        .with_env_overrides()
        .build()?;

    // Convert LociConfig to EngineConfig
    let config = EngineConfig {
        model_path: loci_config.engine.model_path.unwrap(),
        n_ctx: loci_config.engine.context_length as u32,
        n_batch: loci_config.engine.batch_size as u32,
        n_gpu_layers: loci_config.engine.n_gpu_layers,
        n_threads: loci_config.engine.n_threads as u32,
        temperature: 0.7,  // Or read from config
        ..Default::default()
    };

    let engine = LociEngine::new(config)?;

    // ... rest of your code
    Ok(())
}
```

### Environment Variable Overrides

You can override config with environment variables:

```bash
export LOCI_MODEL_PATH="llama-2-7b.Q4_K_M.gguf"
export LOCI_BACKEND="cuda"
export LOCI_N_GPU_LAYERS=-1
export LOCI_LOG_LEVEL="debug"

cargo run
```

---

## Advanced Features

### 1. Adjusting Sampling Parameters

```rust
use loci::{LociEngine, EngineConfig};

// Deterministic output (low temperature)
let config_deterministic = EngineConfig {
    model_path: "model.gguf".to_string(),
    temperature: 0.1,
    top_k: 1,
    top_p: 0.5,
    ..Default::default()
};

// Creative output (high temperature)
let config_creative = EngineConfig {
    model_path: "model.gguf".to_string(),
    temperature: 1.2,
    top_k: 100,
    top_p: 0.95,
    repeat_penalty: 1.05,
    ..Default::default()
};
```

### 2. Performance Monitoring

```rust
let engine = LociEngine::new(config)?;
let response = engine.generate("Tell me a story", 200)?;

let stats = engine.stats();
println!("Prompt processing:");
println!("  Tokens: {}", stats.prompt_eval_count);
println!("  Speed: {:.2} t/s", stats.prompt_tokens_per_second());
println!("Generation:");
println!("  Tokens: {}", stats.eval_count);
println!("  Speed: {:.2} t/s", stats.eval_tokens_per_second());
```

### 3. Backend Selection

```rust
use loci::detect_backend;

let backend = detect_backend();
println!("Auto-detected backend: {}", backend.name());

// Use the detected backend
let engine = LociEngine::new(config)?;
println!("Using: {}", engine.backend_name());
```

---

## Performance Tips

### 1. Choose the Right Quantization

| Format | Size | Speed | Quality | Use Case |
|--------|------|-------|---------|----------|
| Q4_0 | Small | Fast | Good | General use |
| Q4_K_M | Medium | Balanced | Better | Recommended |
| Q5_K_M | Larger | Slower | Best | High quality |
| IQ2_XXS | Smallest | Very Fast | OK | Resource-constrained |

### 2. Optimize GPU Layers

```rust
// Offload all layers to GPU (fastest)
let config = EngineConfig {
    n_gpu_layers: -1,  // All layers
    ..Default::default()
};

// Offload only some layers (saves VRAM)
let config = EngineConfig {
    n_gpu_layers: 32,  // First 32 layers
    ..Default::default()
};

// CPU only (slowest but no GPU needed)
let config = EngineConfig {
    n_gpu_layers: 0,  // No GPU
    ..Default::default()
};
```

### 3. Adjust Batch Size and Context

```rust
// Larger batch = faster but more memory
let config = EngineConfig {
    n_batch: 512,   // Default
    n_ctx: 2048,    // Standard context
    ..Default::default()
};

// Smaller batch = slower but less memory
let config = EngineConfig {
    n_batch: 128,
    n_ctx: 1024,
    ..Default::default()
};

// Long context for complex tasks
let config = EngineConfig {
    n_batch: 512,
    n_ctx: 8192,   // 8k context
    ..Default::default()
};
```

---

## Troubleshooting

### Model Won't Load

**Problem**: `Failed to load model: File not found`

**Solution**: Check the model path is correct and file exists:
```bash
ls -lh tinyllama-1.1b-chat-v1.0.Q4_0.gguf
```

### Out of Memory

**Problem**: `Out of memory error`

**Solutions**:
1. Use smaller quantization (Q4_0 instead of Q5_K_M)
2. Reduce batch size: `batch_size: 128`
3. Reduce context length: `context_length: 1024`
4. Use CPU instead of GPU: `n_gpu_layers: 0`

### Slow Generation

**Problem**: Generation is very slow

**Solutions**:
1. Enable GPU: `n_gpu_layers: -1`
2. Enable kernel fusion: `enable_fusion: true`
3. Increase batch size: `batch_size: 512`
4. Use faster quantization (Q4_0 instead of FP16)

### CUDA/GPU Not Detected

**Problem**: GPU not being used despite `n_gpu_layers: -1`

**Solutions**:
1. Check CUDA installation: `nvidia-smi`
2. Rebuild with CUDA: `cargo build --release --features cuda`
3. Set backend explicitly:
   ```toml
   [backend]
   backend_type = "cuda"
   ```

---

## Next Steps

Now that you have Loci running, explore these topics:

1. **[Configuration Guide](./CONFIGURATION.md)** - Detailed configuration options
2. **[API Reference](./API_REFERENCE.md)** - Complete API documentation
3. **[Plugin Development](./PLUGIN_DEVELOPMENT.md)** - Create custom plugins
4. **[Performance Tuning](./PERFORMANCE_TUNING.md)** - Optimize for your hardware
5. **[Deployment Guide](./DOCKER_DEPLOYMENT.md)** - Deploy to production

---

## Examples

Check out the `examples/` directory for more code samples:

- `examples/basic_generation.rs` - Simple text generation
- `examples/streaming.rs` - Streaming token generation
- `examples/json_constraint.rs` - Structured output with JSON
- `examples/multi_session.rs` - Managing multiple sessions
- `examples/plugin_usage.rs` - Using plugins

Run an example:
```bash
cargo run --example basic_generation
```

---

## Getting Help

- **Documentation**: [docs/](../docs/)
- **GitHub Issues**: [github.com/decade-afk/Loci/issues](https://github.com/decade-afk/Loci/issues)
- **Discord**: [discord.gg/loci](https://discord.gg/loci)
- **Email**: team@loci.dev

---

**Happy Coding with Loci!** 🚀

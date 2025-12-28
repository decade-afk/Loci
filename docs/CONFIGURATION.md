# Loci Configuration Guide

This guide covers all configuration options available in Loci, including file-based configuration, environment variables, and programmatic configuration.

---

## Table of Contents

1. [Configuration Methods](#configuration-methods)
2. [Engine Configuration](#engine-configuration)
3. [Backend Configuration](#backend-configuration)
4. [Memory Configuration](#memory-configuration)
5. [Plugin Configuration](#plugin-configuration)
6. [Logging Configuration](#logging-configuration)
7. [Server Configuration](#server-configuration)
8. [Environment Variables](#environment-variables)
9. [Best Practices](#best-practices)

---

## Configuration Methods

### Method 1: Configuration File (Recommended)

Create a `loci.toml` file:

```toml
[engine]
model_path = "./models/llama-2-7b-q4_k_m.gguf"
batch_size = 512
context_length = 2048
n_gpu_layers = -1

[backend]
backend_type = "cuda"
enable_fusion = true

[logging]
level = "info"
```

Load it in code:

```rust
use loci::ConfigLoader;

let config = ConfigLoader::from_file("loci.toml")?
    .with_env_overrides()
    .build()?;
```

### Method 2: Programmatic Configuration

```rust
use loci::{LociConfig, EngineSettings, BackendSettings};

let config = LociConfig {
    engine: EngineSettings {
        model_path: Some("model.gguf".to_string()),
        batch_size: 512,
        n_gpu_layers: -1,
        ..Default::default()
    },
    backend: BackendSettings {
        backend_type: "cuda".to_string(),
        enable_fusion: true,
        ..Default::default()
    },
    ..Default::default()
};
```

### Method 3: Environment Variables

```bash
export LOCI_MODEL_PATH="./models/llama-2-7b.gguf"
export LOCI_BACKEND="cuda"
export LOCI_N_GPU_LAYERS=-1

# Then load with overrides
let config = ConfigLoader::from_file("loci.toml")?
    .with_env_overrides()  // Apply env vars
    .build()?;
```

---

## Engine Configuration

The `[engine]` section controls core inference settings.

### Available Options

```toml
[engine]
# Path to the GGUF model file (required)
model_path = "./models/llama-2-7b-q4_k_m.gguf"

# Batch size for processing (default: 512)
# Larger = faster but more memory
batch_size = 512

# Number of threads for CPU inference (0 = auto-detect)
# Default: number of CPU cores
n_threads = 0

# Maximum context length in tokens (default: 2048)
# Must be <= model's max context
context_length = 2048

# Number of layers to offload to GPU (default: -1)
# -1 = all layers (fastest)
# 0 = CPU only (slowest)
# 32 = offload 32 layers (hybrid)
n_gpu_layers = -1

# Use memory mapping for model loading (default: true)
# Recommended for fast startup
use_mmap = true

# Lock model in RAM to prevent swapping (default: false)
# Only use if you have enough RAM
use_mlock = false
```

### Programmatic Configuration

```rust
use loci::EngineConfig;

let config = EngineConfig {
    model_path: "path/to/model.gguf".to_string(),
    n_ctx: 4096,           // Context length
    n_batch: 512,          // Batch size
    n_threads: 8,          // CPU threads
    n_gpu_layers: -1,      // Use all GPU layers
    temperature: 0.7,      // Sampling temperature
    top_k: 40,             // Top-K sampling
    top_p: 0.9,            // Top-P sampling
    repeat_penalty: 1.1,   // Repetition penalty
};
```

### Performance Tuning

#### For Maximum Speed (GPU)
```rust
let config = EngineConfig {
    n_gpu_layers: -1,      // All layers on GPU
    n_batch: 512,          // Large batch
    n_ctx: 2048,
    temperature: 0.8,
    ..Default::default()
};
```

#### For Maximum Quality
```rust
let config = EngineConfig {
    n_gpu_layers: -1,
    n_ctx: 4096,           // Longer context
    n_batch: 256,          // Smaller batch for accuracy
    temperature: 0.5,      // More focused
    top_k: 20,
    top_p: 0.85,
    ..Default::default()
};
```

#### For Low Memory
```rust
let config = EngineConfig {
    n_gpu_layers: 0,       // CPU only
    n_batch: 128,          // Small batch
    n_ctx: 1024,           // Shorter context
    temperature: 0.8,
    ..Default::default()
};
```

---

## Backend Configuration

The `[backend]` section controls compute backend selection and optimizations.

### Available Options

```toml
[backend]
# Backend type: "cpu", "cuda", "metal", "rocm", "vulkan"
# Default: "cpu" (auto-detected if not specified)
backend_type = "cuda"

# GPU device ID (default: 0)
# For multi-GPU systems
device_id = 0

# Enable kernel fusion optimizations (default: true)
# Fuses operations like RMSNorm+RoPE for 30% speedup
enable_fusion = true
```

### Backend Selection Guide

| Backend | Platform | Performance | Requirements |
|---------|----------|-------------|--------------|
| `cpu` | All | Baseline | None |
| `cuda` | Linux/Windows | Fastest (NVIDIA) | CUDA 11.8+ |
| `metal` | macOS | Fastest (Apple) | macOS 13+ |
| `rocm` | Linux | Fast (AMD) | ROCm 5.4+ |
| `vulkan` | All | Moderate | Vulkan 1.3+ |

### Auto-Detection

```rust
use loci::detect_backend;

// Automatically selects best backend
let backend = detect_backend();
println!("Using: {}", backend.name());
```

### Multi-GPU Setup

```toml
# Use GPU 0
[backend]
backend_type = "cuda"
device_id = 0

# Or use GPU 1
[backend]
backend_type = "cuda"
device_id = 1
```

---

## Memory Configuration

The `[memory]` section configures memory management for Paged Attention.

### Available Options

```toml
[memory]
# VRAM budget in MB (default: 4096)
vram_mb = 4096

# RAM budget in MB (default: 8192)
ram_mb = 8192

# Block size in KB (default: 256)
# Each block stores BLOCK_SIZE tokens
block_size_kb = 256

# Enable swapping between VRAM and RAM (default: true)
enable_swap = true
```

### Memory Budget Guidelines

#### NVIDIA RTX 4090 (24GB VRAM)
```toml
[memory]
vram_mb = 20480   # 20GB (leave 4GB for system)
ram_mb = 16384    # 16GB
enable_swap = true
```

#### Apple M2 Max (32GB Unified)
```toml
[memory]
vram_mb = 24576   # 24GB
ram_mb = 8192     # 8GB
enable_swap = true
```

#### Consumer GPU (8GB VRAM)
```toml
[memory]
vram_mb = 6144    # 6GB (leave 2GB for system)
ram_mb = 16384    # 16GB RAM for swap
enable_swap = true
```

#### CPU-Only (No GPU)
```toml
[memory]
vram_mb = 0
ram_mb = 8192     # Use system RAM
enable_swap = false
```

---

## Plugin Configuration

The `[plugins]` section configures the plugin system.

### Available Options

```toml
[plugins]
# Plugin directory (default: "./plugins")
plugin_dir = "./plugins"

# Enable plugin system (default: true)
enabled = true

# Auto-load plugins on startup
auto_load = [
    "conflict-guard",
    "json-validator",
    "profanity-filter"
]
```

### Plugin Directory Structure

```
plugins/
├── native/
│   ├── conflict-guard.so      # Linux
│   ├── conflict-guard.dylib   # macOS
│   └── conflict-guard.dll     # Windows
└── wasm/
    ├── json-validator.wasm
    └── profanity-filter.wasm
```

### Programmatic Plugin Loading

```rust
use loci::{PluginRegistry, WasmPlugin};

let mut registry = PluginRegistry::new();

// Load WASM plugin
registry.register_wasm(
    "plugins/wasm/json-validator.wasm",
    None  // No signature verification
)?;

// Load native plugin
registry.register_native(Box::new(MyPlugin))?;
```

---

## Logging Configuration

The `[logging]` section controls logging behavior.

### Available Options

```toml
[logging]
# Log level: "trace", "debug", "info", "warn", "error"
# Default: "info"
level = "info"

# Output format: "text" or "json"
# Default: "text"
format = "text"

# Log file path (optional)
# If not specified, no file logging
file = "./logs/loci.log"

# Enable console output (default: true)
console = true
```

### Log Levels

| Level | Use Case |
|-------|----------|
| `trace` | Very verbose, debugging internals |
| `debug` | Detailed debugging information |
| `info` | General information (recommended) |
| `warn` | Warnings and potential issues |
| `error` | Only errors |

### Examples

#### Development (Verbose)
```toml
[logging]
level = "debug"
format = "text"
console = true
file = "./logs/debug.log"
```

#### Production (Minimal)
```toml
[logging]
level = "warn"
format = "json"
console = false
file = "/var/log/loci/production.log"
```

#### Testing (Trace Everything)
```toml
[logging]
level = "trace"
format = "text"
console = true
```

---

## Server Configuration

The `[server]` section configures the HTTP server (for `loci serve`).

### Available Options

```toml
[server]
# Listen address (default: "127.0.0.1")
host = "127.0.0.1"

# Listen port (default: 8080)
port = 8080

# Enable CORS (default: true)
enable_cors = true

# API key for authentication (optional)
# If not set, no authentication required
api_key = "your-secret-key-here"

# Maximum request size in MB (default: 100)
max_request_size_mb = 100
```

### Server Examples

#### Local Development
```toml
[server]
host = "127.0.0.1"
port = 8080
enable_cors = true
api_key = ""  # No auth
```

#### Production (Secure)
```toml
[server]
host = "0.0.0.0"  # Listen on all interfaces
port = 8443
enable_cors = false
api_key = "prod-secret-key-xyz123"
max_request_size_mb = 50
```

#### Docker Container
```toml
[server]
host = "0.0.0.0"
port = 8080
enable_cors = true
```

### Using API Key

```bash
# With API key
curl http://localhost:8080/v1/completions \
  -H "Authorization: Bearer your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100}'
```

---

## Environment Variables

All configuration options can be overridden with environment variables.

### Available Environment Variables

| Variable | Maps To | Example |
|----------|---------|---------|
| `LOCI_MODEL_PATH` | `engine.model_path` | `./models/llama-2-7b.gguf` |
| `LOCI_BATCH_SIZE` | `engine.batch_size` | `512` |
| `LOCI_N_THREADS` | `engine.n_threads` | `8` |
| `LOCI_N_GPU_LAYERS` | `engine.n_gpu_layers` | `-1` |
| `LOCI_BACKEND` | `backend.backend_type` | `cuda` |
| `LOCI_HOST` | `server.host` | `0.0.0.0` |
| `LOCI_PORT` | `server.port` | `8080` |
| `LOCI_API_KEY` | `server.api_key` | `secret123` |
| `LOCI_LOG_LEVEL` | `logging.level` | `debug` |

### Usage

```bash
# Override model path
export LOCI_MODEL_PATH="./different-model.gguf"

# Override backend
export LOCI_BACKEND="metal"

# Override port
export LOCI_PORT=9000

# Run with overrides
loci serve
```

### In Docker

```bash
docker run -p 8080:8080 \
  -e LOCI_MODEL_PATH=/models/llama-2-7b.gguf \
  -e LOCI_BACKEND=cuda \
  -e LOCI_LOG_LEVEL=info \
  -v ./models:/models \
  loci/loci:latest
```

---

## Best Practices

### 1. Use Configuration Files for Defaults

```toml
# loci.toml - default settings
[engine]
model_path = "./models/default-model.gguf"
batch_size = 512
```

Then override with env vars:
```bash
export LOCI_MODEL_PATH="./models/production-model.gguf"
```

### 2. Separate Configs for Environments

```bash
loci.dev.toml      # Development config
loci.staging.toml  # Staging config
loci.prod.toml     # Production config
```

Load the appropriate one:
```bash
loci serve --config loci.prod.toml
```

### 3. Use Validation

```rust
let config = ConfigLoader::from_file("loci.toml")?
    .with_env_overrides()
    .validate()?  // Catches errors early
    .build()?;
```

### 4. Document Your Configuration

```toml
# loci.toml - Production configuration
# Last updated: 2025-12-28
# Maintainer: ops-team@company.com

[engine]
# Using Llama-2-7B quantized to Q4_K_M for balance
model_path = "./models/llama-2-7b-q4_k_m.gguf"

# Batch size optimized for A100 GPU
batch_size = 512
```

### 5. Version Control Safe

```bash
# .gitignore
loci.local.toml      # Local overrides
*.secret.toml        # Secret configs
.env                 # Environment variables
```

Commit template:
```bash
git add loci.template.toml
```

### 6. Monitor and Adjust

```rust
let stats = engine.get_stats();
println!("Memory used: {} MB", stats.memory_used_mb);

// Adjust batch_size if memory is too high
```

---

## Complete Configuration Example

Here's a complete, production-ready configuration:

```toml
# loci.toml - Production Configuration
# Environment: Production
# Hardware: NVIDIA A100 (40GB)
# Updated: 2025-12-28

[engine]
model_path = "./models/llama-2-7b-q4_k_m.gguf"
batch_size = 512
n_threads = 0  # Auto-detect
context_length = 4096
n_gpu_layers = -1  # All layers on GPU
use_mmap = true
use_mlock = false

[backend]
backend_type = "cuda"
device_id = 0
enable_fusion = true

[memory]
vram_mb = 36864  # 36GB (leave 4GB for system)
ram_mb = 65536   # 64GB
block_size_kb = 256
enable_swap = true

[plugins]
plugin_dir = "./plugins"
enabled = true
auto_load = [
    "profanity-filter",
    "json-validator",
    "rate-limiter"
]

[logging]
level = "info"
format = "json"
file = "/var/log/loci/production.log"
console = false

[server]
host = "0.0.0.0"
port = 8443
enable_cors = false
api_key = "${LOCI_API_KEY}"  # From environment
max_request_size_mb = 50
```

---

## Troubleshooting Configuration

### Configuration Not Loading

**Problem**: Settings are not being applied

**Solution**: Check file path and format:
```bash
# Verify file exists
ls -l loci.toml

# Validate TOML syntax
cat loci.toml
```

### Environment Variables Not Working

**Problem**: Env vars not overriding config

**Solution**: Ensure you call `with_env_overrides()`:
```rust
ConfigLoader::from_file("loci.toml")?
    .with_env_overrides()  // <-- Required
    .build()?
```

### Invalid Configuration

**Problem**: Config validation fails

**Solution**: Check error message:
```rust
match ConfigLoader::from_file("loci.toml")?.validate() {
    Ok(_) => println!("Config is valid"),
    Err(e) => eprintln!("Invalid config: {}", e),
}
```

---

## Next Steps

- **[API Reference](./API_REFERENCE.md)** - Complete API documentation
- **[Quick Start](./QUICK_START.md)** - Get started quickly
- **[Performance Tuning](./PERFORMANCE_TUNING.md)** - Optimize performance
- **[Plugin Development](./PLUGIN_DEVELOPMENT.md)** - Create plugins

---

**Last Updated**: 2025-12-28

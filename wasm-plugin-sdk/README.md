# Loci WASM Plugin SDK

**Secure, sandboxed plugin development for the Loci LLM inference framework**

## Overview

The Loci WASM Plugin SDK enables developers to create powerful, safe plugins for the Loci framework using WebAssembly. WASM plugins run in a sandboxed environment with resource limits, making them ideal for untrusted code and third-party extensions.

## Features

- **🔒 Sandboxed Execution**: WASM provides memory isolation and security
- **⚡ Near-Native Performance**: Compiled WASM runs at near-native speeds
- **🎯 Resource Limits**: CPU (fuel) and memory limits prevent resource exhaustion
- **🌐 Cross-Platform**: WASM plugins work on any platform
- **📦 Small Binary Size**: Optimized for minimal footprint

## Quick Start

### 1. Install Rust and WASM Target

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### 2. Create a New Plugin

```bash
cargo new --lib my-wasm-plugin
cd my-wasm-plugin
```

### 3. Configure Cargo.toml

```toml
[package]
name = "my-wasm-plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
loci-wasm-plugin-sdk = { git = "https://github.com/decade-afk/loci", path = "wasm-plugin-sdk" }

[profile.release]
opt-level = "z"      # Optimize for size
lto = true           # Link-time optimization
codegen-units = 1    # Better optimization
strip = true         # Strip debug symbols
panic = "abort"      # Smaller binary size
```

### 4. Write Your Plugin

```rust
#![no_std]

use loci_wasm_plugin_sdk::*;

// Define plugin metadata
plugin_metadata! {
    name: "my_plugin",
    version: "0.1.0"
}

// Hook: Transform logits before sampling
#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: i32, n_vocab: i32, context_len: i32) -> i32 {
    unsafe {
        let logits = get_mut_slice(logits_ptr, n_vocab);

        // Your transformation logic here
        // Example: Ban token ID 100
        if logits.len() > 100 {
            logits[100] = f32::NEG_INFINITY;
        }
    }

    0 // Success
}

// Hook: Post-process sampled token
#[no_mangle]
pub extern "C" fn post_sample(token_id: i32) -> i32 {
    // Your post-sampling logic here
    token_id
}
```

### 5. Build the Plugin

```bash
cargo build --target wasm32-unknown-unknown --release
```

Your WASM plugin will be in: `target/wasm32-unknown-unknown/release/my_wasm_plugin.wasm`

### 6. Use the Plugin with Loci

```rust
use loci::prelude::*;

// Create WASM plugin configuration
let wasm_config = WasmPluginConfig {
    name: "my_plugin".to_string(),
    version: "0.1.0".to_string(),
    wasm_path: "path/to/my_wasm_plugin.wasm".into(),
    max_memory: 10 * 1024 * 1024,  // 10 MB
    max_fuel: 1_000_000,             // 1M instructions
    enable_wasi: false,              // Disabled for security
    timeout_ms: 5000,                // 5 seconds
};

// Load the plugin
let plugin = WasmPlugin::load(wasm_config)?;

// Or use with PluginRegistry
let mut registry = PluginRegistry::new();
registry.load_wasm_plugin("my_wasm_plugin.wasm")?;
```

## Plugin Hooks

### Required Exports

Every WASM plugin **must** export these functions:

```rust
#[no_mangle]
pub extern "C" fn plugin_name() -> i32;  // Returns pointer to name string

#[no_mangle]
pub extern "C" fn plugin_name_len() -> i32;  // Returns name length

#[no_mangle]
pub extern "C" fn plugin_version() -> i32;  // Returns pointer to version string

#[no_mangle]
pub extern "C" fn plugin_version_len() -> i32;  // Returns version length
```

**Pro tip**: Use the `plugin_metadata!` macro to automatically generate these.

### Optional Hook Functions

Implement any of these hooks based on your plugin's functionality:

#### 1. `transform_logits` - Modify sampling probabilities

```rust
#[no_mangle]
pub extern "C" fn transform_logits(
    logits_ptr: i32,    // Pointer to logits array (f32)
    n_vocab: i32,       // Vocabulary size
    context_len: i32    // Current context length
) -> i32 {
    unsafe {
        let logits = get_mut_slice(logits_ptr, n_vocab);
        // Modify logits here
    }
    0 // Return 0 for success, non-zero for error
}
```

**Use cases**:
- Token banning (content filtering)
- Grammar constraints
- Probability boosting for specific tokens
- Custom sampling distributions

#### 2. `post_sample` - Process sampled token

```rust
#[no_mangle]
pub extern "C" fn post_sample(token_id: i32) -> i32 {
    // Process or replace the token
    token_id  // Return token (modified or original)
}
```

**Use cases**:
- Token replacement rules
- Logging sampled tokens
- Token statistics collection

#### 3. `pre_generate` - Process prompt before generation

```rust
#[no_mangle]
pub extern "C" fn pre_generate(prompt_ptr: i32) -> i32 {
    // Read and process prompt
    // Return pointer to modified prompt (or original)
    prompt_ptr
}
```

**Use cases**:
- Prompt preprocessing
- Template injection
- Context augmentation

## SDK Helper Functions

### Memory Management

```rust
// Allocate memory for host-to-WASM data transfer
pub fn alloc(size: i32) -> i32;

// Deallocate memory
pub fn dealloc(ptr: i32, size: i32);
```

### String Helpers

```rust
// Read string from WASM memory
pub unsafe fn read_string(ptr: i32, len: i32) -> &'static str;

// Write string to WASM memory (allocates)
pub fn write_string(s: &str) -> (i32, i32);
```

### Slice Helpers

```rust
// Get mutable f32 slice (for logits)
pub unsafe fn get_mut_slice(ptr: i32, len: i32) -> &'static mut [f32];

// Get immutable f32 slice
pub unsafe fn get_slice(ptr: i32, len: i32) -> &'static [f32];

// Get i32 slice (for token IDs)
pub unsafe fn get_i32_slice(ptr: i32, len: i32) -> &'static [i32];
```

### Logits Manipulation

```rust
// Apply temperature scaling
pub fn apply_temperature(logits: &mut [f32], temperature: f32);

// Apply softmax normalization
pub fn softmax(logits: &mut [f32]);

// Find top-k tokens
pub fn find_top_k(logits: &[f32], k: usize) -> [i32; 10];
```

### Token Banning

```rust
pub struct TokenBanner {
    banned_tokens: &'static [i32],
}

impl TokenBanner {
    pub const fn new(banned_tokens: &'static [i32]) -> Self;
    pub fn apply(&self, logits: &mut [f32]);
}
```

## Examples

### Example 1: Token Banner

```rust
#![no_std]
use loci_wasm_plugin_sdk::*;

plugin_metadata! {
    name: "token_banner",
    version: "0.1.0"
}

static BANNED_TOKENS: &[i32] = &[100, 200, 300];

#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: i32, n_vocab: i32, _context_len: i32) -> i32 {
    unsafe {
        let logits = get_mut_slice(logits_ptr, n_vocab);
        for &token_id in BANNED_TOKENS {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] = f32::NEG_INFINITY;
            }
        }
    }
    0
}
```

### Example 2: Dynamic Temperature

```rust
#![no_std]
use loci_wasm_plugin_sdk::*;

plugin_metadata! {
    name: "temperature_booster",
    version: "0.1.0"
}

const BASE_TEMPERATURE: f32 = 1.0;
const MAX_TEMPERATURE: f32 = 1.5;

#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: i32, n_vocab: i32, context_len: i32) -> i32 {
    unsafe {
        let logits = get_mut_slice(logits_ptr, n_vocab);

        // Increase temperature as context grows
        let temp_boost = (context_len as f32 / 100.0) * 0.1;
        let temperature = (BASE_TEMPERATURE + temp_boost).min(MAX_TEMPERATURE);

        apply_temperature(logits, temperature);
    }
    0
}
```

### Example 3: Technical Term Booster

```rust
#![no_std]
use loci_wasm_plugin_sdk::*;

plugin_metadata! {
    name: "tech_term_booster",
    version: "0.1.0"
}

// Token IDs for technical terms (example)
static TECH_TOKENS: &[i32] = &[
    1234,  // "algorithm"
    5678,  // "neural"
    9012,  // "optimization"
];

const BOOST_FACTOR: f32 = 1.5;

#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: i32, n_vocab: i32, _context_len: i32) -> i32 {
    unsafe {
        let logits = get_mut_slice(logits_ptr, n_vocab);

        for &token_id in TECH_TOKENS {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] *= BOOST_FACTOR;
            }
        }
    }
    0
}
```

## Resource Limits

WASM plugins run with strict resource limits to prevent abuse:

### Memory Limits

```rust
WasmPluginConfig {
    max_memory: 16 * 1024 * 1024,  // 16 MB (default)
    // ...
}
```

- Default: 16 MB
- Recommended range: 1-64 MB
- Exceeding limit causes plugin termination

### CPU Limits (Fuel)

```rust
WasmPluginConfig {
    max_fuel: 1_000_000,  // 1M instructions (default)
    // ...
}
```

- Default: 1M instructions
- Prevents infinite loops
- Automatically refilled for each hook call

### Timeout

```rust
WasmPluginConfig {
    timeout_ms: 5000,  // 5 seconds (default)
    // ...
}
```

- Wall-clock time limit
- Prevents blocking operations

## Security Best Practices

### ✅ DO

- Keep plugins small and focused
- Use const data when possible
- Test plugins thoroughly before deployment
- Set conservative resource limits
- Disable WASI unless absolutely needed

### ❌ DON'T

- Enable WASI for untrusted plugins
- Use unbounded loops
- Allocate excessive memory
- Assume specific memory layout
- Rely on timing-dependent behavior

## Debugging

### Build with Debug Symbols

```toml
[profile.release]
opt-level = "z"
strip = false  # Keep debug symbols
```

### Enable WASM Logging

```rust
#[no_mangle]
pub extern "C" fn debug_log(msg_ptr: i32, msg_len: i32) {
    // Custom logging implementation
}
```

### Inspect WASM Binary

```bash
# View exports
wasm2wat my_plugin.wasm | grep "(export"

# View imports
wasm2wat my_plugin.wasm | grep "(import"

# Check binary size
ls -lh my_plugin.wasm
```

## Performance Tips

1. **Minimize allocations**: Use stack-allocated buffers when possible
2. **Avoid branching**: Modern CPUs prefer linear code
3. **Use SIMD operations**: WebAssembly supports SIMD (when enabled)
4. **Profile fuel usage**: Monitor `max_fuel` consumption
5. **Optimize for size**: Use `opt-level = "z"` and LTO

## Troubleshooting

### Error: "Missing required export: plugin_name"

**Solution**: Make sure you use the `plugin_metadata!` macro:

```rust
plugin_metadata! {
    name: "my_plugin",
    version: "0.1.0"
}
```

### Error: "WASM memory access out of bounds"

**Solution**: Always check array bounds before accessing:

```rust
if (index as usize) < logits.len() {
    logits[index as usize] = value;
}
```

### Error: "Fuel exhausted"

**Solution**: Increase `max_fuel` or optimize your plugin code:

```rust
WasmPluginConfig {
    max_fuel: 10_000_000,  // Increase limit
    // ...
}
```

## Building with wasm-pack (Alternative)

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build plugin
wasm-pack build --target wasm32-unknown-unknown --release

# Output: pkg/my_plugin_bg.wasm
```

## FAQ

**Q: Can I use standard library crates?**
A: No, WASM plugins use `#![no_std]` for size and security. Use `core` instead.

**Q: Can I use external dependencies?**
A: Yes, but only `no_std` crates. Check the crate documentation.

**Q: How do I debug WASM plugins?**
A: Use `wasmtime` CLI or browser DevTools with source maps.

**Q: Can plugins call external APIs?**
A: Only if WASI is enabled (disabled by default for security).

**Q: What's the performance overhead?**
A: ~5-10% compared to native plugins, mostly from boundary crossings.

## License

This SDK is part of the Loci project and is dual-licensed under MIT or Apache-2.0.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Resources

- [Loci Documentation](../README.md)
- [WASM Specification](https://webassembly.org/specs/)
- [wasmtime Documentation](https://docs.wasmtime.dev/)
- [Rust WASM Book](https://rustwasm.github.io/docs/book/)

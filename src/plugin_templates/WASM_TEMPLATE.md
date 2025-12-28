# Loci Plugin Template: WASM

A template for creating secure, sandboxed WASM plugins for Loci AI engine.

## Why WASM Plugins?

- ✅ **Sandboxed**: Runs in isolated environment, cannot access host filesystem
- ✅ **Portable**: Same binary runs on all platforms
- ✅ **Safe**: Memory-safe by design
- ✅ **Small**: Typical size 20-50KB (vs 2-5MB for native)

## Template Structure

```
my-loci-wasm-plugin/
├── Cargo.toml
├── src/
│   └── lib.rs
├── plugin.toml
└── README.md
```

## Quick Start

### 1. Install WASM Target

```bash
rustup target add wasm32-wasi
```

### 2. Create Plugin Project

```bash
cargo new --lib my-loci-wasm-plugin
cd my-loci-wasm-plugin
```

### 3. Configure Cargo.toml

```toml
[package]
name = "my-loci-wasm-plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
# No dependencies on Loci! WASM plugins use raw ABI

[profile.release]
opt-level = "z"  # Optimize for size
lto = true       # Link-time optimization
strip = true     # Strip symbols
```

### 4. Implement Plugin

Edit `src/lib.rs`:

```rust
use std::slice;
use std::ptr;

// ==================== Plugin Metadata ====================

#[no_mangle]
pub extern "C" fn plugin_metadata(out_ptr: *mut u8, out_len: usize) -> usize {
    let metadata = r#"{
        "name": "My WASM Plugin",
        "version": "0.1.0",
        "author": "Your Name",
        "description": "A WASM plugin for Loci",
        "type": "wasm"
    }"#;

    let bytes = metadata.as_bytes();
    let len = bytes.len().min(out_len);

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), out_ptr, len);
    }

    len
}

// ==================== Pre-process Hook ====================

#[no_mangle]
pub extern "C" fn pre_process(prompt_ptr: *mut u8, prompt_len: usize, out_len: *mut usize) -> i32 {
    // Read input prompt
    let prompt = unsafe {
        let slice = slice::from_raw_parts(prompt_ptr, prompt_len);
        String::from_utf8_lossy(slice).to_string()
    };

    // Transform prompt
    let new_prompt = format!("[WASM Enhanced] {}", prompt);
    let new_bytes = new_prompt.as_bytes();

    // Write back to buffer
    let copy_len = new_bytes.len().min(prompt_len);
    unsafe {
        ptr::copy_nonoverlapping(new_bytes.as_ptr(), prompt_ptr, copy_len);
        *out_len = copy_len;
    }

    0  // Return 0 = Continue
}

// ==================== Transform Logits Hook ====================

#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: *mut f32, logits_len: usize) -> i32 {
    // Read logits array
    let logits = unsafe {
        slice::from_raw_parts_mut(logits_ptr, logits_len)
    };

    // Apply temperature scaling
    let temperature = 0.7;
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }

    0  // Return 0 = Continue, 1 = Suspend, 2 = Break
}

// ==================== On Token Generated Hook ====================

#[no_mangle]
pub extern "C" fn on_token_generated(token_id: i32, token_ptr: *const u8, token_len: usize) -> i32 {
    // Read token text
    let token_text = unsafe {
        let slice = slice::from_raw_parts(token_ptr, token_len);
        String::from_utf_8_lossy(slice)
    };

    // Example: Stop on specific token
    if token_text.contains("STOP") {
        return 2;  // Break
    }

    0  // Continue
}

// ==================== Memory Allocator (Required for WASM) ====================

use std::alloc::{alloc, dealloc, Layout};

#[no_mangle]
pub extern "C" fn wasm_alloc(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { alloc(layout) }
}

#[no_mangle]
pub extern "C" fn wasm_free(ptr: *mut u8, size: usize) {
    let layout = Layout::from_size_align(size, 8).unwrap();
    unsafe { dealloc(ptr, layout) }
}
```

### 5. Build WASM Binary

```bash
cargo build --target wasm32-wasi --release
```

Output: `target/wasm32-wasi/release/my_loci_wasm_plugin.wasm`

### 6. Optimize (Optional)

```bash
# Install wasm-opt (from binaryen)
# Ubuntu/Debian:
sudo apt install binaryen

# macOS:
brew install binaryen

# Optimize
wasm-opt -Oz \
  target/wasm32-wasi/release/my_loci_wasm_plugin.wasm \
  -o target/wasm32-wasi/release/my_loci_wasm_plugin.opt.wasm

# Check size
ls -lh target/wasm32-wasi/release/*.wasm
```

---

## Advanced Examples

### Example 1: Keyword Blocker

```rust
const BLOCKED_KEYWORDS: &[&str] = &["violence", "illegal", "harmful"];

#[no_mangle]
pub extern "C" fn on_token_generated(token_id: i32, token_ptr: *const u8, token_len: usize) -> i32 {
    let token_text = unsafe {
        let slice = slice::from_raw_parts(token_ptr, token_len);
        String::from_utf8_lossy(slice).to_lowercase()
    };

    for keyword in BLOCKED_KEYWORDS {
        if token_text.contains(keyword) {
            // Block generation
            return 2;  // Break
        }
    }

    0  // Continue
}
```

### Example 2: Language Detector

```rust
#[no_mangle]
pub extern "C" fn pre_process(prompt_ptr: *mut u8, prompt_len: usize, out_len: *mut usize) -> i32 {
    let prompt = unsafe {
        let slice = slice::from_raw_parts(prompt_ptr, prompt_len);
        String::from_utf8_lossy(slice).to_string()
    };

    // Detect language (simple heuristic)
    let lang = if prompt.chars().any(|c| c as u32 > 0x4E00 && c as u32 < 0x9FFF) {
        "Chinese"
    } else if prompt.chars().any(|c| c as u32 > 0x3040 && c as u32 < 0x30FF) {
        "Japanese"
    } else {
        "English"
    };

    // Add language hint
    let new_prompt = format!("[Language: {}] {}", lang, prompt);
    let new_bytes = new_prompt.as_bytes();

    let copy_len = new_bytes.len().min(prompt_len);
    unsafe {
        ptr::copy_nonoverlapping(new_bytes.as_ptr(), prompt_ptr, copy_len);
        *out_len = copy_len;
    }

    0
}
```

### Example 3: Token Probability Filter

```rust
#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: *mut f32, logits_len: usize) -> i32 {
    let logits = unsafe {
        slice::from_raw_parts_mut(logits_ptr, logits_len)
    };

    // Find max logit
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Apply top-k filtering (keep top 50)
    const TOP_K: usize = 50;

    // Compute exp(logit - max) for numerical stability
    let mut exp_logits: Vec<(usize, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max_logit).exp()))
        .collect();

    // Sort by probability (descending)
    exp_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Zero out logits outside top-k
    for (i, _) in exp_logits.iter().skip(TOP_K) {
        logits[*i] = f32::NEG_INFINITY;
    }

    0
}
```

---

## WASM ABI Reference

### Required Exports

#### 1. plugin_metadata

```rust
pub extern "C" fn plugin_metadata(out_ptr: *mut u8, out_len: usize) -> usize
```

**Purpose**: Returns JSON metadata about the plugin
**Returns**: Number of bytes written

**Example JSON**:
```json
{
    "name": "My Plugin",
    "version": "1.0.0",
    "author": "Your Name",
    "description": "Plugin description",
    "type": "wasm"
}
```

#### 2. pre_process (Optional)

```rust
pub extern "C" fn pre_process(
    prompt_ptr: *mut u8,
    prompt_len: usize,
    out_len: *mut usize
) -> i32
```

**Purpose**: Modify prompt before inference
**Returns**: 0 = Continue, 1 = Suspend, 2 = Break

#### 3. transform_logits (Optional)

```rust
pub extern "C" fn transform_logits(
    logits_ptr: *mut f32,
    logits_len: usize
) -> i32
```

**Purpose**: Transform logits before sampling
**Returns**: 0 = Continue, 1 = Suspend, 2 = Break

#### 4. on_token_generated (Optional)

```rust
pub extern "C" fn on_token_generated(
    token_id: i32,
    token_ptr: *const u8,
    token_len: usize
) -> i32
```

**Purpose**: React to generated tokens
**Returns**: 0 = Continue, 1 = Suspend, 2 = Break

---

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pre_process() {
        let mut buffer = vec![0u8; 256];
        let prompt = "Hello";
        buffer[..prompt.len()].copy_from_slice(prompt.as_bytes());

        let mut out_len = 0usize;
        let result = pre_process(
            buffer.as_mut_ptr(),
            prompt.len(),
            &mut out_len as *mut usize
        );

        assert_eq!(result, 0);
        assert!(out_len > prompt.len());

        let output = String::from_utf8_lossy(&buffer[..out_len]);
        assert!(output.contains("[WASM Enhanced]"));
    }
}
```

### Integration Testing with Loci

```rust
// In your Loci project
#[test]
fn test_wasm_plugin() {
    let plugin = WasmPlugin::load(
        Path::new("plugins/my_plugin.wasm"),
        PluginMetadata { /* ... */ }
    ).unwrap();

    let mut prompt = "Test".to_string();
    let ctx = PluginContext::default();

    let result = plugin.pre_process(&mut prompt, &ctx).unwrap();

    assert!(matches!(result, PluginControlFlow::Continue));
    assert!(prompt.contains("[WASM Enhanced]"));
}
```

---

## Publishing

### 1. Create plugin.toml

```toml
[plugin]
name = "my-wasm-plugin"
version = "0.1.0"
author = "Your Name <your.email@example.com>"
description = "A WASM plugin for Loci"
license = "MIT"
type = "wasm"

[requirements]
loci_version = ">=0.1.0"
wasm_version = "wasm32-wasi"

[hooks]
pre_process = true
transform_logits = true
on_token_generated = true

[limits]
max_memory = 16  # MB
max_fuel = 1000000  # WASM instructions
timeout = 100  # ms per hook call
```

### 2. Package

```bash
tar -czf my-wasm-plugin-v0.1.0.tar.gz \
  target/wasm32-wasi/release/my_loci_wasm_plugin.wasm \
  plugin.toml \
  README.md
```

### 3. Submit to Marketplace

(See `PLUGIN_MARKETPLACE.md`)

---

## Performance Tips

1. **Minimize Allocations**: Avoid `String::new()`, `Vec::new()` in hot paths
2. **Use Static Data**: Prefer `const` over runtime initialization
3. **Optimize Build**: Use `opt-level = "z"` and `lto = true`
4. **Profile**: Use `wasmtime --profile` to find bottlenecks

## Size Optimization

```bash
# After cargo build:
# 1. Strip with wasm-strip
wasm-strip target/wasm32-wasi/release/my_plugin.wasm

# 2. Optimize with wasm-opt
wasm-opt -Oz target/wasm32-wasi/release/my_plugin.wasm \
  -o target/wasm32-wasi/release/my_plugin.opt.wasm

# 3. Compress with gzip (for distribution)
gzip -9 target/wasm32-wasi/release/my_plugin.opt.wasm
```

**Expected sizes**:
- Before optimization: 100-200 KB
- After optimization: 20-50 KB
- After gzip: 10-25 KB

---

## Security

### WASM Sandbox Guarantees

✅ **Cannot access host filesystem**
✅ **Cannot make network requests**
✅ **Cannot spawn processes**
✅ **Limited memory (configurable)**
✅ **Instruction counting (fuel)**
✅ **Timeout protection**

### Best Practices

1. **Validate all inputs**: Check pointers, lengths
2. **Bounds checking**: Never trust input sizes
3. **Error handling**: Return error codes, don't panic
4. **Resource limits**: Respect memory/fuel limits

---

## Troubleshooting

**Q: "memory access out of bounds"**
- Check all pointer arithmetic
- Ensure buffer sizes are correct
- Use `slice::from_raw_parts` safely

**Q: "unknown import"**
- WASM plugins cannot import host functions
- Use only WASI-compatible functions

**Q: Plugin too large**
- Remove debug symbols: `strip = true`
- Use `opt-level = "z"`
- Run `wasm-opt -Oz`

---

## Resources

- [WebAssembly Specification](https://webassembly.org/specs/)
- [WASI Tutorial](https://github.com/bytecodealliance/wasmtime/blob/main/docs/WASI-tutorial.md)
- [Loci WASM Plugin Examples](../examples/wasm_plugins/)
- [Plugin Marketplace](https://plugins.loci.ai)

---

**Happy WASM Plugin Development!** 🚀

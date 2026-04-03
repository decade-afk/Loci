# Loci Plugin Template: Native (Rust)

A template for creating high-performance native plugins for Loci AI engine.

## Template Structure

```
my-loci-plugin/
├── Cargo.toml
├── src/
│   └── lib.rs
├── plugin.toml
└── README.md
```

## Quick Start

### 1. Create Plugin Project

```bash
cargo new --lib my-loci-plugin
cd my-loci-plugin
```

### 2. Add Loci Dependency

Edit `Cargo.toml`:

```toml
[package]
name = "my-loci-plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]  # Important for dynamic loading

[dependencies]
loci = { path = "../../" }  # Adjust path to Loci
loci_legacy_plugin_api = { path = "../../crates/legacy-plugin-api" }
anyhow = "1.0"
```

### 3. Implement Plugin

Edit `src/lib.rs`:

```rust
use loci::{
    Plugin, PluginMetadata, PluginControlFlow, PluginContext,
    LogitsView, PluginType,
};
use anyhow::Result;

pub struct MyPlugin;

impl Plugin for MyPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "My Custom Plugin".to_string(),
            version: "0.1.0".to_string(),
            author: "Your Name".to_string(),
            description: "A custom Loci plugin".to_string(),
            plugin_type: PluginType::Native,
        }
    }

    fn pre_process(&self, prompt: &mut String, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        // Modify the prompt before inference
        println!("[MyPlugin] Pre-processing prompt: {}", prompt);

        // Example: Add a prefix
        *prompt = format!("[System: Enhanced] {}", prompt);

        Ok(PluginControlFlow::Continue)
    }

    fn transform_logits(&self, logits: &mut LogitsView, ctx: &PluginContext) -> Result<PluginControlFlow> {
        // Transform logits before sampling
        println!("[MyPlugin] Transforming {} logits at step {}", logits.len(), ctx.step);

        // Example: Temperature scaling
        let temperature = 0.8;
        for logit in logits.data.iter_mut() {
            *logit /= temperature;
        }

        Ok(PluginControlFlow::Continue)
    }

    fn on_token_generated(&self, token_id: i32, token_text: &str, ctx: &PluginContext) -> Result<PluginControlFlow> {
        // React to generated tokens
        println!("[MyPlugin] Token {}: '{}'", token_id, token_text);

        // Example: Stop on specific keyword
        if token_text.contains("STOP") {
            println!("[MyPlugin] Stop keyword detected!");
            return Ok(PluginControlFlow::Break);
        }

        Ok(PluginControlFlow::Continue)
    }
}

// Export the stable legacy plugin ABI v2 entrypoint
loci_legacy_plugin_api::export_legacy_plugin_v2!(MyPlugin);
```

### 4. Create Plugin Metadata

Create `plugin.toml`:

```toml
[plugin]
name = "my-custom-plugin"
version = "0.1.0"
author = "Your Name <your.email@example.com>"
description = "A custom Loci plugin"
license = "MIT"
type = "native"

[requirements]
loci_version = ">=0.1.0"
rust_version = ">=1.70"

[hooks]
# Which hooks does this plugin use?
pre_process = true
transform_logits = true
on_token_generated = true

[config]
# Plugin-specific configuration
temperature = 0.8
max_tokens = 1000
```

### 5. Build

```bash
cargo build --release
```

The plugin will be at `target/release/libmy_loci_plugin.so` (Linux) or `.dylib` (macOS) or `.dll` (Windows).

---

## Advanced Examples

### Example 1: Content Filter Plugin

```rust
pub struct ContentFilterPlugin {
    banned_words: Vec<String>,
}

impl ContentFilterPlugin {
    pub fn new(banned_words: Vec<String>) -> Self {
        Self { banned_words }
    }
}

impl Plugin for ContentFilterPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "Content Filter".to_string(),
            version = "1.0.0".to_string(),
            author: "Loci Team".to_string(),
            description: "Filters inappropriate content".to_string(),
            plugin_type: PluginType::Native,
        }
    }

    fn on_token_generated(&self, _token_id: i32, token_text: &str, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        let lower = token_text.to_lowercase();

        for banned in &self.banned_words {
            if lower.contains(banned) {
                eprintln!("[ContentFilter] Blocked: {}", banned);
                return Ok(PluginControlFlow::Break);
            }
        }

        Ok(PluginControlFlow::Continue)
    }
}
```

### Example 2: Citation Injector Plugin

```rust
pub struct CitationPlugin {
    citations: Vec<String>,
}

impl Plugin for CitationPlugin {
    fn pre_process(&self, prompt: &mut String, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        // Inject citations into the prompt
        let citations_text = self.citations.join("\n");
        *prompt = format!(
            "References:\n{}\n\nQuestion: {}",
            citations_text,
            prompt
        );

        Ok(PluginControlFlow::Continue)
    }
}
```

### Example 3: Auto-Retry Plugin

```rust
pub struct AutoRetryPlugin {
    max_retries: usize,
    retry_count: AtomicUsize,
}

impl Plugin for AutoRetryPlugin {
    fn on_token_generated(&self, _token_id: i32, token_text: &str, _ctx: &PluginContext) -> Result<PluginControlFlow> {
        // Check for error indicators
        if token_text.contains("ERROR") || token_text.contains("FAILED") {
            let retries = self.retry_count.fetch_add(1, Ordering::SeqCst);

            if retries < self.max_retries {
                eprintln!("[AutoRetry] Retry attempt {}/{}", retries + 1, self.max_retries);

                return Ok(PluginControlFlow::Suspend {
                    reason: "Auto-retry triggered".to_string(),
                    user_data: Some(format!("retry_{}", retries)),
                });
            } else {
                eprintln!("[AutoRetry] Max retries exceeded");
                return Ok(PluginControlFlow::Break);
            }
        }

        Ok(PluginControlFlow::Continue)
    }
}
```

---

## Testing Your Plugin

Create `tests/integration_test.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_metadata() {
        let plugin = MyPlugin;
        let metadata = plugin.metadata();

        assert_eq!(metadata.name, "My Custom Plugin");
        assert_eq!(metadata.version, "0.1.0");
    }

    #[test]
    fn test_pre_process() {
        let plugin = MyPlugin;
        let mut prompt = "Hello".to_string();
        let ctx = PluginContext::default();

        let result = plugin.pre_process(&mut prompt, &ctx).unwrap();

        assert!(matches!(result, PluginControlFlow::Continue));
        assert!(prompt.starts_with("[System: Enhanced]"));
    }

    #[test]
    fn test_transform_logits() {
        let plugin = MyPlugin;
        let mut logits_data = vec![1.0, 2.0, 3.0, 4.0];
        let mut logits = LogitsView {
            data: &mut logits_data,
            vocab_size: 4,
        };
        let ctx = PluginContext { step: 0, session_id: "test".to_string() };

        let result = plugin.transform_logits(&mut logits, &ctx).unwrap();

        assert!(matches!(result, PluginControlFlow::Continue));
        // Check temperature scaling applied
        assert!(logits.data[0] < 1.5);
    }
}
```

Run tests:

```bash
cargo test
```

---

## Publishing

### 1. Sign Your Plugin (Ed25519)

```bash
# Generate keypair (save securely!)
openssl genpkey -algorithm ed25519 -out plugin.key

# Sign the plugin binary
openssl pkeyutl -sign -inkey plugin.key \
  -in target/release/libmy_loci_plugin.so \
  -out target/release/libmy_loci_plugin.so.sig
```

### 2. Package

```bash
tar -czf my-loci-plugin-v0.1.0.tar.gz \
  target/release/libmy_loci_plugin.so \
  target/release/libmy_loci_plugin.so.sig \
  plugin.toml \
  README.md
```

### 3. Submit to Loci Plugin Marketplace

(See `PLUGIN_MARKETPLACE.md` for submission guidelines)

---

## Best Practices

1. **Performance**: Avoid heavy computations in `transform_logits` (called every token)
2. **Safety**: Validate all inputs, handle errors gracefully
3. **Compatibility**: Test with multiple Loci versions
4. **Documentation**: Provide clear usage examples
5. **Testing**: Comprehensive unit and integration tests

---

## Troubleshooting

**Q: Plugin not loading?**
- Check `plugin.toml` syntax
- Ensure Loci version compatibility
- Verify signature (if required)

**Q: Plugin crashes?**
- Use `RUST_BACKTRACE=1` for debugging
- Check memory safety (no dangling pointers)
- Validate all FFI boundaries

**Q: Performance issues?**
- Profile with `perf` or `cargo flamegraph`
- Minimize allocations in hot paths
- Consider SIMD optimizations

---

## Resources

- [Loci Plugin API Documentation](../docs/plugin_api.md)
- [Example Plugins](../examples/plugins/)
- [Plugin Marketplace](https://plugins.loci.ai)
- [Community Forum](https://discuss.loci.ai)

---

**Happy Plugin Development!** 🚀

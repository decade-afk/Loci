# Loci Plugin Examples

This directory contains example plugins demonstrating the extensibility of Loci's plugin system.

## Available Plugins

### 1. Profanity Filter Plugin (`profanity_filter_plugin.rs`)

Filters offensive language from input prompts and generated output.

**Features:**
- Customizable blocked words list
- Configurable replacement text
- Filter input only, output only, or both
- Case-insensitive filtering

**Usage:**
```rust
use loci::plugin::Plugin;
use loci::examples::plugins::ProfanityFilterPlugin;

// Basic usage
let plugin = ProfanityFilterPlugin::new("profanity_filter");
engine.plugin_manager_mut().register(plugin)?;

// With custom replacement
let plugin = ProfanityFilterPlugin::new("profanity_filter")
    .with_replacement("[FILTERED]".to_string());
engine.plugin_manager_mut().register(plugin)?;

// With custom blocked words
let plugin = ProfanityFilterPlugin::with_custom_blocked_words(
    "profanity_filter",
    vec!["word1".to_string(), "word2".to_string()],
);
engine.plugin_manager_mut().register(plugin)?;
```

### 2. JSON Output Formatter Plugin (`json_output_formatter_plugin.rs`)

Formats model output as structured JSON with optional metadata.

**Features:**
- Wraps response in JSON object
- Includes timestamp and plugin metadata
- Tracks generation timing
- Configurable metadata fields

**Usage:**
```rust
use loci::examples::plugins::JsonFormatterPlugin;

// Basic JSON formatting
let mut plugin = JsonFormatterPlugin::new("json_formatter");
plugin.init()?;
engine.plugin_manager_mut().register(plugin)?;

// With all metadata
let mut plugin = JsonFormatterPlugin::new("json_formatter")
    .with_metadata(true)
    .with_timing(true)
    .with_prompt(true);
plugin.init()?;
engine.plugin_manager_mut().register(plugin)?;
```

**Output Example:**
```json
{
  "content": "The generated response text here...",
  "metadata": {
    "timestamp": "2026-01-02T12:34:56.789Z",
    "plugin": "json_formatter",
    "plugin_version": "1.0.0"
  },
  "timing": {
    "elapsed_ms": 1234
  }
}
```

### 3. Translation Plugin (`translation_plugin.rs`)

Wraps prompts with translation instructions for automated translation.

**Features:**
- Predefined language pairs (English↔Chinese, English↔Spanish, etc.)
- Custom language pairs
- Clean prompt templates

**Usage:**
```rust
use loci::examples::plugins::TranslationPlugin;

// Predefined language pairs
let plugin = TranslationPlugin::english_to_chinese("translator");
engine.plugin_manager_mut().register(plugin)?;

let plugin = TranslationPlugin::chinese_to_english("translator");
engine.plugin_manager_mut().register(plugin)?;

let plugin = TranslationPlugin::english_to_spanish("translator");
engine.plugin_manager_mut().register(plugin)?;

// Custom language pairs
let plugin = TranslationPlugin::new("translator", "German", "French");
engine.plugin_manager_mut().register(plugin)?;
```

### 4. Code Explainer Plugin (`code_explainer_plugin.rs`)

Enhances prompts for code explanation tasks with automatic language detection.

**Features:**
- Automatic code language detection (Python, Rust, JS, Java, C++, Go, etc.)
- Configurable detail level (Brief, Standard, Detailed, Comprehensive)
- Customizable explanation style

**Usage:**
```rust
use loci::examples::plugins::{CodeExplainerPlugin, DetailLevel};

// Standard explanation
let plugin = CodeExplainerPlugin::new("code_explainer");
engine.plugin_manager_mut().register(plugin)?;

// Brief explanation
let plugin = CodeExplainerPlugin::brief("brief_explainer");
engine.plugin_manager_mut().register(plugin)?;

// Detailed explanation
let plugin = CodeExplainerPlugin::detailed("detailed_explainer");
engine.plugin_manager_mut().register(plugin)?;

// Comprehensive explanation
let plugin = CodeExplainerPlugin::comprehensive("comprehensive_explainer");
engine.plugin_manager_mut().register(plugin)?;

// With explicit language
let plugin = CodeExplainerPlugin::new("python_explainer")
    .with_language("Python")
    .with_detail_level(DetailLevel::Detailed);
engine.plugin_manager_mut().register(plugin)?;
```

## Testing

Run tests for all example plugins:

```bash
cargo test --example plugins
```

## Building a Custom Plugin

To create your own plugin, implement the `Plugin` trait:

```rust
use loci::plugin::Plugin;
use loci::error::Result;

struct MyCustomPlugin {
    name: String,
}

impl Plugin for MyCustomPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        // Modify the prompt before inference
        Ok(format!("Enhanced: {}", prompt))
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        // Modify the response after inference
        Ok(format!("{} [processed]", response))
    }

    fn on_token(&self, token: &str) -> Result<String> {
        // Process each token during streaming
        Ok(token.to_string())
    }
}
```

See [PLUGIN_GUIDE.md](../../PLUGIN_GUIDE.md) for complete plugin development documentation.

## Advanced Examples

### Chain Multiple Plugins

```rust
// Apply multiple plugins in sequence
engine.plugin_manager_mut().register(
    TranslationPlugin::english_to_chinese("translator")
)?;

engine.plugin_manager_mut().register(
    JsonFormatterPlugin::new("json_formatter")
)?;

// Execution order:
// 1. Translation plugin processes prompt
// 2. Model generates response
// 3. JSON formatter formats output
```

### Conditional Plugin Logic

```rust
struct ConditionalPlugin {
    condition: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl Plugin for ConditionalPlugin {
    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if (self.condition)(prompt) {
            Ok(format!("[SPECIAL] {}", prompt))
        } else {
            Ok(prompt.to_string())
        }
    }
    // ... other methods
}
```

### Stateful Plugin

```rust
use std::sync::{Arc, Mutex};

struct StatefulPlugin {
    counter: Arc<Mutex<usize>>,
}

impl Plugin for StatefulPlugin {
    fn on_token(&self, token: &str) -> Result<String> {
        let mut count = self.counter.lock().unwrap();
        *count += 1;
        Ok(token.to_string())
    }
    // ... other methods
}
```

## Plugin Hooks Reference

| Hook | When Called | Use Case |
|------|-------------|----------|
| `init()` | On plugin registration | Initialize resources, load config |
| `pre_generate()` | Before model inference | Modify prompts, add templates |
| `on_token()` | During streaming (per token) | Real-time filtering, stats |
| `post_generate()` | After model inference | Format output, filter content |
| `cleanup()` | On plugin unload | Release resources |

## Best Practices

1. **Error Handling**: Always return `Result` and use `?` for error propagation
2. **Performance**: Minimize allocations in hot paths (especially `on_token()`)
3. **Thread Safety**: Use `Arc<Mutex<>>` for shared state
4. **Resource Management**: Implement `cleanup()` to release resources
5. **Testing**: Write unit tests for all plugin logic

## Contributing

To contribute a new example plugin:

1. Create the plugin file in this directory
2. Add a mod entry in `mod.rs`
3. Add documentation to this README
4. Include unit tests
5. Update the list of available plugins

## License

These examples are part of Loci and follow the same license (MIT OR Apache-2.0).
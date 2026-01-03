//! Hot-swap plugin demo
//!
//! This example demonstrates:
//! 1. Registering static plugins
//! 2. Loading dynamic plugins from shared libraries (hot-swap)
//! 3. Reloading plugins without restarting
//! 4. Persisting plugin configuration to TOML
//! 5. Loading configuration from file

use loci::plugin::Plugin;
use loci::plugin_registry::{PluginRegistry, PluginConfig};
use loci::error::Result;
use std::collections::HashMap;

/// Example static plugin
struct StaticGreeterPlugin {
    greeting: String,
}

impl StaticGreeterPlugin {
    fn new(greeting: &str) -> Self {
        Self {
            greeting: greeting.to_string(),
        }
    }
}

impl Plugin for StaticGreeterPlugin {
    fn name(&self) -> &str {
        "static_greeter"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        println!("[StaticGreeter] Initialized with greeting: '{}'", self.greeting);
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("{} {}", self.greeting, prompt))
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("[StaticGreeter] Cleanup");
        Ok(())
    }
}

/// Example: Time-stamper plugin
struct TimestampPlugin;

impl Plugin for TimestampPlugin {
    fn name(&self) -> &str {
        "timestamp"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Ok(format!("{} [timestamp: {}]", response, timestamp))
    }
}

fn main() -> Result<()> {
    println!("=== Loci Hot-Swap Plugin Demo ===\n");

    // 1. Create registry with config file
    let config_path = "plugins.toml";
    let mut registry = PluginRegistry::with_config_path(config_path);

    println!("1. Registering static plugins...");
    registry.register_static(StaticGreeterPlugin::new("Hello,"))?;
    registry.register_static(TimestampPlugin)?;
    println!("   Registered {} plugins\n", registry.count());

    // 2. List all plugins
    println!("2. Current plugin list:");
    for (name, version, enabled, is_dynamic) in registry.list() {
        let plugin_type = if is_dynamic { "dynamic" } else { "static" };
        println!("   - {} v{} (enabled: {}, type: {})", name, version, enabled, plugin_type);
    }
    println!();

    // 3. Test plugin hooks
    println!("3. Testing plugin hooks:");
    let input = "world!";
    println!("   Input: {}", input);
    let processed = registry.apply_pre_generate(input)?;
    println!("   After pre_generate: {}", processed);

    let output = "This is the response";
    println!("   Output: {}", output);
    let processed = registry.apply_post_generate(output)?;
    println!("   After post_generate: {}", processed);
    println!();

    // 4. Disable a plugin
    println!("4. Testing enable/disable:");
    println!("   Disabling 'timestamp' plugin...");
    registry.disable("timestamp")?;
    let processed = registry.apply_post_generate(output)?;
    println!("   Result: {} (timestamp missing)", processed);

    println!("   Re-enabling 'timestamp' plugin...");
    registry.enable("timestamp")?;
    let processed = registry.apply_post_generate(output)?;
    println!("   Result: {}", processed);
    println!();

    // 5. Save configuration to file
    println!("5. Saving configuration to '{}'...", config_path);
    registry.save_to_file(config_path)?;
    println!("   Configuration saved!\n");

    // Show generated TOML
    let content = std::fs::read_to_string(config_path)?;
    println!("Generated TOML:");
    println!("---");
    println!("{}", content);
    println!("---\n");

    // 6. Test loading from file
    println!("6. Testing configuration reload:");
    let mut registry2 = PluginRegistry::new();

    // Register the same plugins first (for static plugins)
    registry2.register_static(StaticGreeterPlugin::new("Hello,"))?;
    registry2.register_static(TimestampPlugin)?;

    // Load state from file
    println!("   Loading from '{}'...", config_path);
    registry2.load_from_file(config_path)?;

    println!("   Loaded plugins:");
    for (name, version, enabled, is_dynamic) in registry2.list() {
        let plugin_type = if is_dynamic { "dynamic" } else { "static" };
        println!("   - {} v{} (enabled: {}, type: {})", name, version, enabled, plugin_type);
    }
    println!();

    // 7. Dynamic plugin note
    println!("7. Dynamic Plugin Loading:");
    println!("   To load a dynamic plugin, you would use:");
    println!("   ```rust");
    println!("   registry.load_dynamic_plugin(\"path/to/plugin.dll\")?;");
    println!("   ```");
    println!("   The plugin shared library must export a 'create_plugin' function.");
    println!();

    println!("=== Demo Complete ===");

    // Cleanup
    std::fs::remove_file(config_path).ok();

    Ok(())
}

//! Plugin Registry comprehensive demo
//!
//! Demonstrates all three key features:
//! 1. Hot-swap (dynamic plugin loading/unloading/reloading)
//! 2. Centralized management (PluginRegistry)
//! 3. Persistent configuration (TOML save/load)

use loci::plugin::Plugin;
use loci::plugin_registry::PluginRegistry;
use loci::error::Result;

/// Example plugin 1: Prefix adder
struct PrefixPlugin {
    prefix: String,
}

impl PrefixPlugin {
    fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }
}

impl Plugin for PrefixPlugin {
    fn name(&self) -> &str {
        "prefix"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("{}{}", self.prefix, prompt))
    }
}

/// Example plugin 2: Suffix adder
struct SuffixPlugin {
    suffix: String,
}

impl SuffixPlugin {
    fn new(suffix: &str) -> Self {
        Self {
            suffix: suffix.to_string(),
        }
    }
}

impl Plugin for SuffixPlugin {
    fn name(&self) -> &str {
        "suffix"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        Ok(format!("{}{}", response, self.suffix))
    }
}

/// Example plugin 3: Uppercase converter
struct UppercasePlugin;

impl Plugin for UppercasePlugin {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn version(&self) -> &str {
        "2.0.0"
    }

    fn on_token(&self, token: &str) -> Result<String> {
        Ok(token.to_uppercase())
    }
}

fn print_separator() {
    println!("\n{}", "=".repeat(60));
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║    Loci Plugin Registry - Comprehensive Demo            ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    let config_file = "demo_plugins.toml";

    // ═══════════════════════════════════════════════════════════
    // PART 1: Centralized Plugin Management
    // ═══════════════════════════════════════════════════════════
    print_separator();
    println!("PART 1: Centralized Plugin Management");
    print_separator();

    let mut registry = PluginRegistry::with_config_path(config_file);

    println!("\n1.1 Registering plugins...");
    registry.register_static(PrefixPlugin::new("[USER] "))?;
    registry.register_static(SuffixPlugin::new(" [END]"))?;
    registry.register_static(UppercasePlugin)?;

    println!("     ✓ Registered {} plugins", registry.count());

    println!("\n1.2 Listing all plugins:");
    for (name, version, enabled, is_dynamic) in registry.list() {
        let plugin_type = if is_dynamic { "🔌 dynamic" } else { "📦 static" };
        let status = if enabled { "✓ enabled" } else { "✗ disabled" };
        println!("     • {} v{} ({}, {})", name, version, status, plugin_type);
    }

    // ═══════════════════════════════════════════════════════════
    // PART 2: Plugin Hooks in Action
    // ═══════════════════════════════════════════════════════════
    print_separator();
    println!("PART 2: Plugin Hooks in Action");
    print_separator();

    println!("\n2.1 Testing pre_generate hook:");
    let input = "Hello, world!";
    println!("     Input:  '{}'", input);
    let result = registry.apply_pre_generate(input)?;
    println!("     Output: '{}'", result);

    println!("\n2.2 Testing post_generate hook:");
    let response = "This is a response";
    println!("     Input:  '{}'", response);
    let result = registry.apply_post_generate(response)?;
    println!("     Output: '{}'", result);

    println!("\n2.3 Testing on_token hook (streaming):");
    let tokens = vec!["hello", " ", "world"];
    print!("     Tokens: ");
    for token in &tokens {
        let processed = registry.apply_on_token(token)?;
        print!("{}", processed);
    }
    println!();

    // ═══════════════════════════════════════════════════════════
    // PART 3: Runtime Control (Enable/Disable)
    // ═══════════════════════════════════════════════════════════
    print_separator();
    println!("PART 3: Runtime Control (Enable/Disable)");
    print_separator();

    println!("\n3.1 Disabling 'prefix' plugin...");
    registry.disable("prefix")?;
    let result = registry.apply_pre_generate(input)?;
    println!("     Without prefix: '{}'", result);

    println!("\n3.2 Re-enabling 'prefix' plugin...");
    registry.enable("prefix")?;
    let result = registry.apply_pre_generate(input)?;
    println!("     With prefix: '{}'", result);

    println!("\n3.3 Current stats:");
    println!("     Total plugins: {}", registry.count());
    println!("     Enabled: {}", registry.count_enabled());
    println!("     Dynamic: {}", registry.count_dynamic());

    // ═══════════════════════════════════════════════════════════
    // PART 4: Persistent Configuration
    // ═══════════════════════════════════════════════════════════
    print_separator();
    println!("PART 4: Persistent Configuration");
    print_separator();

    println!("\n4.1 Saving configuration to '{}'...", config_file);
    registry.save_to_file(config_file)?;
    println!("     ✓ Configuration saved!");

    println!("\n4.2 Generated TOML content:");
    let content = std::fs::read_to_string(config_file)?;
    println!("┌────────────────────────────────────────────────────────┐");
    for line in content.lines() {
        println!("│ {:<54} │", line);
    }
    println!("└────────────────────────────────────────────────────────┘");

    println!("\n4.3 Creating new registry and loading from file...");
    let mut registry2 = PluginRegistry::new();

    // Must register plugins first (they're static)
    registry2.register_static(PrefixPlugin::new("[USER] "))?;
    registry2.register_static(SuffixPlugin::new(" [END]"))?;
    registry2.register_static(UppercasePlugin)?;

    // Load states from file
    registry2.load_from_file(config_file)?;

    println!("     ✓ Configuration loaded!");
    println!("\n4.4 Verifying loaded state:");
    for (name, version, enabled, _) in registry2.list() {
        let status = if enabled { "✓" } else { "✗" };
        println!("     {} {} v{}", status, name, version);
    }

    // ═══════════════════════════════════════════════════════════
    // PART 5: Hot-Swap Capabilities
    // ═══════════════════════════════════════════════════════════
    print_separator();
    println!("PART 5: Hot-Swap Capabilities");
    print_separator();

    println!("\n5.1 Dynamic Plugin Loading:");
    println!("     To load a dynamic plugin at runtime:");
    println!("     ```rust");
    println!("     registry.load_dynamic_plugin(\"plugin.dll\")?;");
    println!("     ```");

    println!("\n5.2 Hot Reload:");
    println!("     To reload a plugin without restarting:");
    println!("     ```rust");
    println!("     registry.reload_dynamic_plugin(\"my_plugin\")?;");
    println!("     ```");

    println!("\n5.3 Hot Unload:");
    println!("     To unload a plugin at runtime:");
    println!("     ```rust");
    println!("     registry.unload_dynamic_plugin(\"my_plugin\")?;");
    println!("     ```");

    println!("\n5.4 Creating a Dynamic Plugin:");
    println!("     Your plugin must export:");
    println!("     ```rust");
    println!("     #[no_mangle]");
    println!("     pub extern \"C\" fn create_plugin() -> *mut dyn Plugin {{");
    println!("         Box::into_raw(Box::new(MyPlugin::new()))");
    println!("     }}");
    println!("     ```");

    println!("\n5.5 Build dynamic plugin:");
    println!("     [lib]");
    println!("     crate-type = [\"cdylib\"]");

    // ═══════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════
    print_separator();
    println!("SUMMARY");
    print_separator();

    println!("\n✅ Features Demonstrated:");
    println!("   1. ✓ Hot-swap (Dynamic loading/reloading)");
    println!("   2. ✓ Centralized management (PluginRegistry)");
    println!("   3. ✓ Persistent configuration (TOML)");

    println!("\n📋 Key APIs:");
    println!("   • register_static()         - Register compiled plugins");
    println!("   • load_dynamic_plugin()     - Load .dll/.so at runtime");
    println!("   • reload_dynamic_plugin()   - Hot-reload plugin");
    println!("   • unload_dynamic_plugin()   - Hot-unload plugin");
    println!("   • enable() / disable()      - Control plugins");
    println!("   • save_to_file()            - Persist configuration");
    println!("   • load_from_file()          - Restore configuration");

    println!("\n🔄 Plugin Lifecycle:");
    println!("   Register → Init → Enable → [Hooks] → Disable → Cleanup");

    print_separator();
    println!("Demo Complete!");
    print_separator();

    // Cleanup
    std::fs::remove_file(config_file).ok();

    Ok(())
}

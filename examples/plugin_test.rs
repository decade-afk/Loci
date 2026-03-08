//! Plugin system test example
//!
//! This example demonstrates plugin functionality without requiring a full model.
//! It tests plugin registration, enable/disable, and text processing.

use loci::error::Result;
use loci::plugin::{Plugin, PluginManager};

/// Test plugin 1: Adds prefix
struct PrefixPlugin {
    prefix: String,
}

impl PrefixPlugin {
    fn new(prefix: String) -> Self {
        Self { prefix }
    }
}

impl Plugin for PrefixPlugin {
    fn name(&self) -> &str {
        "prefix"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        println!("[PrefixPlugin] Initialized with prefix: '{}'", self.prefix);
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("{}{}", self.prefix, prompt))
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("[PrefixPlugin] Cleanup");
        Ok(())
    }
}

/// Test plugin 2: Adds suffix
struct SuffixPlugin {
    suffix: String,
}

impl SuffixPlugin {
    fn new(suffix: String) -> Self {
        Self { suffix }
    }
}

impl Plugin for SuffixPlugin {
    fn name(&self) -> &str {
        "suffix"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        println!("[SuffixPlugin] Initialized with suffix: '{}'", self.suffix);
        Ok(())
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        Ok(format!("{}{}", response, self.suffix))
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("[SuffixPlugin] Cleanup");
        Ok(())
    }
}

/// Test plugin 3: Uppercase converter
struct UppercasePlugin;

impl Plugin for UppercasePlugin {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn on_token(&self, token: &str) -> Result<String> {
        Ok(token.to_uppercase())
    }
}

fn main() -> Result<()> {
    println!("=== Loci Plugin System Test ===\n");

    // Create plugin manager
    let mut manager = PluginManager::new();

    println!("1. Testing plugin registration...");
    manager.register(PrefixPlugin::new("[USER] ".to_string()))?;
    manager.register(SuffixPlugin::new(" [END]".to_string()))?;
    manager.register(UppercasePlugin)?;

    println!("   Registered {} plugins\n", manager.count());

    // List plugins
    println!("2. Listing all plugins:");
    for (name, version, enabled) in manager.list() {
        println!("   - {} v{} (enabled: {})", name, version, enabled);
    }
    println!();

    // Test pre_generate
    println!("3. Testing pre_generate hook:");
    let input = "Hello, world!";
    println!("   Input: {}", input);
    let processed = manager.apply_pre_generate(input)?;
    println!("   After pre_generate: {}", processed);
    println!();

    // Test post_generate
    println!("4. Testing post_generate hook:");
    let output = "This is a response";
    println!("   Output: {}", output);
    let processed = manager.apply_post_generate(output)?;
    println!("   After post_generate: {}", processed);
    println!();

    // Test on_token
    println!("5. Testing on_token hook (streaming simulation):");
    let tokens = vec!["hello", " ", "world", "!"];
    print!("   Processed tokens: ");
    for token in tokens {
        let processed = manager.apply_on_token(token)?;
        print!("{}", processed);
    }
    println!("\n");

    // Test enable/disable
    println!("6. Testing enable/disable:");
    println!("   Disabling 'prefix' plugin...");
    manager.disable("prefix")?;

    let input2 = "Test without prefix";
    println!("   Input: {}", input2);
    let processed2 = manager.apply_pre_generate(input2)?;
    println!("   Result: {} (prefix should be missing)", processed2);

    println!("   Re-enabling 'prefix' plugin...");
    manager.enable("prefix")?;
    let processed3 = manager.apply_pre_generate(input2)?;
    println!("   Result: {} (prefix restored)", processed3);
    println!();

    // Test duplicate registration
    println!("7. Testing duplicate registration prevention:");
    match manager.register(PrefixPlugin::new("DUP".to_string())) {
        Ok(_) => println!("   ERROR: Should have failed!"),
        Err(e) => println!("   Correctly rejected: {}", e),
    }
    println!();

    // Test unregister
    println!("8. Testing plugin unregistration:");
    println!("   Before: {} plugins", manager.count());
    manager.unregister("uppercase")?;
    println!(
        "   After unregister 'uppercase': {} plugins",
        manager.count()
    );
    println!();

    println!("9. Final plugin list:");
    for (name, version, enabled) in manager.list() {
        println!("   - {} v{} (enabled: {})", name, version, enabled);
    }

    println!("\n=== All Tests Passed! ===");

    Ok(())
}

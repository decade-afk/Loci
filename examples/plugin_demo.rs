// Plugin Demo for Loci
// 
// This example demonstrates using multiple plugins with Loci's inference engine.
// Run with: cargo run --example plugin_demo --features plugin-examples

use loci::prelude::*;
use loci::plugin::Plugin;
use loci::error::Result;

#[cfg(feature = "plugin-examples")]
use loci::examples::plugins::*;

fn main() -> Result<()> {
    println!("🚀 Loci Plugin Demo\n");
    println!("This example demonstrates using multiple plugins with Loci.\n");

    // Example 1: Profanity Filter Plugin
    println!("📝 Example 1: Profanity Filter Plugin");
    println!("--------------------------------------");
    
    #[cfg(feature = "plugin-examples")]
    {
        let profanity_plugin = ProfanityFilterPlugin::new("profanity_filter")
            .with_replacement("[FILTERED]".to_string());
        
        let test_input = "This contains badword1 and profanity.";
        let filtered = profanity_plugin.post_generate(test_input)?;
        
        println!("Input:  {}", test_input);
        println!("Output: {}", filtered);
        println!();
    }

    // Example 2: JSON Formatter Plugin
    println!("📝 Example 2: JSON Formatter Plugin");
    println!("------------------------------------");
    
    #[cfg(feature = "plugin-examples")]
    {
        let mut json_plugin = JsonFormatterPlugin::new("json_formatter")
            .with_metadata(true)
            .with_timing(true);
        json_plugin.init()?;
        
        let test_input = "Hello, this is a test response.";
        let json_output = json_plugin.post_generate(test_input)?;
        
        println!("Input:  {}", test_input);
        println!("Output: {}", json_output);
        println!();
    }

    // Example 3: Translation Plugin
    println!("📝 Example 3: Translation Plugin");
    println!("--------------------------------");
    
    #[cfg(feature = "plugin-examples")]
    {
        let translator = TranslationPlugin::english_to_chinese("translator");
        
        let test_input = "Hello, world!";
        let translated_prompt = translator.pre_generate(test_input)?;
        
        println!("Original: {}", test_input);
        println!("Translation Prompt:\n{}", translated_prompt);
        println!();
    }

    // Example 4: Code Explainer Plugin
    println!("📝 Example 4: Code Explainer Plugin");
    println!("-----------------------------------");
    
    #[cfg(feature = "plugin-examples")]
    {
        let explainer = CodeExplainerPlugin::detailed("code_explainer");
        
        let test_code = r#"
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

fn main() {
    println!("{}", greet("World"));
}
"#;
        let explanation_prompt = explainer.pre_generate(test_code)?;
        
        println!("Code:");
        println!("{}", test_code);
        println!("\nExplanation Prompt:\n{}", explanation_prompt);
        println!();
    }

    // Example 5: Chaining Multiple Plugins
    println!("📝 Example 5: Plugin Chain");
    println!("---------------------------");
    
    #[cfg(feature = "plugin-examples")]
    {
        println!("Demonstrating plugin chain: Translation → JSON Format");
        println!();
        
        let translator = TranslationPlugin::english_to_chinese("translator");
        let mut json_formatter = JsonFormatterPlugin::new("json_formatter")
            .with_metadata(true);
        json_formatter.init()?;
        
        let original_input = "Hello, world!";
        println!("Step 1 - Original: {}", original_input);
        
        let translated = translator.pre_generate(original_input)?;
        println!("Step 2 - Translation Prompt Applied");
        
        let formatted = json_formatter.post_generate("Simulated response")?;
        println!("Step 3 - JSON Formatting Applied");
        println!("Output: {}", formatted);
        println!();
    }

    // Example 6: Custom Plugin
    println!("📝 Example 6: Custom Plugin");
    println!("---------------------------");
    
    struct CustomLoggerPlugin {
        name: String,
    }
    
    impl Plugin for CustomLoggerPlugin {
        fn name(&self) -> &str {
            &self.name
        }
        
        fn version(&self) -> &str {
            "1.0.0"
        }
        
        fn pre_generate(&self, prompt: &str) -> Result<String> {
            println!("🔍 [PLUGIN] Processing prompt: {} chars", prompt.len());
            Ok(prompt.to_string())
        }
        
        fn post_generate(&self, response: &str) -> Result<String> {
            println!("✅ [PLUGIN] Generated response: {} chars", response.len());
            Ok(response.to_string())
        }
    }
    
    let custom_plugin = CustomLoggerPlugin {
        name: "custom_logger".to_string(),
    };
    
    let test_input = "Test prompt";
    let _output = custom_plugin.pre_generate(test_input)?;
    let _output = custom_plugin.post_generate("Test response")?;
    println!();

    println!("✅ All examples completed!");
    println!();
    println!("💡 Tips:");
    println!("   - Plugins can be chained for complex processing");
    println!("   - Each plugin has a specific purpose (filtering, formatting, translation)");
    println!("   - Custom plugins can be created by implementing the Plugin trait");
    println!("   - See examples/plugins/ for more examples");
    
    Ok(())
}

#[cfg(not(feature = "plugin-examples"))]
fn main() -> Result<()> {
    println!("⚠️  Plugin examples feature not enabled!");
    println!("Run with: cargo run --example plugin_demo --features plugin-examples");
    Ok(())
}
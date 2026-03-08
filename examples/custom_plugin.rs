//! Custom plugin example
//!
//! This example demonstrates how to create and register custom plugins with Loci.
//! Plugins can be used for:
//! - Prompt engineering and templates
//! - Response filtering and formatting
//! - Token-level processing (streaming)
//! - Logging and analytics
//!
//! Run with:
//! cargo run --example custom_plugin -- -m <model_path> -p "Your prompt"

use loci::plugin::Plugin;
use loci::prelude::*;
use std::path::PathBuf;

/// Example plugin: Adds Markdown formatting to responses
struct MarkdownFormatterPlugin {
    enabled: bool,
}

impl MarkdownFormatterPlugin {
    fn new() -> Self {
        Self { enabled: true }
    }
}

impl Plugin for MarkdownFormatterPlugin {
    fn name(&self) -> &str {
        "markdown_formatter"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        println!(
            "[MarkdownFormatter] Initializing plugin v{}",
            self.version()
        );
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if !self.enabled {
            return Ok(prompt.to_string());
        }

        // Wrap prompt in a structured format
        let formatted = format!("**User Query:**\n{}\n\n**Assistant Response:**\n", prompt);
        Ok(formatted)
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        if !self.enabled {
            return Ok(response.to_string());
        }

        // Add markdown code blocks if response looks like code
        if response.contains("fn ") || response.contains("def ") || response.contains("class ") {
            return Ok(format!("```\n{}\n```", response.trim()));
        }

        Ok(response.to_string())
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("[MarkdownFormatter] Cleaning up plugin");
        Ok(())
    }
}

/// Example plugin: Logs all generated tokens (for debugging/analytics)
struct TokenLoggerPlugin {
    token_count: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl TokenLoggerPlugin {
    fn new() -> Self {
        Self {
            token_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }
}

impl Plugin for TokenLoggerPlugin {
    fn name(&self) -> &str {
        "token_logger"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn on_token(&self, token: &str) -> Result<String> {
        let mut count = self.token_count.lock().unwrap();
        *count += 1;

        // Log every 10 tokens
        if *count % 10 == 0 {
            println!("[TokenLogger] Generated {} tokens", *count);
        }

        Ok(token.to_string())
    }

    fn cleanup(&mut self) -> Result<()> {
        let count = self.token_count.lock().unwrap();
        println!("[TokenLogger] Total tokens generated: {}", *count);
        Ok(())
    }
}

/// Example plugin: Content safety filter
struct ContentFilterPlugin {
    blocked_words: Vec<String>,
}

impl ContentFilterPlugin {
    fn new() -> Self {
        Self {
            blocked_words: vec!["badword1".to_string(), "badword2".to_string()],
        }
    }
}

impl Plugin for ContentFilterPlugin {
    fn name(&self) -> &str {
        "content_filter"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        let mut filtered = response.to_string();

        for word in &self.blocked_words {
            filtered = filtered.replace(word, "[FILTERED]");
        }

        Ok(filtered)
    }
}

fn main() -> anyhow::Result<()> {
    // For this example, we'll use a simple CLI
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 5 {
        eprintln!("Usage: {} -m <model_path> -p <prompt>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run --example custom_plugin -- -m model.gguf -p \"Write a hello world in Python\"");
        std::process::exit(1);
    }

    let mut model_path = PathBuf::new();
    let mut prompt = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-m" => {
                i += 1;
                model_path = PathBuf::from(&args[i]);
            }
            "-p" => {
                i += 1;
                prompt = args[i].clone();
            }
            _ => {}
        }
        i += 1;
    }

    println!("=== Loci Custom Plugin Example ===\n");

    // Create inference engine
    println!("Loading model: {}", model_path.display());
    let config = ModelConfig::new(&model_path)
        .with_context_size(2048)
        .with_gpu_layers(-1);

    let mut engine = InferenceEngine::new(config)?;

    // Register custom plugins
    println!("\nRegistering plugins...");

    let markdown_plugin = MarkdownFormatterPlugin::new();
    engine.plugin_manager_mut().register(markdown_plugin)?;

    let token_logger = TokenLoggerPlugin::new();
    engine.plugin_manager_mut().register(token_logger)?;

    let content_filter = ContentFilterPlugin::new();
    engine.plugin_manager_mut().register(content_filter)?;

    // List registered plugins
    let plugins = engine.plugin_manager().list();
    println!("Registered {} plugins:", plugins.len());
    for (name, version, enabled) in plugins {
        println!(
            "  - {} (v{}) [{}]",
            name,
            version,
            if enabled { "enabled" } else { "disabled" }
        );
    }

    println!("\n--- Generating Response ---\n");

    // Generate response with streaming
    use loci::inference::GenerationParams;

    let params = GenerationParams {
        max_tokens: 256,
        temperature: 0.7,
        ..Default::default()
    };

    engine.generate_stream(&prompt, params, |token| {
        print!("{}", token);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        true
    })?;

    println!("\n\n--- Done ---");

    Ok(())
}

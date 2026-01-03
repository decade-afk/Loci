//! Example: Advanced Logits Manipulation Plugin
//!
//! This demonstrates the power of the new plugin system with logit-level intervention.
//!
//! ## What this shows:
//! - Zero-copy logits manipulation
//! - Token banning/boosting
//! - Context-aware transformations
//! - Plugin composition

use loci::prelude::*;
use loci::sampler::LogitsView;

/// Plugin that bans specific tokens (e.g., profanity filter)
struct TokenBanPlugin {
    banned_tokens: Vec<i32>,
}

impl TokenBanPlugin {
    fn new(banned_tokens: Vec<i32>) -> Self {
        Self { banned_tokens }
    }
}

impl Plugin for TokenBanPlugin {
    fn name(&self) -> &str {
        "token_ban"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn transform_logits(&self, logits: &mut LogitsView, _context: &[i32]) -> Result<()> {
        // Ban specific tokens by setting their logits to -infinity
        logits.apply_mask(&self.banned_tokens);
        println!("[TokenBanPlugin] Banned {} tokens", self.banned_tokens.len());
        Ok(())
    }
}

/// Plugin that boosts technical terms in a coding context
struct TechnicalTermBoostPlugin {
    boost_map: Vec<(i32, f32)>, // (token_id, bias)
}

impl TechnicalTermBoostPlugin {
    fn new(boost_map: Vec<(i32, f32)>) -> Self {
        Self { boost_map }
    }
}

impl Plugin for TechnicalTermBoostPlugin {
    fn name(&self) -> &str {
        "tech_term_boost"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn transform_logits(&self, logits: &mut LogitsView, context: &[i32]) -> Result<()> {
        // Only boost if we're in a technical context (e.g., last tokens suggest code)
        let is_code_context = context.len() > 2; // Simplified heuristic

        if is_code_context {
            for &(token, bias) in &self.boost_map {
                logits.apply_bias(token, bias)?;
            }
            println!("[TechTermBoost] Boosted {} technical terms", self.boost_map.len());
        }

        Ok(())
    }
}

/// Plugin that implements dynamic repetition penalty based on context
struct ContextAwareRepetitionPlugin {
    base_penalty: f32,
    window_size: usize,
}

impl ContextAwareRepetitionPlugin {
    fn new(base_penalty: f32, window_size: usize) -> Self {
        Self {
            base_penalty,
            window_size,
        }
    }
}

impl Plugin for ContextAwareRepetitionPlugin {
    fn name(&self) -> &str {
        "context_aware_repetition"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn transform_logits(&self, logits: &mut LogitsView, context: &[i32]) -> Result<()> {
        // Get recent tokens within window
        let recent_start = context.len().saturating_sub(self.window_size);
        let recent_tokens = &context[recent_start..];

        // Count token frequencies
        let mut token_counts: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
        for &token in recent_tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }

        // Apply stronger penalty to more frequent tokens
        for (&token, &count) in &token_counts {
            if count > 1 {
                let dynamic_penalty = self.base_penalty * (count as f32).sqrt();

                // Apply penalty (llama.cpp convention: divide positive, multiply negative)
                if let Some(logit) = logits.get(token) {
                    let new_value = if logit > 0.0 {
                        logit / dynamic_penalty
                    } else {
                        logit * dynamic_penalty
                    };
                    logits.set(token, new_value)?;
                }
            }
        }

        println!(
            "[ContextRepetition] Applied dynamic penalty to {} repeated tokens",
            token_counts.iter().filter(|(_, &c)| c > 1).count()
        );

        Ok(())
    }
}

/// Plugin that logs token statistics (post-sample hook)
struct TokenStatsPlugin {
    token_count: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl TokenStatsPlugin {
    fn new() -> Self {
        Self {
            token_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }

    fn get_count(&self) -> usize {
        *self.token_count.lock().unwrap()
    }
}

impl Plugin for TokenStatsPlugin {
    fn name(&self) -> &str {
        "token_stats"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        let mut count = self.token_count.lock().unwrap();
        *count += 1;

        if *count % 10 == 0 {
            println!("[TokenStats] Generated {} tokens", *count);
        }

        Ok(token_id) // Pass through unchanged
    }
}

fn main() -> Result<()> {
    println!("=== Loci Logits Plugin Demo ===\n");

    // Note: This is a demonstration of the plugin API
    // In a real scenario, you would:
    // 1. Load a model
    // 2. Register these plugins
    // 3. Run inference

    println!("Available plugin types:");
    println!("1. TokenBanPlugin - Bans specific tokens (e.g., profanity filtering)");
    println!("2. TechnicalTermBoostPlugin - Boosts technical terms in code generation");
    println!("3. ContextAwareRepetitionPlugin - Dynamic repetition penalty");
    println!("4. TokenStatsPlugin - Logs token generation statistics");

    println!("\n=== Plugin Composition Example ===");
    println!("Plugins can be chained together:");
    println!("  Input Logits");
    println!("     ↓");
    println!("  TokenBanPlugin (mask banned tokens)");
    println!("     ↓");
    println!("  TechnicalTermBoostPlugin (boost relevant terms)");
    println!("     ↓");
    println!("  ContextAwareRepetitionPlugin (penalize repetition)");
    println!("     ↓");
    println!("  Sampler (temperature, top-k, top-p)");
    println!("     ↓");
    println!("  TokenStatsPlugin (log statistics)");
    println!("     ↓");
    println!("  Final Token");

    println!("\n=== Zero-Copy Benefits ===");
    println!("- Logits are modified in-place (no allocation)");
    println!("- Plugins receive mutable view (&mut LogitsView)");
    println!("- Multiple plugins compose without copying");
    println!("- FFI-safe integration with llama.cpp");

    println!("\n=== Usage Example (Pseudocode) ===");
    println!(r#"
// Create plugins
let ban_plugin = TokenBanPlugin::new(vec![123, 456]); // Ban tokens 123, 456
let boost_plugin = TechnicalTermBoostPlugin::new(vec![(789, 2.0)]); // Boost token 789
let repeat_plugin = ContextAwareRepetitionPlugin::new(1.2, 32);
let stats_plugin = TokenStatsPlugin::new();

// Register with engine
engine.plugin_manager_mut().register(ban_plugin)?;
engine.plugin_manager_mut().register(boost_plugin)?;
engine.plugin_manager_mut().register(repeat_plugin)?;
engine.plugin_manager_mut().register(stats_plugin)?;

// Generate text - plugins automatically apply
let response = engine.generate("Write a Rust function", params)?;

// Check statistics
println!("Total tokens generated: {}", stats_plugin.get_count());
    "#);

    Ok(())
}

//! Example: Advanced Logits Manipulation Plugin
//!
//! This demonstrates logit-level plugin hooks with the current `LogitsView` API.

use loci::prelude::*;

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
        for &token in &self.banned_tokens {
            if token >= 0 && (token as usize) < logits.vocab_size() {
                logits.set(token, f32::NEG_INFINITY)?;
            }
        }
        println!(
            "[TokenBanPlugin] Banned {} tokens",
            self.banned_tokens.len()
        );
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
        let is_code_context = context.len() > 2;
        if is_code_context {
            for &(token, bias) in &self.boost_map {
                if let Some(current) = logits.get(token) {
                    logits.set(token, current + bias)?;
                }
            }
            println!("[TechTermBoost] Boosted {} terms", self.boost_map.len());
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
        let recent_start = context.len().saturating_sub(self.window_size);
        let recent_tokens = &context[recent_start..];

        let mut token_counts: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
        for &token in recent_tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }

        for (&token, &count) in &token_counts {
            if count > 1 {
                let dynamic_penalty = self.base_penalty * (count as f32).sqrt();
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
            "[ContextRepetition] Penalized {} repeated tokens",
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
        Ok(token_id)
    }
}

fn main() -> Result<()> {
    println!("=== Loci Logits Plugin Demo ===");
    println!("1. TokenBanPlugin");
    println!("2. TechnicalTermBoostPlugin");
    println!("3. ContextAwareRepetitionPlugin");
    println!("4. TokenStatsPlugin");

    let stats_plugin = TokenStatsPlugin::new();
    let _ban_plugin = TokenBanPlugin::new(vec![123, 456]);
    let _boost_plugin = TechnicalTermBoostPlugin::new(vec![(789, 2.0)]);
    let _repeat_plugin = ContextAwareRepetitionPlugin::new(1.2, 32);

    println!(
        "{}",
        r#"Pseudo flow:
engine.plugin_manager_mut().register(ban_plugin)?;
engine.plugin_manager_mut().register(boost_plugin)?;
engine.plugin_manager_mut().register(repeat_plugin)?;
engine.plugin_manager_mut().register(stats_plugin)?;
let response = engine.generate("Write a Rust function", params)?;
println!("Total tokens generated: {}", stats_plugin.get_count());"#
    );

    println!("Current stats count (demo): {}", stats_plugin.get_count());
    Ok(())
}

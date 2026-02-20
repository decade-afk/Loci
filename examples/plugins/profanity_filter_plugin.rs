// Profanity Filter Plugin for Loci
// 
// This plugin filters offensive language from both input prompts and generated output.
// It demonstrates the text processing hooks of Loci's plugin system.

use loci::plugin::Plugin;
use loci::error::Result;
use std::collections::HashSet;

pub struct ProfanityFilterPlugin {
    name: String,
    blocked_words: HashSet<String>,
    replacement: String,
    filter_input: bool,
    filter_output: bool,
}

impl ProfanityFilterPlugin {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            blocked_words: Self::default_blocked_words(),
            replacement: "***".to_string(),
            filter_input: true,
            filter_output: true,
        }
    }

    pub fn with_custom_blocked_words(name: &str, words: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            blocked_words: words.into_iter().collect(),
            replacement: "***".to_string(),
            filter_input: true,
            filter_output: true,
        }
    }

    pub fn with_replacement(mut self, replacement: String) -> Self {
        self.replacement = replacement;
        self
    }

    pub fn filter_input_only(mut self) -> Self {
        self.filter_input = true;
        self.filter_output = false;
        self
    }

    pub fn filter_output_only(mut self) -> Self {
        self.filter_input = false;
        self.filter_output = true;
        self
    }

    fn default_blocked_words() -> HashSet<String> {
        // Default blocked words list
        vec![
            "badword1".to_string(),
            "badword2".to_string(),
            "profanity".to_string(),
        ]
        .into_iter()
        .collect()
    }

    fn filter_text(&self, text: &str) -> String {
        let mut filtered = text.to_string();
        for word in &self.blocked_words {
            filtered = filtered.replace(word, &self.replacement);
        }
        filtered
    }
}

impl Plugin for ProfanityFilterPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if self.filter_input {
            Ok(self.filter_text(prompt))
        } else {
            Ok(prompt.to_string())
        }
   _generate(&self, response: &str) -> Result<String> {
        if self.filter_output {
            Ok(self.filter_text(response))
        } else {
            Ok(response.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_filtering() {
        let plugin = ProfanityFilterPlugin::new("test");
        let input = "This contains badword1 and profanity.";
        let output = plugin.post_generate(input).unwrap();
        assert!(!output.contains("badword1"));
        assert!(!output.contains("profanity"));
    }

    #[test]
    fn test_custom_replacement() {
        let plugin = ProfanityFilterPlugin::new("test")
            .with_replacement("[FILTERED]".to_string());
        let input = "This has profanity.";
        let output = plugin.post_generate(input).unwrap();
        assert!(output.contains("[FILTERED]"));
    }

    #[test]
    fn test_custom_blocked_words() {
        let plugin = ProfanityFilterPlugin::with_custom_blocked_words(
            "test",
            vec!["blocked".to_string()],
        );
        let input = "This is blocked content.";
        let output = plugin.post_generate(input).unwrap();
        assert!(!output.contains("blocked"));
    }
}
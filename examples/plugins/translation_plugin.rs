// Translation Plugin for Loci
// 
// This plugin wraps prompts with translation instructions.
// Supports multiple language pairs.

use loci::plugin::Plugin;
use loci::error::Result;

pub struct TranslationPlugin {
    name: String,
    source_lang: String,
    target_lang: String,
}

impl TranslationPlugin {
    pub fn new(name: &str, source_lang: &str, target_lang: &str) -> Self {
        Self {
            name: name.to_string(),
            source_lang: source_lang.to_string(),
            target_lang: target_lang.to_string(),
        }
    }

    // Common language pairs
    pub fn english_to_chinese(name: &str) -> Self {
        Self::new(name, "English", "Chinese")
    }

    pub fn chinese_to_english(name: &str) -> Self {
        Self::new(name, "Chinese", "English")
    }

    pub fn english_to_spanish(name: &str) -> Self {
        Self::new(name, "English", "Spanish")
    }

    pub fn english_to_french(name: &str) -> Self {
        Self::new(name, "English", "French")
    }

    fn get_prompt_template(&self) -> String {
        format!(
            "Translate the following text from {} to {}. Only provide the translation, no explanations.\n\nText: {{prompt}}\nTranslation:",
            self.source_lang, self.target_lang
        )
    }
}

impl Plugin for TranslationPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        let template = self.get_prompt_template();
        Ok(template.replace("{prompt}", prompt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translation_prompt() {
        let plugin = TranslationPlugin::english_to_chinese("trans_zh");
        let input = "Hello, world!";
        let output = plugin.pre_generate(input).unwrap();
        
        assert!(output.contains("Translate"));
        assert!(output.contains("English"));
        assert!(output.contains("Chinese"));
        assert!(output.contains("Hello, world!"));
    }

    #[test]
    fn test_custom_language_pair() {
        let plugin = TranslationPlugin::new("custom", "Spanish", "German");
        let input = "Hola mundo";
        let output = plugin.pre_generate(input).unwrap();
        
        assert!(output.contains("Spanish"));
        assert!(output.contains("German"));
    }
}
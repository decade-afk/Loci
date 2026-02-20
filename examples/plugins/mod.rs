// Plugin Examples for Loci
// 
// This module contains various example plugins demonstrating
// the extensibility of Loci's plugin system.

pub mod profanity_filter_plugin;
pub mod json_output_formatter_plugin;
pub mod translation_plugin;
pub mod code_explainer_plugin;

pub use profanity_filter_plugin::ProfanityFilterPlugin;
pub use json_output_formatter_plugin::JsonFormatterPlugin;
pub use translation_plugin::TranslationPlugin;
pub use code_explainer_plugin::{CodeExplainerPlugin, DetailLevel};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_plugins() {
        // Test that all plugins can be created
        let profanity = ProfanityFilterPlugin::new("test");
        assert_eq!(profanity.name(), "test");

        let json_formatter = JsonFormatterPlugin::new("test");
        assert_eq!(json_formatter.name(), "test");

        let translator = TranslationPlugin::english_to_chinese("test");
        assert_eq!(translator.name(), "test");

        let explainer = CodeExplainerPlugin::new("test");
        assert_eq!(explainer.name(), "test");
    }
}
//! Built-in example plugins used by documentation and smoke tests.

use crate::error::Result;
use crate::plugin::Plugin;

pub mod plugins {
    use super::*;

    pub struct ProfanityFilterPlugin {
        name: String,
        blocked_terms: Vec<String>,
    }

    impl ProfanityFilterPlugin {
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                blocked_terms: vec!["badword".to_string(), "offensive".to_string()],
            }
        }
    }

    impl Plugin for ProfanityFilterPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn post_generate(&self, response: &str) -> Result<String> {
            let mut output = response.to_string();
            for blocked in &self.blocked_terms {
                output = output.replace(blocked, "***");
            }
            Ok(output)
        }
    }

    pub struct JsonFormatterPlugin {
        name: String,
    }

    impl JsonFormatterPlugin {
        pub fn new(name: impl Into<String>) -> Self {
            Self { name: name.into() }
        }
    }

    impl Plugin for JsonFormatterPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn post_generate(&self, response: &str) -> Result<String> {
            Ok(serde_json::json!({ "response": response }).to_string())
        }
    }

    pub struct TranslationPlugin {
        name: String,
        direction: String,
    }

    impl TranslationPlugin {
        pub fn english_to_chinese(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                direction: "en->zh".to_string(),
            }
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
            Ok(format!("[translation:{}] {}", self.direction, prompt))
        }
    }

    pub struct CodeExplainerPlugin {
        name: String,
        style: String,
    }

    impl CodeExplainerPlugin {
        pub fn detailed(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                style: "detailed".to_string(),
            }
        }
    }

    impl Plugin for CodeExplainerPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn pre_generate(&self, prompt: &str) -> Result<String> {
            Ok(format!("[code-explainer:{}] {}", self.style, prompt))
        }
    }
}

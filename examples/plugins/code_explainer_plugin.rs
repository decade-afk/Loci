// Code Explainer Plugin for Loci
// 
// This plugin enhances prompts for code explanation tasks.
// Automatically detects code languages and formats the prompt.

use loci::plugin::Plugin;
use loci::error::Result;
use std::collections::HashMap;

pub struct CodeExplainerPlugin {
    name: String,
    language: Option<String>,
    detail_level: DetailLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum DetailLevel {
    Brief,
    Standard,
    Detailed,
    Comprehensive,
}

impl DetailLevel {
    fn as_str(&self) -> &str {
        match self {
            DetailLevel::Brief => "brief",
            DetailLevel::Standard => "standard",
            DetailLevel::Detailed => "detailed",
            DetailLevel::Comprehensive => "comprehensive",
        }
    }
}

impl CodeExplainerPlugin {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            language: None,
            detail_level: DetailLevel::Standard,
        }
    }

    pub fn with_language(mut self, language: &str) -> Self {
        self.language = Some(language.to_string());
        self
    }

    pub fn with_detail_level(mut self, level: DetailLevel) -> Self {
        self.detail_level = level;
        self
    }

    // Convenience constructors
    pub fn brief(name: &str) -> Self {
        Self::new(name).with_detail_level(DetailLevel::Brief)
    }

    pub fn detailed(name: &str) -> Self {
        Self::new(name).with_detail_level(DetailLevel::Detailed)
    }

    pub fn comprehensive(name: &str) -> Self {
        Self::new(name).with_detail_level(DetailLevel::Comprehensive)
    }

    fn detect_language(&self, code: &str) -> Option<String> {
        let lang_keywords: HashMap<&str, Vec<&str>> = vec![
            ("Python", vec!["def ", "import ", "from ", "if __name__", "# ", "    ", "print("]),
            ("JavaScript", vec!["function ", "const ", "let ", "=>", "console.log(", "export "]),
            ("Rust", vec!["fn ", "let mut", "impl ", "pub fn", "use ", "struct ", "enum "]),
            ("Java", vec!["public class", "public static void", "System.out.println", "private "]),
            ("C++", vec!["#include", "int main(", "std::", "cout <<", "return 0;", "class "]),
            ("Go", vec!["func ", "package ", "import ", "fmt.", "go ", "defer "]),
            ("TypeScript", vec!["interface ", "type ", ": string", ": number", "interface "]),
            ("C", vec!["#include", "int main(", "printf(", "return 0;", "struct "]),
            ("Swift", vec!["func ", "let ", "var ", "print(", "struct ", "class "]),
            ("Kotlin", vec!["fun ", "val ", "var ", "println(", "class ", "data class "]),
        ]
        .into_iter()
        .collect();

        for (lang, keywords) in &lang_keywords {
            let mut matches = 0;
            for keyword in keywords {
                if code.contains(keyword) {
                    matches += 1;
                }
            }
            if matches >= 2 {
                return Some(lang.to_string());
            }
        }
        None
    }

    fn get_explanation_prompt(&self, code: &str, detected_lang: Option<&str>) -> String {
        let lang = self.language
            .as_deref()
            .or(detected_lang)
            .unwrap_or("this programming language");

        let detail_instruction = match self.detail_level {
            DetailLevel::Brief => "Provide a brief summary of what this code does in 1-2 sentences.",
            DetailLevel::Standard => "Explain the functionality, key concepts, and how the code works.",
            DetailLevel::Detailed => "Provide a detailed explanation including functionality, key concepts, logic flow, and important details.",
            DetailLevel::Comprehensive => "Provide a comprehensive analysis including: functionality, key concepts, logic flow, complexity analysis, best practices, potential issues, and suggestions.",
        };

        format!(
            "You are a code explanation expert. Explain the following {} code.\n\n{} \n\nCode:\n```\n{}\n```\n\nExplanation:",
            lang, detail_instruction, code
        )
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
        let detected_lang = self.detect_language(prompt);
        Ok(self.get_explanation_prompt(prompt, detected_lang.as_deref()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_detection() {
        let plugin = CodeExplainerPlugin::new("test");
        let code = r#"
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
"#;
        let prompt = plugin.pre_generate(code).unwrap();
        assert!(prompt.contains("Python"));
    }

    #[test]
    fn test_rust_detection() {
        let plugin = CodeExplainerPlugin::new("test");
        let code = r#"
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

fn main() {
    println!("{}", greet("World"));
}
"#;
        let prompt = plugin.pre_generate(code).unwrap();
        assert!(prompt.contains("Rust"));
    }

    #[test]
    fn test_javascript_detection() {
        let plugin = CodeExplainerPlugin::new("test");
        let code = r#"
function greet(name) {
    return `Hello, ${name}!`;
}

console.log(greet("World"));
"#;
        let prompt = plugin.pre_generate(code).unwrap();
        assert!(prompt.contains("JavaScript"));
    }

    #[test]
    fn test_brief_explanation() {
        let plugin = CodeExplainerPlugin::brief("test");
        let code = "print('Hello')";
        let prompt = plugin.pre_generate(code).unwrap();
        assert!(prompt.contains("brief"));
        assert!(prompt.contains("1-2 sentences"));
    }

    #[test]
    fn test_comprehensive_explanation() {
        let plugin = CodeExplainerPlugin::comprehensive("test");
        let code = "print('Hello')";
        let prompt = plugin.pre_generate(code).unwrap();
        assert!(prompt.contains("comprehensive"));
        assert!(prompt.contains("complexity analysis"));
    }
}
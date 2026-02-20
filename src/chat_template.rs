//! Chat template system for formatting conversations

use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            name: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Chat template types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplateType {
    /// ChatML format (used by GPT-4, Qwen, etc.)
    ChatML,
    /// Llama 2/3 format
    Llama,
    /// Alpaca format
    Alpaca,
    /// Vicuna format
    Vicuna,
    /// Mistral/Mixtral format
    Mistral,
    /// Zephyr format
    Zephyr,
    /// Custom template
    Custom,
}

/// Chat template for formatting conversations
pub struct ChatTemplate {
    template_type: ChatTemplateType,
    custom_template: Option<String>,
    bos_token: String,
    eos_token: String,
    system_prefix: String,
    system_suffix: String,
    user_prefix: String,
    user_suffix: String,
    assistant_prefix: String,
    assistant_suffix: String,
}

impl ChatTemplate {
    /// Create a new chat template
    pub fn new(template_type: ChatTemplateType) -> Self {
        match template_type {
            ChatTemplateType::ChatML => Self::chatml(),
            ChatTemplateType::Llama => Self::llama(),
            ChatTemplateType::Alpaca => Self::alpaca(),
            ChatTemplateType::Vicuna => Self::vicuna(),
            ChatTemplateType::Mistral => Self::mistral(),
            ChatTemplateType::Zephyr => Self::zephyr(),
            ChatTemplateType::Custom => Self::default(),
        }
    }

    /// ChatML template (GPT-4, Qwen, etc.)
    pub fn chatml() -> Self {
        Self {
            template_type: ChatTemplateType::ChatML,
            custom_template: None,
            bos_token: String::new(),
            eos_token: "<|im_end|>".to_string(),
            system_prefix: "<|im_start|>system\n".to_string(),
            system_suffix: "<|im_end|>\n".to_string(),
            user_prefix: "<|im_start|>user\n".to_string(),
            user_suffix: "<|im_end|>\n".to_string(),
            assistant_prefix: "<|im_start|>assistant\n".to_string(),
            assistant_suffix: "<|im_end|>\n".to_string(),
        }
    }

    /// Llama 2/3 template
    pub fn llama() -> Self {
        Self {
            template_type: ChatTemplateType::Llama,
            custom_template: None,
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            system_prefix: "[INST] <<SYS>>\n".to_string(),
            system_suffix: "\n<</SYS>>\n\n".to_string(),
            user_prefix: "".to_string(),
            user_suffix: " [/INST] ".to_string(),
            assistant_prefix: "".to_string(),
            assistant_suffix: " </s><s>[INST] ".to_string(),
        }
    }

    /// Alpaca template
    pub fn alpaca() -> Self {
        Self {
            template_type: ChatTemplateType::Alpaca,
            custom_template: None,
            bos_token: String::new(),
            eos_token: String::new(),
            system_prefix: "".to_string(),
            system_suffix: "\n\n".to_string(),
            user_prefix: "### Instruction:\n".to_string(),
            user_suffix: "\n\n".to_string(),
            assistant_prefix: "### Response:\n".to_string(),
            assistant_suffix: "\n\n".to_string(),
        }
    }

    /// Vicuna template
    pub fn vicuna() -> Self {
        Self {
            template_type: ChatTemplateType::Vicuna,
            custom_template: None,
            bos_token: String::new(),
            eos_token: "</s>".to_string(),
            system_prefix: "".to_string(),
            system_suffix: "\n\n".to_string(),
            user_prefix: "USER: ".to_string(),
            user_suffix: "\n".to_string(),
            assistant_prefix: "ASSISTANT: ".to_string(),
            assistant_suffix: "</s>\n".to_string(),
        }
    }

    /// Mistral/Mixtral template
    pub fn mistral() -> Self {
        Self {
            template_type: ChatTemplateType::Mistral,
            custom_template: None,
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            system_prefix: "".to_string(),
            system_suffix: "\n\n".to_string(),
            user_prefix: "[INST] ".to_string(),
            user_suffix: " [/INST]".to_string(),
            assistant_prefix: "".to_string(),
            assistant_suffix: "</s>".to_string(),
        }
    }

    /// Zephyr template
    pub fn zephyr() -> Self {
        Self {
            template_type: ChatTemplateType::Zephyr,
            custom_template: None,
            bos_token: "<|system|>".to_string(),
            eos_token: "</s>".to_string(),
            system_prefix: "\n".to_string(),
            system_suffix: "</s>\n".to_string(),
            user_prefix: "<|user|>\n".to_string(),
            user_suffix: "</s>\n".to_string(),
            assistant_prefix: "<|assistant|>\n".to_string(),
            assistant_suffix: "</s>\n".to_string(),
        }
    }

    /// Format a conversation into a prompt
    pub fn format(&self, messages: &[ChatMessage]) -> Result<String> {
        if messages.is_empty() {
            return Err(LociError::InvalidArgument(
                "Cannot format empty message list".to_string(),
            ));
        }

        let mut prompt = self.bos_token.clone();

        for (_i, msg) in messages.iter().enumerate() {
            let (prefix, suffix) = match msg.role.as_str() {
                "system" => (&self.system_prefix, &self.system_suffix),
                "user" => (&self.user_prefix, &self.user_suffix),
                "assistant" => (&self.assistant_prefix, &self.assistant_suffix),
                _ => {
                    return Err(LociError::InvalidArgument(format!(
                        "Unknown role: {}",
                        msg.role
                    )))
                }
            };

            prompt.push_str(prefix);
            prompt.push_str(&msg.content);
            prompt.push_str(suffix);
        }

        // Add assistant prefix for generation
        if let Some(last_msg) = messages.last() {
            if last_msg.role != "assistant" {
                prompt.push_str(&self.assistant_prefix);
            }
        }

        Ok(prompt)
    }

    /// Format a single user message (convenience method)
    pub fn format_user_message(&self, content: &str) -> Result<String> {
        self.format(&[ChatMessage::user(content)])
    }

    /// Format a conversation with system prompt
    pub fn format_with_system(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
    ) -> Result<String> {
        let mut all_messages = vec![ChatMessage::system(system_prompt)];
        all_messages.extend_from_slice(messages);
        self.format(&all_messages)
    }
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self::chatml()
    }
}

/// Builder for creating custom chat templates
pub struct ChatTemplateBuilder {
    bos_token: String,
    eos_token: String,
    system_prefix: String,
    system_suffix: String,
    user_prefix: String,
    user_suffix: String,
    assistant_prefix: String,
    assistant_suffix: String,
}

impl ChatTemplateBuilder {
    pub fn new() -> Self {
        Self {
            bos_token: String::new(),
            eos_token: String::new(),
            system_prefix: String::new(),
            system_suffix: String::new(),
            user_prefix: String::new(),
            user_suffix: String::new(),
            assistant_prefix: String::new(),
            assistant_suffix: String::new(),
        }
    }

    pub fn bos_token(mut self, token: impl Into<String>) -> Self {
        self.bos_token = token.into();
        self
    }

    pub fn eos_token(mut self, token: impl Into<String>) -> Self {
        self.eos_token = token.into();
        self
    }

    pub fn system_format(mut self, prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        self.system_prefix = prefix.into();
        self.system_suffix = suffix.into();
        self
    }

    pub fn user_format(mut self, prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        self.user_prefix = prefix.into();
        self.user_suffix = suffix.into();
        self
    }

    pub fn assistant_format(
        mut self,
        prefix: impl Into<String>,
        suffix: impl Into<String>,
    ) -> Self {
        self.assistant_prefix = prefix.into();
        self.assistant_suffix = suffix.into();
        self
    }

    pub fn build(self) -> ChatTemplate {
        ChatTemplate {
            template_type: ChatTemplateType::Custom,
            custom_template: None,
            bos_token: self.bos_token,
            eos_token: self.eos_token,
            system_prefix: self.system_prefix,
            system_suffix: self.system_suffix,
            user_prefix: self.user_prefix,
            user_suffix: self.user_suffix,
            assistant_prefix: self.assistant_prefix,
            assistant_suffix: self.assistant_suffix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_format() {
        let template = ChatTemplate::chatml();
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello!"),
            ChatMessage::assistant("Hi! How can I help you?"),
            ChatMessage::user("What's the weather?"),
        ];

        let prompt = template.format(&messages).unwrap();
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_llama_format() {
        let template = ChatTemplate::llama();
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello!"),
        ];

        let prompt = template.format(&messages).unwrap();
        assert!(prompt.contains("<s>"));
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("<<SYS>>"));
    }

    #[test]
    fn test_custom_template() {
        let template = ChatTemplateBuilder::new()
            .bos_token("<BOS>")
            .eos_token("<EOS>")
            .user_format("User: ", "\n")
            .assistant_format("Assistant: ", "\n")
            .build();

        let messages = vec![ChatMessage::user("Hello!")];
        let prompt = template.format(&messages).unwrap();
        assert!(prompt.contains("<BOS>"));
        assert!(prompt.contains("User: Hello!"));
    }
}

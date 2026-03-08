//! Chat template demonstration
//!
//! This example shows how to use the chat template system to format conversations
//! for different model types (ChatML, Llama, Alpaca, etc.)

use loci::chat_template::{ChatMessage, ChatTemplate, ChatTemplateBuilder, ChatTemplateType};
use loci::prelude::*;

fn main() -> Result<()> {
    println!("=== Chat Template Demo ===\n");

    // Example 1: ChatML format (used by GPT-4, Qwen, etc.)
    println!("1. ChatML Format:");
    let chatml = ChatTemplate::chatml();
    let messages = vec![
        ChatMessage::system("You are a helpful AI assistant."),
        ChatMessage::user("What is Rust?"),
        ChatMessage::assistant("Rust is a systems programming language."),
        ChatMessage::user("Tell me more about its safety features."),
    ];
    let prompt = chatml.format(&messages)?;
    println!("{}\n", prompt);

    // Example 2: Llama format
    println!("2. Llama 2/3 Format:");
    let llama = ChatTemplate::llama();
    let prompt = llama.format(&messages)?;
    println!("{}\n", prompt);

    // Example 3: Alpaca format
    println!("3. Alpaca Format:");
    let alpaca = ChatTemplate::alpaca();
    let simple_messages = vec![ChatMessage::user(
        "Explain quantum computing in simple terms.",
    )];
    let prompt = alpaca.format(&simple_messages)?;
    println!("{}\n", prompt);

    // Example 4: Custom template
    println!("4. Custom Template:");
    let custom = ChatTemplateBuilder::new()
        .bos_token("<START>")
        .eos_token("<END>")
        .user_format("Human: ", "\n")
        .assistant_format("AI: ", "\n")
        .build();

    let custom_messages = vec![
        ChatMessage::user("Hello!"),
        ChatMessage::assistant("Hi there!"),
    ];
    let prompt = custom.format(&custom_messages)?;
    println!("{}\n", prompt);

    // Example 5: Format with system prompt
    println!("5. With System Prompt:");
    let template = ChatTemplate::new(ChatTemplateType::Vicuna);
    let user_messages = vec![ChatMessage::user("What's the capital of France?")];
    let prompt = template.format_with_system("You are a geography expert.", &user_messages)?;
    println!("{}\n", prompt);

    Ok(())
}

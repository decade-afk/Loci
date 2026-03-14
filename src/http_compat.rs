use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct OpenAiChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiChatCompletionsRequest {
    pub model: Option<String>,
    pub messages: Vec<OpenAiChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatCompletionsResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChatChoice>,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatChoice {
    pub index: u32,
    pub message: OpenAiChatMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatStreamResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAiChatStreamChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiChatStreamChoice {
    pub index: u32,
    pub delta: OpenAiChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct OpenAiChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OpenAiEmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiEmbeddingsRequest {
    pub model: Option<String>,
    pub input: OpenAiEmbeddingInput,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiEmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiEmbeddingData>,
    pub model: String,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiEmbeddingData {
    pub object: &'static str,
    pub index: u32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiModelListResponse {
    pub object: &'static str,
    pub data: Vec<OpenAiModelDescriptor>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiModelDescriptor {
    pub id: String,
    pub object: &'static str,
    pub owned_by: &'static str,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub stream: Option<bool>,
    pub options: Option<OllamaGenerateOptions>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaGenerateOptions {
    pub num_predict: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<&'static str>,
    pub prompt_eval_count: u32,
    pub eval_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelTag>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaModelTag {
    pub name: String,
    pub model: String,
    pub modified_at: String,
    pub size: u64,
}

pub fn openai_chat_messages_to_prompt(messages: &[OpenAiChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        if !prompt.is_empty() {
            prompt.push_str("\n\n");
        }
        prompt.push_str(&format!(
            "{}: {}",
            message.role.trim().to_uppercase(),
            message.content.trim()
        ));
    }
    if !prompt.is_empty() {
        prompt.push_str("\n\nASSISTANT:");
    }
    prompt
}

pub fn normalize_openai_embedding_input(input: &OpenAiEmbeddingInput) -> Vec<String> {
    match input {
        OpenAiEmbeddingInput::Single(value) => vec![value.clone()],
        OpenAiEmbeddingInput::Multiple(values) => values.clone(),
    }
}

pub fn estimate_token_count(text: &str) -> u32 {
    let whitespace = text.split_whitespace().count() as u32;
    let char_chunks = (text.chars().count() as u32).div_ceil(4);
    whitespace.max(char_chunks).max((!text.is_empty()) as u32)
}

pub fn chunk_text_for_streaming(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let total_chars = text.chars().count();
    let target_chunks = estimate_token_count(text).clamp(1, 24) as usize;
    let chunk_chars = ((total_chars + target_chunks - 1) / target_chunks).max(24);

    let mut chunks = Vec::new();
    let mut start_byte = 0usize;
    let mut char_count = 0usize;

    for (byte_index, _) in text.char_indices() {
        if char_count >= chunk_chars {
            chunks.push(text[start_byte..byte_index].to_string());
            start_byte = byte_index;
            char_count = 0;
        }
        char_count += 1;
    }

    if start_byte < text.len() {
        chunks.push(text[start_byte..].to_string());
    }

    chunks.retain(|chunk| !chunk.is_empty());
    chunks
}

pub fn openai_chat_stream_chunk(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&'static str>,
    content: Option<String>,
    finish_reason: Option<&'static str>,
) -> OpenAiChatStreamResponse {
    OpenAiChatStreamResponse {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![OpenAiChatStreamChoice {
            index: 0,
            delta: OpenAiChatStreamDelta { role, content },
            finish_reason,
        }],
    }
}

pub fn ollama_stream_event(
    model: &str,
    created_at: &str,
    response: String,
    done: bool,
    done_reason: Option<&'static str>,
    prompt_eval_count: u32,
    eval_count: u32,
    error: Option<String>,
) -> OllamaGenerateResponse {
    OllamaGenerateResponse {
        model: model.to_string(),
        created_at: created_at.to_string(),
        response,
        done,
        done_reason,
        prompt_eval_count,
        eval_count,
        error,
    }
}

pub fn compatibility_created_at() -> String {
    format!("unix-ms:{}", unix_seconds_now() * 1000)
}

pub fn unix_seconds_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_messages_render_to_prompt() {
        let prompt = openai_chat_messages_to_prompt(&[
            OpenAiChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            OpenAiChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ]);

        assert!(prompt.contains("SYSTEM: You are helpful."));
        assert!(prompt.contains("USER: Hello"));
        assert!(prompt.ends_with("ASSISTANT:"));
    }

    #[test]
    fn normalize_embedding_input_handles_single_and_batch() {
        assert_eq!(
            normalize_openai_embedding_input(&OpenAiEmbeddingInput::Single("hi".to_string())),
            vec!["hi".to_string()]
        );
        assert_eq!(
            normalize_openai_embedding_input(&OpenAiEmbeddingInput::Multiple(vec![
                "a".to_string(),
                "b".to_string()
            ])),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    #[test]
    fn chunk_text_for_streaming_preserves_full_content() {
        let text = "streaming compatibility output should preserve every byte across chunks";
        let chunks = chunk_text_for_streaming(text);

        assert!(!chunks.is_empty());
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn openai_stream_chunk_formats_role_and_finish() {
        let chunk = openai_chat_stream_chunk(
            "chatcmpl-test",
            42,
            "loci/demo",
            Some("assistant"),
            None,
            None,
        );
        let encoded = serde_json::to_string(&chunk).expect("chunk should serialize");
        assert!(encoded.contains("\"chat.completion.chunk\""));
        assert!(encoded.contains("\"role\":\"assistant\""));

        let final_chunk =
            openai_chat_stream_chunk("chatcmpl-test", 42, "loci/demo", None, None, Some("stop"));
        let final_json = serde_json::to_string(&final_chunk).expect("chunk should serialize");
        assert!(final_json.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn ollama_stream_event_can_represent_completion_and_error() {
        let done = ollama_stream_event(
            "loci/demo",
            "unix-ms:42",
            "hello".to_string(),
            true,
            Some("stop"),
            4,
            2,
            None,
        );
        let done_json = serde_json::to_string(&done).expect("event should serialize");
        assert!(done_json.contains("\"done\":true"));
        assert!(done_json.contains("\"done_reason\":\"stop\""));

        let error = ollama_stream_event(
            "loci/demo",
            "unix-ms:42",
            String::new(),
            true,
            Some("error"),
            0,
            0,
            Some("boom".to_string()),
        );
        let error_json = serde_json::to_string(&error).expect("event should serialize");
        assert!(error_json.contains("\"error\":\"boom\""));
    }
}

use loci_protocol::{ImageInput, SessionRequest};

use crate::http_io::{sse_event, sse_response, stream_fragments};
use crate::http_types::{ChatCompletionsRequest, ChatContentPart, ChatMessage, ChatMessageContent};

/// Builds an OpenAI-compatible streaming completion response from a single fully generated text.
pub(crate) fn completion_stream_response(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
) -> String {
    let mut events = Vec::new();
    for fragment in stream_fragments(text) {
        events.push(sse_event(&serde_json::json!({
            "id": id,
            "object": "text_completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "text": fragment,
                "index": 0,
                "finish_reason": serde_json::Value::Null,
            }],
        })));
    }
    events.push(sse_event(&serde_json::json!({
        "id": id,
        "object": "text_completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "text": "",
            "index": 0,
            "finish_reason": "stop",
        }],
    })));
    events.push("data: [DONE]\n\n".to_string());
    sse_response(&events.concat())
}

/// Builds an OpenAI-compatible streaming chat response while preserving the expected delta framing.
pub(crate) fn chat_completion_stream_response(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
) -> String {
    let mut events = vec![sse_event(&serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
            },
            "finish_reason": serde_json::Value::Null,
        }],
    }))];
    for fragment in stream_fragments(text) {
        events.push(sse_event(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": fragment,
                },
                "finish_reason": serde_json::Value::Null,
            }],
        })));
    }
    events.push(sse_event(&serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    })));
    events.push("data: [DONE]\n\n".to_string());
    sse_response(&events.concat())
}

/// Flattens OpenAI-style chat messages into the prompt format expected by the internal inference path.
pub(crate) fn flatten_messages(
    messages: &[ChatMessage],
) -> Result<(String, Vec<ImageInput>), String> {
    let mut prompt = String::new();
    let mut images = Vec::new();
    for message in messages {
        prompt.push_str(&message.role);
        prompt.push_str(": ");
        match &message.content {
            ChatMessageContent::Text(text) => prompt.push_str(text),
            ChatMessageContent::Parts(parts) => {
                let mut first = true;
                for part in parts {
                    if !first {
                        prompt.push(' ');
                    }
                    first = false;
                    match part {
                        ChatContentPart::Text { text } | ChatContentPart::InputText { text } => {
                            prompt.push_str(text)
                        }
                        ChatContentPart::ImageUrl { image_url }
                        | ChatContentPart::InputImage { image_url } => {
                            prompt.push_str("<image>");
                            images.push(ImageInput::Url {
                                url: image_url.url().to_string(),
                            });
                        }
                    }
                }
            }
        }
        prompt.push_str("\n\n");
    }
    prompt.push_str("assistant:");
    Ok((prompt, images))
}

/// Converts an OpenAI-compatible chat request payload into the normalized internal session request.
pub(crate) fn chat_request_from_payload(
    payload: ChatCompletionsRequest,
) -> Result<SessionRequest, String> {
    let (prompt, images) = flatten_messages(&payload.messages)?;
    Ok(SessionRequest {
        prompt,
        max_tokens: payload.max_tokens,
        temperature: payload.temperature,
        target_model: payload.model,
        images,
        structured_output: false,
        tool_calling: false,
    })
}

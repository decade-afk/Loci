use serde::Serialize;
use std::io::{Read, Write};

/// Splits a raw HTTP request buffer into the request line and decoded body payload.
pub(crate) fn parse_request_parts(request: &str) -> (&str, &str) {
    let (head, body) = request.split_once("\r\n\r\n").unwrap_or((request, ""));
    let request_line = head.lines().next().unwrap_or_default();
    (request_line, body)
}

/// Reads a complete HTTP/1.1 request, including chunked bodies and `Expect: 100-continue` flows.
pub(crate) fn read_request(
    stream: &mut (impl Read + Write),
) -> std::result::Result<String, String> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut sent_continue = false;
    loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| bad_request(&format!("read failed: {error}")))?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if !sent_continue {
            if let Some(framing) = request_framing(&buffer) {
                if framing.expect_continue {
                    stream
                        .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                        .map_err(|error| {
                            bad_request(&format!("failed to acknowledge request body: {error}"))
                        })?;
                    stream
                        .flush()
                        .map_err(|error| bad_request(&format!("flush failed: {error}")))?;
                    sent_continue = true;
                }
            }
        }
        if let Some(required_len) = required_request_len(&buffer) {
            if buffer.len() >= required_len {
                break;
            }
        }
    }

    let normalized = normalize_request_body(buffer)?;
    String::from_utf8(normalized).map_err(|_| bad_request("request is not valid utf-8"))
}

/// Serializes a success JSON body into a complete HTTP response.
pub(crate) fn json_response(body: &str) -> String {
    http_response("200 OK", body)
}

/// Serializes a client error as the server's standard JSON error response.
pub(crate) fn bad_request(message: &str) -> String {
    http_response("400 Bad Request", &format!(r#"{{"error":"{}"}}"#, message))
}

/// Serializes a missing-route error as the server's standard JSON error response.
pub(crate) fn not_found(message: &str) -> String {
    http_response("404 Not Found", &format!(r#"{{"error":"{}"}}"#, message))
}

/// Serializes a JSON body with the default JSON content type.
pub(crate) fn http_response(status: &str, body: &str) -> String {
    http_response_with_content_type(status, "application/json", body)
}

/// Serializes a complete Server-Sent Events response body.
pub(crate) fn sse_response(body: &str) -> String {
    http_response_with_content_type(status_ok(), "text/event-stream", body)
}

/// Serializes a response with an explicit content type while always closing the connection.
pub(crate) fn http_response_with_content_type(
    status: &str,
    content_type: &str,
    body: &str,
) -> String {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

/// Encodes one SSE `data:` event and falls back to a structured serialization error event.
pub(crate) fn sse_event<T: Serialize>(value: &T) -> String {
    match serde_json::to_string(value) {
        Ok(body) => format!("data: {body}\n\n"),
        Err(error) => format!(
            "data: {{\"type\":\"serialization_error\",\"message\":{}}}\n\n",
            serde_json::to_string(&error.to_string()).unwrap_or_else(|_| "\"unknown\"".to_string())
        ),
    }
}

/// Chunks a fully generated response into small fragments so streaming endpoints can emit incremental deltas.
pub(crate) fn stream_fragments(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return vec![String::new()];
    }

    chars
        .chunks(24)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect()
}

fn required_request_len(buffer: &[u8]) -> Option<usize> {
    let framing = request_framing(buffer)?;
    match framing.body {
        BodyFraming::ContentLength(content_length) => Some(framing.header_end + content_length),
        BodyFraming::Chunked => {
            let body = &buffer[framing.header_end..];
            body.windows(5)
                .position(|window| window == b"0\r\n\r\n")
                .map(|offset| framing.header_end + offset + 5)
        }
        BodyFraming::Empty => Some(framing.header_end),
    }
}

fn normalize_request_body(buffer: Vec<u8>) -> std::result::Result<Vec<u8>, String> {
    let Some(framing) = request_framing(&buffer) else {
        return Ok(buffer);
    };

    match framing.body {
        BodyFraming::Chunked => {
            let decoded = decode_chunked_body(&buffer[framing.header_end..])?;
            let mut normalized = buffer[..framing.header_end].to_vec();
            normalized.extend_from_slice(&decoded);
            Ok(normalized)
        }
        _ => Ok(buffer),
    }
}

/// Inspects the buffered request head and determines how much body data is still needed.
fn request_framing(buffer: &[u8]) -> Option<RequestFraming> {
    let header_end = buffer.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    let headers = std::str::from_utf8(&buffer[..header_end]).ok()?;
    let mut content_length = None;
    let mut chunked = false;
    let mut expect_continue = false;

    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse::<usize>().ok();
        }
        if name.eq_ignore_ascii_case("transfer-encoding")
            && value.to_ascii_lowercase().contains("chunked")
        {
            chunked = true;
        }
        if name.eq_ignore_ascii_case("expect")
            && value.to_ascii_lowercase().contains("100-continue")
        {
            expect_continue = true;
        }
    }

    let body = if chunked {
        BodyFraming::Chunked
    } else if let Some(content_length) = content_length {
        BodyFraming::ContentLength(content_length)
    } else {
        BodyFraming::Empty
    };

    Some(RequestFraming {
        header_end,
        body,
        expect_continue,
    })
}

/// Decodes an HTTP chunked-transfer body into the plain payload consumed by the router layer.
fn decode_chunked_body(body: &[u8]) -> std::result::Result<Vec<u8>, String> {
    let mut offset = 0usize;
    let mut decoded = Vec::new();

    loop {
        let size_line_end = find_bytes(&body[offset..], b"\r\n")
            .ok_or_else(|| bad_request("malformed chunked request body"))?;
        let size_line = std::str::from_utf8(&body[offset..offset + size_line_end])
            .map_err(|_| bad_request("chunk size is not valid utf-8"))?;
        let size_token = size_line.split(';').next().unwrap_or_default().trim();
        let chunk_size = usize::from_str_radix(size_token, 16)
            .map_err(|_| bad_request("chunk size is not valid hexadecimal"))?;
        offset += size_line_end + 2;

        if chunk_size == 0 {
            return Ok(decoded);
        }

        if body.len() < offset + chunk_size + 2 {
            return Err(bad_request("chunked request body ended unexpectedly"));
        }

        decoded.extend_from_slice(&body[offset..offset + chunk_size]);
        offset += chunk_size;

        if &body[offset..offset + 2] != b"\r\n" {
            return Err(bad_request("chunk delimiter is missing"));
        }
        offset += 2;
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn status_ok() -> &'static str {
    "200 OK"
}

#[derive(Clone, Copy)]
struct RequestFraming {
    header_end: usize,
    body: BodyFraming,
    expect_continue: bool,
}

#[derive(Clone, Copy)]
enum BodyFraming {
    Empty,
    ContentLength(usize),
    Chunked,
}

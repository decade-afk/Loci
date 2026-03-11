use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine as _;
use loci::error::{LociError, Result};
use loci::function_calling::{FunctionCall, FunctionDefinition};
use loci::tool_plugin::{dynamic_tool_plugin_into_opaque, DynamicToolPluginOpaque, ToolPlugin};
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:9515";
const W3C_ELEMENT_KEY: &str = "element-6066-11e4-a52e-4f735466cecf";

#[derive(Debug, Clone)]
struct WebDriverEndpoint {
    host: String,
    port: u16,
    base_path: String,
}

impl WebDriverEndpoint {
    fn parse(raw: &str) -> Result<Self> {
        let trimmed = raw.trim();
        let without_scheme = trimmed.strip_prefix("http://").ok_or_else(|| {
            LociError::InvalidArgument(format!(
                "Unsupported endpoint '{}'. Only http:// is supported",
                trimmed
            ))
        })?;

        let (host_port, path_part) = if let Some(idx) = without_scheme.find('/') {
            (&without_scheme[..idx], &without_scheme[idx..])
        } else {
            (without_scheme, "")
        };
        if host_port.is_empty() {
            return Err(LociError::InvalidArgument(
                "WebDriver endpoint host is empty".to_string(),
            ));
        }

        let (host, port) = if let Some((host, port_raw)) = host_port.rsplit_once(':') {
            if host.is_empty() {
                return Err(LociError::InvalidArgument(
                    "WebDriver endpoint host is empty".to_string(),
                ));
            }
            let port = port_raw.parse::<u16>().map_err(|_| {
                LociError::InvalidArgument(format!("Invalid WebDriver port '{}'", port_raw))
            })?;
            (host.to_string(), port)
        } else {
            (host_port.to_string(), 80)
        };

        let base_path = if path_part.is_empty() {
            String::new()
        } else {
            path_part.trim_end_matches('/').to_string()
        };
        Ok(Self {
            host,
            port,
            base_path,
        })
    }

    fn normalized_url(&self) -> String {
        if self.base_path.is_empty() {
            format!("http://{}:{}", self.host, self.port)
        } else {
            format!("http://{}:{}{}", self.host, self.port, self.base_path)
        }
    }

    fn join_path(&self, path: &str) -> Result<String> {
        if !path.starts_with('/') {
            return Err(LociError::InvalidArgument(format!(
                "WebDriver path must start with '/': {path}"
            )));
        }
        if self.base_path.is_empty() {
            Ok(path.to_string())
        } else {
            Ok(format!("{}{}", self.base_path, path))
        }
    }

    fn send_json(&self, method: &str, path: &str, payload: Option<&Value>) -> Result<Value> {
        let full_path = self.join_path(path)?;
        let body = match payload {
            Some(value) => serde_json::to_string(value)
                .map_err(|e| LociError::SerializationError(e.to_string()))?,
            None => String::new(),
        };

        let mut stream = TcpStream::connect((self.host.as_str(), self.port))
            .map_err(LociError::IoError)?;
        let mut req = format!(
            "{method} {full_path} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n",
            self.host, self.port
        );
        if body.is_empty() {
            req.push_str("Content-Length: 0\r\n\r\n");
        } else {
            req.push_str("Content-Type: application/json\r\n");
            req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
            req.push_str(&body);
        }
        stream.write_all(req.as_bytes()).map_err(LociError::IoError)?;
        stream.flush().map_err(LociError::IoError)?;

        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes).map_err(LociError::IoError)?;
        if bytes.is_empty() {
            return Err(LociError::Other(
                "Empty response from WebDriver server".to_string(),
            ));
        }

        let split = bytes
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| LociError::Other("Invalid HTTP response from WebDriver".to_string()))?;
        let header = String::from_utf8_lossy(&bytes[..split]).to_string();
        let body = String::from_utf8_lossy(&bytes[(split + 4)..]).to_string();
        let status_line = header
            .lines()
            .next()
            .ok_or_else(|| LociError::Other("Missing status line from WebDriver".to_string()))?;
        let status = status_line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| LociError::Other(format!("Malformed status line: {status_line}")))?
            .parse::<u16>()
            .map_err(|_| LociError::Other(format!("Invalid status line: {status_line}")))?;

        let payload = if body.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str::<Value>(&body)
                .map_err(|e| LociError::SerializationError(e.to_string()))?
        };

        if !(200..300).contains(&status) {
            return Err(LociError::Other(format!(
                "WebDriver request failed (HTTP {}): {}",
                status,
                webdriver_error_message(&payload)
            )));
        }

        extract_webdriver_value(payload)
    }
}

fn webdriver_error_message(payload: &Value) -> String {
    payload
        .get("value")
        .and_then(|v| v.get("message"))
        .and_then(Value::as_str)
        .or_else(|| payload.get("message").and_then(Value::as_str))
        .unwrap_or("unknown webdriver error")
        .to_string()
}

fn extract_webdriver_value(payload: Value) -> Result<Value> {
    if payload.is_null() {
        return Ok(Value::Null);
    }
    if let Some(value) = payload.get("value") {
        if let Some(error) = value.get("error").and_then(Value::as_str) {
            let message = value
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown webdriver error");
            return Err(LociError::Other(format!("WebDriver error {error}: {message}")));
        }
        return Ok(value.clone());
    }
    Ok(payload)
}

fn parse_required_string(call: &FunctionCall, key: &str) -> Result<String> {
    call.get_string(key).ok_or_else(|| {
        LociError::InvalidArgument(format!("Missing or invalid string argument: {key}"))
    })
}

fn parse_optional_endpoint(call: &FunctionCall) -> Result<WebDriverEndpoint> {
    let raw = call
        .get_string("endpoint")
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string());
    WebDriverEndpoint::parse(&raw)
}

fn parse_optional_u32(call: &FunctionCall, key: &str) -> Result<Option<u32>> {
    let Some(value) = call.get_number(key) else {
        return Ok(None);
    };
    if value < 0.0 {
        return Err(LociError::InvalidArgument(format!(
            "Argument '{key}' must be non-negative"
        )));
    }
    Ok(Some(value as u32))
}

fn parse_optional_i32(call: &FunctionCall, key: &str) -> Result<Option<i32>> {
    let Some(value) = call.get_number(key) else {
        return Ok(None);
    };
    if value < i32::MIN as f64 || value > i32::MAX as f64 {
        return Err(LociError::InvalidArgument(format!(
            "Argument '{key}' out of i32 range"
        )));
    }
    Ok(Some(value as i32))
}

fn parse_optional_string_array(call: &FunctionCall, key: &str) -> Result<Vec<String>> {
    let Some(value) = call.get_argument(key) else {
        return Ok(Vec::new());
    };
    let array = value.as_array().ok_or_else(|| {
        LociError::InvalidArgument(format!("Argument '{key}' must be an array of strings"))
    })?;
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        let text = item.as_str().ok_or_else(|| {
            LociError::InvalidArgument(format!(
                "Argument '{key}' must contain only strings"
            ))
        })?;
        out.push(text.to_string());
    }
    Ok(out)
}

fn find_element(endpoint: &WebDriverEndpoint, session_id: &str, selector: &str) -> Result<String> {
    let value = endpoint.send_json(
        "POST",
        &format!("/session/{session_id}/element"),
        Some(&json!({"using":"css selector","value":selector})),
    )?;
    if let Some(id) = value
        .get(W3C_ELEMENT_KEY)
        .and_then(Value::as_str)
        .or_else(|| value.get("ELEMENT").and_then(Value::as_str))
    {
        Ok(id.to_string())
    } else {
        Err(LociError::Other(
            "WebDriver did not return an element id".to_string(),
        ))
    }
}

fn webdriver_key_value(raw: &str) -> String {
    let key = raw.trim().to_ascii_lowercase();
    match key.as_str() {
        "enter" => "\u{E007}".to_string(),
        "tab" => "\u{E004}".to_string(),
        "escape" | "esc" => "\u{E00C}".to_string(),
        "backspace" => "\u{E003}".to_string(),
        "delete" => "\u{E017}".to_string(),
        "arrow_up" | "up" => "\u{E013}".to_string(),
        "arrow_down" | "down" => "\u{E015}".to_string(),
        "arrow_left" | "left" => "\u{E012}".to_string(),
        "arrow_right" | "right" => "\u{E014}".to_string(),
        "home" => "\u{E011}".to_string(),
        "end" => "\u{E010}".to_string(),
        "page_up" => "\u{E00E}".to_string(),
        "page_down" => "\u{E00F}".to_string(),
        "shift" => "\u{E008}".to_string(),
        "control" | "ctrl" => "\u{E009}".to_string(),
        "alt" => "\u{E00A}".to_string(),
        "meta" | "win" | "command" => "\u{E03D}".to_string(),
        "space" => " ".to_string(),
        _ => raw.to_string(),
    }
}

pub struct BrowserToolPlugin;

impl BrowserToolPlugin {
    fn send_actions(
        &self,
        endpoint: &WebDriverEndpoint,
        session_id: &str,
        actions: Value,
    ) -> Result<()> {
        endpoint.send_json(
            "POST",
            &format!("/session/{session_id}/actions"),
            Some(&json!({ "actions": actions })),
        )?;
        Ok(())
    }

    fn release_actions(&self, endpoint: &WebDriverEndpoint, session_id: &str) -> Result<()> {
        endpoint.send_json("DELETE", &format!("/session/{session_id}/actions"), None)?;
        Ok(())
    }

    fn open_session(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let browser = call
            .get_string("browser")
            .unwrap_or_else(|| "chrome".to_string())
            .to_ascii_lowercase();
        let headless = call.get_bool("headless").unwrap_or(false);
        let incognito = call.get_bool("incognito").unwrap_or(false);
        let width = parse_optional_u32(call, "window_width")?;
        let height = parse_optional_u32(call, "window_height")?;
        let binary = call.get_string("binary");
        let extra_args = parse_optional_string_array(call, "args")?;

        let options_key = match browser.as_str() {
            "chrome" => "goog:chromeOptions",
            "msedge" => "ms:edgeOptions",
            "firefox" => "moz:firefoxOptions",
            _ => {
                return Err(LociError::InvalidArgument(format!(
                    "Unsupported browser '{}'; expected chrome/msedge/firefox",
                    browser
                )))
            }
        };

        let mut args = Vec::new();
        if headless {
            if browser == "firefox" {
                args.push("-headless".to_string());
            } else {
                args.push("--headless=new".to_string());
            }
        }
        if incognito {
            if browser == "firefox" {
                args.push("-private".to_string());
            } else {
                args.push("--incognito".to_string());
            }
        }
        if let (Some(w), Some(h)) = (width, height) {
            args.push(format!("--window-size={},{}", w, h));
        }
        args.extend(extra_args);

        let mut always_match = serde_json::Map::new();
        always_match.insert("browserName".to_string(), json!(browser));

        let mut options = serde_json::Map::new();
        if !args.is_empty() {
            options.insert("args".to_string(), json!(args));
        }
        if let Some(binary) = binary {
            options.insert("binary".to_string(), json!(binary));
        }
        if !options.is_empty() {
            always_match.insert(options_key.to_string(), Value::Object(options));
        }

        let value = endpoint.send_json(
            "POST",
            "/session",
            Some(&json!({"capabilities":{"alwaysMatch":Value::Object(always_match)}})),
        )?;

        let session_id = value
            .get("sessionId")
            .and_then(Value::as_str)
            .or_else(|| {
                value
                    .get("value")
                    .and_then(|v| v.get("sessionId"))
                    .and_then(Value::as_str)
            })
            .ok_or_else(|| {
                LociError::Other("WebDriver did not return a session id".to_string())
            })?;

        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "capabilities": value.get("capabilities").cloned().unwrap_or(Value::Null)
        }))
    }

    fn close_session(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        endpoint.send_json("DELETE", &format!("/session/{session_id}"), None)?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "closed": true
        }))
    }

    fn navigate(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let url = parse_required_string(call, "url")?;

        endpoint.send_json(
            "POST",
            &format!("/session/{session_id}/url"),
            Some(&json!({ "url": url })),
        )?;
        let current_url = endpoint.send_json("GET", &format!("/session/{session_id}/url"), None)?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "url": current_url.as_str().unwrap_or("")
        }))
    }

    fn get_title(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let title = endpoint.send_json("GET", &format!("/session/{session_id}/title"), None)?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "title": title.as_str().unwrap_or("")
        }))
    }

    fn click(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let selector = parse_required_string(call, "selector")?;
        let element_id = find_element(&endpoint, &session_id, &selector)?;
        endpoint.send_json(
            "POST",
            &format!("/session/{session_id}/element/{element_id}/click"),
            Some(&json!({})),
        )?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "selector": selector,
            "clicked": true
        }))
    }

    fn type_text(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let selector = parse_required_string(call, "selector")?;
        let text = parse_required_string(call, "text")?;
        let clear_first = call.get_bool("clear_first").unwrap_or(true);

        let element_id = find_element(&endpoint, &session_id, &selector)?;
        if clear_first {
            endpoint.send_json(
                "POST",
                &format!("/session/{session_id}/element/{element_id}/clear"),
                Some(&json!({})),
            )?;
        }

        let chars = text.chars().map(|c| c.to_string()).collect::<Vec<_>>();
        endpoint.send_json(
            "POST",
            &format!("/session/{session_id}/element/{element_id}/value"),
            Some(&json!({
                "text": text,
                "value": chars
            })),
        )?;

        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "selector": selector,
            "typed_chars": text.chars().count(),
            "clear_first": clear_first
        }))
    }

    fn read_text(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let selector = parse_required_string(call, "selector")?;
        let element_id = find_element(&endpoint, &session_id, &selector)?;
        let text = endpoint.send_json(
            "GET",
            &format!("/session/{session_id}/element/{element_id}/text"),
            None,
        )?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "selector": selector,
            "text": text.as_str().unwrap_or("")
        }))
    }

    fn screenshot(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let output_path = call.get_string("output_path");
        let include_base64 = call.get_bool("include_base64").unwrap_or(false);

        let png_base64 = endpoint
            .send_json(
                "POST",
                &format!("/session/{session_id}/screenshot"),
                Some(&json!({})),
            )?
            .as_str()
            .ok_or_else(|| LociError::Other("Invalid screenshot response payload".to_string()))?
            .to_string();

        let mut out = json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "png_base64_len": png_base64.len()
        });

        if let Some(path) = output_path {
            let bytes = BASE64_STANDARD
                .decode(&png_base64)
                .map_err(|e| LociError::Other(format!("Failed to decode screenshot base64: {e}")))?;
            if let Some(parent) = Path::new(&path).parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).map_err(LociError::IoError)?;
                }
            }
            fs::write(&path, &bytes).map_err(LociError::IoError)?;
            out["saved_to"] = json!(path);
            out["bytes"] = json!(bytes.len());
        }
        if include_base64 {
            out["png_base64"] = json!(png_base64);
        }
        Ok(out)
    }

    fn key_press(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let key = parse_required_string(call, "key")?;
        let value = webdriver_key_value(&key);
        self.send_actions(
            &endpoint,
            &session_id,
            json!([{
                "type": "key",
                "id": "keyboard",
                "actions": [
                    { "type": "keyDown", "value": value },
                    { "type": "keyUp", "value": webdriver_key_value(&key) }
                ]
            }]),
        )?;
        self.release_actions(&endpoint, &session_id)?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "key": key,
            "sent": true
        }))
    }

    fn send_keys(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let text = parse_required_string(call, "text")?;
        let selector = call.get_string("selector");

        if let Some(selector) = selector {
            let element_id = find_element(&endpoint, &session_id, &selector)?;
            let chars = text.chars().map(|c| c.to_string()).collect::<Vec<_>>();
            endpoint.send_json(
                "POST",
                &format!("/session/{session_id}/element/{element_id}/value"),
                Some(&json!({
                    "text": text,
                    "value": chars
                })),
            )?;
            Ok(json!({
                "endpoint": endpoint.normalized_url(),
                "session_id": session_id,
                "selector": selector,
                "typed_chars": text.chars().count()
            }))
        } else {
            let mut key_actions = Vec::with_capacity(text.chars().count() * 2);
            for c in text.chars() {
                let v = c.to_string();
                key_actions.push(json!({ "type": "keyDown", "value": v }));
                key_actions.push(json!({ "type": "keyUp", "value": c.to_string() }));
            }

            self.send_actions(
                &endpoint,
                &session_id,
                json!([{
                    "type": "key",
                    "id": "keyboard",
                    "actions": key_actions
                }]),
            )?;
            self.release_actions(&endpoint, &session_id)?;
            Ok(json!({
                "endpoint": endpoint.normalized_url(),
                "session_id": session_id,
                "typed_chars": text.chars().count()
            }))
        }
    }

    fn mouse_move(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let x = parse_optional_i32(call, "x")?
            .ok_or_else(|| LociError::InvalidArgument("Missing x".to_string()))?;
        let y = parse_optional_i32(call, "y")?
            .ok_or_else(|| LociError::InvalidArgument("Missing y".to_string()))?;
        let duration_ms = parse_optional_u32(call, "duration_ms")?.unwrap_or(0);
        let origin = call
            .get_string("origin")
            .unwrap_or_else(|| "viewport".to_string());

        self.send_actions(
            &endpoint,
            &session_id,
            json!([{
                "type": "pointer",
                "id": "mouse",
                "parameters": { "pointerType": "mouse" },
                "actions": [
                    {
                        "type": "pointerMove",
                        "duration": duration_ms,
                        "x": x,
                        "y": y,
                        "origin": origin
                    }
                ]
            }]),
        )?;
        self.release_actions(&endpoint, &session_id)?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "x": x,
            "y": y,
            "origin": origin
        }))
    }

    fn mouse_click_at(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let x = parse_optional_i32(call, "x")?
            .ok_or_else(|| LociError::InvalidArgument("Missing x".to_string()))?;
        let y = parse_optional_i32(call, "y")?
            .ok_or_else(|| LociError::InvalidArgument("Missing y".to_string()))?;
        let button_raw = parse_optional_u32(call, "button")?.unwrap_or(0);
        if button_raw > 2 {
            return Err(LociError::InvalidArgument(
                "button must be 0(left), 1(middle), or 2(right)".to_string(),
            ));
        }
        let button = button_raw as i32;
        let duration_ms = parse_optional_u32(call, "duration_ms")?.unwrap_or(0);
        let double_click = call.get_bool("double_click").unwrap_or(false);
        let mut action_items = vec![
            json!({
                "type": "pointerMove",
                "duration": duration_ms,
                "x": x,
                "y": y,
                "origin": "viewport"
            }),
            json!({ "type": "pointerDown", "button": button }),
            json!({ "type": "pointerUp", "button": button }),
        ];
        if double_click {
            action_items.push(json!({ "type": "pointerDown", "button": button }));
            action_items.push(json!({ "type": "pointerUp", "button": button }));
        }

        self.send_actions(
            &endpoint,
            &session_id,
            json!([{
                "type": "pointer",
                "id": "mouse",
                "parameters": { "pointerType": "mouse" },
                "actions": action_items
            }]),
        )?;
        self.release_actions(&endpoint, &session_id)?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "x": x,
            "y": y,
            "button": button,
            "double_click": double_click
        }))
    }

    fn mouse_wheel(&self, call: &FunctionCall) -> Result<Value> {
        let endpoint = parse_optional_endpoint(call)?;
        let session_id = parse_required_string(call, "session_id")?;
        let delta_x = parse_optional_i32(call, "delta_x")?.unwrap_or(0);
        let delta_y = parse_optional_i32(call, "delta_y")?
            .ok_or_else(|| LociError::InvalidArgument("Missing delta_y".to_string()))?;
        let x = parse_optional_i32(call, "x")?.unwrap_or(0);
        let y = parse_optional_i32(call, "y")?.unwrap_or(0);
        let duration_ms = parse_optional_u32(call, "duration_ms")?.unwrap_or(0);

        self.send_actions(
            &endpoint,
            &session_id,
            json!([{
                "type": "wheel",
                "id": "wheel",
                "actions": [{
                    "type": "scroll",
                    "x": x,
                    "y": y,
                    "deltaX": delta_x,
                    "deltaY": delta_y,
                    "duration": duration_ms
                }]
            }]),
        )?;
        self.release_actions(&endpoint, &session_id)?;
        Ok(json!({
            "endpoint": endpoint.normalized_url(),
            "session_id": session_id,
            "x": x,
            "y": y,
            "delta_x": delta_x,
            "delta_y": delta_y
        }))
    }
}

impl ToolPlugin for BrowserToolPlugin {
    fn name(&self) -> &str {
        "browser_tool_plugin"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn functions(&self) -> Vec<FunctionDefinition> {
        let mut open = FunctionDefinition::new(
            "browser_open_session",
            "Open a browser session via WebDriver",
        )
        .add_parameter(
            "endpoint",
            "string",
            "WebDriver endpoint (default http://127.0.0.1:9515)",
            false,
        )
        .add_parameter(
            "browser",
            "string",
            "Browser name: chrome/msedge/firefox",
            false,
        )
        .add_parameter("headless", "boolean", "Run in headless mode", false)
        .add_parameter("incognito", "boolean", "Open private/incognito session", false)
        .add_parameter("window_width", "number", "Window width", false)
        .add_parameter("window_height", "number", "Window height", false)
        .add_parameter("binary", "string", "Browser binary path", false)
        .add_parameter("args", "array", "Additional launch args", false);
        if let Some(param) = open.parameters.get_mut("browser") {
            param.enum_values = Some(vec![
                "chrome".to_string(),
                "msedge".to_string(),
                "firefox".to_string(),
            ]);
        }

        vec![
            open,
            FunctionDefinition::new(
                "browser_close_session",
                "Close an existing WebDriver browser session",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true),
            FunctionDefinition::new(
                "browser_navigate",
                "Navigate active tab to URL",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("url", "string", "Target URL", true),
            FunctionDefinition::new(
                "browser_get_title",
                "Get current page title",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true),
            FunctionDefinition::new(
                "browser_click",
                "Click first element matching CSS selector",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("selector", "string", "CSS selector", true),
            FunctionDefinition::new(
                "browser_type",
                "Type text into first element matching CSS selector",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("selector", "string", "CSS selector", true)
            .add_parameter("text", "string", "Text to input", true)
            .add_parameter("clear_first", "boolean", "Clear existing value first", false),
            FunctionDefinition::new(
                "browser_read_text",
                "Read text from first element matching CSS selector",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("selector", "string", "CSS selector", true),
            FunctionDefinition::new(
                "browser_screenshot",
                "Capture screenshot from current tab",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("output_path", "string", "Optional PNG output path", false)
            .add_parameter(
                "include_base64",
                "boolean",
                "Include base64 in tool response",
                false,
            ),
            FunctionDefinition::new(
                "browser_key_press",
                "Send one keyboard key press to current focused context",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("key", "string", "Key name or raw key value", true),
            FunctionDefinition::new(
                "browser_send_keys",
                "Send key sequence. Optional selector for direct element input",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("text", "string", "Text/key sequence", true)
            .add_parameter("selector", "string", "Optional CSS selector", false),
            FunctionDefinition::new(
                "browser_mouse_move",
                "Move mouse pointer inside browser viewport",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("x", "number", "Target x coordinate", true)
            .add_parameter("y", "number", "Target y coordinate", true)
            .add_parameter("duration_ms", "number", "Move duration in ms", false)
            .add_parameter("origin", "string", "viewport/pointer", false),
            FunctionDefinition::new(
                "browser_mouse_click_at",
                "Click at viewport coordinate using mouse actions",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("x", "number", "Target x coordinate", true)
            .add_parameter("y", "number", "Target y coordinate", true)
            .add_parameter("button", "number", "0 left / 1 middle / 2 right", false)
            .add_parameter("double_click", "boolean", "Double click at position", false)
            .add_parameter("duration_ms", "number", "Move duration in ms", false),
            FunctionDefinition::new(
                "browser_mouse_wheel",
                "Scroll mouse wheel in browser viewport",
            )
            .add_parameter("endpoint", "string", "WebDriver endpoint", false)
            .add_parameter("session_id", "string", "Session id", true)
            .add_parameter("delta_y", "number", "Scroll delta Y", true)
            .add_parameter("delta_x", "number", "Scroll delta X", false)
            .add_parameter("x", "number", "Viewport x reference", false)
            .add_parameter("y", "number", "Viewport y reference", false)
            .add_parameter("duration_ms", "number", "Scroll duration in ms", false),
        ]
    }

    fn execute(&self, call: &FunctionCall) -> Result<Value> {
        match call.name.as_str() {
            "browser_open_session" => self.open_session(call),
            "browser_close_session" => self.close_session(call),
            "browser_navigate" => self.navigate(call),
            "browser_get_title" => self.get_title(call),
            "browser_click" => self.click(call),
            "browser_type" => self.type_text(call),
            "browser_read_text" => self.read_text(call),
            "browser_screenshot" => self.screenshot(call),
            "browser_key_press" => self.key_press(call),
            "browser_send_keys" => self.send_keys(call),
            "browser_mouse_move" => self.mouse_move(call),
            "browser_mouse_click_at" => self.mouse_click_at(call),
            "browser_mouse_wheel" => self.mouse_wheel(call),
            _ => Err(LociError::InvalidArgument(format!(
                "Unsupported tool call: {}",
                call.name
            ))),
        }
    }
}

#[no_mangle]
pub extern "C" fn create_tool_plugin_v1() -> DynamicToolPluginOpaque {
    dynamic_tool_plugin_into_opaque(Box::new(BrowserToolPlugin))
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "C" fn create_tool_plugin() -> *mut dyn ToolPlugin {
    Box::into_raw(Box::new(BrowserToolPlugin))
}

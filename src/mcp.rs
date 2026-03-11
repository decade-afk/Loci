//! MCP (Model Context Protocol) client integration for tool calling.
//!
//! This module provides:
//! - A stdio MCP client implementation
//! - MCP tool schema -> `FunctionDefinition` mapping
//! - Registration bridge into `FunctionCallingManager`

use crate::error::{LociError, Result};
use crate::function_calling::{
    FunctionCall, FunctionCallingManager, FunctionDefinition, FunctionHandler,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;

/// Default MCP protocol version used by the stdio client.
pub const DEFAULT_MCP_PROTOCOL_VERSION: &str = "2025-03-26";

/// Configuration for connecting to one MCP server over stdio.
#[derive(Debug, Clone)]
pub struct McpStdioServerConfig {
    pub server_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub working_directory: Option<PathBuf>,
    pub env: HashMap<String, String>,
    pub protocol_version: String,
    pub client_name: String,
    pub tool_prefix: Option<String>,
}

impl McpStdioServerConfig {
    pub fn new(server_name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            server_name: server_name.into(),
            command: command.into(),
            args: Vec::new(),
            working_directory: None,
            env: HashMap::new(),
            protocol_version: DEFAULT_MCP_PROTOCOL_VERSION.to_string(),
            client_name: "loci".to_string(),
            tool_prefix: None,
        }
    }
}

/// Result of registering MCP tools into the function manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRegistrationReport {
    pub server_name: String,
    pub tool_prefix: String,
    pub registered_tools: Vec<String>,
}

/// Registration policy for mapping MCP tool names to local function names.
#[derive(Debug, Clone, Default)]
pub struct McpToolRegistrationOptions {
    pub tool_prefix: Option<String>,
}

/// Minimal MCP client capability used by Loci's function-calling engine.
pub trait McpClient: Send {
    fn server_name(&self) -> &str;
    fn list_tools(&mut self) -> Result<Vec<McpTool>>;
    fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value>;
}

/// MCP tool input schema subset used for function mapping.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpInputSchema {
    #[serde(default)]
    pub properties: HashMap<String, Value>,
    #[serde(default)]
    pub required: Vec<String>,
}

/// MCP tool descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema", default)]
    pub input_schema: McpInputSchema,
}

/// MCP stdio client implementation.
pub struct StdioMcpClient {
    server_name: String,
    protocol_version: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_request_id: u64,
}

impl StdioMcpClient {
    pub fn connect(config: McpStdioServerConfig) -> Result<Self> {
        if config.server_name.trim().is_empty() {
            return Err(LociError::ConfigError(
                "MCP server_name cannot be empty".to_string(),
            ));
        }
        if config.command.trim().is_empty() {
            return Err(LociError::ConfigError(
                "MCP command cannot be empty".to_string(),
            ));
        }

        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(cwd) = &config.working_directory {
            command.current_dir(cwd);
        }
        for (k, v) in &config.env {
            command.env(k, v);
        }

        let mut child = command.spawn().map_err(|e| {
            LociError::NetworkError(format!(
                "Failed to start MCP server '{}' ({}): {}",
                config.server_name, config.command, e
            ))
        })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            LociError::NetworkError("MCP server stdin is unavailable".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            LociError::NetworkError("MCP server stdout is unavailable".to_string())
        })?;

        let mut client = Self {
            server_name: config.server_name,
            protocol_version: config.protocol_version,
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_request_id: 1,
        };

        let initialize_result = client.send_request(
            "initialize",
            json!({
                "protocolVersion": client.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": config.client_name,
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }),
        )?;

        if let Some(server_version) = initialize_result
            .get("protocolVersion")
            .and_then(Value::as_str)
        {
            client.protocol_version = server_version.to_string();
        }

        client.send_notification("notifications/initialized", None)?;
        Ok(client)
    }

    fn send_notification(&mut self, method: &str, params: Option<Value>) -> Result<()> {
        let mut payload = json!({
            "jsonrpc": "2.0",
            "method": method,
        });
        if let Some(params) = params {
            payload["params"] = params;
        }
        self.write_message(&payload)
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(1);

        let mut payload = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if !params.is_null() {
            payload["params"] = params;
        }
        self.write_message(&payload)?;
        self.read_response_for_id(id)
    }

    fn write_message(&mut self, payload: &Value) -> Result<()> {
        let encoded =
            serde_json::to_string(payload).map_err(|e| LociError::SerializationError(e.to_string()))?;
        if encoded.contains('\n') {
            return Err(LociError::SerializationError(
                "MCP stdio payload contains embedded newline".to_string(),
            ));
        }
        self.stdin
            .write_all(encoded.as_bytes())
            .map_err(LociError::IoError)?;
        self.stdin.write_all(b"\n").map_err(LociError::IoError)?;
        self.stdin.flush().map_err(LociError::IoError)?;
        Ok(())
    }

    fn read_response_for_id(&mut self, expected_id: u64) -> Result<Value> {
        let expected = json!(expected_id);
        loop {
            let message = self.read_message()?;
            if let Some(result) = self.process_message(expected.clone(), message)? {
                return Ok(result);
            }
        }
    }

    fn read_message(&mut self) -> Result<Value> {
        loop {
            let mut line = String::new();
            let read_bytes = self.stdout.read_line(&mut line).map_err(LociError::IoError)?;
            if read_bytes == 0 {
                return Err(LociError::NetworkError(format!(
                    "MCP server '{}' closed stdout",
                    self.server_name
                )));
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            return serde_json::from_str::<Value>(trimmed)
                .map_err(|e| LociError::SerializationError(format!("Invalid MCP JSON: {e}")));
        }
    }

    fn process_message(&mut self, expected_id: Value, message: Value) -> Result<Option<Value>> {
        if let Some(batch) = message.as_array() {
            for item in batch {
                if let Some(result) = self.process_message(expected_id.clone(), item.clone())? {
                    return Ok(Some(result));
                }
            }
            return Ok(None);
        }

        let id = message.get("id").cloned();
        let method = message.get("method").and_then(Value::as_str);

        if let (Some(id), Some(method)) = (id.clone(), method) {
            self.reply_method_not_found(id, method)?;
            return Ok(None);
        }

        if let Some(id) = id {
            if id != expected_id {
                return Ok(None);
            }

            if let Some(error) = message.get("error") {
                let msg = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown MCP error");
                return Err(LociError::NetworkError(format!(
                    "MCP request failed on '{}': {}",
                    self.server_name, msg
                )));
            }

            if let Some(result) = message.get("result") {
                return Ok(Some(result.clone()));
            }

            return Err(LociError::SerializationError(
                "MCP response missing both result and error".to_string(),
            ));
        }

        Ok(None)
    }

    fn reply_method_not_found(&mut self, id: Value, method: &str) -> Result<()> {
        let response = json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("Method not supported by loci MCP client: {method}")
            }
        });
        self.write_message(&response)
    }
}

impl McpClient for StdioMcpClient {
    fn server_name(&self) -> &str {
        &self.server_name
    }

    fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let result = self.send_request("tools/list", params)?;
            let Some(raw_tools) = result.get("tools").and_then(Value::as_array) else {
                return Err(LociError::SerializationError(
                    "MCP tools/list result missing 'tools' array".to_string(),
                ));
            };

            for raw in raw_tools {
                let tool = serde_json::from_value::<McpTool>(raw.clone()).map_err(|e| {
                    LociError::SerializationError(format!("Invalid MCP tool schema: {e}"))
                })?;
                tools.push(tool);
            }

            cursor = result
                .get("nextCursor")
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            if cursor.is_none() {
                break;
            }
        }

        Ok(tools)
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let args = match arguments {
            Value::Object(map) => Value::Object(map),
            Value::Null => json!({}),
            other => {
                return Err(LociError::InvalidArgument(format!(
                    "MCP tool arguments must be JSON object, got: {}",
                    other
                )))
            }
        };

        let result = self.send_request(
            "tools/call",
            json!({
                "name": name,
                "arguments": args,
            }),
        )?;

        if result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let details = extract_mcp_text_content(&result)
                .unwrap_or_else(|| serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string()));
            return Err(LociError::Other(format!(
                "MCP tool '{}' on server '{}' returned error: {}",
                name, self.server_name, details
            )));
        }

        if let Some(structured) = result.get("structuredContent") {
            return Ok(structured.clone());
        }

        Ok(result)
    }
}

impl Drop for StdioMcpClient {
    fn drop(&mut self) {
        let _ = self.stdin.flush();
        let _ = self.child.try_wait();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Convert one MCP tool into Loci function definition.
pub fn mcp_tool_to_function_definition(
    tool: &McpTool,
    local_name: impl Into<String>,
) -> FunctionDefinition {
    let description = tool
        .description
        .clone()
        .unwrap_or_else(|| format!("MCP tool '{}'", tool.name));
    let mut definition = FunctionDefinition::new(local_name.into(), description);

    let mut param_names = tool
        .input_schema
        .properties
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    param_names.sort();

    for param_name in param_names {
        let schema = tool
            .input_schema
            .properties
            .get(&param_name)
            .cloned()
            .unwrap_or(Value::Null);
        let param_type = extract_schema_type(&schema);
        let description = extract_schema_description(&schema);
        let required = tool.input_schema.required.iter().any(|x| x == &param_name);

        definition = definition.add_parameter(
            &param_name,
            param_type,
            description.unwrap_or_else(|| "MCP tool parameter".to_string()),
            required,
        );

        if let Some(enum_values) = extract_schema_enum(&schema) {
            if let Some(param) = definition.parameters.get_mut(&param_name) {
                param.enum_values = Some(enum_values);
            }
        }
    }

    definition
}

/// Register all tools exposed by one MCP client to the function manager.
pub fn register_mcp_client_tools(
    manager: &mut FunctionCallingManager,
    client: Arc<Mutex<Box<dyn McpClient>>>,
    options: McpToolRegistrationOptions,
) -> Result<McpRegistrationReport> {
    let server_name = client.lock().server_name().to_string();
    let tool_prefix = options
        .tool_prefix
        .unwrap_or_else(|| format!("mcp.{}.", sanitize_identifier(&server_name)));

    let remote_tools = client.lock().list_tools()?;
    let mut registered_tools = Vec::with_capacity(remote_tools.len());

    for tool in remote_tools {
        let local_name = format!("{}{}", tool_prefix, tool.name);
        let remote_name = tool.name.clone();
        let definition = mcp_tool_to_function_definition(&tool, local_name.clone());
        let handler = McpToolHandler {
            client: Arc::clone(&client),
            remote_tool_name: remote_name,
        };
        manager.register_function_with_handler(definition, handler)?;
        registered_tools.push(local_name);
    }

    Ok(McpRegistrationReport {
        server_name,
        tool_prefix,
        registered_tools,
    })
}

/// Connect to MCP server over stdio and register all remote tools.
pub fn connect_and_register_stdio_server(
    manager: &mut FunctionCallingManager,
    config: McpStdioServerConfig,
) -> Result<McpRegistrationReport> {
    let options = McpToolRegistrationOptions {
        tool_prefix: config.tool_prefix.clone(),
    };
    let client: Box<dyn McpClient> = Box::new(StdioMcpClient::connect(config)?);
    register_mcp_client_tools(manager, Arc::new(Mutex::new(client)), options)
}

struct McpToolHandler {
    client: Arc<Mutex<Box<dyn McpClient>>>,
    remote_tool_name: String,
}

impl FunctionHandler for McpToolHandler {
    fn execute(&self, call: &FunctionCall) -> Result<Value> {
        let mut args = serde_json::Map::new();
        for (k, v) in &call.arguments {
            args.insert(k.clone(), v.clone());
        }
        self.client
            .lock()
            .call_tool(&self.remote_tool_name, Value::Object(args))
    }
}

fn sanitize_identifier(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "server".to_string()
    } else {
        out
    }
}

fn extract_schema_type(schema: &Value) -> String {
    if let Some(t) = schema.get("type") {
        if let Some(one) = t.as_str() {
            return one.to_string();
        }
        if let Some(arr) = t.as_array() {
            for item in arr {
                if let Some(name) = item.as_str() {
                    if name != "null" {
                        return name.to_string();
                    }
                }
            }
        }
    }
    "string".to_string()
}

fn extract_schema_description(schema: &Value) -> Option<String> {
    schema
        .get("description")
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .or_else(|| {
            schema
                .get("title")
                .and_then(Value::as_str)
                .map(|s| s.to_string())
        })
}

fn extract_schema_enum(schema: &Value) -> Option<Vec<String>> {
    let values = schema.get("enum")?.as_array()?;
    let mut out = Vec::new();
    for value in values {
        if let Some(s) = value.as_str() {
            out.push(s.to_string());
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn extract_mcp_text_content(result: &Value) -> Option<String> {
    let blocks = result.get("content")?.as_array()?;
    let mut texts = Vec::new();
    for block in blocks {
        if block.get("type").and_then(Value::as_str) == Some("text") {
            if let Some(text) = block.get("text").and_then(Value::as_str) {
                texts.push(text.to_string());
            }
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(texts.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_calling::FunctionCall;
    use serde_json::json;

    struct MockMcpClient {
        server_name: String,
        tools: Vec<McpTool>,
    }

    impl MockMcpClient {
        fn new() -> Self {
            Self {
                server_name: "mock".to_string(),
                tools: vec![McpTool {
                    name: "sum".to_string(),
                    description: Some("Add two numbers".to_string()),
                    input_schema: McpInputSchema {
                        properties: HashMap::from([
                            ("a".to_string(), json!({"type":"number"})),
                            ("b".to_string(), json!({"type":"number"})),
                        ]),
                        required: vec!["a".to_string(), "b".to_string()],
                    },
                }],
            }
        }
    }

    impl McpClient for MockMcpClient {
        fn server_name(&self) -> &str {
            &self.server_name
        }

        fn list_tools(&mut self) -> Result<Vec<McpTool>> {
            Ok(self.tools.clone())
        }

        fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
            if name != "sum" {
                return Err(LociError::InvalidArgument(format!("unknown tool: {name}")));
            }
            let Some(args) = arguments.as_object() else {
                return Err(LociError::InvalidArgument(
                    "arguments should be object".to_string(),
                ));
            };
            let a = args
                .get("a")
                .and_then(Value::as_f64)
                .ok_or_else(|| LociError::InvalidArgument("missing a".to_string()))?;
            let b = args
                .get("b")
                .and_then(Value::as_f64)
                .ok_or_else(|| LociError::InvalidArgument("missing b".to_string()))?;
            Ok(json!({ "sum": a + b }))
        }
    }

    #[test]
    fn map_mcp_tool_to_function_definition_preserves_enum_and_required() {
        let tool = McpTool {
            name: "weather".to_string(),
            description: Some("Get weather".to_string()),
            input_schema: McpInputSchema {
                properties: HashMap::from([
                    ("location".to_string(), json!({"type":"string"})),
                    (
                        "unit".to_string(),
                        json!({
                            "type":"string",
                            "description":"temperature unit",
                            "enum":["celsius","fahrenheit"]
                        }),
                    ),
                ]),
                required: vec!["location".to_string()],
            },
        };

        let def = mcp_tool_to_function_definition(&tool, "mcp.mock.weather");
        assert_eq!(def.name, "mcp.mock.weather");
        assert!(def.required.contains(&"location".to_string()));
        assert_eq!(
            def.parameters
                .get("unit")
                .and_then(|p| p.enum_values.clone())
                .unwrap_or_default(),
            vec!["celsius".to_string(), "fahrenheit".to_string()]
        );
    }

    #[test]
    fn register_and_execute_mcp_tool_handler() {
        let mut manager = FunctionCallingManager::new();
        let client: Arc<Mutex<Box<dyn McpClient>>> =
            Arc::new(Mutex::new(Box::new(MockMcpClient::new())));
        let report = register_mcp_client_tools(
            &mut manager,
            client,
            McpToolRegistrationOptions::default(),
        )
        .unwrap();

        assert_eq!(report.server_name, "mock");
        assert_eq!(report.registered_tools.len(), 1);
        let local_name = report.registered_tools[0].clone();

        let call = FunctionCall::new(local_name)
            .with_argument("a", json!(2))
            .with_argument("b", json!(5));
        let out = manager.execute_function_call(&call).unwrap();
        assert_eq!(out["sum"].as_f64(), Some(7.0));
    }
}

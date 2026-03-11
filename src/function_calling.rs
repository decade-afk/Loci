//! Function calling support for LLMs

use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Function parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParameter {
    #[serde(rename = "type")]
    pub param_type: String,
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// Function definition for LLM function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, FunctionParameter>,
    pub required: Vec<String>,
}

impl FunctionDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: HashMap::new(),
            required: Vec::new(),
        }
    }

    pub fn add_parameter(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let param_name = name.into();
        self.parameters.insert(
            param_name.clone(),
            FunctionParameter {
                param_type: param_type.into(),
                description: Some(description.into()),
                enum_values: None,
            },
        );
        if required {
            self.required.push(param_name);
        }
        self
    }

    pub fn to_json_schema(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        })
    }
}

/// Function call result from LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

impl FunctionCall {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: HashMap::new(),
        }
    }

    pub fn with_argument(mut self, key: impl Into<String>, value: Value) -> Self {
        self.arguments.insert(key.into(), value);
        self
    }

    pub fn get_argument(&self, key: &str) -> Option<&Value> {
        self.arguments.get(key)
    }

    pub fn get_string(&self, key: &str) -> Option<String> {
        self.arguments.get(key)?.as_str().map(|s| s.to_string())
    }

    pub fn get_number(&self, key: &str) -> Option<f64> {
        self.arguments.get(key)?.as_f64()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.arguments.get(key)?.as_bool()
    }
}

/// Executable handler for function calls.
pub trait FunctionHandler: Send + Sync {
    fn execute(&self, call: &FunctionCall) -> Result<Value>;
}

struct FnFunctionHandler<F>
where
    F: Fn(&FunctionCall) -> Result<Value> + Send + Sync + 'static,
{
    executor: F,
}

impl<F> FunctionHandler for FnFunctionHandler<F>
where
    F: Fn(&FunctionCall) -> Result<Value> + Send + Sync + 'static,
{
    fn execute(&self, call: &FunctionCall) -> Result<Value> {
        (self.executor)(call)
    }
}

fn parse_required_number(call: &FunctionCall, key: &str) -> Result<f64> {
    call.get_number(key).ok_or_else(|| {
        LociError::InvalidArgument(format!("Missing or invalid numeric argument: {key}"))
    })
}

fn parse_required_string(call: &FunctionCall, key: &str) -> Result<String> {
    call.get_string(key).ok_or_else(|| {
        LociError::InvalidArgument(format!("Missing or invalid string argument: {key}"))
    })
}

fn builtin_echo_definition() -> FunctionDefinition {
    FunctionDefinition::new("echo", "Echo back the input text").add_parameter(
        "text",
        "string",
        "Text to return unchanged",
        true,
    )
}

fn builtin_calculator_definition() -> FunctionDefinition {
    let mut definition =
        FunctionDefinition::new("calculator", "Perform arithmetic operation on two numbers")
            .add_parameter("a", "number", "First operand", true)
            .add_parameter("b", "number", "Second operand", true)
            .add_parameter(
                "operation",
                "string",
                "One of add/sub/mul/div/pow/mod",
                true,
            );

    if let Some(param) = definition.parameters.get_mut("operation") {
        param.enum_values = Some(vec![
            "add".to_string(),
            "sub".to_string(),
            "mul".to_string(),
            "div".to_string(),
            "pow".to_string(),
            "mod".to_string(),
        ]);
    }
    definition
}

fn builtin_timestamp_definition() -> FunctionDefinition {
    FunctionDefinition::new("timestamp_now", "Get current unix timestamp in seconds")
}

fn builtin_text_stats_definition() -> FunctionDefinition {
    FunctionDefinition::new("text_stats", "Compute simple text statistics").add_parameter(
        "text",
        "string",
        "Input text",
        true,
    )
}

fn builtin_read_text_file_definition() -> FunctionDefinition {
    FunctionDefinition::new("read_text_file", "Read UTF-8 text file content")
        .add_parameter("path", "string", "File path to read", true)
        .add_parameter(
            "max_bytes",
            "number",
            "Optional cap for bytes read (default 65536)",
            false,
        )
}

fn builtin_list_directory_definition() -> FunctionDefinition {
    FunctionDefinition::new("list_directory", "List files/directories under a path").add_parameter(
        "path",
        "string",
        "Directory path",
        true,
    )
}

/// Function calling manager
pub struct FunctionCallingManager {
    functions: HashMap<String, FunctionDefinition>,
    handlers: HashMap<String, Arc<dyn FunctionHandler>>,
}

impl FunctionCallingManager {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            handlers: HashMap::new(),
        }
    }

    pub fn with_builtin_tools() -> Self {
        let mut manager = Self::new();
        manager.register_builtin_tools().ok();
        manager
    }

    pub fn register_function(&mut self, function: FunctionDefinition) {
        let name = function.name.clone();
        self.functions.insert(name.clone(), function);
        self.handlers.remove(&name);
    }

    pub fn register_handler<H>(&mut self, name: impl Into<String>, handler: H) -> Result<()>
    where
        H: FunctionHandler + 'static,
    {
        self.register_handler_arc(name, Arc::new(handler))
    }

    pub fn register_handler_arc(
        &mut self,
        name: impl Into<String>,
        handler: Arc<dyn FunctionHandler>,
    ) -> Result<()> {
        let name = name.into();
        if !self.functions.contains_key(&name) {
            return Err(LociError::InvalidArgument(format!(
                "Cannot register handler for unknown function: {name}"
            )));
        }
        self.handlers.insert(name, handler);
        Ok(())
    }

    pub fn register_function_with_handler<H>(
        &mut self,
        function: FunctionDefinition,
        handler: H,
    ) -> Result<()>
    where
        H: FunctionHandler + 'static,
    {
        let name = function.name.clone();
        if self.functions.contains_key(&name) {
            return Err(LociError::InvalidArgument(format!(
                "Function already registered: {name}"
            )));
        }
        self.functions.insert(name.clone(), function);
        self.handlers.insert(name, Arc::new(handler));
        Ok(())
    }

    pub fn register_closure_tool<F>(
        &mut self,
        function: FunctionDefinition,
        executor: F,
    ) -> Result<()>
    where
        F: Fn(&FunctionCall) -> Result<Value> + Send + Sync + 'static,
    {
        self.register_function_with_handler(function, FnFunctionHandler { executor })
    }

    pub fn register_builtin_tools(&mut self) -> Result<()> {
        self.register_closure_tool(builtin_echo_definition(), |call| {
            let text = parse_required_string(call, "text")?;
            Ok(json!({ "text": text }))
        })?;

        self.register_closure_tool(builtin_calculator_definition(), |call| {
            let a = parse_required_number(call, "a")?;
            let b = parse_required_number(call, "b")?;
            let operation = parse_required_string(call, "operation")?;

            let result = match operation.as_str() {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "div" => {
                    if b == 0.0 {
                        return Err(LociError::InvalidArgument("Division by zero".to_string()));
                    }
                    a / b
                }
                "pow" => a.powf(b),
                "mod" => {
                    if b == 0.0 {
                        return Err(LociError::InvalidArgument("Modulo by zero".to_string()));
                    }
                    a % b
                }
                _ => {
                    return Err(LociError::InvalidArgument(format!(
                        "Unsupported operation: {operation}"
                    )))
                }
            };

            Ok(json!({
                "operation": operation,
                "a": a,
                "b": b,
                "result": result
            }))
        })?;

        self.register_closure_tool(builtin_timestamp_definition(), |_call| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| LociError::Other(format!("system time error: {e}")))?
                .as_secs();
            Ok(json!({ "unix_seconds": now }))
        })?;

        self.register_closure_tool(builtin_text_stats_definition(), |call| {
            let text = parse_required_string(call, "text")?;
            let chars = text.chars().count();
            let words = text.split_whitespace().count();
            let lines = text.lines().count();
            Ok(json!({
                "chars": chars,
                "words": words,
                "lines": lines
            }))
        })?;

        self.register_closure_tool(builtin_read_text_file_definition(), |call| {
            let path = parse_required_string(call, "path")?;
            let max_bytes = call
                .get_number("max_bytes")
                .map(|v| v.max(1.0).min(4_194_304.0) as usize)
                .unwrap_or(65_536);

            let bytes = std::fs::read(&path).map_err(|e| LociError::IoError(e))?;
            let bytes = if bytes.len() > max_bytes {
                bytes[..max_bytes].to_vec()
            } else {
                bytes
            };

            let text = String::from_utf8(bytes)
                .map_err(|e| LociError::InvalidArgument(format!("File is not valid UTF-8: {e}")))?;
            Ok(json!({
                "path": path,
                "content": text
            }))
        })?;

        self.register_closure_tool(builtin_list_directory_definition(), |call| {
            let path = parse_required_string(call, "path")?;
            let mut entries = Vec::new();
            let iter = std::fs::read_dir(&path).map_err(LociError::IoError)?;
            for entry in iter {
                let entry = entry.map_err(LociError::IoError)?;
                let metadata = entry.metadata().map_err(LociError::IoError)?;
                let name = entry.file_name().to_string_lossy().to_string();
                entries.push(json!({
                    "name": name,
                    "is_dir": metadata.is_dir(),
                    "is_file": metadata.is_file(),
                    "len": metadata.len()
                }));
            }
            Ok(json!({
                "path": path,
                "entries": entries
            }))
        })?;

        Ok(())
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionDefinition> {
        self.functions.get(name)
    }

    pub fn unregister_function(&mut self, name: &str) -> Option<FunctionDefinition> {
        self.handlers.remove(name);
        self.functions.remove(name)
    }

    pub fn list_functions(&self) -> Vec<&FunctionDefinition> {
        let mut list: Vec<&FunctionDefinition> = self.functions.values().collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }

    pub fn format_functions_for_prompt(&self) -> String {
        let mut prompt = String::from("Available functions:\n\n");

        for func in self.list_functions() {
            prompt.push_str(&format!("Function: {}\n", func.name));
            prompt.push_str(&format!("Description: {}\n", func.description));
            prompt.push_str("Parameters:\n");

            for (param_name, param) in &func.parameters {
                let required = if func.required.contains(param_name) {
                    " (required)"
                } else {
                    ""
                };
                prompt.push_str(&format!(
                    "  - {}: {}{}\n",
                    param_name, param.param_type, required
                ));
                if let Some(desc) = &param.description {
                    prompt.push_str(&format!("    {}\n", desc));
                }
            }
            prompt.push_str("\n");
        }

        prompt.push_str("To call a function, respond with JSON in this format:\n");
        prompt.push_str("{\n");
        prompt.push_str("  \"function\": \"function_name\",\n");
        prompt.push_str("  \"arguments\": {\n");
        prompt.push_str("    \"param1\": \"value1\",\n");
        prompt.push_str("    \"param2\": \"value2\"\n");
        prompt.push_str("  }\n");
        prompt.push_str("}\n");

        prompt
    }

    pub fn parse_function_call(&self, response: &str) -> Result<Option<FunctionCall>> {
        let trimmed = response.trim();

        if !trimmed.starts_with('{') {
            if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
                if start < end {
                    return self.parse_function_call(&trimmed[start..=end]);
                }
            }
            return Ok(None);
        }

        let json: Value = serde_json::from_str(trimmed)
            .map_err(|e| LociError::SerializationError(e.to_string()))?;

        let function_name = json["function"]
            .as_str()
            .ok_or_else(|| LociError::InvalidArgument("Missing function name".to_string()))?;

        if !self.functions.contains_key(function_name) {
            return Err(LociError::InvalidArgument(format!(
                "Unknown function: {}",
                function_name
            )));
        }

        let arguments = json["arguments"]
            .as_object()
            .ok_or_else(|| LociError::InvalidArgument("Missing arguments".to_string()))?;

        let mut call = FunctionCall::new(function_name);
        for (key, value) in arguments {
            call = call.with_argument(key, value.clone());
        }

        Ok(Some(call))
    }

    pub fn validate_function_call(&self, call: &FunctionCall) -> Result<()> {
        let func = self.functions.get(&call.name).ok_or_else(|| {
            LociError::InvalidArgument(format!("Unknown function: {}", call.name))
        })?;

        for required_param in &func.required {
            if !call.arguments.contains_key(required_param) {
                return Err(LociError::InvalidArgument(format!(
                    "Missing required parameter: {}",
                    required_param
                )));
            }
        }

        Ok(())
    }

    pub fn execute_function_call(&self, call: &FunctionCall) -> Result<Value> {
        self.validate_function_call(call)?;

        let handler = self.handlers.get(&call.name).ok_or_else(|| {
            LociError::UnsupportedOperation(format!(
                "No execution handler registered for function: {}",
                call.name
            ))
        })?;

        handler.execute(call)
    }
}

impl Default for FunctionCallingManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_definition() {
        let func = FunctionDefinition::new("get_weather", "Get current weather")
            .add_parameter("location", "string", "City name", true)
            .add_parameter("unit", "string", "Temperature unit", false);

        assert_eq!(func.name, "get_weather");
        assert_eq!(func.parameters.len(), 2);
        assert_eq!(func.required.len(), 1);
    }

    #[test]
    fn test_function_call() {
        let call = FunctionCall::new("get_weather")
            .with_argument("location", Value::String("London".to_string()))
            .with_argument("unit", Value::String("celsius".to_string()));

        assert_eq!(call.name, "get_weather");
        assert_eq!(call.get_string("location"), Some("London".to_string()));
    }

    #[test]
    fn test_function_calling_manager() {
        let mut manager = FunctionCallingManager::new();

        let func = FunctionDefinition::new("test_func", "Test function").add_parameter(
            "param1",
            "string",
            "Test param",
            true,
        );

        manager.register_function(func);

        assert!(manager.get_function("test_func").is_some());
        assert_eq!(manager.list_functions().len(), 1);
    }

    #[test]
    fn test_register_function_with_handler_and_execute() {
        let mut manager = FunctionCallingManager::new();
        let func = FunctionDefinition::new("sum2", "Add two numbers")
            .add_parameter("a", "number", "A", true)
            .add_parameter("b", "number", "B", true);
        manager
            .register_closure_tool(func, |call| {
                let a = call
                    .get_number("a")
                    .ok_or_else(|| LociError::InvalidArgument("missing a".to_string()))?;
                let b = call
                    .get_number("b")
                    .ok_or_else(|| LociError::InvalidArgument("missing b".to_string()))?;
                Ok(json!({ "result": a + b }))
            })
            .unwrap();

        let call = FunctionCall::new("sum2")
            .with_argument("a", json!(2))
            .with_argument("b", json!(5));
        let out = manager.execute_function_call(&call).unwrap();
        assert_eq!(out["result"].as_f64(), Some(7.0));
    }

    #[test]
    fn test_parse_function_call() {
        let mut manager = FunctionCallingManager::new();
        let func = FunctionDefinition::new("test", "Test")
            .add_parameter("arg", "string", "Argument", true);
        manager.register_function(func);

        let json = r#"{"function": "test", "arguments": {"arg": "value"}}"#;
        let call = manager.parse_function_call(json).unwrap();

        assert!(call.is_some());
        let call = call.unwrap();
        assert_eq!(call.name, "test");
        assert_eq!(call.get_string("arg"), Some("value".to_string()));
    }

    #[test]
    fn test_parse_function_call_inside_text() {
        let mut manager = FunctionCallingManager::new();
        let func = FunctionDefinition::new("test", "Test")
            .add_parameter("arg", "string", "Argument", true);
        manager.register_function(func);

        let text = "I will call now: {\"function\": \"test\", \"arguments\": {\"arg\": \"value\"}}";
        let call = manager.parse_function_call(text).unwrap().unwrap();
        assert_eq!(call.name, "test");
    }

    #[test]
    fn test_builtin_calculator_execution() {
        let manager = FunctionCallingManager::with_builtin_tools();
        let call = FunctionCall::new("calculator")
            .with_argument("a", json!(9))
            .with_argument("b", json!(3))
            .with_argument("operation", json!("div"));
        let out = manager.execute_function_call(&call).unwrap();
        assert_eq!(out["result"].as_f64(), Some(3.0));
    }
}

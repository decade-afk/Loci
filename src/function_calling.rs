//! Function calling support for LLMs

use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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

/// Function calling manager
pub struct FunctionCallingManager {
    functions: HashMap<String, FunctionDefinition>,
}

impl FunctionCallingManager {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    pub fn register_function(&mut self, function: FunctionDefinition) {
        self.functions.insert(function.name.clone(), function);
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionDefinition> {
        self.functions.get(name)
    }

    pub fn list_functions(&self) -> Vec<&FunctionDefinition> {
        self.functions.values().collect()
    }

    pub fn format_functions_for_prompt(&self) -> String {
        let mut prompt = String::from("Available functions:\n\n");
        
        for func in self.functions.values() {
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
                    param_name,
                    param.param_type,
                    required
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
        let func = self
            .functions
            .get(&call.name)
            .ok_or_else(|| LociError::InvalidArgument(format!("Unknown function: {}", call.name)))?;

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
        
        let func = FunctionDefinition::new("test_func", "Test function")
            .add_parameter("param1", "string", "Test param", true);
        
        manager.register_function(func);
        
        assert!(manager.get_function("test_func").is_some());
        assert_eq!(manager.list_functions().len(), 1);
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
}

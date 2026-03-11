//! OpenClaw-style agent adapter plugin for Loci.
//!
//! Goal:
//! - Inject a stable tool-calling contract into prompts.
//! - Normalize model output into strict JSON envelopes for host orchestration.
//! - Keep host-side executor independent (web/tool/sandbox).
//!
//! Build:
//!   cargo build --release --manifest-path examples/openclaw_adapter_plugin/Cargo.toml
//!
//! Load:
//!   loci.exe plugin load examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll
//!
//! Optional environment variables:
//! - LOCI_OPENCLAW_TOOLS_PATH: path to JSON tool schema file
//! - LOCI_OPENCLAW_STRICT_JSON: 1/true/yes/on to force JSON envelope (default: true)
//! - LOCI_OPENCLAW_SYSTEM_PROMPT: extra system prefix text

use loci::error::Result;
use loci::plugin::{dynamic_plugin_into_opaque, DynamicPluginOpaque, Plugin};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

const TOOLS_PATH_ENV: &str = "LOCI_OPENCLAW_TOOLS_PATH";
const STRICT_JSON_ENV: &str = "LOCI_OPENCLAW_STRICT_JSON";
const SYSTEM_PROMPT_ENV: &str = "LOCI_OPENCLAW_SYSTEM_PROMPT";

#[derive(Debug, Clone, Deserialize)]
struct ToolSpec {
    name: String,
    description: String,
    #[serde(default)]
    parameters: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolSpecRoot {
    tools: Vec<ToolSpec>,
}

pub struct OpenClawAdapterPlugin {
    name: String,
    tools: Vec<ToolSpec>,
    strict_json: bool,
    system_prefix: String,
}

impl OpenClawAdapterPlugin {
    pub fn new() -> Self {
        let tools = Self::load_tools_from_env();
        let strict_json = env_bool(STRICT_JSON_ENV, true);
        let system_prefix = std::env::var(SYSTEM_PROMPT_ENV).unwrap_or_else(|_| {
            "You are an agent runtime inside Loci. Use tools when necessary and keep responses deterministic.".to_string()
        });

        Self {
            name: "openclaw_adapter".to_string(),
            tools,
            strict_json,
            system_prefix,
        }
    }

    fn load_tools_from_env() -> Vec<ToolSpec> {
        let path = match std::env::var(TOOLS_PATH_ENV) {
            Ok(v) => PathBuf::from(v),
            Err(_) => return Vec::new(),
        };

        let body = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        if let Ok(list) = serde_json::from_str::<Vec<ToolSpec>>(&body) {
            return list;
        }

        if let Ok(root) = serde_json::from_str::<ToolSpecRoot>(&body) {
            return root.tools;
        }

        Vec::new()
    }

    fn tools_value(&self) -> Value {
        let mut items = Vec::new();
        for t in &self.tools {
            items.push(json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            }));
        }
        Value::Array(items)
    }

    fn build_contract(&self) -> String {
        let tools_json = serde_json::to_string_pretty(&self.tools_value())
            .unwrap_or_else(|_| "[]".to_string());

        format!(
            "__OPENCLAW_CONTRACT_V1__\n\
             {system}\n\
             \n\
             Available tools:\n\
             {tools}\n\
             \n\
             Output MUST be JSON only (no markdown/code fences).\n\
             Use exactly one of:\n\
             1) Tool call envelope:\n\
             {{\"type\":\"tool_call\",\"name\":\"<tool>\",\"arguments\":{{...}},\"id\":\"<opaque-id>\"}}\n\
             2) Final answer envelope:\n\
             {{\"type\":\"final\",\"content\":\"<final answer text>\"}}\n",
            system = self.system_prefix,
            tools = tools_json
        )
    }

    fn strip_markdown_fence(raw: &str) -> String {
        let s = raw.trim();
        if !s.starts_with("```") {
            return s.to_string();
        }

        let mut lines = s.lines();
        let _first = lines.next();
        let mut buf = String::new();
        for line in lines {
            if line.trim_start().starts_with("```") {
                break;
            }
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(line);
        }
        buf.trim().to_string()
    }

    fn normalize_response_json(&self, response: &str) -> String {
        let cleaned = Self::strip_markdown_fence(response);

        if let Ok(v) = serde_json::from_str::<Value>(&cleaned) {
            if let Some(obj) = v.as_object() {
                if let Some(ty) = obj.get("type").and_then(|x| x.as_str()) {
                    if ty == "tool_call" || ty == "final" {
                        return serde_json::to_string(&v).unwrap_or(cleaned);
                    }
                }

                // Compatibility bridge: OpenAI-like function_call payload.
                let function_name = obj.get("function").and_then(|x| x.as_str());
                let arguments = obj.get("arguments");
                if let Some(name) = function_name {
                    let bridged = json!({
                        "type": "tool_call",
                        "name": name,
                        "arguments": arguments.cloned().unwrap_or_else(|| json!({})),
                        "id": "compat-function-call"
                    });
                    return serde_json::to_string(&bridged).unwrap_or_else(|_| cleaned.clone());
                }
            }
        }

        let wrapped = json!({
            "type": "final",
            "content": cleaned
        });
        serde_json::to_string(&wrapped).unwrap_or(cleaned)
    }
}

impl Plugin for OpenClawAdapterPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        println!(
            "[OpenClawAdapterPlugin] init: strict_json={}, tools={}",
            self.strict_json,
            self.tools.len()
        );
        Ok(())
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if prompt.contains("__OPENCLAW_CONTRACT_V1__") {
            return Ok(prompt.to_string());
        }

        let wrapped = format!(
            "{}\nUser request:\n{}\n",
            self.build_contract(),
            prompt
        );
        Ok(wrapped)
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        if self.strict_json {
            Ok(self.normalize_response_json(response))
        } else {
            Ok(response.to_string())
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("[OpenClawAdapterPlugin] cleanup");
        Ok(())
    }
}

fn env_bool(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

#[no_mangle]
pub extern "C" fn create_plugin_v1() -> DynamicPluginOpaque {
    dynamic_plugin_into_opaque(Box::new(OpenClawAdapterPlugin::new()))
}

#[no_mangle]
pub extern "C" fn create_plugin() -> DynamicPluginOpaque {
    create_plugin_v1()
}

#[cfg(test)]
mod tests {
    use super::OpenClawAdapterPlugin;

    #[test]
    fn normalize_plain_text_into_final_envelope() {
        let plugin = OpenClawAdapterPlugin::new();
        let out = plugin.normalize_response_json("hello world");
        assert!(out.contains("\"type\":\"final\""));
        assert!(out.contains("\"hello world\""));
    }

    #[test]
    fn normalize_openai_function_payload_into_tool_call() {
        let plugin = OpenClawAdapterPlugin::new();
        let out = plugin.normalize_response_json(
            r#"{"function":"web_search","arguments":{"q":"loci"}} "#,
        );
        assert!(out.contains("\"type\":\"tool_call\""));
        assert!(out.contains("\"name\":\"web_search\""));
    }

    #[test]
    fn strip_markdown_code_fence() {
        let fenced = "```json\n{\"type\":\"final\",\"content\":\"ok\"}\n```";
        let out = OpenClawAdapterPlugin::strip_markdown_fence(fenced);
        assert_eq!(out, "{\"type\":\"final\",\"content\":\"ok\"}");
    }
}

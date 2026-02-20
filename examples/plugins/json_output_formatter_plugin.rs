// JSON Output Formatter Plugin for Loci
// 
// This plugin formats model output as JSON with metadata.
// Useful for API integration and structured responses.

use loci::plugin::Plugin;
use loci::error::Result;
use serde_json::{json, Value};
use chrono::Utc;

pub struct JsonFormatterPlugin {
    name: String,
    include_metadata: bool,
    include_timing: bool,
    include_prompt: bool,
    start_time: Option<i64>,
}

impl JsonFormatterPlugin {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            include_metadata: true,
            include_timing: true,
            include_prompt: false,
            start_time: None,
        }
    }

    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    pub fn with_timing(mut self, include: bool) -> Self {
        self.include_timing = include;
        self
    }

    pub fn with_prompt(mut self, include: bool) -> Self {
        self.include_prompt = include;
        self
    }

    fn record_start_time(&mut self) {
        self.start_time = Some(Utc::now().timestamp_millis());
    }

    fn get_elapsed_ms(&self) -> i64 {
        match self.start_time {
            Some(start) => Utc::now().timestamp_millis() - start,
            None => 0,
        }
    }
}

impl Plugin for JsonFormatterPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        // Store prompt if needed
        Ok(prompt.to_string())
    }

    fn post_generate(&mut self, response: &str) -> Result<String> {
        let elapsed = self.get_elapsed_ms();
        
        let mut json_obj = json!({
            "content": response,
        });

        if self.include_metadata {
            json_obj["metadata"] = json!({
                "timestamp": Utc::now().to_rfc3339(),
                "plugin": self.name,
                "plugin_version": self.version(),
            });
        }

        if self.include_timing {
            json_obj["timing"] = json!({
                "elapsed_ms": elapsed,
            });
        }

        Ok(json_obj.to_string())
    }

    fn init(&mut self) -> Result<()> {
        self.start_time = Some(Utc::now().timestamp_millis());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_json_output() {
        let mut plugin = JsonFormatterPlugin::new("test");
        plugin.init().unwrap();
        
        let input = "Hello, world!";
        let output = plugin.post_generate(input).unwrap();
        
        let json: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(json["content"], "Hello, world!");
    }

    #[test]
    fn test_json_with_metadata() {
        let mut plugin = JsonFormatterPlugin::new("test")
            .with_metadata(true);
        plugin.init().unwrap();
        
        let input = "Test response";
        let output = plugin.post_generate(input).unwrap();
        
        let json: Value = serde_json::from_str(&output).unwrap();
        assert!(json["metadata"].is_object());
        assert!(json["metadata"]["timestamp"].is_string());
    }

    #[test]
    fn test_json_with_timing() {
        let mut plugin = JsonFormatterPlugin::new("test")
            .with_timing(true);
        plugin.init().unwrap();
        
        let input = "Test response";
        let output = plugin.post_generate(input).unwrap();
        
        let json: Value = serde_json::from_str(&output).unwrap();
        assert!(json["timing"].is_object());
        assert!(json["timing"]["elapsed_ms"].is_number());
    }
}
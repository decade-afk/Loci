//! Config Module
//!
//! This module provides core functionality for the Loci project.
//!


use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;




#[derive(Debug, Clone, Serialize, Deserialize)]
    /// LociConfig structure
pub struct LociConfig {
    
    #[serde(default)]
    pub engine: EngineSettings,

    
    #[serde(default)]
    pub backend: BackendSettings,

    
    #[serde(default)]
    pub memory: MemorySettings,

    
    #[serde(default)]
    pub plugins: PluginSettings,

    
    #[serde(default)]
    pub logging: LoggingSettings,

    
    #[serde(default)]
    pub server: ServerSettings,
}

// Implementation for Default
impl Default for LociConfig {
    fn default() -> Self {
        Self {
            engine: EngineSettings::default(),
            backend: BackendSettings::default(),
            memory: MemorySettings::default(),
            plugins: PluginSettings::default(),
            logging: LoggingSettings::default(),
            server: ServerSettings::default(),
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// EngineSettings structure
pub struct EngineSettings {
    
    pub model_path: Option<String>,

    
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    
    #[serde(default)]
    pub n_threads: usize,

    
    #[serde(default = "default_context_length")]
    pub context_length: usize,

    
    #[serde(default = "default_gpu_layers")]
    pub n_gpu_layers: i32,

    
    #[serde(default = "default_true")]
    pub use_mmap: bool,

    
    #[serde(default)]
    pub use_mlock: bool,
}

// Implementation for Default
impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            model_path: None,
            batch_size: default_batch_size(),
            n_threads: num_cpus::get(),
            context_length: default_context_length(),
            n_gpu_layers: default_gpu_layers(),
            use_mmap: true,
            use_mlock: false,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// BackendSettings structure
pub struct BackendSettings {
    
    #[serde(default = "default_backend")]
    pub backend_type: String,

    
    #[serde(default)]
    pub device_id: usize,

    
    #[serde(default = "default_true")]
    pub enable_fusion: bool,
}

// Implementation for Default
impl Default for BackendSettings {
    fn default() -> Self {
        Self {
            backend_type: default_backend(),
            device_id: 0,
            enable_fusion: true,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// MemorySettings structure
pub struct MemorySettings {
    
    #[serde(default = "default_vram")]
    pub vram_mb: u64,

    
    #[serde(default = "default_ram")]
    pub ram_mb: u64,

    
    #[serde(default = "default_block_size")]
    pub block_size_kb: usize,

    
    #[serde(default = "default_true")]
    pub enable_swap: bool,
}

// Implementation for Default
impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            vram_mb: default_vram(),
            ram_mb: default_ram(),
            block_size_kb: default_block_size(),
            enable_swap: true,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// PluginSettings structure
pub struct PluginSettings {
    
    #[serde(default = "default_plugin_dir")]
    pub plugin_dir: String,

    
    #[serde(default = "default_true")]
    pub enabled: bool,

    
    #[serde(default)]
    pub auto_load: Vec<String>,
}

// Implementation for Default
impl Default for PluginSettings {
    fn default() -> Self {
        Self {
            plugin_dir: default_plugin_dir(),
            enabled: true,
            auto_load: Vec::new(),
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// LoggingSettings structure
pub struct LoggingSettings {
    
    #[serde(default = "default_log_level")]
    pub level: String,

    
    #[serde(default = "default_log_format")]
    pub format: String,

    
    pub file: Option<String>,

    
    #[serde(default = "default_true")]
    pub console: bool,
}

// Implementation for Default
impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            file: None,
            console: true,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
    /// ServerSettings structure
pub struct ServerSettings {
    
    #[serde(default = "default_host")]
    pub host: String,

    
    #[serde(default = "default_port")]
    pub port: u16,

    
    #[serde(default = "default_true")]
    pub enable_cors: bool,

    
    pub api_key: Option<String>,

    
    #[serde(default = "default_max_request_size")]
    pub max_request_size_mb: usize,
}

// Implementation for Default
impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            enable_cors: true,
            api_key: None,
            max_request_size_mb: default_max_request_size(),
        }
    }
}



fn default_batch_size() -> usize { 512 }
fn default_context_length() -> usize { 2048 }
fn default_gpu_layers() -> i32 { -1 }
fn default_true() -> bool { true }
fn default_backend() -> String { "cpu".to_string() }
fn default_vram() -> u64 { 4096 }
fn default_ram() -> u64 { 8192 }
fn default_block_size() -> usize { 256 }
fn default_plugin_dir() -> String { "./plugins".to_string() }
fn default_log_level() -> String { "info".to_string() }
fn default_log_format() -> String { "text".to_string() }
fn default_host() -> String { "127.0.0.1".to_string() }
fn default_port() -> u16 { 8080 }
fn default_max_request_size() -> usize { 100 }




    /// ConfigLoader structure
pub struct ConfigLoader {
    config: LociConfig,
    #[allow(dead_code)]
    env_overrides: HashMap<String, String>,
}

// Implementation for ConfigLoader
impl ConfigLoader {
    
    /// new function
    pub fn new() -> Self {
        Self {
            config: LociConfig::default(),
            env_overrides: HashMap::new(),
        }
    }

    
    /// from_file function
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {:?}", path))?;

        let config: LociConfig = match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::from_str(&content)
                .context("Failed to parse TOML config")?,
            Some("json") => serde_json::from_str(&content)
                .context("Failed to parse JSON config")?,
            _ => bail!("Unsupported config file format. Use .toml or .json"),
        };

        Ok(Self {
            config,
            env_overrides: HashMap::new(),
        })
    }

    
    /// with_env_overrides function
    pub fn with_env_overrides(mut self) -> Self {
        
        if let Ok(val) = std::env::var("LOCI_MODEL_PATH") {
            self.config.engine.model_path = Some(val);
        }
        if let Ok(val) = std::env::var("LOCI_BATCH_SIZE") {
            if let Ok(size) = val.parse() {
                self.config.engine.batch_size = size;
            }
        }
        if let Ok(val) = std::env::var("LOCI_N_THREADS") {
            if let Ok(threads) = val.parse() {
                self.config.engine.n_threads = threads;
            }
        }
        if let Ok(val) = std::env::var("LOCI_N_GPU_LAYERS") {
            if let Ok(layers) = val.parse() {
                self.config.engine.n_gpu_layers = layers;
            }
        }

        
        if let Ok(val) = std::env::var("LOCI_BACKEND") {
            self.config.backend.backend_type = val;
        }

        
        if let Ok(val) = std::env::var("LOCI_HOST") {
            self.config.server.host = val;
        }
        if let Ok(val) = std::env::var("LOCI_PORT") {
            if let Ok(port) = val.parse() {
                self.config.server.port = port;
            }
        }
        if let Ok(val) = std::env::var("LOCI_API_KEY") {
            self.config.server.api_key = Some(val);
        }

        
        if let Ok(val) = std::env::var("LOCI_LOG_LEVEL") {
            self.config.logging.level = val;
        }

        self
    }

    
    /// validate function
    pub fn validate(&self) -> Result<()> {
        
        if let Some(ref path) = self.config.engine.model_path {
            if !Path::new(path).exists() {
                bail!("Model file not found: {}", path);
            }
        }

        
        if self.config.engine.batch_size == 0 {
            bail!("Batch size must be greater than 0");
        }

        
        if self.config.engine.context_length == 0 {
            bail!("Context length must be greater than 0");
        }

        
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.config.logging.level.as_str()) {
            bail!("Invalid log level: {}. Must be one of: {:?}",
                self.config.logging.level, valid_levels);
        }

        
        let valid_backends = ["cpu", "cuda", "metal", "rocm", "vulkan"];
        if !valid_backends.contains(&self.config.backend.backend_type.as_str()) {
            bail!("Invalid backend: {}. Must be one of: {:?}",
                self.config.backend.backend_type, valid_backends);
        }

        Ok(())
    }

    
    /// build function
    pub fn build(self) -> Result<LociConfig> {
        self.validate()?;
        Ok(self.config)
    }

    
    /// save function
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let content = match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::to_string_pretty(&self.config)
                .context("Failed to serialize to TOML")?,
            Some("json") => serde_json::to_string_pretty(&self.config)
                .context("Failed to serialize to JSON")?,
            _ => bail!("Unsupported config file format. Use .toml or .json"),
        };

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {:?}", path))?;

        Ok(())
    }
}

// Implementation for Default
impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}



// Implementation for LociConfig
impl LociConfig {
    
    /// example_toml function
    pub fn example_toml() -> String {
        r#"# Loci 配置文件示例

[engine]
model_path = "./models/llama-2-7b-q4_k_m.gguf"
batch_size = 512
n_threads = 0  # 0 = 自动检测
context_length = 2048
n_gpu_layers = -1  # -1 = 全部使用 GPU
use_mmap = true
use_mlock = false

[backend]
backend_type = "cpu"  # cpu/cuda/metal/rocm/vulkan
device_id = 0
enable_fusion = true

[memory]
vram_mb = 4096
ram_mb = 8192
block_size_kb = 256
enable_swap = true

[plugins]
plugin_dir = "./plugins"
enabled = true
auto_load = ["conflict-guard", "json-validator"]

[logging]
level = "info"  # trace/debug/info/warn/error
format = "text"  # text/json
file = "./logs/loci.log"  # 留空则不写文件
console = true

[server]
host = "127.0.0.1"
port = 8080
enable_cors = true
api_key = "your-secret-key-here"  # 留空则不需要认证
max_request_size_mb = 100
"#.to_string()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LociConfig::default();
        assert_eq!(config.engine.batch_size, 512);
        assert_eq!(config.backend.backend_type, "cpu");
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_config_validation() {
        let loader = ConfigLoader::new();
        assert!(loader.validate().is_ok());
    }

    #[test]
    fn test_toml_serialization() {
        let config = LociConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("[engine]"));
    }

    #[test]
    fn test_json_serialization() {
        let config = LociConfig::default();
        let json_str = serde_json::to_string_pretty(&config).unwrap();
        assert!(json_str.contains("\"engine\""));
    }

    #[test]
    fn test_example_toml() {
        let example = LociConfig::example_toml();
        assert!(example.contains("[engine]"));
        assert!(example.contains("model_path"));
    }
}

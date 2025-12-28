/**
 * Loci 配置管理系统
 *
 * 核心特性：
 * 1. 多格式支持（TOML/JSON/环境变量）
 * 2. 配置验证
 * 3. 环境变量覆盖
 * 4. 热重载支持
 */

use anyhow::{Result, Context, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashMap;

// ==================== 配置结构 ====================

/// Loci 全局配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LociConfig {
    /// 引擎配置
    #[serde(default)]
    pub engine: EngineSettings,

    /// 后端配置
    #[serde(default)]
    pub backend: BackendSettings,

    /// 内存配置
    #[serde(default)]
    pub memory: MemorySettings,

    /// 插件配置
    #[serde(default)]
    pub plugins: PluginSettings,

    /// 日志配置
    #[serde(default)]
    pub logging: LoggingSettings,

    /// HTTP 服务器配置
    #[serde(default)]
    pub server: ServerSettings,
}

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

/// 引擎设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSettings {
    /// 模型路径
    pub model_path: Option<String>,

    /// 批次大小
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// 线程数（0 = 自动检测）
    #[serde(default)]
    pub n_threads: usize,

    /// 上下文长度
    #[serde(default = "default_context_length")]
    pub context_length: usize,

    /// GPU 层数（-1 = 全部）
    #[serde(default = "default_gpu_layers")]
    pub n_gpu_layers: i32,

    /// 是否使用 mmap
    #[serde(default = "default_true")]
    pub use_mmap: bool,

    /// 是否使用 mlock
    #[serde(default)]
    pub use_mlock: bool,
}

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

/// 后端设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSettings {
    /// 后端类型（cpu/cuda/metal/rocm）
    #[serde(default = "default_backend")]
    pub backend_type: String,

    /// 设备 ID
    #[serde(default)]
    pub device_id: usize,

    /// 是否启用 Kernel 融合
    #[serde(default = "default_true")]
    pub enable_fusion: bool,
}

impl Default for BackendSettings {
    fn default() -> Self {
        Self {
            backend_type: default_backend(),
            device_id: 0,
            enable_fusion: true,
        }
    }
}

/// 内存设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    /// VRAM 大小（MB）
    #[serde(default = "default_vram")]
    pub vram_mb: u64,

    /// RAM 大小（MB）
    #[serde(default = "default_ram")]
    pub ram_mb: u64,

    /// Block 大小（KB）
    #[serde(default = "default_block_size")]
    pub block_size_kb: usize,

    /// 是否启用 Swap
    #[serde(default = "default_true")]
    pub enable_swap: bool,
}

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

/// 插件设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSettings {
    /// 插件目录
    #[serde(default = "default_plugin_dir")]
    pub plugin_dir: String,

    /// 是否启用插件
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// 自动加载插件列表
    #[serde(default)]
    pub auto_load: Vec<String>,
}

impl Default for PluginSettings {
    fn default() -> Self {
        Self {
            plugin_dir: default_plugin_dir(),
            enabled: true,
            auto_load: Vec::new(),
        }
    }
}

/// 日志设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// 日志级别（trace/debug/info/warn/error）
    #[serde(default = "default_log_level")]
    pub level: String,

    /// 日志输出格式（text/json）
    #[serde(default = "default_log_format")]
    pub format: String,

    /// 日志文件路径
    pub file: Option<String>,

    /// 是否输出到控制台
    #[serde(default = "default_true")]
    pub console: bool,
}

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

/// HTTP 服务器设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    /// 监听地址
    #[serde(default = "default_host")]
    pub host: String,

    /// 监听端口
    #[serde(default = "default_port")]
    pub port: u16,

    /// 是否启用 CORS
    #[serde(default = "default_true")]
    pub enable_cors: bool,

    /// API 密钥
    pub api_key: Option<String>,

    /// 最大请求大小（MB）
    #[serde(default = "default_max_request_size")]
    pub max_request_size_mb: usize,
}

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

// ==================== 默认值函数 ====================

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

// ==================== 配置加载器 ====================

/// 配置加载器
pub struct ConfigLoader {
    config: LociConfig,
    #[allow(dead_code)]
    env_overrides: HashMap<String, String>,
}

impl ConfigLoader {
    /// 创建默认配置加载器
    pub fn new() -> Self {
        Self {
            config: LociConfig::default(),
            env_overrides: HashMap::new(),
        }
    }

    /// 从文件加载配置
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

    /// 从环境变量加载配置
    pub fn with_env_overrides(mut self) -> Self {
        // 引擎配置
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

        // 后端配置
        if let Ok(val) = std::env::var("LOCI_BACKEND") {
            self.config.backend.backend_type = val;
        }

        // 服务器配置
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

        // 日志配置
        if let Ok(val) = std::env::var("LOCI_LOG_LEVEL") {
            self.config.logging.level = val;
        }

        self
    }

    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        // 验证模型路径
        if let Some(ref path) = self.config.engine.model_path {
            if !Path::new(path).exists() {
                bail!("Model file not found: {}", path);
            }
        }

        // 验证批次大小
        if self.config.engine.batch_size == 0 {
            bail!("Batch size must be greater than 0");
        }

        // 验证上下文长度
        if self.config.engine.context_length == 0 {
            bail!("Context length must be greater than 0");
        }

        // 验证日志级别
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.config.logging.level.as_str()) {
            bail!("Invalid log level: {}. Must be one of: {:?}",
                self.config.logging.level, valid_levels);
        }

        // 验证后端类型
        let valid_backends = ["cpu", "cuda", "metal", "rocm", "vulkan"];
        if !valid_backends.contains(&self.config.backend.backend_type.as_str()) {
            bail!("Invalid backend: {}. Must be one of: {:?}",
                self.config.backend.backend_type, valid_backends);
        }

        Ok(())
    }

    /// 构建最终配置
    pub fn build(self) -> Result<LociConfig> {
        self.validate()?;
        Ok(self.config)
    }

    /// 保存配置到文件
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

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 配置示例生成 ====================

impl LociConfig {
    /// 生成示例配置
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

// ==================== 单元测试 ====================

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

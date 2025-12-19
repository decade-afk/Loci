use thiserror::Error;

/// AI 服务相关错误类型
#[derive(Error, Debug)]
pub enum AIError {
    #[error("模型未加载")]
    ModelNotLoaded,

    #[error("模型已加载")]
    ModelAlreadyLoaded,

    #[error("模型加载失败: {0}")]
    ModelLoadError(String),

    #[error("推理失败: {0}")]
    InferenceError(String),

    #[error("生成已取消")]
    GenerationCancelled,

    #[error("配置错误: {0}")]
    ConfigError(String),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("锁错误: 无法获取锁")]
    LockError,

    #[error("未知错误: {0}")]
    Unknown(String),
}

/// Agent 系统相关错误类型
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("模型未找到: {0}")]
    ModelNotFound(String),

    #[error("Agent 未找到: {0}")]
    AgentNotFound(String),

    #[error("会话未找到: {0}")]
    SessionNotFound(String),

    #[error("模型已存在: {0}")]
    ModelAlreadyExists(String),

    #[error("Agent 已存在: {0}")]
    AgentAlreadyExists(String),

    #[error("模型加载失败: {0}")]
    ModelLoadError(String),

    #[error("推理失败: {0}")]
    InferenceError(String),

    #[error("上下文溢出: 需要 {required} tokens，但只有 {available} tokens 可用")]
    ContextOverflow { required: usize, available: usize },

    #[error("配置错误: {0}")]
    ConfigError(String),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("锁错误: 无法获取锁")]
    LockError,

    #[error("未知错误: {0}")]
    Unknown(String),
}

/// 系统信息相关错误类型
#[derive(Error, Debug)]
pub enum SystemInfoError {
    #[error("无法检测系统信息: {0}")]
    DetectionError(String),

    #[error("GPU 不可用")]
    GpuNotAvailable,

    #[error("内存不足: 需要 {required} GB，但只有 {available} GB 可用")]
    InsufficientMemory { required: f64, available: f64 },

    #[error("未知错误: {0}")]
    Unknown(String),
}

/// 导出相关错误类型
#[derive(Error, Debug)]
pub enum ExportError {
    #[error("不支持的导出格式: {0}")]
    UnsupportedFormat(String),

    #[error("导出失败: {0}")]
    ExportFailed(String),

    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    #[error("序列化错误: {0}")]
    SerializationError(String),

    #[error("未知错误: {0}")]
    Unknown(String),
}

// 实现从各种错误类型到 String 的转换，用于 Tauri 命令返回
impl From<AIError> for String {
    fn from(err: AIError) -> Self {
        err.to_string()
    }
}

impl From<AgentError> for String {
    fn from(err: AgentError) -> Self {
        err.to_string()
    }
}

impl From<SystemInfoError> for String {
    fn from(err: SystemInfoError) -> Self {
        err.to_string()
    }
}

impl From<ExportError> for String {
    fn from(err: ExportError) -> Self {
        err.to_string()
    }
}

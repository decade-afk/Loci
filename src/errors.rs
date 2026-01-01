//! Error types for the Loci AI inference framework.
//!
//! This module defines comprehensive error types for various operations within the system:
//! - `AIError`: Core AI/ML model operations including loading, inference, and generation
//! - `AgentError`: Agent system operations including session management and model interactions
//! - `SystemInfoError`: System information detection and resource validation
//! - `ExportError`: Model and data export operations
//!
//! All error types implement `std::error::Error` and `std::fmt::Debug` for proper error handling.

use thiserror::Error;


/// Errors that can occur during AI/ML model operations.
///
/// This enum covers errors related to model loading, inference execution,
/// generation processes, and configuration management.
#[derive(Error, Debug)]
pub enum AIError {
    /// Occurs when attempting to perform operations on a model that hasn't been loaded.
    #[error("模型未加载")]
    ModelNotLoaded,

    /// Occurs when attempting to load a model that is already loaded in memory.
    #[error("模型已加载")]
    ModelAlreadyLoaded,

    /// Occurs when the model loading process fails due to invalid files, memory issues, or format problems.
    #[error("模型加载失败: {0}")]
    ModelLoadError(String),

    /// Occurs during inference execution due to runtime errors, invalid inputs, or internal failures.
    #[error("推理失败: {0}")]
    InferenceError(String),

    /// Occurs when a text generation operation is cancelled by the user or system.
    #[error("生成已取消")]
    GenerationCancelled,

    /// Occurs when configuration parameters are invalid, missing, or incompatible.
    #[error("配置错误: {0}")]
    ConfigError(String),

    /// Occurs during file I/O operations such as reading or writing model files.
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    /// Occurs when unable to acquire a lock for concurrent access to shared resources.
    #[error("锁错误: 无法获取锁")]
    LockError,

    /// Occurs when an unexpected error that doesn't fit other categories is encountered.
    #[error("未知错误: {0}")]
    Unknown(String),
}


/// Errors that can occur during Agent system operations.
///
/// This enum covers errors related to agent lifecycle management,
/// session handling, model interactions, and context management.
#[derive(Error, Debug)]
pub enum AgentError {
    /// Occurs when attempting to access a model that doesn't exist in the registry.
    #[error("模型未找到: {0}")]
    ModelNotFound(String),

    /// Occurs when attempting to access an agent that hasn't been registered or created.
    #[error("Agent 未找到: {0}")]
    AgentNotFound(String),

    /// Occurs when attempting to access a session that doesn't exist or has expired.
    #[error("会话未找到: {0}")]
    SessionNotFound(String),

    /// Occurs when attempting to register a model with an ID that already exists.
    #[error("模型已存在: {0}")]
    ModelAlreadyExists(String),

    /// Occurs when attempting to create an agent with an ID that already exists.
    #[error("Agent 已存在: {0}")]
    AgentAlreadyExists(String),

    /// Occurs when loading a model for an agent fails.
    #[error("模型加载失败: {0}")]
    ModelLoadError(String),

    /// Occurs when inference execution within an agent context fails.
    #[error("推理失败: {0}")]
    InferenceError(String),

    /// Occurs when the context window is exceeded during generation.
    /// This happens when the required tokens exceed the available context capacity.
    #[error("上下文溢出: 需要 {required} tokens，但只有 {available} tokens 可用")]
    ContextOverflow { required: usize, available: usize },

    /// Occurs when agent configuration is invalid or incomplete.
    #[error("配置错误: {0}")]
    ConfigError(String),

    /// Occurs during file I/O operations for agent data persistence.
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    /// Occurs when unable to acquire a lock for concurrent agent operations.
    #[error("锁错误: 无法获取锁")]
    LockError,

    /// Occurs when an unexpected error in agent operations is encountered.
    #[error("未知错误: {0}")]
    Unknown(String),
}


/// Errors that can occur during system information detection and resource validation.
///
/// This enum covers errors related to hardware detection, GPU availability,
/// memory capacity checks, and system resource validation.
#[derive(Error, Debug)]
pub enum SystemInfoError {
    /// Occurs when the system fails to detect or retrieve hardware information.
    /// This can happen due to permission issues, missing drivers, or unsupported platforms.
    #[error("无法检测系统信息: {0}")]
    DetectionError(String),

    /// Occurs when attempting to use GPU acceleration but no compatible GPU is available.
    /// This may be due to missing CUDA/ROCm drivers, unsupported hardware, or disabled GPU support.
    #[error("GPU 不可用")]
    GpuNotAvailable,

    /// Occurs when the system doesn't have sufficient memory resources for the requested operation.
    /// This includes both RAM and GPU memory validation.
    #[error("内存不足: 需要 {required} GB，但只有 {available} GB 可用")]
    InsufficientMemory { required: f64, available: f64 },

    /// Occurs when an unexpected error in system information operations is encountered.
    #[error("未知错误: {0}")]
    Unknown(String),
}


/// Errors that can occur during model and data export operations.
///
/// This enum covers errors related to exporting models, configurations,
/// and data to various formats and destinations.
#[derive(Error, Debug)]
pub enum ExportError {
    /// Occurs when attempting to export to a format that is not supported by the system.
    /// Supported formats may include GGUF, ONNX, and other serialization formats.
    #[error("不支持的导出格式: {0}")]
    UnsupportedFormat(String),

    /// Occurs when the export process fails due to internal errors, invalid data,
    /// or destination issues.
    #[error("导出失败: {0}")]
    ExportFailed(String),

    /// Occurs during file I/O operations such as reading source files or writing export files.
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    /// Occurs when converting model data or configurations to the target format fails.
    /// This includes JSON, binary serialization, and other format-specific serialization errors.
    #[error("序列化错误: {0}")]
    SerializationError(String),

    /// Occurs when an unexpected error in export operations is encountered.
    #[error("未知错误: {0}")]
    Unknown(String),
}


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

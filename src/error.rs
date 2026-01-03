//! Error types for Loci

use thiserror::Error;

/// Result type alias for Loci operations
pub type Result<T> = std::result::Result<T, LociError>;

/// Main error type for Loci
#[derive(Error, Debug)]
pub enum LociError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("llama.cpp error: {0}")]
    LlamaCppError(String),

    #[error("Plugin error: {0}")]
    PluginError(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid token ID: {0}")]
    InvalidToken(i32),

    #[error("Model not found")]
    ModelNotFound,

    #[error("Session not found")]
    SessionNotFound,

    #[error("Invalid model path")]
    InvalidModelPath,

    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    #[error("Invalid block ID: {0}")]
    InvalidBlockId(u64),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<String> for LociError {
    fn from(s: String) -> Self {
        LociError::Other(s)
    }
}

impl From<&str> for LociError {
    fn from(s: &str) -> Self {
        LociError::Other(s.to_string())
    }
}

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

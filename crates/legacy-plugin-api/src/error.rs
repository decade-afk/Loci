use thiserror::Error;

pub type Result<T> = std::result::Result<T, LociError>;

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
    #[error("Invalid session state: {0}")]
    InvalidSessionState(String),
    #[error("Invalid model path")]
    InvalidModelPath,
    #[error("Out of memory: {0}")]
    OutOfMemory(String),
    #[error("Invalid block ID: {0}")]
    InvalidBlockId(u64),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Network error: {0}")]
    NetworkError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    #[error("Plugin initialization failed: {0}")]
    PluginInitError(String),
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),
    #[error("Model format error: {0}")]
    ModelFormatError(String),
    #[error("Context overflow: {0}")]
    ContextOverflow(String),
    #[error("Other error: {0}")]
    Other(String),
}

impl From<String> for LociError {
    fn from(value: String) -> Self {
        Self::Other(value)
    }
}

impl From<&str> for LociError {
    fn from(value: &str) -> Self {
        Self::Other(value.to_string())
    }
}

impl From<serde_json::Error> for LociError {
    fn from(value: serde_json::Error) -> Self {
        Self::SerializationError(value.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LociError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("backend not available: {0}")]
    BackendNotAvailable(String),
    #[error("backend error: {0}")]
    BackendError(String),
    #[error("model load error: {0}")]
    ModelLoadError(String),
    #[error("inference error: {0}")]
    InferenceError(String),
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, LociError>;

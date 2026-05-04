//! Error types used by the Loci core runtime.

use thiserror::Error;

/// Enumerates the user-visible orchestration failures returned by `loci-core`.
#[derive(Debug, Error)]
pub enum LociError {
    #[error("no backend is available for the current build")]
    NoBackendAvailable,
    #[error("no model has been registered")]
    NoModelsRegistered,
    #[error("requested model `{0}` is not registered")]
    RequestedModelMissing(String),
    #[error("no compatible backend is available for model `{model}` with format `{format}`")]
    NoCompatibleBackend { model: String, format: String },
    #[error("backend execution failed: {0}")]
    Backend(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

/// Standard result alias used throughout the core crate.
pub type Result<T> = std::result::Result<T, LociError>;

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

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum ErrorRecovery {
    /// Retry the operation
    Retry { max_attempts: u32, delay_ms: u64 },
    /// Fallback to alternative approach
    Fallback { alternative: String },
    /// Reset to safe state
    Reset,
    /// Abort operation
    Abort,
}

/// Error context for better debugging
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// Component that failed
    pub component: String,
    /// Additional context data
    pub context: std::collections::HashMap<String, String>,
    /// Suggested recovery strategy
    pub recovery: Option<ErrorRecovery>,
}

impl LociError {
    /// Create error with context
    pub fn with_context(self, context: ErrorContext) -> LociErrorWithContext {
        LociErrorWithContext {
            error: self,
            context,
        }
    }

    /// Create a model loading error with context
    pub fn model_load_with_context(msg: String, path: String) -> LociErrorWithContext {
        let mut context_data = std::collections::HashMap::new();
        context_data.insert("model_path".to_string(), path);
        
        let context = ErrorContext {
            operation: "model_loading".to_string(),
            component: "backend".to_string(),
            context: context_data,
            recovery: Some(ErrorRecovery::Fallback {
                alternative: "Try a different model format or check file permissions".to_string(),
            }),
        };
        
        LociError::ModelLoadError(msg).with_context(context)
    }

    /// Create an inference error with context
    pub fn inference_with_context(msg: String, session_id: Option<String>) -> LociErrorWithContext {
        let mut context_data = std::collections::HashMap::new();
        if let Some(id) = session_id {
            context_data.insert("session_id".to_string(), id);
        }
        
        let context = ErrorContext {
            operation: "inference".to_string(),
            component: "engine".to_string(),
            context: context_data,
            recovery: Some(ErrorRecovery::Retry {
                max_attempts: 3,
                delay_ms: 100,
            }),
        };
        
        LociError::InferenceError(msg).with_context(context)
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            LociError::ModelLoadError(_) => false,
            LociError::InferenceError(_) => true,
            LociError::ConfigError(_) => false,
            LociError::IoError(_) => true,
            LociError::LlamaCppError(_) => true,
            LociError::PluginError(_) => true,
            LociError::BackendError(_) => true,
            LociError::UnsupportedOperation(_) => false,
            LociError::InvalidToken(_) => false,
            LociError::ModelNotFound => false,
            LociError::SessionNotFound => false,
            LociError::InvalidSessionState(_) => true,
            LociError::InvalidModelPath => false,
            LociError::OutOfMemory(_) => false,
            LociError::InvalidBlockId(_) => false,
            LociError::InvalidArgument(_) => false,
            LociError::ResourceExhausted(_) => true,
            LociError::Timeout(_) => true,
            LociError::NetworkError(_) => true,
            LociError::SerializationError(_) => false,
            LociError::ConstraintViolation(_) => true,
            LociError::PluginInitError(_) => true,
            LociError::BackendNotAvailable(_) => false,
            LociError::ModelFormatError(_) => false,
            LociError::ContextOverflow(_) => true,
            LociError::Other(_) => false,
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            LociError::ModelLoadError(_) => ErrorSeverity::Critical,
            LociError::InferenceError(_) => ErrorSeverity::High,
            LociError::ConfigError(_) => ErrorSeverity::Medium,
            LociError::IoError(_) => ErrorSeverity::Medium,
            LociError::LlamaCppError(_) => ErrorSeverity::High,
            LociError::PluginError(_) => ErrorSeverity::Medium,
            LociError::BackendError(_) => ErrorSeverity::High,
            LociError::UnsupportedOperation(_) => ErrorSeverity::Low,
            LociError::InvalidToken(_) => ErrorSeverity::Low,
            LociError::ModelNotFound => ErrorSeverity::Critical,
            LociError::SessionNotFound => ErrorSeverity::Medium,
            LociError::InvalidSessionState(_) => ErrorSeverity::Medium,
            LociError::InvalidModelPath => ErrorSeverity::High,
            LociError::OutOfMemory(_) => ErrorSeverity::Critical,
            LociError::InvalidBlockId(_) => ErrorSeverity::Low,
            LociError::InvalidArgument(_) => ErrorSeverity::Low,
            LociError::ResourceExhausted(_) => ErrorSeverity::High,
            LociError::Timeout(_) => ErrorSeverity::Medium,
            LociError::NetworkError(_) => ErrorSeverity::Medium,
            LociError::SerializationError(_) => ErrorSeverity::Low,
            LociError::ConstraintViolation(_) => ErrorSeverity::Low,
            LociError::PluginInitError(_) => ErrorSeverity::Medium,
            LociError::BackendNotAvailable(_) => ErrorSeverity::Critical,
            LociError::ModelFormatError(_) => ErrorSeverity::High,
            LociError::ContextOverflow(_) => ErrorSeverity::Medium,
            LociError::Other(_) => ErrorSeverity::Low,
        }
    }
}

/// Error with additional context
#[derive(Debug)]
pub struct LociErrorWithContext {
    pub error: LociError,
    pub context: ErrorContext,
}

impl std::fmt::Display for LociErrorWithContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (in {} operation on {})", 
               self.error, 
               self.context.operation, 
               self.context.component)?;
        
        if !self.context.context.is_empty() {
            write!(f, " - Context: {:?}", self.context.context)?;
        }
        
        if let Some(ref recovery) = self.context.recovery {
            write!(f, " - Suggested recovery: {:?}", recovery)?;
        }
        
        Ok(())
    }
}

impl std::error::Error for LociErrorWithContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
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

impl From<serde_json::Error> for LociError {
    fn from(e: serde_json::Error) -> Self {
        LociError::SerializationError(e.to_string())
    }
}

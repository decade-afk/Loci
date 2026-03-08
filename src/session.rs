//! Session management for multi-session inference
//!
//! This module provides session management capabilities that enable:
//! - Multiple concurrent inference sessions
//! - Session-model decoupling (multiple sessions can share one model)
//! - Independent session state and context
//! - Per-session plugin management

use crate::error::{LociError, Result};
use crate::model_registry::{ModelId, ModelRegistry};
use crate::plugin::PluginManager;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Unique identifier for an inference session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

impl SessionId {
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SessionId({})", self.0)
    }
}

/// Session execution state
///
/// Tracks the current state of an inference session, enabling
/// suspend/resume functionality for tool calls and external interactions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionState {
    /// Session is actively running inference
    Running,

    /// Session is suspended, waiting for external input/tool results
    ///
    /// Contains:
    /// - `reason`: Why the session was suspended (e.g., "tool_call", "user_input")
    /// - `data`: Optional data for the external handler (e.g., tool parameters as JSON)
    AwaitingExternal {
        reason: String,
        data: Option<String>,
    },

    /// Session is resuming from suspension with external data
    ///
    /// Contains the external data/tool result that was injected via `resume_session()`
    Resuming {
        external_data: String,
    },

    /// Session completed successfully
    Completed,

    /// Session encountered an error
    Error {
        message: String,
    },
}

impl Default for SessionState {
    fn default() -> Self {
        SessionState::Running
    }
}

impl SessionState {
    /// Check if session can accept new input
    pub fn can_generate(&self) -> bool {
        matches!(self, SessionState::Running | SessionState::Resuming { .. })
    }

    /// Check if session is waiting for external input
    pub fn is_suspended(&self) -> bool {
        matches!(self, SessionState::AwaitingExternal { .. })
    }

    /// Check if session is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, SessionState::Completed | SessionState::Error { .. })
    }
}

/// An inference session with its own state and context
///
/// Each session maintains:
/// - Reference to a model (via ModelId)
/// - Independent context tokens
/// - Session-specific plugin manager
/// - KV cache state (future)
/// - Execution state (Running/Suspended/Resuming)
pub struct InferenceSession {
    session_id: SessionId,
    model_id: ModelId,
    context_tokens: Vec<i32>,
    plugin_manager: PluginManager,
    max_context: u32,
    /// Current execution state
    state: SessionState,
    /// Suspended generation context (preserved during suspension)
    suspended_context: Option<SuspendedContext>,
}

/// Context preserved during session suspension
#[derive(Debug, Clone)]
struct SuspendedContext {
    /// Partial prompt/generation when suspended
    partial_output: String,
    /// Number of tokens generated before suspension
    tokens_generated: usize,
    /// Maximum tokens requested
    max_tokens: usize,
}

impl InferenceSession {
    /// Create a new session
    fn new(session_id: SessionId, model_id: ModelId, max_context: u32) -> Self {
        Self {
            session_id,
            model_id,
            context_tokens: Vec::new(),
            plugin_manager: PluginManager::new(),
            max_context,
            state: SessionState::default(),
            suspended_context: None,
        }
    }

    /// Get session ID
    pub fn id(&self) -> SessionId {
        self.session_id
    }

    /// Get associated model ID
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Get current context length
    pub fn context_length(&self) -> usize {
        self.context_tokens.len()
    }

    /// Get maximum context size
    pub fn max_context(&self) -> u32 {
        self.max_context
    }

    /// Get reference to plugin manager
    pub fn plugin_manager(&self) -> &PluginManager {
        &self.plugin_manager
    }

    /// Get mutable reference to plugin manager
    pub fn plugin_manager_mut(&mut self) -> &mut PluginManager {
        &mut self.plugin_manager
    }

    /// Get current session state
    pub fn state(&self) -> &SessionState {
        &self.state
    }

    /// Check if session can accept new generation requests
    pub fn can_generate(&self) -> bool {
        self.state.can_generate()
    }

    /// Check if session is suspended and waiting for external input
    pub fn is_suspended(&self) -> bool {
        self.state.is_suspended()
    }

    /// Generate text using this session
    ///
    /// # Arguments
    ///
    /// * `registry` - Model registry to access the model
    /// * `prompt` - Input text prompt
    /// * `max_tokens` - Maximum tokens to generate
    ///
    /// # Returns
    ///
    /// Generated text response
    ///
    /// # Note
    ///
    /// This is currently a placeholder implementation that demonstrates the API.
    /// Full integration with LlamaCppModel requires backend coordination that
    /// will be implemented in a future update.
    #[allow(unused_variables)]
    pub fn generate(
        &mut self,
        registry: &ModelRegistry,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        // Verify model exists
        if !registry.has_model(self.model_id) {
            return Err(LociError::ModelNotFound);
        }

        // Pre-generation hook
        let processed_prompt = self
            .plugin_manager
            .apply_pre_generate(prompt)
            .unwrap_or_else(|_| prompt.to_string());

        // TODO: Actual model inference would go here
        // let model = registry.get_model(self.model_id).unwrap();
        // let response = model.generate(&processed_prompt, max_tokens)?;

        // For now, return a placeholder indicating the architecture works
        let response = format!("[Session {} response to: {}]", self.session_id, processed_prompt);

        // Post-generation hook
        let processed_response = self
            .plugin_manager
            .apply_post_generate(&response)
            .unwrap_or(response.clone());

        // Simulate token tracking (in real impl, these come from tokenizer)
        let estimated_tokens = processed_prompt.len() / 4 + processed_response.len() / 4;
        if self.context_tokens.len() + estimated_tokens > self.max_context as usize {
            let keep = self.max_context as usize - estimated_tokens;
            if keep > 0 && keep < self.context_tokens.len() {
                self.context_tokens = self.context_tokens[self.context_tokens.len() - keep..].to_vec();
            }
        }

        Ok(processed_response)
    }

    /// Resume a suspended session with external data/tool results
    ///
    /// # Arguments
    ///
    /// * `external_data` - Data from external source (e.g., tool call result, user input)
    ///
    /// # Returns
    ///
    /// Ok if session was successfully resumed, Err if session is not suspended
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Session suspends for tool call
    /// let result = session.generate(&registry, "Use calculator to add 5+3", 100);
    /// // => Session enters AwaitingExternal state with tool call data
    ///
    /// // External system executes tool
    /// let tool_result = "8";
    ///
    /// // Resume session with tool result
    /// session.resume_session(tool_result.to_string())?;
    ///
    /// // Continue generation
    /// let final_result = session.generate(&registry, "", 100)?;
    /// ```
    pub fn resume_session(&mut self, external_data: String) -> Result<()> {
        // Verify session is in suspended state
        if !self.is_suspended() {
            return Err(LociError::InvalidSessionState(format!(
                "Cannot resume session {} - not in suspended state (current: {:?})",
                self.session_id, self.state
            )));
        }

        // Transition to Resuming state
        self.state = SessionState::Resuming { external_data };

        Ok(())
    }

    /// Manually suspend the session
    ///
    /// This can be used by plugins or application logic to suspend a session.
    ///
    /// # Arguments
    ///
    /// * `reason` - Reason for suspension
    /// * `data` - Optional data for external handler
    pub fn suspend(&mut self, reason: String, data: Option<String>) {
        self.state = SessionState::AwaitingExternal { reason, data };
    }

    /// Clear session context
    pub fn clear_context(&mut self) {
        self.context_tokens.clear();
    }

    /// Get context tokens (read-only)
    pub fn context_tokens(&self) -> &[i32] {
        &self.context_tokens
    }
}

/// Session manager for coordinating multiple inference sessions
///
/// The SessionManager provides:
/// - Session lifecycle management (create, destroy)
/// - Model registry integration
/// - Multi-session coordination
///
/// # Examples
///
/// ```ignore
/// use loci::session::SessionManager;
///
/// let mut manager = SessionManager::new();
/// let model_id = manager.load_model("qwen-0.5b.gguf", 2048)?;
/// let session1 = manager.create_session(model_id)?;
/// let session2 = manager.create_session(model_id)?; // Shares same model
/// ```
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<SessionId, InferenceSession>>>,
    model_registry: Arc<RwLock<ModelRegistry>>,
    next_session_id: AtomicU64,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            model_registry: Arc::new(RwLock::new(ModelRegistry::new())),
            next_session_id: AtomicU64::new(1),
        }
    }

    /// Load a model into the registry
    ///
    /// # Arguments
    ///
    /// * `path` - Path to GGUF model file
    /// * `n_ctx` - Context size
    ///
    /// # Returns
    ///
    /// ModelId that can be used to create sessions
    pub fn load_model(&self, path: &str, n_ctx: u32) -> Result<ModelId> {
        self.model_registry.read().load_model(path, n_ctx)
    }

    /// Unload a model from the registry
    pub fn unload_model(&self, model_id: ModelId) -> Result<()> {
        self.model_registry.read().unload_model(model_id)
    }

    /// Create a new inference session
    ///
    /// # Arguments
    ///
    /// * `model_id` - ID of the model to use
    ///
    /// # Returns
    ///
    /// SessionId for the newly created session
    pub fn create_session(&self, model_id: ModelId) -> Result<SessionId> {
        // Verify model exists
        if !self.model_registry.read().has_model(model_id) {
            return Err(LociError::ModelNotFound);
        }

        let session_id = SessionId(self.next_session_id.fetch_add(1, Ordering::SeqCst));

        // Get model info to determine max context
        let max_context = self
            .model_registry
            .read()
            .get_model_info(model_id)
            .map(|_| 2048u32) // Default context size
            .ok_or(LociError::ModelNotFound)?;

        let session = InferenceSession::new(session_id, model_id, max_context);

        let mut sessions = self.sessions.write();
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Get a session by ID
    ///
    /// Returns None if session doesn't exist
    pub fn get_session(&self, session_id: SessionId) -> Option<SessionHandle> {
        if self.sessions.read().contains_key(&session_id) {
            Some(SessionHandle {
                session_id,
                sessions: Arc::clone(&self.sessions),
                model_registry: Arc::clone(&self.model_registry),
            })
        } else {
            None
        }
    }

    /// Destroy a session and free its resources
    pub fn destroy_session(&self, session_id: SessionId) -> Result<()> {
        let mut sessions = self.sessions.write();

        if let Some(_session) = sessions.remove(&session_id) {
            Ok(())
        } else {
            Err(LociError::SessionNotFound)
        }
    }

    /// Get number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.read().len()
    }

    /// Get number of loaded models
    pub fn model_count(&self) -> usize {
        self.model_registry.read().model_count()
    }

    /// List all active sessions
    pub fn list_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.read();
        sessions
            .values()
            .map(|s| SessionInfo {
                session_id: s.session_id,
                model_id: s.model_id,
                context_length: s.context_length(),
                max_context: s.max_context,
            })
            .collect()
    }

    /// Check if a session exists
    pub fn has_session(&self, session_id: SessionId) -> bool {
        self.sessions.read().contains_key(&session_id)
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to access a session
///
/// This handle provides safe concurrent access to a session
pub struct SessionHandle {
    session_id: SessionId,
    sessions: Arc<RwLock<HashMap<SessionId, InferenceSession>>>,
    model_registry: Arc<RwLock<ModelRegistry>>,
}

impl SessionHandle {
    /// Generate text using this session
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut sessions = self.sessions.write();
        let session = sessions
            .get_mut(&self.session_id)
            .ok_or(LociError::SessionNotFound)?;

        let registry = self.model_registry.read();
        session.generate(&registry, prompt, max_tokens)
    }

    /// Clear session context
    pub fn clear_context(&self) -> Result<()> {
        let mut sessions = self.sessions.write();
        let session = sessions
            .get_mut(&self.session_id)
            .ok_or(LociError::SessionNotFound)?;

        session.clear_context();
        Ok(())
    }

    /// Get session info
    pub fn info(&self) -> Result<SessionInfo> {
        let sessions = self.sessions.read();
        let session = sessions
            .get(&self.session_id)
            .ok_or(LociError::SessionNotFound)?;

        Ok(SessionInfo {
            session_id: session.session_id,
            model_id: session.model_id,
            context_length: session.context_length(),
            max_context: session.max_context,
        })
    }

    /// Get session ID
    pub fn id(&self) -> SessionId {
        self.session_id
    }
}

/// Information about a session
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: SessionId,
    pub model_id: ModelId,
    pub context_length: usize,
    pub max_context: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_manager_creation() {
        let manager = SessionManager::new();
        assert_eq!(manager.session_count(), 0);
        assert_eq!(manager.model_count(), 0);
    }

    #[test]
    fn test_session_id_display() {
        let id = SessionId(42);
        assert_eq!(format!("{}", id), "SessionId(42)");
    }

    #[test]
    fn test_session_manager_operations() {
        let manager = SessionManager::new();

        // Test initial state
        assert_eq!(manager.session_count(), 0);
        assert!(!manager.has_session(SessionId(1)));

        // Test list empty
        let sessions = manager.list_sessions();
        assert_eq!(sessions.len(), 0);
    }
}

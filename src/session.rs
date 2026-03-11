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
use crate::session_store::{
    InMemorySessionStore, SessionStore, SessionStoreConfig, SessionStoreRegistry,
    SqliteSessionStore,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
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

impl From<u64> for SessionId {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

/// Role label for a conversation message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionRole {
    User,
    Assistant,
    Tool,
}

impl SessionRole {
    fn as_label(&self) -> &'static str {
        match self {
            SessionRole::User => "user",
            SessionRole::Assistant => "assistant",
            SessionRole::Tool => "tool",
        }
    }
}

/// One conversation record persisted in session memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub role: SessionRole,
    pub content: String,
}

/// Session execution state
///
/// Tracks the current state of an inference session, enabling
/// suspend/resume functionality for tool calls and external interactions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
/// - Conversation records (user/assistant/tool)
/// - Session-specific plugin manager
/// - Execution state (Running/Suspended/Resuming)
pub struct InferenceSession {
    session_id: SessionId,
    model_id: ModelId,
    context_tokens: Vec<i32>,
    conversation: Vec<MemoryRecord>,
    conversation_tokens: usize,
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

/// Serializable snapshot for suspended generation state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSuspendedSnapshot {
    pub partial_output: String,
    pub tokens_generated: usize,
    pub max_tokens: usize,
}

/// Serializable snapshot of an inference session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub session_id: u64,
    pub model_path: String,
    pub model_n_ctx: u32,
    pub state: SessionState,
    pub records: Vec<SessionRecord>,
    pub suspended_context: Option<SessionSuspendedSnapshot>,
}

#[derive(Debug, Clone)]
struct MemoryRecord {
    role: SessionRole,
    content: String,
    estimated_tokens: usize,
}

impl InferenceSession {
    /// Create a new session
    fn new(session_id: SessionId, model_id: ModelId, max_context: u32) -> Self {
        Self {
            session_id,
            model_id,
            context_tokens: Vec::new(),
            conversation: Vec::new(),
            conversation_tokens: 0,
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

    fn estimate_tokens(text: &str) -> usize {
        let whitespace_tokens = text.split_whitespace().count();
        let char_tokens = (text.chars().count() + 3) / 4;
        whitespace_tokens.max(char_tokens).max(1)
    }

    fn sync_context_tokens(&mut self) {
        let target = self.conversation_tokens.min(self.max_context as usize);
        self.context_tokens.resize(target, 0);
    }

    fn append_record(&mut self, role: SessionRole, content: String) {
        let estimated_tokens = Self::estimate_tokens(&content);
        self.conversation_tokens += estimated_tokens;
        self.conversation.push(MemoryRecord {
            role,
            content,
            estimated_tokens,
        });
    }

    fn prune_conversation_to_budget(&mut self, reserve_tokens: usize) {
        let max_ctx = self.max_context as usize;
        if reserve_tokens >= max_ctx {
            self.conversation.clear();
            self.conversation_tokens = 0;
            self.sync_context_tokens();
            return;
        }

        let budget = max_ctx - reserve_tokens;
        while self.conversation_tokens > budget {
            if self.conversation.is_empty() {
                break;
            }
            let first = self.conversation.remove(0);
            self.conversation_tokens = self.conversation_tokens.saturating_sub(first.estimated_tokens);
        }
        self.sync_context_tokens();
    }

    fn render_conversation_prompt(&self) -> String {
        let mut out = String::new();
        for record in &self.conversation {
            out.push('[');
            out.push_str(record.role.as_label());
            out.push_str("] ");
            out.push_str(&record.content);
            out.push('\n');
        }
        out
    }

    pub fn records(&self) -> Vec<SessionRecord> {
        self.conversation
            .iter()
            .map(|r| SessionRecord {
                role: r.role,
                content: r.content.clone(),
            })
            .collect()
    }

    fn suspended_snapshot(&self) -> Option<SessionSuspendedSnapshot> {
        self.suspended_context
            .as_ref()
            .map(|ctx| SessionSuspendedSnapshot {
                partial_output: ctx.partial_output.clone(),
                tokens_generated: ctx.tokens_generated,
                max_tokens: ctx.max_tokens,
            })
    }

    fn suspended_from_snapshot(
        snapshot: Option<SessionSuspendedSnapshot>,
    ) -> Option<SuspendedContext> {
        snapshot.map(|ctx| SuspendedContext {
            partial_output: ctx.partial_output,
            tokens_generated: ctx.tokens_generated,
            max_tokens: ctx.max_tokens,
        })
    }

    pub fn to_snapshot(&self, model_path: String, model_n_ctx: u32) -> SessionSnapshot {
        SessionSnapshot {
            session_id: self.session_id.as_u64(),
            model_path,
            model_n_ctx,
            state: self.state.clone(),
            records: self.records(),
            suspended_context: self.suspended_snapshot(),
        }
    }

    pub fn from_snapshot(snapshot: SessionSnapshot, model_id: ModelId, max_context: u32) -> Self {
        let mut conversation = Vec::with_capacity(snapshot.records.len());
        let mut conversation_tokens = 0usize;
        for record in snapshot.records {
            let estimated_tokens = Self::estimate_tokens(&record.content);
            conversation_tokens += estimated_tokens;
            conversation.push(MemoryRecord {
                role: record.role,
                content: record.content,
                estimated_tokens,
            });
        }

        let mut session = Self {
            session_id: SessionId::from(snapshot.session_id),
            model_id,
            context_tokens: Vec::new(),
            conversation,
            conversation_tokens,
            plugin_manager: PluginManager::new(),
            max_context,
            state: snapshot.state,
            suspended_context: Self::suspended_from_snapshot(snapshot.suspended_context),
        };
        session.prune_conversation_to_budget(0);
        session
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
    pub fn generate(
        &mut self,
        registry: &ModelRegistry,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<String> {
        // Ensure session is in a runnable state.
        match &self.state {
            SessionState::Running | SessionState::Resuming { .. } => {}
            state => {
                return Err(LociError::InvalidSessionState(format!(
                    "Session {} cannot generate in state {:?}",
                    self.session_id, state
                )));
            }
        }

        // Verify model exists
        if !registry.has_model(self.model_id) {
            return Err(LociError::ModelNotFound);
        }

        // Consume resume payload exactly once and move back to Running.
        let external_data = match std::mem::replace(&mut self.state, SessionState::Running) {
            SessionState::Resuming { external_data } => Some(external_data),
            SessionState::Running => None,
            other => {
                self.state = other;
                None
            }
        };
        if let Some(data) = external_data {
            self.append_record(SessionRole::Tool, data);
        }

        // Pre-generation hook
        let processed_prompt = self
            .plugin_manager
            .apply_pre_generate(prompt)
            .unwrap_or_else(|_| prompt.to_string());
        if !processed_prompt.is_empty() {
            self.append_record(SessionRole::User, processed_prompt);
        }

        // Build the full model prompt from conversation memory.
        self.prune_conversation_to_budget(max_tokens);
        let model_prompt = self.render_conversation_prompt();

        let response = registry.generate(self.model_id, &model_prompt, max_tokens)?;

        // Post-generation hook
        let processed_response = self
            .plugin_manager
            .apply_post_generate(&response)
            .unwrap_or(response.clone());

        self.append_record(SessionRole::Assistant, processed_response.clone());
        self.prune_conversation_to_budget(0);
        self.suspended_context = None;

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
        self.suspended_context = Some(SuspendedContext {
            partial_output: self
                .conversation
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default(),
            tokens_generated: self.conversation_tokens,
            max_tokens: self.max_context as usize,
        });
        self.state = SessionState::AwaitingExternal { reason, data };
    }

    /// Clear session context
    pub fn clear_context(&mut self) {
        self.context_tokens.clear();
        self.conversation.clear();
        self.conversation_tokens = 0;
        self.suspended_context = None;
        self.state = SessionState::Running;
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
    sessions: Arc<RwLock<HashMap<SessionId, Arc<RwLock<InferenceSession>>>>>,
    model_registry: Arc<RwLock<ModelRegistry>>,
    store: Arc<dyn SessionStore>,
    next_session_id: AtomicU64,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new() -> Self {
        Self::with_store(Arc::new(InMemorySessionStore::new()))
    }

    /// Create a new session manager with a custom persistence store.
    pub fn with_store(store: Arc<dyn SessionStore>) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            model_registry: Arc::new(RwLock::new(ModelRegistry::new())),
            store,
            next_session_id: AtomicU64::new(1),
        }
    }

    /// Create a new session manager with a SQLite persistence store.
    pub fn with_sqlite_store<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        Ok(Self::with_store(Arc::new(SqliteSessionStore::new(db_path)?)))
    }

    /// Create a new session manager by resolving a store plugin from builtin registry.
    ///
    /// Supported builtin kinds:
    /// - `memory`
    /// - `sqlite` (requires `path`)
    /// - `redis` (requires `url`, optional `prefix`) when feature `redis-store` is enabled
    pub fn with_store_plugin(kind: &str, options: HashMap<String, String>) -> Result<Self> {
        let registry = SessionStoreRegistry::with_builtin_factories();
        Self::with_store_plugin_from_registry(&registry, kind, options)
    }

    /// Create a new session manager by resolving a store plugin from a custom registry.
    pub fn with_store_plugin_from_registry(
        registry: &SessionStoreRegistry,
        kind: &str,
        options: HashMap<String, String>,
    ) -> Result<Self> {
        let config = SessionStoreConfig::new(options);
        let store = registry.create(kind, &config)?;
        Ok(Self::with_store(store))
    }

    /// Create a session manager by dynamically loading one store plugin library.
    ///
    /// This initializes a temporary registry with builtin factories, loads the
    /// dynamic factory from `library_path`, then instantiates it with `options`.
    pub fn with_dynamic_store_plugin<P: AsRef<Path>>(
        library_path: P,
        options: HashMap<String, String>,
    ) -> Result<Self> {
        let registry = SessionStoreRegistry::with_builtin_factories();
        let kind = registry.load_dynamic_factory(library_path)?;
        Self::with_store_plugin_from_registry(&registry, &kind, options)
    }

    /// List store plugin kinds available from builtin registry.
    pub fn available_store_plugins() -> Vec<String> {
        SessionStoreRegistry::with_builtin_factories().list_kinds()
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
        if self
            .sessions
            .read()
            .values()
            .any(|session| session.read().model_id == model_id)
        {
            return Err(LociError::InvalidSessionState(format!(
                "Cannot unload model {} while active sessions still reference it",
                model_id
            )));
        }
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
        let registry = self.model_registry.read();
        if !registry.has_model(model_id) {
            return Err(LociError::ModelNotFound);
        }

        let session_id = SessionId(self.next_session_id.fetch_add(1, Ordering::SeqCst));

        // Get model info to determine max context
        let max_context = registry
            .get_model_info(model_id)
            .map(|info| info.n_ctx)
            .ok_or(LociError::ModelNotFound)?;
        registry.acquire_model(model_id)?;
        drop(registry);

        let session = Arc::new(RwLock::new(InferenceSession::new(
            session_id,
            model_id,
            max_context,
        )));

        let mut sessions = self.sessions.write();
        sessions.insert(session_id, session);

        Ok(session_id)
    }

    /// Persist one active session into the configured session store.
    pub fn save_session(&self, session_id: SessionId) -> Result<()> {
        let session = self
            .sessions
            .read()
            .get(&session_id)
            .cloned()
            .ok_or(LociError::SessionNotFound)?;

        let session = session.read();
        let model_info = self
            .model_registry
            .read()
            .get_model_info(session.model_id)
            .ok_or(LociError::ModelNotFound)?;
        let snapshot = session.to_snapshot(model_info.path, model_info.n_ctx);
        drop(session);

        self.store.save(&snapshot)
    }

    /// Persist all active sessions.
    pub fn save_all_sessions(&self) -> Result<usize> {
        let session_ids: Vec<SessionId> = self.sessions.read().keys().copied().collect();
        for session_id in &session_ids {
            self.save_session(*session_id)?;
        }
        Ok(session_ids.len())
    }

    /// Restore one session from the configured session store into memory.
    pub fn restore_session(&self, session_id: SessionId) -> Result<()> {
        if self.has_session(session_id) {
            return Err(LociError::InvalidSessionState(format!(
                "Session {} is already active in memory",
                session_id
            )));
        }

        let snapshot = self
            .store
            .load(session_id)?
            .ok_or(LociError::SessionNotFound)?;
        let restored_session_id = SessionId::from(snapshot.session_id);
        if restored_session_id != session_id {
            return Err(LociError::SerializationError(format!(
                "Session snapshot id mismatch: requested {}, got {}",
                session_id, restored_session_id
            )));
        }

        let model_id = self
            .model_registry
            .read()
            .load_model(&snapshot.model_path, snapshot.model_n_ctx)?;
        let max_context = self
            .model_registry
            .read()
            .get_model_info(model_id)
            .map(|info| info.n_ctx)
            .ok_or(LociError::ModelNotFound)?;

        let session = Arc::new(RwLock::new(InferenceSession::from_snapshot(
            snapshot,
            model_id,
            max_context,
        )));

        let mut sessions = self.sessions.write();
        if sessions.contains_key(&session_id) {
            return Err(LociError::InvalidSessionState(format!(
                "Session {} became active during restore",
                session_id
            )));
        }
        sessions.insert(session_id, session);
        drop(sessions);

        self.next_session_id
            .fetch_max(session_id.as_u64().saturating_add(1), Ordering::SeqCst);
        Ok(())
    }

    /// Restore all sessions from the configured session store.
    pub fn restore_all_sessions(&self) -> Result<Vec<SessionId>> {
        let persisted_ids = self.store.list_ids()?;
        let mut restored = Vec::new();
        for session_id in persisted_ids {
            if self.has_session(session_id) {
                continue;
            }
            self.restore_session(session_id)?;
            restored.push(session_id);
        }
        Ok(restored)
    }

    /// Delete one persisted session snapshot without affecting in-memory sessions.
    pub fn delete_persisted_session(&self, session_id: SessionId) -> Result<()> {
        self.store.delete(session_id)
    }

    /// List all persisted session IDs from the configured session store.
    pub fn list_persisted_sessions(&self) -> Result<Vec<SessionId>> {
        self.store.list_ids()
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

        if let Some(session) = sessions.remove(&session_id) {
            let model_id = session.read().model_id;
            drop(sessions);
            self.model_registry.read().release_model(model_id)?;
            self.store.delete(session_id)?;
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
            .map(|session| {
                let s = session.read();
                SessionInfo {
                    session_id: s.session_id,
                    model_id: s.model_id,
                    context_length: s.context_length(),
                    max_context: s.max_context,
                    state: s.state.clone(),
                    message_count: s.conversation.len(),
                }
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
    sessions: Arc<RwLock<HashMap<SessionId, Arc<RwLock<InferenceSession>>>>>,
    model_registry: Arc<RwLock<ModelRegistry>>,
}

impl SessionHandle {
    fn resolve_session(&self) -> Result<Arc<RwLock<InferenceSession>>> {
        let sessions = self.sessions.read();
        sessions
            .get(&self.session_id)
            .cloned()
            .ok_or(LociError::SessionNotFound)
    }

    /// Generate text using this session
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let session = self.resolve_session()?;
        let mut session = session.write();

        let registry = self.model_registry.read();
        session.generate(&registry, prompt, max_tokens)
    }

    /// Clear session context
    pub fn clear_context(&self) -> Result<()> {
        let session = self.resolve_session()?;
        let mut session = session.write();

        session.clear_context();
        Ok(())
    }

    /// Suspend this session with external wait reason.
    pub fn suspend(&self, reason: String, data: Option<String>) -> Result<()> {
        let session = self.resolve_session()?;
        let mut session = session.write();
        session.suspend(reason, data);
        Ok(())
    }

    /// Resume this session with external tool/user data.
    pub fn resume(&self, external_data: String) -> Result<()> {
        let session = self.resolve_session()?;
        let mut session = session.write();
        session.resume_session(external_data)
    }

    /// Get conversation records for this session.
    pub fn records(&self) -> Result<Vec<SessionRecord>> {
        let session = self.resolve_session()?;
        let session = session.read();
        Ok(session.records())
    }

    /// Get session info
    pub fn info(&self) -> Result<SessionInfo> {
        let session = self.resolve_session()?;
        let session = session.read();

        Ok(SessionInfo {
            session_id: session.session_id,
            model_id: session.model_id,
            context_length: session.context_length(),
            max_context: session.max_context,
            state: session.state.clone(),
            message_count: session.conversation.len(),
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
    pub state: SessionState,
    pub message_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

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
        assert_eq!(SessionId::from(42).as_u64(), 42);
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

    #[test]
    fn test_available_store_plugins_contains_builtin() {
        let plugins = SessionManager::available_store_plugins();
        assert!(plugins.contains(&"memory".to_string()));
        assert!(plugins.contains(&"sqlite".to_string()));
    }

    #[test]
    fn test_manager_with_store_plugin_memory() {
        let manager = SessionManager::with_store_plugin("memory", HashMap::new()).unwrap();
        assert_eq!(manager.session_count(), 0);
        assert_eq!(manager.model_count(), 0);
    }

    #[test]
    fn test_manager_with_store_plugin_sqlite() {
        let db_path = std::env::temp_dir().join("loci-session-manager-plugin.sqlite");
        let mut options = HashMap::new();
        options.insert("path".to_string(), db_path.to_string_lossy().to_string());

        let manager = SessionManager::with_store_plugin("sqlite", options).unwrap();
        assert_eq!(manager.session_count(), 0);

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn test_manager_with_dynamic_store_plugin_missing_library() {
        let err = SessionManager::with_dynamic_store_plugin(
            "missing_session_store_factory_plugin.dll",
            HashMap::new(),
        )
        .err()
        .expect("missing dynamic store plugin should fail");
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn test_session_suspend_resume_flow() {
        let mut session = InferenceSession::new(SessionId(1), ModelId::from_u64(1), 1024);
        assert!(session.can_generate());
        assert!(!session.is_suspended());

        session.suspend("tool_call".to_string(), Some("{\"name\":\"weather\"}".to_string()));
        assert!(session.is_suspended());

        session.resume_session("sunny".to_string()).unwrap();
        assert!(matches!(session.state(), SessionState::Resuming { .. }));
    }

    #[test]
    fn test_clear_context_resets_state() {
        let mut session = InferenceSession::new(SessionId(1), ModelId::from_u64(1), 1024);
        session.suspend("pause".to_string(), None);
        session.clear_context();
        assert!(matches!(session.state(), SessionState::Running));
        assert_eq!(session.context_length(), 0);
        assert_eq!(session.records().len(), 0);
    }

    #[test]
    fn test_session_snapshot_roundtrip() {
        let mut session = InferenceSession::new(SessionId(11), ModelId::from_u64(7), 256);
        session.append_record(SessionRole::User, "hello".to_string());
        session.append_record(SessionRole::Assistant, "world".to_string());
        session.suspend("tool_call".to_string(), Some("{\"k\":\"v\"}".to_string()));

        let snapshot = session.to_snapshot("model.gguf".to_string(), 256);
        let encoded = serde_json::to_string(&snapshot).unwrap();
        let decoded: SessionSnapshot = serde_json::from_str(&encoded).unwrap();

        let restored = InferenceSession::from_snapshot(decoded, ModelId::from_u64(9), 256);
        assert_eq!(restored.id(), SessionId(11));
        assert_eq!(restored.model_id(), ModelId::from_u64(9));
        assert_eq!(restored.records().len(), 2);
        assert!(matches!(
            restored.state(),
            SessionState::AwaitingExternal { .. }
        ));
    }

    #[test]
    fn test_manager_restore_from_shared_store() {
        let store = Arc::new(InMemorySessionStore::new());

        let manager_a = SessionManager::with_store(store.clone());
        let model_id = manager_a.load_model("mock-model.gguf", 512).unwrap();
        let session_id = manager_a.create_session(model_id).unwrap();
        let handle = manager_a.get_session(session_id).unwrap();
        handle
            .suspend("external_call".to_string(), Some("{}".to_string()))
            .unwrap();
        manager_a.save_session(session_id).unwrap();

        drop(manager_a);

        let manager_b = SessionManager::with_store(store);
        let restored = manager_b.restore_all_sessions().unwrap();
        assert_eq!(restored, vec![session_id]);
        assert!(manager_b.has_session(session_id));

        let info = manager_b.get_session(session_id).unwrap().info().unwrap();
        assert!(matches!(info.state, SessionState::AwaitingExternal { .. }));

        let new_session = manager_b.create_session(info.model_id).unwrap();
        assert!(new_session.as_u64() > session_id.as_u64());
    }

    #[test]
    fn test_destroy_session_deletes_persisted_snapshot() {
        let store = Arc::new(InMemorySessionStore::new());
        let manager = SessionManager::with_store(store);

        let model_id = manager.load_model("mock-model.gguf", 512).unwrap();
        let session_id = manager.create_session(model_id).unwrap();
        manager.save_session(session_id).unwrap();
        assert_eq!(manager.list_persisted_sessions().unwrap(), vec![session_id]);

        manager.destroy_session(session_id).unwrap();
        assert!(manager.list_persisted_sessions().unwrap().is_empty());
    }
}

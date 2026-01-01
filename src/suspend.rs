//! Suspend Module
//!
//! This module provides core functionality for the Loci project.
//!


use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Instant, Duration};
use anyhow::{Result, bail};
use serde::{Serialize, Deserialize};




#[derive(Debug, Clone, PartialEq, Eq)]
    /// ControlFlow enumeration
pub enum ControlFlow {
    
    Continue,

    
    Suspend(SuspendReason),

    
    Stop(StopReason),
}


#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    /// SuspendReason enumeration
pub enum SuspendReason {
    
    ToolCall {
        
        tool_name: String,

        
        arguments: String,

        
        call_id: String,
    },

    
    HumanInput {
        
        prompt: String,

        
        expected_type: String,
    },

    
    Custom {
        
        reason: String,

        
        data: String,
    },
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// StopReason enumeration
pub enum StopReason {
    
    EndOfSequence,

    
    MaxLength,

    
    StopSequence,

    
    UserCancelled,
}




#[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// SessionState enumeration
pub enum SessionState {
    
    Idle,

    
    Running,

    
    AwaitingExternal,

    
    Resuming,

    
    Completed,

    
    Cancelled,
}

// Implementation for SessionState
impl SessionState {
    
    /// can_start function
    pub fn can_start(&self) -> bool {
        matches!(self, SessionState::Idle | SessionState::Completed | SessionState::Cancelled)
    }

    
    /// can_suspend function
    pub fn can_suspend(&self) -> bool {
        matches!(self, SessionState::Running)
    }

    
    /// can_resume function
    pub fn can_resume(&self) -> bool {
        matches!(self, SessionState::AwaitingExternal)
    }

    
    /// is_active function
    pub fn is_active(&self) -> bool {
        matches!(self, SessionState::Running | SessionState::Resuming)
    }
}




#[derive(Debug, Clone)]
    /// ResumeContext structure
pub struct ResumeContext {
    
    pub injection: String,

    
    pub injection_type: InjectionType,

    
    pub metadata: HashMap<String, String>,

    
    pub created_at: Instant,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// InjectionType enumeration
pub enum InjectionType {
    
    ToolResult,

    
    UserInput,

    
    SystemMessage,

    
    Custom,
}

// Implementation for ResumeContext
impl ResumeContext {
    
    /// tool_result function
    pub fn tool_result(result: String, tool_name: String) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("tool_name".to_string(), tool_name);

        Self {
            injection: result,
            injection_type: InjectionType::ToolResult,
            metadata,
            created_at: Instant::now(),
        }
    }

    
    /// user_input function
    pub fn user_input(input: String) -> Self {
        Self {
            injection: input,
            injection_type: InjectionType::UserInput,
            metadata: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    
    /// system_message function
    pub fn system_message(message: String) -> Self {
        Self {
            injection: message,
            injection_type: InjectionType::SystemMessage,
            metadata: HashMap::new(),
            created_at: Instant::now(),
        }
    }
}




    /// SuspendableSession structure
pub struct SuspendableSession {
    
    pub session_id: String,

    
    state: SessionState,

    
    suspend_reason: Option<SuspendReason>,

    
    generated_tokens: Vec<i32>,

    
    generated_text: String,

    
    suspended_at: Option<Instant>,

    
    resume_history: Vec<ResumeContext>,

    
    state_history: Vec<(SessionState, Instant)>,
}

// Implementation for SuspendableSession
impl SuspendableSession {
    
    /// new function
    pub fn new(session_id: String) -> Self {
        let mut state_history = Vec::new();
        state_history.push((SessionState::Idle, Instant::now()));

        Self {
            session_id,
            state: SessionState::Idle,
            suspend_reason: None,
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            suspended_at: None,
            resume_history: Vec::new(),
            state_history,
        }
    }

    
    /// start function
    pub fn start(&mut self) -> Result<()> {
        if !self.state.can_start() {
            bail!("Cannot start session in state {:?}", self.state);
        }

        self.transition_to(SessionState::Running);
        self.generated_tokens.clear();
        self.generated_text.clear();
        self.resume_history.clear();

        Ok(())
    }

    
    /// suspend function
    pub fn suspend(&mut self, reason: SuspendReason) -> Result<()> {
        if !self.state.can_suspend() {
            bail!("Cannot suspend session in state {:?}", self.state);
        }

        self.suspend_reason = Some(reason);
        self.suspended_at = Some(Instant::now());
        self.transition_to(SessionState::AwaitingExternal);

        eprintln!("🔄 Session {} suspended: {:?}", self.session_id, self.suspend_reason);

        Ok(())
    }

    
    /// resume function
    pub fn resume(&mut self, context: ResumeContext) -> Result<()> {
        if !self.state.can_resume() {
            bail!("Cannot resume session in state {:?}", self.state);
        }

        
        self.resume_history.push(context.clone());

        
        self.generated_text.push_str(&context.injection);

        
        self.transition_to(SessionState::Resuming);

        eprintln!("🔄 Session {} resumed with {} chars injection",
                 self.session_id, context.injection.len());

        
        self.transition_to(SessionState::Running);

        
        self.suspend_reason = None;
        self.suspended_at = None;

        Ok(())
    }

    
    /// complete function
    pub fn complete(&mut self, reason: StopReason) {
        self.transition_to(SessionState::Completed);
        eprintln!("✅ Session {} completed: {:?}", self.session_id, reason);
    }

    
    /// cancel function
    pub fn cancel(&mut self) {
        self.transition_to(SessionState::Cancelled);
        eprintln!("❌ Session {} cancelled", self.session_id);
    }

    
    /// add_token function
    pub fn add_token(&mut self, token_id: i32, token_text: &str) {
        self.generated_tokens.push(token_id);
        self.generated_text.push_str(token_text);
    }

    
    /// state function
    pub fn state(&self) -> SessionState {
        self.state
    }

    
    /// suspend_reason function
    pub fn suspend_reason(&self) -> Option<&SuspendReason> {
        self.suspend_reason.as_ref()
    }

    
    /// suspend_duration function
    pub fn suspend_duration(&self) -> Option<Duration> {
        self.suspended_at.map(|t| Instant::now().duration_since(t))
    }

    
    /// generated_text function
    pub fn generated_text(&self) -> &str {
        &self.generated_text
    }

    
    /// resume_count function
    pub fn resume_count(&self) -> usize {
        self.resume_history.len()
    }

    
    fn transition_to(&mut self, new_state: SessionState) {
        self.state = new_state;
        self.state_history.push((new_state, Instant::now()));
    }
}




    /// SuspendableSessionManager structure
pub struct SuspendableSessionManager {
    
    sessions: Arc<RwLock<HashMap<String, SuspendableSession>>>,

    
    suspended_sessions: Arc<RwLock<Vec<String>>>,
}

// Implementation for SuspendableSessionManager
impl SuspendableSessionManager {
    
    /// new function
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            suspended_sessions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    
    /// create_session function
    pub fn create_session(&self, session_id: String) -> Result<()> {
        let mut sessions = self.sessions.write()
            .map_err(|e| anyhow!("Failed to acquire sessions write lock: {}", e))?;

        if sessions.contains_key(&session_id) {
            bail!("Session {} already exists", session_id);
        }

        let session = SuspendableSession::new(session_id.clone());
        sessions.insert(session_id, session);

        Ok(())
    }

    
    /// start_session function
    pub fn start_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write()
            .map_err(|e| anyhow!("Failed to acquire sessions write lock: {}", e))?;
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

        session.start()
    }

    
    /// suspend_session function
    pub fn suspend_session(&self, session_id: &str, reason: SuspendReason) -> Result<()> {
        let mut sessions = self.sessions.write()
            .map_err(|e| anyhow!("Failed to acquire sessions write lock: {}", e))?;
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

        session.suspend(reason)?;

        // Track suspended session
        let mut suspended = self.suspended_sessions.write()
            .map_err(|e| anyhow!("Failed to acquire suspended_sessions write lock: {}", e))?;
        if !suspended.contains(&session_id.to_string()) {
            suspended.push(session_id.to_string());
        }

        Ok(())
    }

    
    /// resume_session function
    pub fn resume_session(&self, session_id: &str, context: ResumeContext) -> Result<()> {
        let mut sessions = self.sessions.write()
            .map_err(|e| anyhow!("Failed to acquire sessions write lock: {}", e))?;
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

        session.resume(context)?;

        // Remove from suspended list
        let mut suspended = self.suspended_sessions.write()
            .map_err(|e| anyhow!("Failed to acquire suspended_sessions write lock: {}", e))?;
        suspended.retain(|id| id != session_id);

        Ok(())
    }

    
    /// complete_session function
    pub fn complete_session(&self, session_id: &str, reason: StopReason) -> Result<()> {
            let mut sessions = self.sessions.write()
                .map_err(|e| anyhow!("Failed to acquire sessions write lock: {}", e))?;
            let session = sessions.get_mut(session_id)
                .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;
    
            session.complete(reason);
            Ok(())
        }

    
    /// get_session_state function
    pub fn get_session_state(&self, session_id: &str) -> Option<SessionState> {
        let sessions = self.sessions.read()
            .ok()
            .and_then(|s| s.get(session_id).map(|s| s.state()))
    }


    /// suspended_sessions function
    pub fn suspended_sessions(&self) -> Vec<String> {
        self.suspended_sessions.read()
            .ok()
            .map(|s| s.clone())
            .unwrap_or_default()
    }


    /// get_session_info function
    pub fn get_session_info(&self, session_id: &str) -> Option<SessionInfo> {
        let sessions = self.sessions.read()
            .ok()
            .and_then(|s| s.get(session_id).map(|s| SessionInfo {
                session_id: s.session_id.clone(),
                state: s.state,
            suspend_reason: s.suspend_reason.clone(),
            generated_length: s.generated_text.len(),
            suspend_duration: s.suspend_duration(),
            resume_count: s.resume_history.len(),
        })
    }
}

// Implementation for Default
impl Default for SuspendableSessionManager {
    fn default() -> Self {
        Self::new()
    }
}


#[derive(Debug, Clone)]
    /// SessionInfo structure
pub struct SessionInfo {
    pub session_id: String,
    pub state: SessionState,
    pub suspend_reason: Option<SuspendReason>,
    pub generated_length: usize,
    pub suspend_duration: Option<Duration>,
    pub resume_count: usize,
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_transitions() {
        assert!(SessionState::Idle.can_start());
        assert!(!SessionState::Running.can_start());
        assert!(SessionState::Running.can_suspend());
        assert!(SessionState::AwaitingExternal.can_resume());
    }

    #[test]
    fn test_suspendable_session_lifecycle() {
        let mut session = SuspendableSession::new("test-1".to_string());

        
        assert!(session.start().is_ok());
        assert_eq!(session.state(), SessionState::Running);

        
        let reason = SuspendReason::ToolCall {
            tool_name: "search".to_string(),
            arguments: r#"{"query": "test"}"#.to_string(),
            call_id: "call-1".to_string(),
        };
        assert!(session.suspend(reason).is_ok());
        assert_eq!(session.state(), SessionState::AwaitingExternal);

        
        let context = ResumeContext::tool_result(
            "Search results: ...".to_string(),
            "search".to_string(),
        );
        assert!(session.resume(context).is_ok());
        assert_eq!(session.state(), SessionState::Running);

        
        session.complete(StopReason::EndOfSequence);
        assert_eq!(session.state(), SessionState::Completed);
    }

    #[test]
    fn test_session_manager() {
        let manager = SuspendableSessionManager::new();

        // Create session
        manager.create_session("session-1".to_string())
            .expect("Failed to create session");

        // Start session
        manager.start_session("session-1")
            .expect("Failed to start session");
        assert_eq!(
            manager.get_session_state("session-1"),
            Some(SessionState::Running)
        );

        // Suspend session
        let reason = SuspendReason::HumanInput {
            prompt: "Please confirm".to_string(),
            expected_type: "yes/no".to_string(),
        };
        manager.suspend_session("session-1", reason)
            .expect("Failed to suspend session");

        let suspended = manager.suspended_sessions();
        assert_eq!(suspended.len(), 1);
        assert_eq!(suspended[0], "session-1");

        // Resume session
        let context = ResumeContext::user_input("yes".to_string());
        manager.resume_session("session-1", context)
            .expect("Failed to resume session");

        let suspended = manager.suspended_sessions();
        assert_eq!(suspended.len(), 0);
    }

    #[test]
    fn test_resume_context_creation() {
        let tool_result = ResumeContext::tool_result(
            "Result data".to_string(),
            "test_tool".to_string(),
        );
        assert_eq!(tool_result.injection_type, InjectionType::ToolResult);
        assert_eq!(
            tool_result.metadata.get("tool_name").expect("tool_name not found"),
            "test_tool"
        );

        let user_input = ResumeContext::user_input("Hello".to_string());
        assert_eq!(user_input.injection_type, InjectionType::UserInput);
    }
}

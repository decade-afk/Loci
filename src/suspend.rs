/**
 * Phase 2 Week 3: 推理挂起/恢复机制
 *
 * 特性：
 * 1. ControlFlow：挂起信号与控制流
 * 2. SessionState：状态机（Running → AwaitingExternal → Resuming）
 * 3. ResumeContext：恢复上下文（工具结果注入）
 * 4. SuspendReason：挂起原因（工具调用、人工介入等）
 * 5. 完整的 ReAct 循环支持
 *
 * 设计原则：
 * - 状态机驱动：明确的状态转换
 * - 上下文保留：挂起时保留完整推理状态
 * - 零拷贝注入：恢复时直接追加 token
 * - 插件驱动：通过插件触发挂起
 */

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Instant, Duration};
use anyhow::{Result, bail};
use serde::{Serialize, Deserialize};

// ==================== 控制流定义 ====================

/// 控制流：推理循环的执行控制
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlFlow {
    /// 继续推理（正常流程）
    Continue,

    /// 挂起推理（等待外部输入）
    Suspend(SuspendReason),

    /// 停止推理（生成结束）
    Stop(StopReason),
}

/// 挂起原因
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuspendReason {
    /// 工具调用（Function Calling）
    ToolCall {
        /// 工具名称
        tool_name: String,

        /// 工具参数（JSON）
        arguments: String,

        /// 调用 ID（用于追踪）
        call_id: String,
    },

    /// 人工介入（Human-in-the-Loop）
    HumanInput {
        /// 提示信息
        prompt: String,

        /// 期待的输入类型
        expected_type: String,
    },

    /// 自定义挂起
    Custom {
        /// 原因描述
        reason: String,

        /// 附加数据（JSON）
        data: String,
    },
}

/// 停止原因
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// EOS token
    EndOfSequence,

    /// 达到最大长度
    MaxLength,

    /// 停止序列匹配
    StopSequence,

    /// 用户取消
    UserCancelled,
}

// ==================== Session 状态机 ====================

/// Session 状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// 空闲（未开始推理）
    Idle,

    /// 运行中（正在推理）
    Running,

    /// 等待外部输入（已挂起）
    AwaitingExternal,

    /// 恢复中（正在注入外部输入）
    Resuming,

    /// 已完成（推理结束）
    Completed,

    /// 已取消
    Cancelled,
}

impl SessionState {
    /// 是否可以开始推理
    pub fn can_start(&self) -> bool {
        matches!(self, SessionState::Idle | SessionState::Completed | SessionState::Cancelled)
    }

    /// 是否可以挂起
    pub fn can_suspend(&self) -> bool {
        matches!(self, SessionState::Running)
    }

    /// 是否可以恢复
    pub fn can_resume(&self) -> bool {
        matches!(self, SessionState::AwaitingExternal)
    }

    /// 是否处于活跃状态
    pub fn is_active(&self) -> bool {
        matches!(self, SessionState::Running | SessionState::Resuming)
    }
}

// ==================== 恢复上下文 ====================

/// 恢复上下文：外部输入注入到推理流中
#[derive(Debug, Clone)]
pub struct ResumeContext {
    /// 注入的文本（工具结果、用户输入等）
    pub injection: String,

    /// 注入类型
    pub injection_type: InjectionType,

    /// 元数据（可选）
    pub metadata: HashMap<String, String>,

    /// 创建时间
    pub created_at: Instant,
}

/// 注入类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionType {
    /// 工具调用结果
    ToolResult,

    /// 用户输入
    UserInput,

    /// 系统消息
    SystemMessage,

    /// 自定义
    Custom,
}

impl ResumeContext {
    /// 创建工具结果注入
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

    /// 创建用户输入注入
    pub fn user_input(input: String) -> Self {
        Self {
            injection: input,
            injection_type: InjectionType::UserInput,
            metadata: HashMap::new(),
            created_at: Instant::now(),
        }
    }

    /// 创建系统消息注入
    pub fn system_message(message: String) -> Self {
        Self {
            injection: message,
            injection_type: InjectionType::SystemMessage,
            metadata: HashMap::new(),
            created_at: Instant::now(),
        }
    }
}

// ==================== Suspendable Session ====================

/// 可挂起的推理会话
pub struct SuspendableSession {
    /// Session ID
    pub session_id: String,

    /// 当前状态
    state: SessionState,

    /// 挂起原因（如果已挂起）
    suspend_reason: Option<SuspendReason>,

    /// 当前生成的 token 序列
    generated_tokens: Vec<i32>,

    /// 当前生成的文本
    generated_text: String,

    /// 挂起时间
    suspended_at: Option<Instant>,

    /// 恢复历史（用于调试）
    resume_history: Vec<ResumeContext>,

    /// 状态转换历史
    state_history: Vec<(SessionState, Instant)>,
}

impl SuspendableSession {
    /// 创建新的可挂起会话
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

    /// 开始推理
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

    /// 挂起推理
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

    /// 恢复推理
    pub fn resume(&mut self, context: ResumeContext) -> Result<()> {
        if !self.state.can_resume() {
            bail!("Cannot resume session in state {:?}", self.state);
        }

        // 记录恢复历史
        self.resume_history.push(context.clone());

        // 注入文本到生成序列
        self.generated_text.push_str(&context.injection);

        // 转换到恢复状态
        self.transition_to(SessionState::Resuming);

        eprintln!("🔄 Session {} resumed with {} chars injection",
                 self.session_id, context.injection.len());

        // 立即转回运行状态（实际实现中会在下一个 token 生成时转换）
        self.transition_to(SessionState::Running);

        // 清除挂起信息
        self.suspend_reason = None;
        self.suspended_at = None;

        Ok(())
    }

    /// 完成推理
    pub fn complete(&mut self, reason: StopReason) {
        self.transition_to(SessionState::Completed);
        eprintln!("✅ Session {} completed: {:?}", self.session_id, reason);
    }

    /// 取消推理
    pub fn cancel(&mut self) {
        self.transition_to(SessionState::Cancelled);
        eprintln!("❌ Session {} cancelled", self.session_id);
    }

    /// 添加生成的 token
    pub fn add_token(&mut self, token_id: i32, token_text: &str) {
        self.generated_tokens.push(token_id);
        self.generated_text.push_str(token_text);
    }

    /// 获取当前状态
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// 获取挂起原因
    pub fn suspend_reason(&self) -> Option<&SuspendReason> {
        self.suspend_reason.as_ref()
    }

    /// 获取挂起时长
    pub fn suspend_duration(&self) -> Option<Duration> {
        self.suspended_at.map(|t| Instant::now().duration_since(t))
    }

    /// 获取生成的文本
    pub fn generated_text(&self) -> &str {
        &self.generated_text
    }

    /// 获取恢复历史记录
    pub fn resume_count(&self) -> usize {
        self.resume_history.len()
    }

    /// 状态转换（内部方法）
    fn transition_to(&mut self, new_state: SessionState) {
        self.state = new_state;
        self.state_history.push((new_state, Instant::now()));
    }
}

// ==================== Session Manager（扩展） ====================

/// Session 管理器（支持挂起/恢复）
pub struct SuspendableSessionManager {
    /// 所有会话
    sessions: Arc<RwLock<HashMap<String, SuspendableSession>>>,

    /// 挂起的会话列表
    suspended_sessions: Arc<RwLock<Vec<String>>>,
}

impl SuspendableSessionManager {
    /// 创建新的管理器
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            suspended_sessions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 创建新会话
    pub fn create_session(&self, session_id: String) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();

        if sessions.contains_key(&session_id) {
            bail!("Session {} already exists", session_id);
        }

        let session = SuspendableSession::new(session_id.clone());
        sessions.insert(session_id, session);

        Ok(())
    }

    /// 开始会话
    pub fn start_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

        session.start()
    }

    /// 挂起会话
    pub fn suspend_session(&self, session_id: &str, reason: SuspendReason) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

        session.suspend(reason)?;

        // 添加到挂起列表
        let mut suspended = self.suspended_sessions.write().unwrap();
        if !suspended.contains(&session_id.to_string()) {
            suspended.push(session_id.to_string());
        }

        Ok(())
    }

    /// 恢复会话
    pub fn resume_session(&self, session_id: &str, context: ResumeContext) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

        session.resume(context)?;

        // 从挂起列表移除
        let mut suspended = self.suspended_sessions.write().unwrap();
        suspended.retain(|id| id != session_id);

        Ok(())
    }

    /// 完成会话
    pub fn complete_session(&self, session_id: &str, reason: StopReason) -> Result<()> {
        let mut sessions = self.sessions.write().unwrap();
        let session = sessions.get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {} not found", session_id))?;

        session.complete(reason);

        Ok(())
    }

    /// 获取会话状态
    pub fn get_session_state(&self, session_id: &str) -> Option<SessionState> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(session_id).map(|s| s.state())
    }

    /// 获取所有挂起的会话
    pub fn suspended_sessions(&self) -> Vec<String> {
        self.suspended_sessions.read().unwrap().clone()
    }

    /// 获取会话信息
    pub fn get_session_info(&self, session_id: &str) -> Option<SessionInfo> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(session_id).map(|s| SessionInfo {
            session_id: s.session_id.clone(),
            state: s.state,
            suspend_reason: s.suspend_reason.clone(),
            generated_length: s.generated_text.len(),
            suspend_duration: s.suspend_duration(),
            resume_count: s.resume_history.len(),
        })
    }
}

impl Default for SuspendableSessionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Session 信息（用于查询）
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: String,
    pub state: SessionState,
    pub suspend_reason: Option<SuspendReason>,
    pub generated_length: usize,
    pub suspend_duration: Option<Duration>,
    pub resume_count: usize,
}

// ==================== 测试模块 ====================

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

        // 开始
        assert!(session.start().is_ok());
        assert_eq!(session.state(), SessionState::Running);

        // 挂起
        let reason = SuspendReason::ToolCall {
            tool_name: "search".to_string(),
            arguments: r#"{"query": "test"}"#.to_string(),
            call_id: "call-1".to_string(),
        };
        assert!(session.suspend(reason).is_ok());
        assert_eq!(session.state(), SessionState::AwaitingExternal);

        // 恢复
        let context = ResumeContext::tool_result(
            "Search results: ...".to_string(),
            "search".to_string(),
        );
        assert!(session.resume(context).is_ok());
        assert_eq!(session.state(), SessionState::Running);

        // 完成
        session.complete(StopReason::EndOfSequence);
        assert_eq!(session.state(), SessionState::Completed);
    }

    #[test]
    fn test_session_manager() {
        let manager = SuspendableSessionManager::new();

        // 创建会话
        manager.create_session("session-1".to_string()).unwrap();

        // 开始会话
        manager.start_session("session-1").unwrap();
        assert_eq!(
            manager.get_session_state("session-1"),
            Some(SessionState::Running)
        );

        // 挂起会话
        let reason = SuspendReason::HumanInput {
            prompt: "Please confirm".to_string(),
            expected_type: "yes/no".to_string(),
        };
        manager.suspend_session("session-1", reason).unwrap();

        let suspended = manager.suspended_sessions();
        assert_eq!(suspended.len(), 1);
        assert_eq!(suspended[0], "session-1");

        // 恢复会话
        let context = ResumeContext::user_input("yes".to_string());
        manager.resume_session("session-1", context).unwrap();

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
        assert_eq!(tool_result.metadata.get("tool_name").unwrap(), "test_tool");

        let user_input = ResumeContext::user_input("Hello".to_string());
        assert_eq!(user_input.injection_type, InjectionType::UserInput);
    }
}

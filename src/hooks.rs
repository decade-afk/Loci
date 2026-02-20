//! Deep Programmable Hook System for Loci
//!
//! This module defines a comprehensive hook system that allows plugins to intercept
//! and modify Loci's behavior at every critical point in the inference pipeline.
//!
//! ## Design Philosophy
//!
//! Loci is designed to be **deeply programmable**. Every significant operation exposes
//! hooks that plugins can use to:
//! - Inspect internal state
//! - Modify data in-flight
//! - Make routing decisions
//! - Inject custom logic
//! - Observe and log behavior
//!
//! ## Hook Categories
//!
//! 1. **Lifecycle Hooks**: Plugin initialization, shutdown, configuration
//! 2. **Model Hooks**: Model loading, unloading, switching
//! 3. **Inference Hooks**: Pre/post processing, streaming, interruption
//! 4. **Token Hooks**: Sampling, logits manipulation, decoding
//! 5. **Memory Hooks**: KV cache operations, memory management
//! 6. **Backend Hooks**: Backend selection, initialization, fallback
//! 7. **Session Hooks**: Multi-session management, context switching
//! 8. **Error Hooks**: Error handling, recovery, logging
//!
//! ## Hook Execution Order
//!
//! Hooks are executed in **registration order** (FIFO) unless priority is specified.
//! Plugins can specify execution priority:
//! - `Critical` (highest): Security, validation
//! - `High`: Core transformations
//! - `Normal` (default): General plugins
//! - `Low`: Logging, metrics
//! - `Monitor`: Observation-only (cannot modify data)

use crate::backend::{BackendParams, InferenceParams, ModelMetadata};
use crate::error::Result;
use crate::sampler::LogitsView;
use std::path::Path;

/// Hook execution priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HookPriority {
    /// Monitor-only hooks (cannot modify data, executed last)
    Monitor = 0,
    /// Low priority hooks (logging, metrics)
    Low = 1,
    /// Normal priority (default)
    Normal = 2,
    /// High priority (core transformations)
    High = 3,
    /// Critical priority (security, validation)
    Critical = 4,
}

impl Default for HookPriority {
    fn default() -> Self {
        HookPriority::Normal
    }
}

/// Hook execution control
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookControl {
    /// Continue to next hook
    Continue,
    /// Stop hook chain and return current value
    Break,
    /// Skip remaining hooks and use original value
    SkipRemaining,
    /// Suspend inference and wait for external tool/input
    ///
    /// This allows plugins to request external data (e.g., tool calls, user input)
    /// before continuing generation. The session will enter `AwaitingExternal` state.
    Suspend {
        /// Reason for suspension (e.g., "tool_call", "user_input")
        reason: String,
        /// Data to be passed to external handler (e.g., tool call parameters)
        data: Option<String>,
    },
}

/// Context information passed to hooks
#[derive(Debug, Clone)]
pub struct HookContext {
    /// Plugin name that registered this hook
    pub plugin_name: String,
    /// Hook priority
    pub priority: HookPriority,
    /// Sequential execution number (0-indexed)
    pub execution_index: usize,
    /// Total number of hooks in chain
    pub total_hooks: usize,
    /// Read-only metadata (arbitrary key-value pairs)
    pub metadata: std::collections::HashMap<String, String>,
}

impl HookContext {
    pub fn new(plugin_name: String, priority: HookPriority) -> Self {
        Self {
            plugin_name,
            priority,
            execution_index: 0,
            total_hooks: 0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Check if this is the first hook in the chain
    pub fn is_first(&self) -> bool {
        self.execution_index == 0
    }

    /// Check if this is the last hook in the chain
    pub fn is_last(&self) -> bool {
        self.execution_index == self.total_hooks - 1
    }
}

// ============================================================================
// 1. LIFECYCLE HOOKS
// ============================================================================

/// Lifecycle hooks for plugin management
pub trait LifecycleHooks: Send + Sync {
    /// Called when plugin is first registered (before init)
    fn on_register(&mut self, _ctx: &HookContext) -> Result<()> {
        Ok(())
    }

    /// Called when plugin is initialized
    fn on_init(&mut self, _ctx: &HookContext) -> Result<()> {
        Ok(())
    }

    /// Called when plugin configuration changes
    fn on_config_change(
        &mut self,
        _ctx: &HookContext,
        _old_config: &std::collections::HashMap<String, String>,
        _new_config: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        Ok(())
    }

    /// Called before plugin is unloaded
    fn on_shutdown(&mut self, _ctx: &HookContext) -> Result<()> {
        Ok(())
    }

    /// Called when plugin is enabled/disabled
    fn on_state_change(&mut self, _ctx: &HookContext, _enabled: bool) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// 2. MODEL HOOKS
// ============================================================================

/// Model lifecycle hooks
pub trait ModelHooks: Send + Sync {
    /// Called before model loading starts
    ///
    /// Plugins can:
    /// - Validate model path
    /// - Modify backend parameters
    /// - Download missing files
    fn on_model_load_start(
        &mut self,
        _ctx: &HookContext,
        _model_path: &Path,
        _params: &mut BackendParams,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }

    /// Called after model is loaded successfully
    fn on_model_loaded(
        &mut self,
        _ctx: &HookContext,
        _metadata: &ModelMetadata,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when model loading fails
    fn on_model_load_error(
        &mut self,
        _ctx: &HookContext,
        _model_path: &Path,
        _error: &crate::error::LociError,
    ) -> Result<()> {
        Ok(())
    }

    /// Called before model is unloaded
    fn on_model_unload(&mut self, _ctx: &HookContext) -> Result<()> {
        Ok(())
    }

    /// Called when switching between models
    fn on_model_switch(
        &mut self,
        _ctx: &HookContext,
        _from_model: Option<&ModelMetadata>,
        _to_model: &ModelMetadata,
    ) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// 3. INFERENCE HOOKS
// ============================================================================

/// Inference pipeline hooks
pub trait InferenceHooks: Send + Sync {
    /// Called before inference starts (after pre_generate plugin hooks)
    ///
    /// Plugins can:
    /// - Modify final prompt
    /// - Adjust inference parameters
    /// - Route to different backends
    fn on_inference_start(
        &mut self,
        _ctx: &HookContext,
        _prompt: &mut String,
        _params: &mut InferenceParams,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }

    /// Called after each token is generated (streaming)
    fn on_token_generated(
        &mut self,
        _ctx: &HookContext,
        _token_id: i32,
        _token_text: &str,
        _position: usize,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }

    /// Called after inference completes successfully
    fn on_inference_complete(
        &mut self,
        _ctx: &HookContext,
        _prompt: &str,
        _response: &str,
        _tokens_generated: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when inference is interrupted (user stop, token limit, etc.)
    fn on_inference_interrupted(
        &mut self,
        _ctx: &HookContext,
        _reason: &str,
        _partial_response: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when inference fails with error
    fn on_inference_error(
        &mut self,
        _ctx: &HookContext,
        _prompt: &str,
        _error: &crate::error::LociError,
    ) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// 4. TOKEN & SAMPLING HOOKS
// ============================================================================

/// Token sampling and processing hooks
pub trait TokenHooks: Send + Sync {
    /// Called before logits are sampled
    ///
    /// This is the most powerful hook for controlling model output.
    /// Plugins can:
    /// - Ban tokens (set to -inf)
    /// - Boost tokens (add bias)
    /// - Implement grammar constraints
    /// - Apply custom sampling logic
    fn on_logits_pre_sample(
        &mut self,
        _ctx: &HookContext,
        _logits: &mut LogitsView,
        _context_tokens: &[i32],
        _position: usize,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }

    /// Called after token is sampled but before decoding
    fn on_token_sampled(
        &mut self,
        _ctx: &HookContext,
        _token_id: i32,
        _logits: &LogitsView,
        _position: usize,
    ) -> Result<i32> {
        Ok(_token_id)
    }

    /// Called after token is decoded to text
    fn on_token_decoded(
        &mut self,
        _ctx: &HookContext,
        _token_id: i32,
        _token_text: &str,
        _position: usize,
    ) -> Result<String> {
        Ok(_token_text.to_string())
    }

    /// Called when special token is encountered (EOS, BOS, etc.)
    fn on_special_token(
        &mut self,
        _ctx: &HookContext,
        _token_id: i32,
        _token_type: SpecialTokenType,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }
}

/// Special token types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialTokenType {
    /// End of sequence
    EOS,
    /// Beginning of sequence
    BOS,
    /// Padding
    PAD,
    /// Unknown token
    UNK,
    /// Custom special token
    Custom,
}

// ============================================================================
// 5. MEMORY & KV CACHE HOOKS
// ============================================================================

/// Memory and KV cache management hooks
pub trait MemoryHooks: Send + Sync {
    /// Called when KV cache block is allocated
    fn on_kv_block_allocate(
        &mut self,
        _ctx: &HookContext,
        _block_id: u64,
        _block_size: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when KV cache block is freed
    fn on_kv_block_free(&mut self, _ctx: &HookContext, _block_id: u64) -> Result<()> {
        Ok(())
    }

    /// Called when KV cache eviction is triggered (LRU)
    fn on_kv_eviction_start(
        &mut self,
        _ctx: &HookContext,
        _reason: &str,
        _current_usage: usize,
        _threshold: usize,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }

    /// Called after KV cache blocks are evicted
    fn on_kv_eviction_complete(
        &mut self,
        _ctx: &HookContext,
        _blocks_evicted: usize,
        _memory_freed: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when memory allocation fails
    fn on_memory_pressure(
        &mut self,
        _ctx: &HookContext,
        _requested: usize,
        _available: usize,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }

    /// Called when memory budget is updated
    fn on_memory_budget_change(
        &mut self,
        _ctx: &HookContext,
        _old_budget: usize,
        _new_budget: usize,
    ) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// 6. BACKEND HOOKS
// ============================================================================

/// Backend selection and management hooks
pub trait BackendHooks: Send + Sync {
    /// Called when backend is being selected
    ///
    /// Plugins can:
    /// - Override backend selection
    /// - Add custom backends
    /// - Implement fallback logic
    fn on_backend_select(
        &mut self,
        _ctx: &HookContext,
        _requested_backend: &str,
        _available_backends: &[String],
    ) -> Result<Option<String>> {
        Ok(None)
    }

    /// Called when backend is initialized
    fn on_backend_init(&mut self, _ctx: &HookContext, _backend_name: &str) -> Result<()> {
        Ok(())
    }

    /// Called when backend initialization fails
    fn on_backend_init_error(
        &mut self,
        _ctx: &HookContext,
        _backend_name: &str,
        _error: &crate::error::LociError,
    ) -> Result<HookControl> {
        Ok(HookControl::Continue)
    }

    /// Called when backend is shutdown
    fn on_backend_shutdown(&mut self, _ctx: &HookContext, _backend_name: &str) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// 7. SESSION HOOKS
// ============================================================================

/// Multi-session management hooks
pub trait SessionHooks: Send + Sync {
    /// Called when new session is created
    fn on_session_create(
        &mut self,
        _ctx: &HookContext,
        _session_id: &str,
        _metadata: &std::collections::HashMap<String, String>,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when switching to different session
    fn on_session_switch(
        &mut self,
        _ctx: &HookContext,
        _from_session: Option<&str>,
        _to_session: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when session is destroyed
    fn on_session_destroy(&mut self, _ctx: &HookContext, _session_id: &str) -> Result<()> {
        Ok(())
    }

    /// Called when session context is saved
    fn on_session_save(
        &mut self,
        _ctx: &HookContext,
        _session_id: &str,
        _save_path: &Path,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when session context is loaded
    fn on_session_load(
        &mut self,
        _ctx: &HookContext,
        _session_id: &str,
        _load_path: &Path,
    ) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// 8. ERROR & RECOVERY HOOKS
// ============================================================================

/// Error handling and recovery hooks
pub trait ErrorHooks: Send + Sync {
    /// Called when any error occurs in Loci
    ///
    /// Plugins can:
    /// - Log errors
    /// - Attempt recovery
    /// - Trigger fallback behavior
    /// - Transform error messages
    fn on_error(
        &mut self,
        _ctx: &HookContext,
        _error: &crate::error::LociError,
        _context: &str,
    ) -> Result<ErrorRecovery> {
        Ok(ErrorRecovery::Propagate)
    }

    /// Called when error recovery is attempted
    fn on_recovery_attempt(
        &mut self,
        _ctx: &HookContext,
        _original_error: &crate::error::LociError,
        _attempt_number: u32,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when recovery succeeds
    fn on_recovery_success(
        &mut self,
        _ctx: &HookContext,
        _original_error: &crate::error::LociError,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when recovery fails
    fn on_recovery_failure(
        &mut self,
        _ctx: &HookContext,
        _original_error: &crate::error::LociError,
        _recovery_error: &crate::error::LociError,
    ) -> Result<()> {
        Ok(())
    }
}

/// Error recovery strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorRecovery {
    /// Propagate error to caller
    Propagate,
    /// Retry operation
    Retry,
    /// Use fallback value/behavior
    Fallback,
    /// Suppress error (dangerous, use carefully)
    Suppress,
}

// ============================================================================
// 9. MASTER HOOK TRAIT
// ============================================================================

/// Master hook trait that combines all hook categories
///
/// Plugins implement this trait to receive hooks for all categories.
/// Default implementations are provided - plugins only override hooks they need.
pub trait DeepProgrammableHooks:
    LifecycleHooks
    + ModelHooks
    + InferenceHooks
    + TokenHooks
    + MemoryHooks
    + BackendHooks
    + SessionHooks
    + ErrorHooks
    + Send
    + Sync
{
    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Get hook priority
    fn priority(&self) -> HookPriority {
        HookPriority::Normal
    }

    /// Check if plugin is read-only (monitor mode)
    fn is_monitor_only(&self) -> bool {
        self.priority() == HookPriority::Monitor
    }
}

// ============================================================================
// 10. HOOK MANAGER
// ============================================================================

/// Manager for all hooks in the system
///
/// This is the central registry that:
/// - Stores all registered hooks
/// - Executes hooks in priority order
/// - Manages hook lifecycle
/// - Provides hook introspection
pub struct HookManager {
    hooks: Vec<Box<dyn DeepProgrammableHooks>>,
    enabled_hooks: std::collections::HashSet<String>,
}

impl HookManager {
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            enabled_hooks: std::collections::HashSet::new(),
        }
    }

    /// Register a new hook
    pub fn register(&mut self, hook: Box<dyn DeepProgrammableHooks>) -> Result<()> {
        let name = hook.name().to_string();

        // Check for duplicates
        if self.hooks.iter().any(|h| h.name() == name) {
            return Err(crate::error::LociError::PluginError(format!(
                "Hook '{}' already registered",
                name
            )));
        }

        self.enabled_hooks.insert(name);
        self.hooks.push(hook);

        // Sort by priority (highest first)
        self.hooks.sort_by(|a, b| b.priority().cmp(&a.priority()));

        Ok(())
    }

    /// Unregister a hook by name
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        self.hooks.retain(|h| h.name() != name);
        self.enabled_hooks.remove(name);
        Ok(())
    }

    /// Enable a hook
    pub fn enable(&mut self, name: &str) {
        self.enabled_hooks.insert(name.to_string());
    }

    /// Disable a hook
    pub fn disable(&mut self, name: &str) {
        self.enabled_hooks.remove(name);
    }

    /// Check if hook is enabled
    pub fn is_enabled(&self, name: &str) -> bool {
        self.enabled_hooks.contains(name)
    }

    /// Get all registered hooks
    pub fn list(&self) -> Vec<(&str, &str, HookPriority, bool)> {
        self.hooks
            .iter()
            .map(|h| {
                (
                    h.name(),
                    h.version(),
                    h.priority(),
                    self.is_enabled(h.name()),
                )
            })
            .collect()
    }

    /// Execute lifecycle hooks
    pub fn execute_lifecycle<F>(&mut self, executor: F) -> Result<()>
    where
        F: Fn(&mut dyn LifecycleHooks, &HookContext) -> Result<()>,
    {
        let total = self.hooks.len();
        let enabled_hooks = self.enabled_hooks.clone();

        for (idx, hook) in self.hooks.iter_mut().enumerate() {
            if !enabled_hooks.contains(hook.name()) {
                continue;
            }

            let mut ctx = HookContext::new(hook.name().to_string(), hook.priority());
            ctx.execution_index = idx;
            ctx.total_hooks = total;

            executor(hook.as_mut(), &ctx)?;
        }
        Ok(())
    }

    /// Execute model hooks
    ///
    /// Returns `Ok(HookControl::Continue)` if all hooks complete normally,
    /// or the first `Suspend`/`Break`/`SkipRemaining` control signal encountered.
    pub fn execute_model<F>(&mut self, executor: F) -> Result<HookControl>
    where
        F: Fn(&mut dyn ModelHooks, &HookContext) -> Result<HookControl>,
    {
        let total = self.hooks.len();
        let enabled_hooks = self.enabled_hooks.clone();

        for (idx, hook) in self.hooks.iter_mut().enumerate() {
            if !enabled_hooks.contains(hook.name()) {
                continue;
            }

            let mut ctx = HookContext::new(hook.name().to_string(), hook.priority());
            ctx.execution_index = idx;
            ctx.total_hooks = total;

            match executor(hook.as_mut(), &ctx)? {
                HookControl::Break => return Ok(HookControl::Break),
                HookControl::SkipRemaining => return Ok(HookControl::SkipRemaining),
                HookControl::Continue => continue,
                suspend @ HookControl::Suspend { .. } => {
                    // Return suspend signal to caller so they can handle it
                    return Ok(suspend);
                }
            }
        }
        Ok(HookControl::Continue)
    }

    /// Get hook count
    pub fn count(&self) -> usize {
        self.hooks.len()
    }

    /// Get enabled hook count
    pub fn count_enabled(&self) -> usize {
        self.enabled_hooks.len()
    }
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHook {
        name: String,
        priority: HookPriority,
    }

    impl TestHook {
        fn new(name: &str, priority: HookPriority) -> Self {
            Self {
                name: name.to_string(),
                priority,
            }
        }
    }

    impl LifecycleHooks for TestHook {}
    impl ModelHooks for TestHook {}
    impl InferenceHooks for TestHook {}
    impl TokenHooks for TestHook {}
    impl MemoryHooks for TestHook {}
    impl BackendHooks for TestHook {}
    impl SessionHooks for TestHook {}
    impl ErrorHooks for TestHook {}

    impl DeepProgrammableHooks for TestHook {
        fn name(&self) -> &str {
            &self.name
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn priority(&self) -> HookPriority {
            self.priority
        }
    }

    #[test]
    fn test_hook_priority_ordering() {
        let mut manager = HookManager::new();

        manager
            .register(Box::new(TestHook::new("low", HookPriority::Low)))
            .unwrap();
        manager
            .register(Box::new(TestHook::new("critical", HookPriority::Critical)))
            .unwrap();
        manager
            .register(Box::new(TestHook::new("normal", HookPriority::Normal)))
            .unwrap();

        let hooks = manager.list();
        assert_eq!(hooks[0].0, "critical"); // Highest priority first
        assert_eq!(hooks[1].0, "normal");
        assert_eq!(hooks[2].0, "low");
    }

    #[test]
    fn test_hook_enable_disable() {
        let mut manager = HookManager::new();
        manager
            .register(Box::new(TestHook::new("test", HookPriority::Normal)))
            .unwrap();

        assert!(manager.is_enabled("test"));

        manager.disable("test");
        assert!(!manager.is_enabled("test"));

        manager.enable("test");
        assert!(manager.is_enabled("test"));
    }
}

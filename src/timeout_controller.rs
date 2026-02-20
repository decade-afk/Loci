//! Timeout control for inference operations
//!
//! This module provides timeout mechanisms for long-running inference:
//! - Per-operation timeout control
//! - Graceful cancellation
//! - Timeout statistics

use crate::error::{LociError, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Timeout configuration
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Default timeout for inference operations
    pub default_timeout: Duration,
    /// Minimum allowed timeout
    pub min_timeout: Duration,
    /// Maximum allowed timeout
    pub max_timeout: Duration,
    /// Enable timeout monitoring
    pub enabled: bool,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(60),
            min_timeout: Duration::from_millis(100),
            max_timeout: Duration::from_secs(300),
            enabled: true,
        }
    }
}

impl TimeoutConfig {
    /// Create new timeout configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default timeout
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Set minimum timeout
    pub fn with_min_timeout(mut self, timeout: Duration) -> Self {
        self.min_timeout = timeout;
        self
    }

    /// Set maximum timeout
    pub fn with_max_timeout(mut self, timeout: Duration) -> Self {
        self.max_timeout = timeout;
        self
    }

    /// Enable/disable timeout
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Validate and clamp timeout to configured limits
    pub fn clamp_timeout(&self, timeout: Duration) -> Duration {
        if timeout < self.min_timeout {
            self.min_timeout
        } else if timeout > self.max_timeout {
            self.max_timeout
        } else {
            timeout
        }
    }
}

/// Timeout statistics
#[derive(Debug, Clone, Default)]
pub struct TimeoutStats {
    /// Total operations timed out
    pub timeouts: u64,
    /// Total operations completed
    pub completed: u64,
    /// Total operations cancelled
    pub cancelled: u64,
    /// Average completion time (milliseconds)
    pub avg_completion_ms: f64,
}

impl TimeoutStats {
    /// Calculate timeout rate
    pub fn timeout_rate(&self) -> f64 {
        let total = self.timeouts + self.completed;
        if total == 0 {
            0.0
        } else {
            (self.timeouts as f64) / (total as f64)
        }
    }
}

/// Timeout controller for managing operation timeouts
pub struct TimeoutController {
    config: TimeoutConfig,
    stats: Arc<parking_lot::Mutex<TimeoutStats>>,
    completion_times: Arc<parking_lot::Mutex<Vec<u64>>>,
}

impl TimeoutController {
    /// Create a new timeout controller with default configuration
    pub fn new() -> Self {
        Self::with_config(TimeoutConfig::default())
    }

    /// Create a new timeout controller with custom configuration
    pub fn with_config(config: TimeoutConfig) -> Self {
        Self {
            config,
            stats: Arc::new(parking_lot::Mutex::new(TimeoutStats::default())),
            completion_times: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
    }

    /// Set timeout configuration
    pub fn set_config(&mut self, config: TimeoutConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &TimeoutConfig {
        &self.config
    }

    /// Create a new timeout context for an operation
    pub fn create_context(&self, timeout: Option<Duration>) -> Result<TimeoutContext> {
        if !self.config.enabled {
            return Ok(TimeoutContext::disabled());
        }

        let timeout = timeout.unwrap_or(self.config.default_timeout);
        let timeout = self.config.clamp_timeout(timeout);

        Ok(TimeoutContext::new(timeout))
    }

    /// Record a successful operation completion
    pub fn record_completion(&self, duration_ms: u64) {
        let mut stats = self.stats.lock();
        stats.completed += 1;

        let mut times = self.completion_times.lock();
        times.push(duration_ms);
        
        // Keep only last 100 completion times
        if times.len() > 100 {
            times.remove(0);
        }

        // Update average
        stats.avg_completion_ms = times.iter().map(|&t| t as f64).sum::<f64>() / times.len() as f64;
    }

    /// Record a timeout
    pub fn record_timeout(&self) {
        let mut stats = self.stats.lock();
        stats.timeouts += 1;
    }

    /// Record a cancellation
    pub fn record_cancellation(&self) {
        let mut stats = self.stats.lock();
        stats.cancelled += 1;
    }

    /// Get timeout statistics
    pub fn stats(&self) -> TimeoutStats {
        self.stats.lock().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock();
        *stats = TimeoutStats::default();
        self.completion_times.lock().clear();
    }

    /// Check if timeout is likely based on history
    pub fn is_likely_timeout(&self, estimated_duration: Duration) -> bool {
        let stats = self.stats.lock();
        let avg_ms = stats.avg_completion_ms;
        
        // If average completion time is much higher than estimated, 
        // timeout is likely
        avg_ms > 0.0 && estimated_duration.as_millis() as f64 < avg_ms * 0.5
    }

    /// Get recommended timeout based on history
    pub fn recommended_timeout(&self) -> Duration {
        let stats = self.stats.lock();
        let avg_ms = stats.avg_completion_ms;
        
        if avg_ms > 0.0 {
            // Add 50% buffer
            let recommended = avg_ms * 1.5;
            Duration::from_millis(recommended as u64)
        } else {
            self.config.default_timeout
        }
    }
}

impl Default for TimeoutController {
    fn default() -> Self {
        Self::new()
    }
}

/// Timeout context for a single operation
pub struct TimeoutContext {
    deadline: Option<Instant>,
    cancelled: Arc<AtomicBool>,
}

impl TimeoutContext {
    /// Create a new timeout context with a deadline
    pub fn new(timeout: Duration) -> Self {
        Self {
            deadline: Some(Instant::now() + timeout),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a disabled timeout context (no timeout)
    pub fn disabled() -> Self {
        Self {
            deadline: None,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Check if the operation has timed out
    pub fn is_timeout(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Instant::now() > deadline
        } else {
            false
        }
    }

    /// Check if the operation has been cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Cancel the operation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Get remaining time until timeout
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// Get the deadline
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Check if should continue (not timed out and not cancelled)
    pub fn should_continue(&self) -> bool {
        !self.is_timeout() && !self.is_cancelled()
    }

    /// Create a cancellation handle
    pub fn cancellation_handle(&self) -> CancellationHandle {
        CancellationHandle {
            cancelled: self.cancelled.clone(),
        }
    }

    /// Convert to a result based on current state
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(LociError::Timeout("Operation cancelled".to_string()));
        }
        
        if self.is_timeout() {
            return Err(LociError::Timeout("Operation timed out".to_string()));
        }

        Ok(())
    }
}

/// Handle for cancelling a timeout context
#[derive(Clone)]
pub struct CancellationHandle {
    cancelled: Arc<AtomicBool>,
}

impl CancellationHandle {
    /// Cancel the associated operation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

/// RAII guard that records completion when dropped
pub struct TimeoutGuard<'a> {
    controller: &'a TimeoutController,
    start: Instant,
}

impl<'a> TimeoutGuard<'a> {
    /// Create a new timeout guard
    pub fn new(controller: &'a TimeoutController) -> Self {
        Self {
            controller,
            start: Instant::now(),
        }
    }

    /// Mark operation as successful
    pub fn mark_success(self) {
        drop(self);
    }

    /// Mark operation as timed out
    pub fn mark_timeout(self) {
        self.controller.record_timeout();
    }

    /// Mark operation as cancelled
    pub fn mark_cancelled(self) {
        self.controller.record_cancellation();
    }
}

impl<'a> Drop for TimeoutGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let duration_ms = duration.as_millis() as u64;
        self.controller.record_completion(duration_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timeout_context() {
        let ctx = TimeoutContext::new(Duration::from_millis(100));
        assert!(!ctx.is_timeout());
        
        thread::sleep(Duration::from_millis(150));
        assert!(ctx.is_timeout());
    }

    #[test]
    fn test_timeout_disabled() {
        let ctx = TimeoutContext::disabled();
        assert!(!ctx.is_timeout());
        assert!(ctx.remaining().is_none());
    }

    #[test]
    fn test_cancellation() {
        let ctx = TimeoutContext::new(Duration::from_secs(60));
        let handle = ctx.cancellation_handle();
        
        assert!(!ctx.is_cancelled());
        handle.cancel();
        assert!(ctx.is_cancelled());
    }

    #[test]
    fn test_should_continue() {
        let ctx = TimeoutContext::new(Duration::from_secs(60));
        assert!(ctx.should_continue());
        
        ctx.cancel();
        assert!(!ctx.should_continue());
    }

    #[test]
    fn test_remaining_time() {
        let ctx = TimeoutContext::new(Duration::from_millis(100));
        let remaining = ctx.remaining().unwrap();
        assert!(remaining.as_millis() > 0);
    }

    #[test]
    fn test_timeout_config_clamp() {
        let config = TimeoutConfig::new()
            .with_min_timeout(Duration::from_millis(100))
            .with_max_timeout(Duration::from_millis(500));

        // Too short - should clamp to min
        let clamped = config.clamp_timeout(Duration::from_millis(50));
        assert_eq!(clamped, Duration::from_millis(100));

        // Too long - should clamp to max
        let clamped = config.clamp_timeout(Duration::from_millis(1000));
        assert_eq!(clamped, Duration::from_millis(500));

        // Within range - no change
        let clamped = config.clamp_timeout(Duration::from_millis(300));
        assert_eq!(clamped, Duration::from_millis(300));
    }

    #[test]
    fn test_timeout_stats() {
        let controller = TimeoutController::new();
        
        controller.record_completion(100);
        controller.record_completion(200);
        
        let stats = controller.stats();
        assert_eq!(stats.completed, 2);
        assert_eq!(stats.timeouts, 0);
        assert!((stats.avg_completion_ms - 150.0).abs() < 0.1);
    }

    #[test]
    fn test_timeout_guard() {
        let controller = TimeoutController::new();
        
        {
            let _guard = TimeoutGuard::new(&controller);
            // Guard records completion when dropped
        }
        
        let stats = controller.stats();
        assert_eq!(stats.completed, 1);
    }
}
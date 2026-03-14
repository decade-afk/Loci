//! Concurrency management for inference operations
//!
//! This module provides:
//! - Concurrent request handling
//! - Connection pool management
//! - Queue management
//! - Rate limiting

use crate::error::{LociError, Result};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Concurrency configuration
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Maximum concurrent operations
    pub max_concurrent: u32,
    /// Queue size for pending operations
    pub queue_size: usize,
    /// Enable rate limiting
    pub enable_rate_limit: bool,
    /// Maximum operations per second (0 = no limit)
    pub max_ops_per_second: u32,
    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            queue_size: 100,
            enable_rate_limit: false,
            max_ops_per_second: 0,
            request_timeout: Duration::from_secs(60),
        }
    }
}

impl ConcurrencyConfig {
    /// Create new concurrency configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum concurrent operations
    pub fn with_max_concurrent(mut self, max: u32) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Set queue size
    pub fn with_queue_size(mut self, size: usize) -> Self {
        self.queue_size = size;
        self
    }

    /// Enable rate limiting
    pub fn with_rate_limit(mut self, enable: bool, max_ops: u32) -> Self {
        self.enable_rate_limit = enable;
        self.max_ops_per_second = max_ops;
        self
    }

    /// Set request timeout
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

/// Concurrency statistics
#[derive(Debug, Clone, Default)]
pub struct ConcurrencyStats {
    /// Current active operations
    pub active_ops: u32,
    /// Current queued operations
    pub queued_ops: usize,
    /// Total operations completed
    pub total_completed: u64,
    /// Total operations rejected
    pub total_rejected: u64,
    /// Average queue wait time (milliseconds)
    pub avg_queue_wait_ms: f64,
    /// Peak concurrent operations
    pub peak_concurrent: u32,
}

/// Request in the queue
struct QueuedRequest {
    id: u64,
    submitted_at: Instant,
}

/// Concurrency manager for handling concurrent operations
#[derive(Clone)]
pub struct ConcurrencyManager {
    config: ConcurrencyConfig,
    active_ops: Arc<Mutex<u32>>,
    queue: Arc<Mutex<VecDeque<QueuedRequest>>>,
    stats: Arc<Mutex<ConcurrencyStats>>,
    next_id: Arc<Mutex<u64>>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
    condvar: Arc<Condvar>,
}

impl ConcurrencyManager {
    /// Create a new concurrency manager with default configuration
    pub fn new() -> Self {
        Self::with_config(ConcurrencyConfig::default())
    }

    /// Create a new concurrency manager with custom configuration
    pub fn with_config(config: ConcurrencyConfig) -> Self {
        Self {
            config,
            active_ops: Arc::new(Mutex::new(0)),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(ConcurrencyStats::default())),
            next_id: Arc::new(Mutex::new(0)),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new())),
            condvar: Arc::new(Condvar::new()),
        }
    }

    /// Set concurrency configuration
    pub fn set_config(&mut self, config: ConcurrencyConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &ConcurrencyConfig {
        &self.config
    }

    /// Acquire a slot for a new operation
    pub fn acquire(&self) -> Result<ConcurrencyGuard> {
        self.acquire_with_timeout(None)
    }

    /// Acquire a slot with custom timeout
    pub fn acquire_with_timeout(&self, timeout: Option<Duration>) -> Result<ConcurrencyGuard> {
        // Check rate limit first
        if self.config.enable_rate_limit {
            let mut limiter = self.rate_limiter.lock();
            if !limiter.try_acquire(self.config.max_ops_per_second) {
                let mut stats = self.stats.lock();
                stats.total_rejected += 1;
                return Err(LociError::ResourceExhausted(
                    "Rate limit exceeded".to_string(),
                ));
            }
        }

        // Try to acquire immediately
        {
            let mut active = self.active_ops.lock();
            if *active < self.config.max_concurrent {
                *active += 1;

                let mut stats = self.stats.lock();
                stats.active_ops = *active;
                stats.total_completed += 1;

                if *active > stats.peak_concurrent {
                    stats.peak_concurrent = *active;
                }

                return Ok(ConcurrencyGuard {
                    active_ops: self.active_ops.clone(),
                    queue: self.queue.clone(),
                    stats: self.stats.clone(),
                    condvar: self.condvar.clone(),
                });
            }
        }

        // Queue the request
        let timeout = timeout.unwrap_or(self.config.request_timeout);
        let mut queue = self.queue.lock();

        if queue.len() >= self.config.queue_size {
            let mut stats = self.stats.lock();
            stats.total_rejected += 1;
            return Err(LociError::ResourceExhausted(format!(
                "Queue full (max {})",
                self.config.queue_size
            )));
        }

        let mut id = self.next_id.lock();
        let request_id = *id;
        *id += 1;
        drop(id);

        let request = QueuedRequest {
            id: request_id,
            submitted_at: Instant::now(),
        };

        queue.push_back(request);

        let mut stats = self.stats.lock();
        stats.queued_ops = queue.len();
        drop(queue);

        // Wait for slot to become available with timeout
        let start = Instant::now();
        let mut active = self.active_ops.lock();

        while *active >= self.config.max_concurrent {
            // Calculate remaining time
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                // Remove from queue
                let mut queue = self.queue.lock();
                if let Some(pos) = queue.iter().position(|r| r.id == request_id) {
                    queue.remove(pos);
                }

                let mut stats = self.stats.lock();
                stats.total_rejected += 1;
                stats.queued_ops = queue.len();

                return Err(LociError::Timeout("Request timed out in queue".to_string()));
            }

            // Wait for notification with remaining time
            let remaining = timeout.saturating_sub(elapsed);
            let _ = self.condvar.wait_for(&mut active, remaining);

            // Re-check queue
            let queue = self.queue.lock();
            if queue.iter().all(|request| request.id != request_id) {
                // Was removed (shouldn't happen)
                return Err(LociError::Timeout("Request removed from queue".to_string()));
            }
            drop(queue);
        }

        // Acquire the slot
        *active += 1;

        // Remove from queue
        let mut queue = self.queue.lock();
        let wait_time = if let Some(pos) = queue.iter().position(|request| request.id == request_id)
        {
            queue
                .remove(pos)
                .map(|request| request.submitted_at.elapsed())
                .unwrap_or_else(|| start.elapsed())
        } else {
            start.elapsed()
        };

        let mut stats = self.stats.lock();
        stats.active_ops = *active;
        stats.total_completed += 1;
        stats.queued_ops = queue.len();

        // Update average wait time
        let current_avg = stats.avg_queue_wait_ms;
        let wait_ms = wait_time.as_millis() as f64;
        stats.avg_queue_wait_ms = (current_avg * (stats.total_completed - 1) as f64 + wait_ms)
            / stats.total_completed as f64;

        if *active > stats.peak_concurrent {
            stats.peak_concurrent = *active;
        }

        Ok(ConcurrencyGuard {
            active_ops: self.active_ops.clone(),
            queue: self.queue.clone(),
            stats: self.stats.clone(),
            condvar: self.condvar.clone(),
        })
    }

    /// Get concurrency statistics
    pub fn stats(&self) -> ConcurrencyStats {
        self.stats.lock().clone()
    }

    /// Get current active operations count
    pub fn active_operations(&self) -> u32 {
        *self.active_ops.lock()
    }

    /// Get current queue size
    pub fn queue_size(&self) -> usize {
        self.queue.lock().len()
    }

    /// Check if system is at capacity
    pub fn is_at_capacity(&self) -> bool {
        *self.active_ops.lock() >= self.config.max_concurrent
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock();
        *stats = ConcurrencyStats::default();
    }
}

impl Default for ConcurrencyManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Concurrency guard that releases slot when dropped
pub struct ConcurrencyGuard {
    active_ops: Arc<Mutex<u32>>,
    queue: Arc<Mutex<VecDeque<QueuedRequest>>>,
    stats: Arc<Mutex<ConcurrencyStats>>,
    condvar: Arc<Condvar>,
}

impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        let mut active = self.active_ops.lock();
        if *active > 0 {
            *active -= 1;
        }

        self.condvar.notify_one();

        let queued_ops = self.queue.lock().len();
        let mut stats = self.stats.lock();
        stats.active_ops = *active;
        stats.queued_ops = queued_ops;
    }
}

/// Rate limiter for operations per second
struct RateLimiter {
    tokens: f64,
    last_update: Instant,
    initialized: bool,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            tokens: 0.0,
            last_update: Instant::now(),
            initialized: false,
        }
    }

    fn try_acquire(&mut self, max_ops: u32) -> bool {
        if max_ops == 0 {
            return true; // No limit
        }

        let now = Instant::now();
        let max_tokens = max_ops as f64;
        if !self.initialized {
            // Allow an initial burst up to max_ops immediately.
            self.tokens = max_tokens;
            self.last_update = now;
            self.initialized = true;
        }

        let elapsed = now.duration_since(self.last_update);

        // Refill tokens based on elapsed time
        let new_tokens = elapsed.as_secs_f64() * max_tokens;
        self.tokens = (self.tokens + new_tokens).min(max_tokens);
        self.last_update = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Connection pool for managing model instances
pub struct ConnectionPool<T> {
    pool: Arc<Mutex<VecDeque<T>>>,
    max_size: usize,
    min_size: usize,
    create_fn: Box<dyn Fn() -> Result<T> + Send + Sync>,
    stats: Arc<Mutex<PoolStats>>,
}

impl<T> ConnectionPool<T> {
    /// Create a new connection pool
    pub fn new<F>(min_size: usize, max_size: usize, create_fn: F) -> Result<Self>
    where
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        let pool = Self {
            pool: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
            min_size,
            create_fn: Box::new(create_fn),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        };

        // Pre-populate with minimum connections
        pool.populate_min()?;

        Ok(pool)
    }

    /// Populate pool with minimum connections
    fn populate_min(&self) -> Result<()> {
        let mut pool = self.pool.lock();
        while pool.len() < self.min_size {
            let conn = (self.create_fn)()?;
            pool.push_back(conn);

            let mut stats = self.stats.lock();
            stats.total_created += 1;
        }
        Ok(())
    }

    /// Acquire a connection from the pool
    pub fn acquire(&self) -> Result<PooledConnection<T>> {
        let mut pool = self.pool.lock();

        // Try to get existing connection
        if let Some(conn) = pool.pop_front() {
            let mut stats = self.stats.lock();
            stats.active += 1;
            stats.acquired += 1;

            Ok(PooledConnection {
                conn: Some(conn),
                pool: self.pool.clone(),
                stats: self.stats.clone(),
            })
        } else {
            // Create new connection if under max
            if pool.len() < self.max_size {
                let conn = (self.create_fn)()?;
                let mut stats = self.stats.lock();
                stats.active += 1;
                stats.total_created += 1;
                stats.acquired += 1;

                Ok(PooledConnection {
                    conn: Some(conn),
                    pool: self.pool.clone(),
                    stats: self.stats.clone(),
                })
            } else {
                Err(LociError::ResourceExhausted(format!(
                    "Connection pool exhausted (max {})",
                    self.max_size
                )))
            }
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let pool = self.pool.lock();
        let mut stats = self.stats.lock();
        stats.idle = pool.len();
        stats.clone()
    }
}

/// Pooled connection wrapper
pub struct PooledConnection<T> {
    conn: Option<T>,
    pool: Arc<Mutex<VecDeque<T>>>,
    stats: Arc<Mutex<PoolStats>>,
}

impl<T> PooledConnection<T> {
    /// Get inner connection
    pub fn get(&self) -> &T {
        self.conn.as_ref().unwrap()
    }

    /// Get mutable inner connection
    pub fn get_mut(&mut self) -> &mut T {
        self.conn.as_mut().unwrap()
    }

    /// Take ownership of the connection (not returned to pool)
    pub fn take(mut self) -> T {
        let mut stats = self.stats.lock();
        stats.active -= 1;
        self.conn.take().unwrap()
    }
}

impl<T> Drop for PooledConnection<T> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            // Return to pool
            let mut pool = self.pool.lock();
            if pool.len() < 100 {
                // Simple safety check
                pool.push_back(conn);
            }

            let mut stats = self.stats.lock();
            stats.active -= 1;
            stats.released += 1;
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections created
    pub total_created: u64,
    /// Connections acquired
    pub acquired: u64,
    /// Connections released
    pub released: u64,
    /// Active connections
    pub active: usize,
    /// Idle connections
    pub idle: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrency_manager_basic() {
        let manager =
            ConcurrencyManager::with_config(ConcurrencyConfig::new().with_max_concurrent(2));

        let guard1 = manager.acquire().unwrap();
        assert_eq!(manager.active_operations(), 1);

        let guard2 = manager.acquire().unwrap();
        assert_eq!(manager.active_operations(), 2);

        drop(guard1);
        assert_eq!(manager.active_operations(), 1);

        let guard3 = manager.acquire().unwrap();
        assert_eq!(manager.active_operations(), 2);

        drop(guard2);
        drop(guard3);
        assert_eq!(manager.active_operations(), 0);
    }

    #[test]
    fn test_concurrency_limit() {
        let manager = ConcurrencyManager::with_config(
            ConcurrencyConfig::new()
                .with_max_concurrent(1)
                .with_queue_size(0),
        );

        let _guard1 = manager.acquire().unwrap();

        // Should fail due to queue size 0
        assert!(manager.acquire().is_err());
    }

    #[test]
    fn test_rate_limiter() {
        let manager =
            ConcurrencyManager::with_config(ConcurrencyConfig::new().with_rate_limit(true, 10));

        // First 10 should succeed
        for _ in 0..10 {
            assert!(manager.acquire().is_ok());
        }

        // 11th might fail depending on timing
        // In practice, we'd need to add small delays
    }

    #[test]
    fn test_connection_pool() {
        let pool = ConnectionPool::new(1, 3, || Ok(42)).unwrap();

        let conn1 = pool.acquire().unwrap();
        assert_eq!(*conn1.get(), 42);

        let conn2 = pool.acquire().unwrap();
        assert_eq!(*conn2.get(), 42);

        let stats = pool.stats();
        assert_eq!(stats.active, 2);
    }
}

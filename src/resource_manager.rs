//! Resource management and monitoring
//!
//! This module provides:
//! - Memory monitoring and tracking
//! - CPU usage monitoring
//! - Resource limits and quotas
//! - Automatic resource cleanup

use crate::error::{LociError, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memory allocated (bytes)
    pub total_allocated: u64,
    /// Current memory in use (bytes)
    pub current_usage: u64,
    /// Peak memory usage (bytes)
    pub peak_usage: u64,
    /// Memory usage percentage (0-100)
    pub usage_percent: f32,
    /// Available memory (bytes)
    pub available: u64,
}

/// CPU usage statistics
#[derive(Debug, Clone, Default)]
pub struct CpuStats {
    /// Current CPU usage percentage (0-100)
    pub usage_percent: f32,
    /// Number of threads in use
    pub active_threads: u32,
    /// Total threads available
    pub total_threads: u32,
}

/// Resource usage statistics
#[derive(Debug, Clone, Default)]
pub struct ResourceStats {
    /// Memory statistics
    pub memory: MemoryStats,
    /// CPU statistics
    pub cpu: CpuStats,
    /// Timestamp of measurement
    pub timestamp: Instant,
}

/// Resource limits configuration
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes (0 = no limit)
    pub max_memory_bytes: u64,
    /// Maximum memory percentage (0-100, 0 = no limit)
    pub max_memory_percent: f32,
    /// Maximum CPU usage percentage (0-100, 0 = no limit)
    pub max_cpu_percent: f32,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
    /// Grace period before enforcement (milliseconds)
    pub grace_period_ms: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 0,
            max_memory_percent: 80.0,
            max_cpu_percent: 90.0,
            max_concurrent_ops: 4,
            grace_period_ms: 1000,
        }
    }
}

impl ResourceLimits {
    /// Create new resource limits
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum memory bytes
    pub fn with_max_memory_bytes(mut self, bytes: u64) -> Self {
        self.max_memory_bytes = bytes;
        self
    }

    /// Set maximum memory percentage
    pub fn with_max_memory_percent(mut self, percent: f32) -> Self {
        self.max_memory_percent = percent;
        self
    }

    /// Set maximum CPU percentage
    pub fn with_max_cpu_percent(mut self, percent: f32) -> Self {
        self.max_cpu_percent = percent;
        self
    }

    /// Set maximum concurrent operations
    pub fn with_max_concurrent_ops(mut self, ops: u32) -> Self {
        self.max_concurrent_ops = ops;
        self
    }

    /// Set grace period
    pub fn with_grace_period(mut self, ms: u64) -> Self {
        self.grace_period_ms = ms;
        self
    }
}

/// Resource monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Update interval for statistics
    pub update_interval: Duration,
    /// Enable automatic monitoring
    pub enabled: bool,
    /// Track memory usage
    pub track_memory: bool,
    /// Track CPU usage
    pub track_cpu: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(5),
            enabled: true,
            track_memory: true,
            track_cpu: true,
        }
    }
}

/// Resource manager for tracking and limiting resource usage
pub struct ResourceManager {
    limits: ResourceLimits,
    config: MonitorConfig,
    stats: Arc<Mutex<ResourceStats>>,
    current_ops: Arc<Mutex<u32>>,
    peak_memory: Arc<Mutex<u64>>,
    allocations: Arc<Mutex<HashMap<String, u64>>>,
    last_update: Arc<Mutex<Instant>>,
}

impl ResourceManager {
    /// Create a new resource manager with default configuration
    pub fn new() -> Self {
        Self::with_limits(ResourceLimits::default())
    }

    /// Create a new resource manager with custom limits
    pub fn with_limits(limits: ResourceLimits) -> Self {
        Self {
            limits,
            config: MonitorConfig::default(),
            stats: Arc::new(Mutex::new(ResourceStats::default())),
            current_ops: Arc::new(Mutex::new(0)),
            peak_memory: Arc::new(Mutex::new(0)),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            last_update: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Set resource limits
    pub fn set_limits(&mut self, limits: ResourceLimits) {
        self.limits = limits;
    }

    /// Get current resource limits
    pub fn limits(&self) -> &ResourceLimits {
        &self.limits
    }

    /// Set monitoring configuration
    pub fn set_monitor_config(&mut self, config: MonitorConfig) {
        self.config = config;
    }

    /// Acquire a resource slot for an operation
    pub fn acquire(&self) -> Result<ResourceGuard> {
        let mut ops = self.current_ops.lock();
        
        if *ops >= self.limits.max_concurrent_ops {
            return Err(LociError::ResourceExhausted(
                format!("Maximum concurrent operations ({}) reached", self.limits.max_concurrent_ops)
            ));
        }

        *ops += 1;
        drop(ops);

        // Check resource limits
        self.check_limits()?;

        Ok(ResourceGuard {
            ops: self.current_ops.clone(),
        })
    }

    /// Check if current usage exceeds limits
    fn check_limits(&self) -> Result<()> {
        let stats = self.get_stats();

        // Check memory limits
        if self.limits.max_memory_bytes > 0 && stats.memory.current_usage > self.limits.max_memory_bytes {
            return Err(LociError::OutOfMemory(
                format!("Memory usage {} exceeds limit {}", 
                    stats.memory.current_usage, self.limits.max_memory_bytes)
            ));
        }

        if self.limits.max_memory_percent > 0.0 && stats.memory.usage_percent > self.limits.max_memory_percent {
            return Err(LociError::OutOfMemory(
                format!("Memory usage {:.1}% exceeds limit {:.1}%", 
                    stats.memory.usage_percent, self.limits.max_memory_percent)
            ));
        }

        // Check CPU limits
        if self.limits.max_cpu_percent > 0.0 && stats.cpu.usage_percent > self.limits.max_cpu_percent {
            return Err(LociError::ResourceExhausted(
                format!("CPU usage {:.1}% exceeds limit {:.1}%", 
                    stats.cpu.usage_percent, self.limits.max_cpu_percent)
            ));
        }

        Ok(())
    }

    /// Register a memory allocation
    pub fn register_allocation(&self, name: String, size: u64) {
        let mut allocs = self.allocations.lock();
        allocs.insert(name, size);
        self.update_memory_stats(size as i64);
    }

    /// Unregister a memory allocation
    pub fn unregister_allocation(&self, name: &str) {
        let mut allocs = self.allocations.lock();
        if let Some(size) = allocs.remove(name) {
            self.update_memory_stats(-(size as i64));
        }
    }

    /// Update memory statistics
    fn update_memory_stats(&self, delta: i64) {
        let mut stats = self.stats.lock();
        let mut peak = self.peak_memory.lock();

        if delta > 0 {
            stats.memory.total_allocated += delta as u64;
            stats.memory.current_usage += delta as u64;
        } else {
            stats.memory.current_usage = stats.memory.current_usage.saturating_sub((-delta) as u64);
        }

        // Update peak
        if stats.memory.current_usage > *peak {
            *peak = stats.memory.current_usage;
        }
        stats.memory.peak_usage = *peak;

        // Update usage percentage (approximate based on typical system memory)
        // In production, this would use actual system memory info
        stats.memory.usage_percent = if stats.memory.total_allocated > 0 {
            (stats.memory.current_usage as f32 / stats.memory.total_allocated as f32) * 100.0
        } else {
            0.0
        };
    }

    /// Update resource statistics
    pub fn update_stats(&self) {
        if !self.config.enabled {
            return;
        }

        let now = Instant::now();
        let mut last_update = self.last_update.lock();
        
        if now.duration_since(*last_update) < self.config.update_interval {
            return;
        }

        *last_update = now;
        drop(last_update);

        let mut stats = self.stats.lock();
        stats.timestamp = now;

        // Update CPU stats (approximate)
        if self.config.track_cpu {
            let ops = *self.current_ops.lock();
            let total_threads = num_cpus::get() as u32;
            stats.cpu.active_threads = ops.min(total_threads);
            stats.cpu.total_threads = total_threads;
            stats.cpu.usage_percent = if total_threads > 0 {
                (stats.cpu.active_threads as f32 / total_threads as f32) * 100.0
            } else {
                0.0
            };
        }

        // Update memory stats
        if self.config.track_memory {
            let allocs = self.allocations.lock();
            stats.memory.total_allocated = allocs.values().sum();
            stats.memory.available = self.limits.max_memory_bytes.saturating_sub(stats.memory.current_usage);
        }
    }

    /// Get current resource statistics
    pub fn get_stats(&self) -> ResourceStats {
        self.update_stats();
        self.stats.lock().clone()
    }

    /// Get peak memory usage
    pub fn peak_memory(&self) -> u64 {
        *self.peak_memory.lock()
    }

    /// Get current number of active operations
    pub fn active_operations(&self) -> u32 {
        *self.current_ops.lock()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock();
        *stats = ResourceStats::default();
        *self.peak_memory.lock() = 0;
        self.allocations.lock().clear();
    }

    /// Check if system is under load
    pub fn is_under_load(&self) -> bool {
        let stats = self.get_stats();
        stats.memory.usage_percent > 70.0 || stats.cpu.usage_percent > 70.0
    }

    /// Get resource usage summary
    pub fn summary(&self) -> String {
        let stats = self.get_stats();
        format!(
            "Memory: {} / {} ({:.1}%) | CPU: {:.1}% | Active Ops: {}",
            bytes_to_human(stats.memory.current_usage),
            bytes_to_human(stats.memory.total_allocated),
            stats.memory.usage_percent,
            stats.cpu.usage_percent,
            self.active_operations()
        )
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource guard that automatically releases resources when dropped
pub struct ResourceGuard {
    ops: Arc<Mutex<u32>>,
}

impl ResourceGuard {
    /// Create a new resource guard
    fn new(ops: Arc<Mutex<u32>>) -> Self {
        Self { ops }
    }
}

impl Drop for ResourceGuard {
    fn drop(&mut self) {
        let mut ops = self.ops.lock();
        if *ops > 0 {
            *ops -= 1;
        }
    }
}

/// Convert bytes to human-readable format
fn bytes_to_human(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_creation() {
        let manager = ResourceManager::new();
        assert_eq!(manager.active_operations(), 0);
    }

    #[test]
    fn test_acquire_release() {
        let manager = ResourceManager::with_limits(
            ResourceLimits::new().with_max_concurrent_ops(2)
        );

        let guard1 = manager.acquire().unwrap();
        assert_eq!(manager.active_operations(), 1);

        let guard2 = manager.acquire().unwrap();
        assert_eq!(manager.active_operations(), 2);

        // Third acquire should fail
        assert!(manager.acquire().is_err());

        drop(guard1);
        assert_eq!(manager.active_operations(), 1);

        drop(guard2);
        assert_eq!(manager.active_operations(), 0);
    }

    #[test]
    fn test_memory_tracking() {
        let manager = ResourceManager::new();
        
        manager.register_allocation("test".to_string(), 1024);
        let stats = manager.get_stats();
        assert_eq!(stats.memory.current_usage, 1024);

        manager.unregister_allocation("test");
        let stats = manager.get_stats();
        assert_eq!(stats.memory.current_usage, 0);
    }

    #[test]
    fn test_peak_memory() {
        let manager = ResourceManager::new();
        
        manager.register_allocation("alloc1".to_string(), 1000);
        assert_eq!(manager.peak_memory(), 1000);

        manager.register_allocation("alloc2".to_string(), 500);
        assert_eq!(manager.peak_memory(), 1500);

        manager.unregister_allocation("alloc1");
        assert_eq!(manager.peak_memory(), 1500); // Peak doesn't decrease
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::new()
            .with_max_memory_bytes(1024)
            .with_max_concurrent_ops(1);
        
        let manager = ResourceManager::with_limits(limits);
        
        manager.register_allocation("test".to_string(), 1024);
        
        // Should fail due to memory limit
        assert!(manager.acquire().is_err());
    }
}
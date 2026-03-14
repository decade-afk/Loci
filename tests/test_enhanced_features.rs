//! Tests for enhanced inference engine features

use loci::prelude::*;
use std::time::Duration;

#[test]
fn test_cache_configuration() {
    let config = CacheConfig::new()
        .with_max_entries(100)
        .with_ttl(Duration::from_secs(3600))
        .with_stats(true);

    assert_eq!(config.max_entries, 100);
    assert_eq!(config.ttl, Duration::from_secs(3600));
    assert!(config.enable_stats);
}

#[test]
fn test_resource_limits() {
    let limits = ResourceLimits::new()
        .with_max_memory_bytes(8_000_000_000) // 8GB
        .with_max_memory_percent(80.0)
        .with_max_cpu_percent(90.0)
        .with_max_concurrent_ops(4);

    assert_eq!(limits.max_memory_bytes, 8_000_000_000);
    assert_eq!(limits.max_memory_percent, 80.0);
    assert_eq!(limits.max_cpu_percent, 90.0);
    assert_eq!(limits.max_concurrent_ops, 4);
}

#[test]
fn test_timeout_configuration() {
    let config = TimeoutConfig::new()
        .with_default_timeout(Duration::from_secs(60))
        .with_min_timeout(Duration::from_millis(100))
        .with_max_timeout(Duration::from_secs(300))
        .with_enabled(true);

    assert_eq!(config.default_timeout, Duration::from_secs(60));
    assert_eq!(config.min_timeout, Duration::from_millis(100));
    assert_eq!(config.max_timeout, Duration::from_secs(300));
    assert!(config.enabled);
}

#[test]
fn test_timeout_clamping() {
    let config = TimeoutConfig::new()
        .with_min_timeout(Duration::from_millis(100))
        .with_max_timeout(Duration::from_millis(500));

    // Test clamping
    assert_eq!(
        config.clamp_timeout(Duration::from_millis(50)),
        Duration::from_millis(100)
    );
    assert_eq!(
        config.clamp_timeout(Duration::from_millis(1000)),
        Duration::from_millis(500)
    );
    assert_eq!(
        config.clamp_timeout(Duration::from_millis(300)),
        Duration::from_millis(300)
    );
}

#[test]
fn test_concurrency_configuration() {
    let config = ConcurrencyConfig::new()
        .with_max_concurrent(4)
        .with_queue_size(100)
        .with_rate_limit(true, 10)
        .with_request_timeout(Duration::from_secs(60));

    assert_eq!(config.max_concurrent, 4);
    assert_eq!(config.queue_size, 100);
    assert!(config.enable_rate_limit);
    assert_eq!(config.max_ops_per_second, 10);
    assert_eq!(config.request_timeout, Duration::from_secs(60));
}

#[test]
fn test_inference_cache_basic() {
    let mut cache = InferenceCache::new();
    let params = InferenceParams::default();

    let key = cache.generate_key("test prompt", &params);
    cache.insert(key, "cached result".to_string());

    let result = cache.get(key);
    assert_eq!(result, Some("cached result".to_string()));
}

#[test]
fn test_timeout_context() {
    let ctx = TimeoutContext::new(Duration::from_millis(100));
    assert!(!ctx.is_timeout());

    std::thread::sleep(Duration::from_millis(150));
    assert!(ctx.is_timeout());
}

#[test]
fn test_timeout_cancellation() {
    let ctx = TimeoutContext::new(Duration::from_secs(60));
    let handle = ctx.cancellation_handle();

    assert!(!ctx.is_cancelled());
    handle.cancel();
    assert!(ctx.is_cancelled());
}

#[test]
fn test_resource_manager() {
    let manager = ResourceManager::new();
    assert_eq!(manager.active_operations(), 0);

    let guard = manager.acquire().unwrap();
    assert_eq!(manager.active_operations(), 1);

    drop(guard);
    assert_eq!(manager.active_operations(), 0);
}

#[test]
fn test_concurrency_manager() {
    let manager = ConcurrencyManager::with_config(ConcurrencyConfig::new().with_max_concurrent(2));

    let guard1 = manager.acquire().unwrap();
    assert_eq!(manager.active_operations(), 1);

    let guard2 = manager.acquire().unwrap();
    assert_eq!(manager.active_operations(), 2);

    drop(guard1);
    assert_eq!(manager.active_operations(), 1);
    drop(guard2);
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

#[test]
fn test_cache_stats() {
    let mut cache = InferenceCache::new();

    cache.insert(1, "value1".to_string());
    cache.get(1); // hit
    cache.get(2); // miss

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.entries, 1);
    assert!(stats.hit_rate() > 0.0);
}

#[test]
fn test_cache_lru_eviction() {
    let config = CacheConfig::new()
        .with_max_entries(2)
        .with_ttl(Duration::from_secs(3600));
    let mut cache = InferenceCache::with_config(config);

    cache.insert(1, "value1".to_string());
    cache.insert(2, "value2".to_string());
    cache.insert(3, "value3".to_string());

    // First entry should be evicted
    assert_eq!(cache.get(1), None);
    assert_eq!(cache.get(2), Some("value2".to_string()));
    assert_eq!(cache.get(3), Some("value3".to_string()));
}

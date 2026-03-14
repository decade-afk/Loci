//! Inference result caching system
//!
//! This module provides a multi-level caching strategy for inference results:
//! - In-memory LRU cache for frequently used prompts
//! - Configurable TTL and size limits
//! - Cache key generation based on prompt hash and parameters
//! - Cache statistics and management

use crate::backend::InferenceParams;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};
use xxhash_rust::xxh64::Xxh64;

/// Cache entry with value and metadata
#[derive(Clone)]
struct CacheEntry {
    /// Cached result
    value: String,
    /// Creation timestamp
    created_at: Instant,
    /// Last access timestamp
    last_access: Instant,
    /// Access count
    access_count: u64,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in cache
    pub max_entries: usize,
    /// Time-to-live for cache entries
    pub ttl: Duration,
    /// Enable cache statistics
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Duration::from_secs(3600), // 1 hour
            enable_stats: true,
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum entries
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Enable/disable statistics
    pub fn with_stats(mut self, enable: bool) -> Self {
        self.enable_stats = enable;
        self
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current entry count
    pub entries: usize,
    /// Total bytes stored
    pub bytes: u64,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64) / (total as f64)
        }
    }
}

/// Inference result cache
pub struct InferenceCache {
    cache: HashMap<u64, CacheEntry>,
    config: CacheConfig,
    stats: CacheStats,
    access_order: Vec<u64>, // For LRU eviction
}

impl InferenceCache {
    /// Create a new cache with default configuration
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a new cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Self {
        Self {
            cache: HashMap::new(),
            config,
            stats: CacheStats::default(),
            access_order: Vec::new(),
        }
    }

    /// Generate cache key from prompt and parameters
    pub fn generate_key(&self, prompt: &str, params: &InferenceParams) -> u64 {
        let mut hasher = Xxh64::new(0);
        prompt.hash(&mut hasher);

        // Hash relevant parameters
        hasher.write_u32(params.max_tokens);
        hasher.write_u32(params.temperature.to_bits());
        hasher.write_u32(params.top_p.to_bits());
        hasher.write_u32(params.top_k);
        hasher.write_u32(params.repeat_penalty.to_bits());

        hasher.finish()
    }

    /// Get cached result if available and not expired
    pub fn get(&mut self, key: u64) -> Option<String> {
        let now = Instant::now();

        if let Some(entry) = self.cache.get_mut(&key) {
            // Check TTL
            if now.duration_since(entry.created_at) > self.config.ttl {
                // Entry expired
                self.cache.remove(&key);
                self.access_order.retain(|&k| k != key);
                if self.config.enable_stats {
                    self.stats.misses += 1;
                    self.stats.entries = self.cache.len();
                }
                return None;
            }

            // Update access info
            entry.last_access = now;
            entry.access_count += 1;

            // Update access order for LRU
            if let Some(pos) = self.access_order.iter().position(|&k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key);

            if self.config.enable_stats {
                self.stats.hits += 1;
            }

            Some(entry.value.clone())
        } else {
            if self.config.enable_stats {
                self.stats.misses += 1;
            }
            None
        }
    }

    /// Insert a new entry into the cache
    pub fn insert(&mut self, key: u64, value: String) {
        let now = Instant::now();
        let entry = CacheEntry {
            value: value.clone(),
            created_at: now,
            last_access: now,
            access_count: 1,
        };

        // Check if we need to evict
        while self.cache.len() >= self.config.max_entries && !self.cache.is_empty() {
            self.evict_lru();
        }

        // Insert new entry
        self.cache.insert(key, entry);
        self.access_order.push(key);

        if self.config.enable_stats {
            self.stats.entries = self.cache.len();
            self.stats.bytes = self.cache.values().map(|e| e.value.len() as u64).sum();
        }
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some(key) = self.access_order.first() {
            let key = *key;
            self.cache.remove(&key);
            self.access_order.remove(0);

            if self.config.enable_stats {
                self.stats.evictions += 1;
                self.stats.entries = self.cache.len();
            }
        }
    }

    /// Clear all entries from the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        if self.config.enable_stats {
            self.stats.entries = 0;
            self.stats.bytes = 0;
        }
    }

    /// Remove expired entries
    pub fn cleanup_expired(&mut self) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        for (&key, entry) in &self.cache {
            if now.duration_since(entry.created_at) > self.config.ttl {
                to_remove.push(key);
            }
        }

        for key in to_remove {
            self.cache.remove(&key);
            self.access_order.retain(|&k| k != key);
        }

        if self.config.enable_stats {
            self.stats.entries = self.cache.len();
            self.stats.bytes = self.cache.values().map(|e| e.value.len() as u64).sum();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset cache statistics
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }

    /// Get current cache size in bytes
    pub fn size_bytes(&self) -> u64 {
        self.stats.bytes
    }

    /// Get number of entries
    pub fn entry_count(&self) -> usize {
        self.cache.len()
    }
}

impl Default for InferenceCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_get() {
        let mut cache = InferenceCache::new();
        let key = 12345;
        let value = "test result".to_string();

        cache.insert(key, value.clone());
        assert_eq!(cache.get(key), Some(value));
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = InferenceCache::new();
        assert_eq!(cache.get(99999), None);
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

    #[test]
    fn test_cache_ttl() {
        let config = CacheConfig::new()
            .with_max_entries(100)
            .with_ttl(Duration::from_millis(100));
        let mut cache = InferenceCache::with_config(config);

        cache.insert(1, "value1".to_string());
        assert_eq!(cache.get(1), Some("value1".to_string()));

        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(cache.get(1), None);
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
    fn test_key_generation() {
        let cache = InferenceCache::new();
        let params = InferenceParams::default();

        let key1 = cache.generate_key("hello", &params);
        let key2 = cache.generate_key("hello", &params);
        let key3 = cache.generate_key("world", &params);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}

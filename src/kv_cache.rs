//! Paged KV Cache implementation
//!
//! This module provides a block-based (paged) KV cache system that enables:
//! - Efficient memory management with 32-token blocks
//! - KV cache sharing between sessions (prefix caching)
//! - Dynamic allocation and deallocation
//! - Zero-copy block sharing via reference counting

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Number of tokens per KV cache block
pub const BLOCK_SIZE: usize = 32;

/// A single KV cache block containing 32 tokens
///
/// Each block stores key-value pairs for all layers of the model.
/// Blocks are reference-counted to enable sharing between sessions.
#[derive(Clone)]
pub struct KVBlock {
    /// Unique block ID
    pub block_id: u64,

    /// Key-value cache data
    /// Shape: [num_layers, BLOCK_SIZE, hidden_dim]
    /// In practice, this is managed by llama.cpp internally,
    /// so we use a placeholder for now
    data: Arc<Vec<f32>>,

    /// Reference count for sharing
    ref_count: Arc<AtomicUsize>,

    /// Number of valid tokens in this block (0-32)
    valid_tokens: usize,
}

impl KVBlock {
    /// Create a new empty KV block
    pub fn new(block_id: u64, num_layers: usize, hidden_dim: usize) -> Self {
        let capacity = num_layers * BLOCK_SIZE * hidden_dim;
        Self {
            block_id,
            data: Arc::new(vec![0.0; capacity]),
            ref_count: Arc::new(AtomicUsize::new(1)),
            valid_tokens: 0,
        }
    }

    /// Get the number of valid tokens in this block
    pub fn valid_tokens(&self) -> usize {
        self.valid_tokens
    }

    /// Set the number of valid tokens
    pub fn set_valid_tokens(&mut self, count: usize) {
        assert!(count <= BLOCK_SIZE);
        self.valid_tokens = count;
    }

    /// Check if block is full
    pub fn is_full(&self) -> bool {
        self.valid_tokens == BLOCK_SIZE
    }

    /// Check if block is empty
    pub fn is_empty(&self) -> bool {
        self.valid_tokens == 0
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Increment reference count
    pub fn acquire(&self) {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement reference count and return new count
    pub fn release(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::SeqCst) - 1
    }

    /// Clone block data (copy-on-write)
    pub fn clone_data(&self) -> Self {
        Self {
            block_id: self.block_id,
            data: Arc::clone(&self.data),
            ref_count: Arc::new(AtomicUsize::new(1)),
            valid_tokens: self.valid_tokens,
        }
    }
}

/// Paged KV cache manager
///
/// Manages a pool of KV blocks that can be allocated to sessions
pub struct PagedKVCache {
    /// All allocated blocks
    blocks: Vec<KVBlock>,

    /// IDs of free blocks available for allocation
    free_blocks: Vec<u64>,

    /// Next block ID to allocate
    next_block_id: u64,

    /// Model configuration
    num_layers: usize,
    hidden_dim: usize,
}

impl PagedKVCache {
    /// Create a new paged KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension size
    /// * `initial_blocks` - Number of blocks to pre-allocate
    pub fn new(num_layers: usize, hidden_dim: usize, initial_blocks: usize) -> Self {
        let mut cache = Self {
            blocks: Vec::new(),
            free_blocks: Vec::new(),
            next_block_id: 0,
            num_layers,
            hidden_dim,
        };

        // Pre-allocate blocks
        for _ in 0..initial_blocks {
            let block_id = cache.allocate_block_id();
            let block = KVBlock::new(block_id, num_layers, hidden_dim);
            cache.blocks.push(block);
            cache.free_blocks.push(block_id);
        }

        cache
    }

    /// Allocate a new block ID
    fn allocate_block_id(&mut self) -> u64 {
        let id = self.next_block_id;
        self.next_block_id += 1;
        id
    }

    /// Allocate a free block
    ///
    /// Returns None if no free blocks available
    pub fn allocate(&mut self) -> Option<u64> {
        if let Some(block_id) = self.free_blocks.pop() {
            return Some(block_id);
        }

        // No free blocks, allocate a new one
        let block_id = self.allocate_block_id();
        let block = KVBlock::new(block_id, self.num_layers, self.hidden_dim);
        self.blocks.push(block);
        Some(block_id)
    }

    /// Free a block back to the pool
    pub fn free(&mut self, block_id: u64) {
        if let Some(block) = self.get_block(block_id) {
            if block.release() == 0 {
                // Last reference, return to free pool
                self.free_blocks.push(block_id);
            }
        }
    }

    /// Get a block by ID
    pub fn get_block(&self, block_id: u64) -> Option<&KVBlock> {
        self.blocks.iter().find(|b| b.block_id == block_id)
    }

    /// Get a mutable block by ID
    pub fn get_block_mut(&mut self, block_id: u64) -> Option<&mut KVBlock> {
        self.blocks.iter_mut().find(|b| b.block_id == block_id)
    }

    /// Get total number of blocks
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get number of free blocks
    pub fn free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get number of used blocks
    pub fn used_blocks(&self) -> usize {
        self.total_blocks() - self.free_blocks()
    }
}

/// Session-specific KV cache
///
/// Maintains a logical sequence of tokens mapped to physical blocks
pub struct SessionKVCache {
    /// Block table: maps logical token positions to physical block IDs
    /// Index i = logical block number, value = physical block ID
    block_table: Vec<u64>,

    /// Total number of tokens in cache
    num_tokens: usize,
}

impl SessionKVCache {
    /// Create a new empty session KV cache
    pub fn new() -> Self {
        Self {
            block_table: Vec::new(),
            num_tokens: 0,
        }
    }

    /// Get number of tokens in cache
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get number of blocks used
    pub fn num_blocks(&self) -> usize {
        self.block_table.len()
    }

    /// Add tokens to the cache
    ///
    /// Returns the number of new blocks needed
    pub fn add_tokens(&mut self, count: usize, pool: &mut PagedKVCache) -> Vec<u64> {
        let mut new_blocks = Vec::new();
        self.num_tokens += count;

        // Calculate how many blocks we need
        let required_blocks = (self.num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Allocate new blocks if needed
        while self.block_table.len() < required_blocks {
            if let Some(block_id) = pool.allocate() {
                self.block_table.push(block_id);
                new_blocks.push(block_id);
            } else {
                // Out of memory
                break;
            }
        }

        new_blocks
    }

    /// Remove tokens from the end
    pub fn remove_tokens(&mut self, count: usize, pool: &mut PagedKVCache) {
        if count >= self.num_tokens {
            self.clear(pool);
            return;
        }

        self.num_tokens -= count;

        // Calculate how many blocks we still need
        let required_blocks = (self.num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Free excess blocks
        while self.block_table.len() > required_blocks {
            if let Some(block_id) = self.block_table.pop() {
                pool.free(block_id);
            }
        }
    }

    /// Clear all cached tokens
    pub fn clear(&mut self, pool: &mut PagedKVCache) {
        for block_id in &self.block_table {
            pool.free(*block_id);
        }
        self.block_table.clear();
        self.num_tokens = 0;
    }

    /// Get the block table
    pub fn block_table(&self) -> &[u64] {
        &self.block_table
    }

    /// Share prefix blocks with another session
    ///
    /// Returns the block IDs that can be shared
    pub fn share_prefix(&self, prefix_len: usize) -> Vec<u64> {
        let num_blocks = (prefix_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        self.block_table.iter().take(num_blocks).copied().collect()
    }

    /// Acquire shared blocks from another session
    pub fn acquire_prefix(&mut self, shared_blocks: &[u64], pool: &mut PagedKVCache) {
        // Release existing blocks before replacing with shared blocks.
        for block_id in &self.block_table {
            pool.free(*block_id);
        }
        self.block_table.clear();
        self.num_tokens = 0;

        // Acquire references to shared blocks
        for &block_id in shared_blocks {
            if let Some(block) = pool.get_block(block_id) {
                block.acquire();
                self.block_table.push(block_id);
            }
        }

        self.num_tokens = shared_blocks.len() * BLOCK_SIZE;
    }
}

impl Default for SessionKVCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_block_creation() {
        let block = KVBlock::new(0, 32, 4096);
        assert_eq!(block.block_id, 0);
        assert_eq!(block.valid_tokens(), 0);
        assert!(block.is_empty());
        assert!(!block.is_full());
        assert_eq!(block.ref_count(), 1);
    }

    #[test]
    fn test_kv_block_tokens() {
        let mut block = KVBlock::new(0, 32, 4096);
        block.set_valid_tokens(16);
        assert_eq!(block.valid_tokens(), 16);
        assert!(!block.is_empty());
        assert!(!block.is_full());

        block.set_valid_tokens(BLOCK_SIZE);
        assert!(block.is_full());
    }

    #[test]
    fn test_kv_block_refcount() {
        let block = KVBlock::new(0, 32, 4096);
        assert_eq!(block.ref_count(), 1);

        block.acquire();
        assert_eq!(block.ref_count(), 2);

        block.release();
        assert_eq!(block.ref_count(), 1);
    }

    #[test]
    fn test_paged_kv_cache() {
        let mut cache = PagedKVCache::new(32, 4096, 10);
        assert_eq!(cache.total_blocks(), 10);
        assert_eq!(cache.free_blocks(), 10);
        assert_eq!(cache.used_blocks(), 0);
    }

    #[test]
    fn test_block_allocation() {
        let mut cache = PagedKVCache::new(32, 4096, 5);

        let block1 = cache.allocate();
        assert!(block1.is_some());
        assert_eq!(cache.free_blocks(), 4);

        let block2 = cache.allocate();
        assert!(block2.is_some());
        assert_eq!(cache.free_blocks(), 3);

        // Free a block
        cache.free(block1.unwrap());
        assert_eq!(cache.free_blocks(), 4);
    }

    #[test]
    fn test_session_kv_cache() {
        let mut pool = PagedKVCache::new(32, 4096, 10);
        let mut session_cache = SessionKVCache::new();

        assert_eq!(session_cache.num_tokens(), 0);
        assert_eq!(session_cache.num_blocks(), 0);

        // Add 50 tokens (should allocate 2 blocks: 32 + 18)
        session_cache.add_tokens(50, &mut pool);
        assert_eq!(session_cache.num_tokens(), 50);
        assert_eq!(session_cache.num_blocks(), 2);

        // Remove some tokens
        session_cache.remove_tokens(20, &mut pool);
        assert_eq!(session_cache.num_tokens(), 30);
        assert_eq!(session_cache.num_blocks(), 1);

        // Clear
        session_cache.clear(&mut pool);
        assert_eq!(session_cache.num_tokens(), 0);
        assert_eq!(session_cache.num_blocks(), 0);
    }

    #[test]
    fn test_prefix_sharing() {
        let mut pool = PagedKVCache::new(32, 4096, 10);
        let mut cache1 = SessionKVCache::new();

        // Session 1 adds 100 tokens
        cache1.add_tokens(100, &mut pool);

        // Share first 64 tokens (2 blocks)
        let shared = cache1.share_prefix(64);
        assert_eq!(shared.len(), 2);

        // Session 2 acquires the shared prefix
        let mut cache2 = SessionKVCache::new();
        cache2.acquire_prefix(&shared, &mut pool);
        assert_eq!(cache2.num_tokens(), 64);
        assert_eq!(cache2.num_blocks(), 2);
    }
}

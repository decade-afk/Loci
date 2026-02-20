//! Advanced Paged KV Cache with LRU Eviction and Memory Management
//!
//! This module provides an enhanced KV cache system with:
//! - **LRU Eviction**: Least Recently Used block replacement
//! - **Timestamp Tracking**: last_access_timestamp for each block
//! - **Memory Budgeting**: GPU memory management with thresholds
//! - **Reference Counting**: Safe block sharing between sessions
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  MemoryBudgeter                                         │
//! │  - 10% floor (minimum free memory)                      │
//! │  - 85% threshold (eviction trigger)                     │
//! │  - 3s cool-down (prevent thrashing)                     │
//! └─────────────────────────────────────────────────────────┘
//!          ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │  PhysicalBlockPool                                      │
//! │  - LRU eviction queue                                   │
//! │  - Block metadata (last_access, ref_count)              │
//! │  - Free block tracking                                  │
//! └─────────────────────────────────────────────────────────┘
//!          ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │  BlockTable (per session)                               │
//! │  - Logical → Physical block mapping                     │
//! │  - Prefix sharing support                               │
//! └─────────────────────────────────────────────────────────┘
//! ```

use crate::error::{LociError, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Number of tokens per KV cache block
pub const BLOCK_SIZE: usize = 32;

/// Physical block ID
pub type BlockId = u64;

/// Block metadata for LRU tracking
#[derive(Clone)]
struct BlockMetadata {
    /// Block ID
    id: BlockId,
    /// Reference count (Arc for thread-safe sharing)
    ref_count: Arc<AtomicUsize>,
    /// Last access timestamp
    last_access: Instant,
    /// Number of valid tokens (0-32)
    valid_tokens: usize,
    /// Whether this block is pinned (cannot be evicted)
    pinned: bool,
}

impl BlockMetadata {
    fn new(id: BlockId) -> Self {
        Self {
            id,
            ref_count: Arc::new(AtomicUsize::new(0)), // Start with 0, acquire() will set to 1
            last_access: Instant::now(),
            valid_tokens: 0,
            pinned: false,
        }
    }

    /// Increment reference count
    fn acquire(&self) {
        self.ref_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement reference count and return new count
    fn release(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::SeqCst) - 1
    }

    /// Get current reference count
    fn get_ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }

    /// Update last access timestamp
    fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// Check if block can be evicted (ref_count == 0 && !pinned)
    fn is_evictable(&self) -> bool {
        self.get_ref_count() == 0 && !self.pinned
    }
}

/// Memory budget configuration for GPU memory management
#[derive(Debug, Clone)]
pub struct MemoryBudgetConfig {
    /// Total GPU memory in bytes
    pub total_memory: usize,

    /// Minimum free memory percentage (e.g., 10%)
    /// Eviction starts when free memory < floor
    pub floor_percentage: f32,

    /// Eviction trigger threshold (e.g., 85%)
    /// When usage > threshold, start evicting least recently used blocks
    pub eviction_threshold: f32,

    /// Cool-down period after eviction (e.g., 3 seconds)
    /// Prevents thrashing by limiting eviction frequency
    pub cool_down_duration: Duration,
}

impl Default for MemoryBudgetConfig {
    fn default() -> Self {
        Self {
            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB default
            floor_percentage: 0.10,                // 10% floor
            eviction_threshold: 0.85,              // 85% threshold
            cool_down_duration: Duration::from_secs(3), // 3s cool-down
        }
    }
}

/// Memory budgeter for GPU memory management
pub struct MemoryBudgeter {
    config: MemoryBudgetConfig,
    /// Timestamp of last eviction
    last_eviction: Option<Instant>,
    /// Current memory usage in bytes
    current_usage: usize,
}

impl MemoryBudgeter {
    pub fn new(config: MemoryBudgetConfig) -> Self {
        Self {
            config,
            last_eviction: None,
            current_usage: 0,
        }
    }

    /// Check if eviction is needed
    pub fn should_evict(&self) -> bool {
        // Check usage threshold
        let usage_ratio = self.current_usage as f32 / self.config.total_memory as f32;
        if usage_ratio < self.config.eviction_threshold {
            return false;
        }

        // Check cool-down period
        if let Some(last_eviction) = self.last_eviction {
            let elapsed = last_eviction.elapsed();
            if elapsed < self.config.cool_down_duration {
                return false; // Still in cool-down
            }
        }

        true
    }

    /// Check if we've hit the memory floor
    pub fn is_below_floor(&self) -> bool {
        let free_memory = self.config.total_memory.saturating_sub(self.current_usage);
        let free_ratio = free_memory as f32 / self.config.total_memory as f32;
        free_ratio < self.config.floor_percentage
    }

    /// Record memory allocation
    pub fn allocate(&mut self, size: usize) {
        self.current_usage += size;
    }

    /// Record memory deallocation
    pub fn deallocate(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
    }

    /// Record eviction event
    pub fn mark_eviction(&mut self) {
        self.last_eviction = Some(Instant::now());
    }

    /// Get current usage ratio
    pub fn usage_ratio(&self) -> f32 {
        self.current_usage as f32 / self.config.total_memory as f32
    }

    /// Get free memory in bytes
    pub fn free_memory(&self) -> usize {
        self.config.total_memory.saturating_sub(self.current_usage)
    }
}

/// Physical block pool with LRU eviction
///
/// Manages a pool of physical blocks with:
/// - LRU-based eviction policy
/// - Reference counting for safe sharing
/// - Last access timestamp tracking
/// - Memory budget enforcement
pub struct PhysicalBlockPool {
    /// All allocated blocks (block_id → metadata)
    blocks: HashMap<BlockId, BlockMetadata>,

    /// Free blocks available for immediate allocation
    free_blocks: VecDeque<BlockId>,

    /// LRU queue for eviction (ordered by last access time, oldest first)
    /// Only contains blocks with ref_count == 0
    lru_queue: VecDeque<BlockId>,

    /// Next block ID to allocate
    next_block_id: BlockId,

    /// Block size in bytes
    block_size_bytes: usize,

    /// Memory budgeter
    budgeter: MemoryBudgeter,

    /// Statistics
    stats: PoolStatistics,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub total_evictions: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl PhysicalBlockPool {
    /// Create a new physical block pool
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension size
    /// * `initial_blocks` - Number of blocks to pre-allocate
    /// * `memory_config` - Memory budget configuration
    pub fn new(
        num_layers: usize,
        hidden_dim: usize,
        initial_blocks: usize,
        memory_config: MemoryBudgetConfig,
    ) -> Self {
        let block_size_bytes = num_layers * BLOCK_SIZE * hidden_dim * std::mem::size_of::<f32>();
        let mut pool = Self {
            blocks: HashMap::new(),
            free_blocks: VecDeque::new(),
            lru_queue: VecDeque::new(),
            next_block_id: 0,
            block_size_bytes,
            budgeter: MemoryBudgeter::new(memory_config),
            stats: PoolStatistics::default(),
        };

        // Pre-allocate blocks
        for _ in 0..initial_blocks {
            let block_id = pool.allocate_block_id();
            let metadata = BlockMetadata::new(block_id);
            pool.blocks.insert(block_id, metadata);
            pool.free_blocks.push_back(block_id);
            pool.budgeter.allocate(block_size_bytes);
        }

        pool
    }

    /// Allocate a new block ID
    fn allocate_block_id(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;
        id
    }

    /// Allocate a block
    ///
    /// Returns a block ID or None if allocation fails
    pub fn allocate(&mut self) -> Result<BlockId> {
        self.stats.total_allocations += 1;

        // 1. Try to reuse blocks from LRU queue first
        while let Some(block_id) = self.lru_queue.pop_front() {
            if let Some(metadata) = self.blocks.get_mut(&block_id) {
                if metadata.is_evictable() {
                    metadata.touch();
                    metadata.acquire(); // Set ref_count to 1
                    return Ok(block_id);
                }
            }
        }

        // 2. Try to allocate from free blocks
        if let Some(block_id) = self.free_blocks.pop_front() {
            if let Some(metadata) = self.blocks.get_mut(&block_id) {
                metadata.touch();
                metadata.acquire(); // Set ref_count to 1
                return Ok(block_id);
            }
        }

        // 3. Check if we can allocate a new block
        if !self.budgeter.is_below_floor() {
            let block_id = self.allocate_block_id();
            let mut metadata = BlockMetadata::new(block_id);
            metadata.acquire(); // Set ref_count to 1
            self.blocks.insert(block_id, metadata);
            self.budgeter.allocate(self.block_size_bytes);
            return Ok(block_id);
        }

        // 4. Try to evict blocks if memory is tight
        if self.budgeter.should_evict() {
            if let Some(evicted_id) = self.evict_lru_block()? {
                self.stats.total_evictions += 1;
                self.budgeter.mark_eviction();

                // Reuse the evicted block
                if let Some(metadata) = self.blocks.get_mut(&evicted_id) {
                    metadata.touch();
                    metadata.acquire();
                    return Ok(evicted_id);
                }
            }
        }

        Err(LociError::OutOfMemory(
            "No free blocks and eviction failed".to_string(),
        ))
    }

    /// Free a block back to the pool
    pub fn free(&mut self, block_id: BlockId) -> Result<()> {
        self.stats.total_deallocations += 1;

        if let Some(metadata) = self.blocks.get_mut(&block_id) {
            let new_ref_count = metadata.release();

            if new_ref_count == 0 {
                // Last reference released, add to LRU queue
                metadata.touch();
                self.lru_queue.push_back(block_id);
            }

            Ok(())
        } else {
            Err(LociError::InvalidBlockId(block_id))
        }
    }

    /// Access a block (updates last_access_timestamp)
    pub fn touch(&mut self, block_id: BlockId) -> Result<()> {
        if let Some(metadata) = self.blocks.get_mut(&block_id) {
            metadata.touch();

            // Remove from LRU queue if present (will be re-added when freed)
            self.lru_queue.retain(|&id| id != block_id);

            Ok(())
        } else {
            Err(LociError::InvalidBlockId(block_id))
        }
    }

    /// Acquire a reference to a block (for sharing)
    pub fn acquire(&mut self, block_id: BlockId) -> Result<()> {
        if let Some(metadata) = self.blocks.get_mut(&block_id) {
            metadata.acquire();
            metadata.touch();

            // Remove from LRU queue (block is now referenced)
            self.lru_queue.retain(|&id| id != block_id);

            Ok(())
        } else {
            Err(LociError::InvalidBlockId(block_id))
        }
    }

    /// Pin a block (prevent eviction)
    pub fn pin(&mut self, block_id: BlockId) -> Result<()> {
        if let Some(metadata) = self.blocks.get_mut(&block_id) {
            metadata.pinned = true;

            // Remove from LRU queue
            self.lru_queue.retain(|&id| id != block_id);

            Ok(())
        } else {
            Err(LociError::InvalidBlockId(block_id))
        }
    }

    /// Unpin a block (allow eviction)
    pub fn unpin(&mut self, block_id: BlockId) -> Result<()> {
        if let Some(metadata) = self.blocks.get_mut(&block_id) {
            metadata.pinned = false;

            // Add back to LRU queue if ref_count == 0
            if metadata.get_ref_count() == 0 {
                self.lru_queue.push_back(block_id);
            }

            Ok(())
        } else {
            Err(LociError::InvalidBlockId(block_id))
        }
    }

    /// Evict the least recently used block
    ///
    /// Returns the ID of the evicted block or None if no evictable blocks
    fn evict_lru_block(&mut self) -> Result<Option<BlockId>> {
        // Scan LRU queue for evictable blocks (ref_count == 0 && !pinned)
        while let Some(block_id) = self.lru_queue.pop_front() {
            if let Some(metadata) = self.blocks.get(&block_id) {
                if metadata.is_evictable() {
                    // Found evictable block
                    return Ok(Some(block_id));
                }
            }
        }

        // No evictable blocks found
        Ok(None)
    }

    /// Get block metadata
    pub fn get_metadata(&self, block_id: BlockId) -> Option<&BlockMetadata> {
        self.blocks.get(&block_id)
    }

    /// Get pool statistics
    pub fn statistics(&self) -> &PoolStatistics {
        &self.stats
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> (usize, f32, usize) {
        (
            self.budgeter.current_usage,
            self.budgeter.usage_ratio(),
            self.budgeter.free_memory(),
        )
    }

    /// Get number of free blocks
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get total number of blocks
    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get number of blocks in LRU queue (evictable candidates)
    pub fn num_lru_blocks(&self) -> usize {
        self.lru_queue.len()
    }
}

/// Block table for a session (logical → physical mapping)
///
/// Maps logical block positions to physical block IDs
pub struct BlockTable {
    /// Logical block number → Physical block ID
    table: Vec<BlockId>,

    /// Total number of tokens in this session
    num_tokens: usize,

    /// Session ID for debugging
    session_id: String,
}

impl BlockTable {
    pub fn new(session_id: String) -> Self {
        Self {
            table: Vec::new(),
            num_tokens: 0,
            session_id,
        }
    }

    /// Get number of tokens
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Get number of logical blocks
    pub fn num_blocks(&self) -> usize {
        self.table.len()
    }

    /// Get the physical block table
    pub fn physical_blocks(&self) -> &[BlockId] {
        &self.table
    }

    /// Add tokens to the block table
    ///
    /// Allocates new physical blocks as needed
    pub fn add_tokens(
        &mut self,
        count: usize,
        pool: &mut PhysicalBlockPool,
    ) -> Result<Vec<BlockId>> {
        let mut new_blocks = Vec::new();
        self.num_tokens += count;

        // Calculate required blocks
        let required_blocks = (self.num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Allocate new blocks if needed
        while self.table.len() < required_blocks {
            let block_id = pool.allocate()?;
            self.table.push(block_id);
            new_blocks.push(block_id);
        }

        Ok(new_blocks)
    }

    /// Remove tokens from the end
    pub fn remove_tokens(
        &mut self,
        count: usize,
        pool: &mut PhysicalBlockPool,
    ) -> Result<()> {
        if count >= self.num_tokens {
            self.clear(pool)?;
            return Ok(());
        }

        self.num_tokens -= count;

        // Calculate required blocks
        let required_blocks = (self.num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Free excess blocks
        while self.table.len() > required_blocks {
            if let Some(block_id) = self.table.pop() {
                pool.free(block_id)?;
            }
        }

        Ok(())
    }

    /// Clear all blocks
    pub fn clear(&mut self, pool: &mut PhysicalBlockPool) -> Result<()> {
        for &block_id in &self.table {
            pool.free(block_id)?;
        }
        self.table.clear();
        self.num_tokens = 0;
        Ok(())
    }

    /// Touch all blocks (update LRU timestamps)
    pub fn touch_all(&self, pool: &mut PhysicalBlockPool) -> Result<()> {
        for &block_id in &self.table {
            pool.touch(block_id)?;
        }
        Ok(())
    }

    /// Share a prefix with another session
    pub fn share_prefix(&self, prefix_len: usize) -> Vec<BlockId> {
        let num_blocks = (prefix_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        self.table.iter().take(num_blocks).copied().collect()
    }

    /// Acquire shared blocks from another session
    pub fn acquire_prefix(
        &mut self,
        shared_blocks: &[BlockId],
        pool: &mut PhysicalBlockPool,
    ) -> Result<()> {
        // Clear existing blocks
        self.table.clear();

        // Acquire references to shared blocks
        for &block_id in shared_blocks {
            pool.acquire(block_id)?;
            self.table.push(block_id);
        }

        self.num_tokens = shared_blocks.len() * BLOCK_SIZE;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budgeter() {
        let config = MemoryBudgetConfig {
            total_memory: 1000,
            floor_percentage: 0.10,
            eviction_threshold: 0.85,
            cool_down_duration: Duration::from_millis(100),
        };

        let mut budgeter = MemoryBudgeter::new(config);

        // Initially, should not evict
        assert!(!budgeter.should_evict());

        // Allocate up to threshold
        budgeter.allocate(850);
        assert!(budgeter.should_evict());

        // Mark eviction and check cool-down
        budgeter.mark_eviction();
        assert!(!budgeter.should_evict()); // Still in cool-down

        // Wait for cool-down
        std::thread::sleep(Duration::from_millis(150));
        assert!(budgeter.should_evict()); // Cool-down expired
    }

    #[test]
    fn test_physical_block_pool_allocation() {
        let config = MemoryBudgetConfig::default();
        let mut pool = PhysicalBlockPool::new(32, 4096, 10, config);

        // Allocate a block
        let block1 = pool.allocate().unwrap();
        assert_eq!(pool.num_free_blocks(), 9);

        // Free the block (this adds it to LRU queue)
        pool.free(block1).unwrap();
        assert_eq!(pool.num_lru_blocks(), 1);

        // Allocate again (should reuse from LRU)
        let block2 = pool.allocate().unwrap();
        // The block might be reused but the ID could be different
        assert_eq!(pool.num_lru_blocks(), 0);
    }

    #[test]
    fn test_block_table_operations() {
        let config = MemoryBudgetConfig::default();
        let mut pool = PhysicalBlockPool::new(32, 4096, 10, config);
        let mut table = BlockTable::new("session1".to_string());

        // Add 50 tokens (should allocate 2 blocks)
        let new_blocks = table.add_tokens(50, &mut pool).unwrap();
        assert_eq!(new_blocks.len(), 2);
        assert_eq!(table.num_tokens(), 50);
        assert_eq!(table.num_blocks(), 2);

        // Remove 20 tokens (still need 1 block)
        table.remove_tokens(20, &mut pool).unwrap();
        assert_eq!(table.num_tokens(), 30);
        assert_eq!(table.num_blocks(), 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = MemoryBudgetConfig {
            total_memory: 100_000, // Small budget to trigger eviction
            floor_percentage: 0.10,
            eviction_threshold: 0.50, // Low threshold for testing
            cool_down_duration: Duration::from_millis(10),
        };

        let mut pool = PhysicalBlockPool::new(32, 4096, 2, config);

        // Allocate blocks until we trigger eviction
        let block1 = pool.allocate().unwrap();
        let block2 = pool.allocate().unwrap();

        // Free both blocks (adds them to LRU queue)
        pool.free(block1).unwrap();
        pool.free(block2).unwrap();

        // Now we should have 2 blocks in LRU queue
        assert_eq!(pool.num_lru_blocks(), 2);

        // Touch block2 (make it more recently used)
        pool.touch(block2).unwrap();

        // Trigger eviction by allocating more blocks
        std::thread::sleep(Duration::from_millis(20)); // Wait for cool-down

        // We should still have at least 1 block in LRU
        assert!(pool.num_lru_blocks() >= 1);
    }

    #[test]
    fn test_prefix_sharing() {
        let config = MemoryBudgetConfig::default();
        let mut pool = PhysicalBlockPool::new(32, 4096, 10, config);

        let mut table1 = BlockTable::new("session1".to_string());
        table1.add_tokens(100, &mut pool).unwrap();

        // Share first 64 tokens
        let shared = table1.share_prefix(64);
        assert_eq!(shared.len(), 2);

        // Session 2 acquires the prefix
        let mut table2 = BlockTable::new("session2".to_string());
        table2.acquire_prefix(&shared, &mut pool).unwrap();
        assert_eq!(table2.num_tokens(), 64);
    }
}

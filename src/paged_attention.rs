

//! # Paged Attention Module
//!
//! This module implements a Paged Attention mechanism with efficient memory management
//! for large language model inference. It provides block-based memory allocation,
//! automatic swapping between VRAM and RAM, and memory budgeting to prevent OOM errors.
//!
//! ## Key Components
//!
//! - **PhysicalBlockAllocator**: Manages physical memory blocks in VRAM and RAM
//! - **BlockTable**: Maps logical blocks to physical blocks per session
//! - **MemoryBudgeter**: Monitors VRAM usage and triggers swapping when needed
//! - **SessionManager**: Orchestrates block allocation and session lifecycle
//!
//! ## Memory Management Strategy
//!
//! The allocator uses a two-tier memory hierarchy:
//! 1. VRAM (fast, limited capacity) - preferred for active blocks
//! 2. RAM (slower, larger capacity) - overflow storage for less active blocks
//!
//! When VRAM is full, the system automatically swaps least-recently-used (LRU) blocks
//! to RAM based on the memory budgeter's thresholds.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use std::time::{Instant, Duration};
use anyhow::{Result, bail};

/// Number of token slots per physical block
/// Each block can store 32 tokens, which is a common choice for efficient memory access
pub const BLOCK_SIZE: usize = 32;

/// Minimum percentage of total VRAM reserved for system use
/// This ensures the system always has some VRAM available for critical operations
pub const SYSTEM_FLOOR_RATIO: f32 = 0.10;

/// Minimum system floor in megabytes
/// Even with a small total VRAM, at least 1024 MB is reserved for system use
pub const SYSTEM_FLOOR_MIN_MB: u64 = 1024;

/// VRAM utilization threshold that triggers block swapping
/// When VRAM usage exceeds 85% of total capacity, the system will start swapping blocks to RAM
pub const SWAP_THRESHOLD_RATIO: f32 = 0.85;

/// Cooldown period in seconds after a swap operation
/// Prevents rapid consecutive swaps that could cause thrashing
pub const COOLDOWN_DURATION_SECS: u64 = 3;




#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockId(pub u32);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogicalBlockId(pub u32);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLocation {
    
    Vram,
    
    Ram,
    
    Free,
}


#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    
    pub id: PhysicalBlockId,

    
    pub location: BlockLocation,

    
    pub last_access: Instant,

    
    pub ref_count: u32,

    
    pub owner_session: Option<SessionId>,

    
    
    pub data_ptr: Option<usize>,
}

impl PhysicalBlock {
    
    pub fn new_free(id: PhysicalBlockId) -> Self {
        Self {
            id,
            location: BlockLocation::Free,
            last_access: Instant::now(),
            ref_count: 0,
            owner_session: None,
            data_ptr: None,
        }
    }

    
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    
    pub fn is_free(&self) -> bool {
        self.location == BlockLocation::Free && self.ref_count == 0
    }

    
    pub fn idle_duration(&self) -> Duration {
        Instant::now().duration_since(self.last_access)
    }
}




#[derive(Debug)]
pub struct BlockTable {
    
    pub session_id: SessionId,

    
    pub mappings: HashMap<LogicalBlockId, PhysicalBlockId>,

    
    pub num_blocks: usize,

    
    pub last_access: Instant,
}

impl BlockTable {
    
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            mappings: HashMap::new(),
            num_blocks: 0,
            last_access: Instant::now(),
        }
    }

    
    pub fn allocate_logical_block(&mut self) -> LogicalBlockId {
        let logical_id = LogicalBlockId(self.num_blocks as u32);
        self.num_blocks += 1;
        logical_id
    }

    
    pub fn map(&mut self, logical: LogicalBlockId, physical: PhysicalBlockId) {
        self.mappings.insert(logical, physical);
        self.touch();
    }

    
    pub fn get_physical(&self, logical: LogicalBlockId) -> Option<PhysicalBlockId> {
        self.mappings.get(&logical).copied()
    }

    
    pub fn unmap(&mut self, logical: LogicalBlockId) -> Option<PhysicalBlockId> {
        self.mappings.remove(&logical)
    }

    
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    
    pub fn idle_duration(&self) -> Duration {
        Instant::now().duration_since(self.last_access)
    }
}




pub struct PhysicalBlockAllocator {
    
    blocks: Vec<PhysicalBlock>,

    
    vram_free_list: VecDeque<PhysicalBlockId>,

    
    ram_free_list: VecDeque<PhysicalBlockId>,

    
    vram_capacity: usize,

    
    ram_capacity: usize,

    
    vram_used: usize,

    
    ram_used: usize,
}

impl PhysicalBlockAllocator {
    
    
    
    
    
    
    pub fn new(vram_mb: u64, ram_mb: u64, block_size_kb: usize) -> Self {
        
        let vram_capacity = (vram_mb * 1024 / block_size_kb as u64) as usize;
        let ram_capacity = (ram_mb * 1024 / block_size_kb as u64) as usize;

        let total_blocks = vram_capacity + ram_capacity;

        
        let mut blocks = Vec::with_capacity(total_blocks);
        let mut vram_free_list = VecDeque::with_capacity(vram_capacity);
        let mut ram_free_list = VecDeque::with_capacity(ram_capacity);

        
        for i in 0..vram_capacity {
            let id = PhysicalBlockId(i as u32);
            let mut block = PhysicalBlock::new_free(id);
            block.location = BlockLocation::Vram;
            blocks.push(block);
            vram_free_list.push_back(id);
        }

        
        for i in 0..ram_capacity {
            let id = PhysicalBlockId((vram_capacity + i) as u32);
            let mut block = PhysicalBlock::new_free(id);
            block.location = BlockLocation::Ram;
            blocks.push(block);
            ram_free_list.push_back(id);
        }

        eprintln!("📦 PhysicalBlockAllocator initialized:");
        eprintln!("   VRAM: {} MB ({} blocks)", vram_mb, vram_capacity);
        eprintln!("   RAM: {} MB ({} blocks)", ram_mb, ram_capacity);

        Self {
            blocks,
            vram_free_list,
            ram_free_list,
            vram_capacity,
            ram_capacity,
            vram_used: 0,
            ram_used: 0,
        }
    }

    
    pub fn allocate_vram(&mut self, session_id: SessionId) -> Option<PhysicalBlockId> {
        let id = self.vram_free_list.pop_front()?;
        let block = &mut self.blocks[id.0 as usize];

        block.location = BlockLocation::Vram;
        block.owner_session = Some(session_id);
        block.ref_count = 1;
        block.touch();

        self.vram_used += 1;

        Some(id)
    }

    
    pub fn allocate_ram(&mut self, session_id: SessionId) -> Option<PhysicalBlockId> {
        let id = self.ram_free_list.pop_front()?;
        let block = &mut self.blocks[id.0 as usize];

        block.location = BlockLocation::Ram;
        block.owner_session = Some(session_id);
        block.ref_count = 1;
        block.touch();

        self.ram_used += 1;

        Some(id)
    }

    
    pub fn free(&mut self, id: PhysicalBlockId) -> Result<()> {
        let block = self.blocks.get_mut(id.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid block ID: {:?}", id))?;

        if block.ref_count > 0 {
            block.ref_count -= 1;
        }

        if block.ref_count == 0 {
            match block.location {
                BlockLocation::Vram => {
                    self.vram_free_list.push_back(id);
                    self.vram_used = self.vram_used.saturating_sub(1);
                }
                BlockLocation::Ram => {
                    self.ram_free_list.push_back(id);
                    self.ram_used = self.ram_used.saturating_sub(1);
                }
                BlockLocation::Free => {
                    
                }
            }

            block.location = BlockLocation::Free;
            block.owner_session = None;
            block.data_ptr = None;
        }

        Ok(())
    }

    
    pub fn swap_to_ram(&mut self, id: PhysicalBlockId) -> Result<()> {
        let block = self.blocks.get_mut(id.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid block ID: {:?}", id))?;

        if block.location != BlockLocation::Vram {
            bail!("Block {:?} is not in VRAM", id);
        }

        
        
        block.location = BlockLocation::Ram;
        self.vram_used = self.vram_used.saturating_sub(1);
        self.ram_used += 1;

        eprintln!("🔄 Swapped block {:?} to RAM", id);

        Ok(())
    }

    
    pub fn promote_to_vram(&mut self, id: PhysicalBlockId) -> Result<()> {
        let block = self.blocks.get_mut(id.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid block ID: {:?}", id))?;

        if block.location != BlockLocation::Ram {
            bail!("Block {:?} is not in RAM", id);
        }

        
        block.location = BlockLocation::Vram;
        self.ram_used = self.ram_used.saturating_sub(1);
        self.vram_used += 1;

        eprintln!("⬆️  Promoted block {:?} to VRAM", id);

        Ok(())
    }

    
    pub fn get_block(&self, id: PhysicalBlockId) -> Option<&PhysicalBlock> {
        self.blocks.get(id.0 as usize)
    }

    
    pub fn get_block_mut(&mut self, id: PhysicalBlockId) -> Option<&mut PhysicalBlock> {
        self.blocks.get_mut(id.0 as usize)
    }

    
    pub fn vram_utilization(&self) -> f32 {
        if self.vram_capacity == 0 {
            return 0.0;
        }
        self.vram_used as f32 / self.vram_capacity as f32
    }

    
    pub fn ram_utilization(&self) -> f32 {
        if self.ram_capacity == 0 {
            return 0.0;
        }
        self.ram_used as f32 / self.ram_capacity as f32
    }

    
    pub fn vram_used(&self) -> usize {
        self.vram_used
    }

    
    pub fn ram_used(&self) -> usize {
        self.ram_used
    }

    
    pub fn find_lru_vram_block(&self, exclude_sessions: &[SessionId]) -> Option<PhysicalBlockId> {
        self.blocks
            .iter()
            .filter(|b| {
                b.location == BlockLocation::Vram &&
                b.ref_count > 0 &&
                !exclude_sessions.contains(&b.owner_session.unwrap_or(SessionId(u64::MAX)))
            })
            .min_by_key(|b| b.last_access)
            .map(|b| b.id)
    }
}




pub struct MemoryBudgeter {
    
    total_vram_mb: u64,

    
    system_floor_mb: u64,

    
    swap_threshold_mb: u64,

    
    cooldown_until: Option<Instant>,

    
    swap_enabled: bool,
}

impl MemoryBudgeter {
    
    pub fn new(total_vram_mb: u64) -> Self {
        let system_floor_mb = (total_vram_mb as f32 * SYSTEM_FLOOR_RATIO) as u64;
        let system_floor_mb = system_floor_mb.max(SYSTEM_FLOOR_MIN_MB);

        let swap_threshold_mb = (total_vram_mb as f32 * SWAP_THRESHOLD_RATIO) as u64;

        eprintln!("📊 MemoryBudgeter initialized:");
        eprintln!("   Total VRAM: {} MB", total_vram_mb);
        eprintln!("   System Floor: {} MB ({:.1}%)",
                 system_floor_mb,
                 (system_floor_mb as f32 / total_vram_mb as f32) * 100.0);
        eprintln!("   Swap Threshold: {} MB ({:.1}%)",
                 swap_threshold_mb,
                 (swap_threshold_mb as f32 / total_vram_mb as f32) * 100.0);

        Self {
            total_vram_mb,
            system_floor_mb,
            swap_threshold_mb,
            cooldown_until: None,
            swap_enabled: true,
        }
    }

    
    pub fn should_swap(&self, current_usage_mb: u64) -> bool {
        if !self.swap_enabled {
            return false;
        }

        
        if let Some(cooldown_end) = self.cooldown_until {
            if Instant::now() < cooldown_end {
                return false;
            }
        }

        current_usage_mb >= self.swap_threshold_mb
    }

    
    pub fn start_cooldown(&mut self) {
        self.cooldown_until = Some(Instant::now() + Duration::from_secs(COOLDOWN_DURATION_SECS));
        eprintln!("❄️  Cool-down started ({} seconds)", COOLDOWN_DURATION_SECS);
    }

    
    pub fn available_vram_mb(&self) -> u64 {
        self.total_vram_mb.saturating_sub(self.system_floor_mb)
    }

    
    pub fn set_swap_enabled(&mut self, enabled: bool) {
        self.swap_enabled = enabled;
    }
}




pub struct SessionManager {
    
    sessions: HashMap<SessionId, BlockTable>,

    
    allocator: Arc<Mutex<PhysicalBlockAllocator>>,

    
    budgeter: Arc<RwLock<MemoryBudgeter>>,

    
    next_session_id: u64,
}

impl SessionManager {
    
    pub fn new(
        vram_mb: u64,
        ram_mb: u64,
        block_size_kb: usize,
    ) -> Self {
        let allocator = PhysicalBlockAllocator::new(vram_mb, ram_mb, block_size_kb);
        let budgeter = MemoryBudgeter::new(vram_mb);

        Self {
            sessions: HashMap::new(),
            allocator: Arc::new(Mutex::new(allocator)),
            budgeter: Arc::new(RwLock::new(budgeter)),
            next_session_id: 1,
        }
    }

    
    pub fn create_session(&mut self) -> SessionId {
        let session_id = SessionId(self.next_session_id);
        self.next_session_id += 1;

        let block_table = BlockTable::new(session_id);
        self.sessions.insert(session_id, block_table);

        eprintln!("✨ Created session {:?}", session_id);

        session_id
    }

    
    pub fn allocate_block(&mut self, session_id: SessionId) -> Result<LogicalBlockId> {
        let block_table = self.sessions.get_mut(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {:?} not found", session_id))?;

        let logical_id = block_table.allocate_logical_block();

        
        let mut allocator = self.allocator.lock();
        let physical_id = if let Some(id) = allocator.allocate_vram(session_id) {
            id
        } else {
            
            let budgeter = self.budgeter.read();
            let vram_used_mb = (allocator.vram_used * 256) as u64 / 1024; 

            if budgeter.should_swap(vram_used_mb) {
                drop(budgeter);

                
                eprintln!("⚠️  VRAM full, triggering LRU swap...");
                let exclude = vec![session_id];

                if let Some(victim_id) = allocator.find_lru_vram_block(&exclude) {
                    allocator.swap_to_ram(victim_id)?;

                    
                    allocator.allocate_vram(session_id)
                        .ok_or_else(|| anyhow::anyhow!("VRAM allocation failed after swap"))?
                } else {
                    
                    allocator.allocate_ram(session_id)
                        .ok_or_else(|| anyhow::anyhow!("RAM allocation failed"))?
                }
            } else {
                
                allocator.allocate_ram(session_id)
                    .ok_or_else(|| anyhow::anyhow!("RAM allocation failed"))?
            }
        };

        block_table.map(logical_id, physical_id);

        Ok(logical_id)
    }

    
    pub fn free_session(&mut self, session_id: SessionId) -> Result<()> {
        let block_table = self.sessions.remove(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {:?} not found", session_id))?;

        let mut allocator = self.allocator.lock();

        for (_, physical_id) in block_table.mappings {
            allocator.free(physical_id)?;
        }

        eprintln!("🗑️  Freed session {:?} ({} blocks)", session_id, block_table.num_blocks);

        Ok(())
    }

    
    pub fn get_block_table(&self, session_id: SessionId) -> Option<&BlockTable> {
        self.sessions.get(&session_id)
    }

    
    pub fn stats(&self) -> SessionManagerStats {
        let allocator = self.allocator.lock();

        SessionManagerStats {
            total_sessions: self.sessions.len(),
            vram_used_blocks: allocator.vram_used,
            vram_total_blocks: allocator.vram_capacity,
            ram_used_blocks: allocator.ram_used,
            ram_total_blocks: allocator.ram_capacity,
            vram_utilization: allocator.vram_utilization(),
            ram_utilization: allocator.ram_utilization(),
        }
    }
}


#[derive(Debug, Clone)]
pub struct SessionManagerStats {
    pub total_sessions: usize,
    pub vram_used_blocks: usize,
    pub vram_total_blocks: usize,
    pub ram_used_blocks: usize,
    pub ram_total_blocks: usize,
    pub vram_utilization: f32,
    pub ram_utilization: f32,
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_basic() {
        let session_id = SessionId(1);
        let mut table = BlockTable::new(session_id);

        let logical = table.allocate_logical_block();
        let physical = PhysicalBlockId(42);

        table.map(logical, physical);
        assert_eq!(table.get_physical(logical), Some(physical));

        table.unmap(logical);
        assert_eq!(table.get_physical(logical), None);
    }

    #[test]
    fn test_physical_block_allocator() {
        let mut allocator = PhysicalBlockAllocator::new(1024, 2048, 256);
        let session_id = SessionId(1);

        
        let id = allocator.allocate_vram(session_id).unwrap();
        assert_eq!(allocator.vram_used, 1);

        
        allocator.free(id).unwrap();
        assert_eq!(allocator.vram_used, 0);
    }

    #[test]
    fn test_session_manager_basic() {
        let mut manager = SessionManager::new(1024, 2048, 256);
        let session_id = manager.create_session();

        
        for _ in 0..10 {
            manager.allocate_block(session_id).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.total_sessions, 1);
        assert!(stats.vram_used_blocks > 0);

        
        manager.free_session(session_id).unwrap();
        let stats = manager.stats();
        assert_eq!(stats.total_sessions, 0);
    }

    #[test]
    fn test_memory_budgeter() {
        let mut budgeter = MemoryBudgeter::new(10000);

        
        assert!(!budgeter.should_swap(5000));

        
        assert!(budgeter.should_swap(9000));

        
        budgeter.start_cooldown();
        assert!(!budgeter.should_swap(9000));
    }
}

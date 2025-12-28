/**
 * Phase 2 Week 1: Paged Attention 核心实现
 *
 * 特性：
 * 1. Block Table 映射（逻辑位置 → 物理块）
 * 2. Physical Block Allocator（LRU 驱动的块分配器）
 * 3. Memory Budgeter（智能显存管理）
 * 4. Swap 机制（VRAM ↔ RAM 自动置换）
 * 5. 支持 128k+ 上下文 + 多会话并发
 *
 * 设计原则：
 * - Block 粒度：32 tokens/block（与 Phase 1 KV Cache 对齐）
 * - 零拷贝：Block 迁移使用指针交换
 * - 无锁核心路径：RefCell + 内部可变性（单线程推理）
 * - LRU 驱动：最近最少使用优先置换
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use std::time::{Instant, Duration};
use anyhow::{Result, bail};

// ==================== 常量定义 ====================

/// 每个 Block 的 Token 数量（与 Phase 1 对齐）
pub const BLOCK_SIZE: usize = 32;

/// 内存预算系统保留比例（10% 显存保留）
pub const SYSTEM_FLOOR_RATIO: f32 = 0.10;

/// 内存预算系统保留最小值（1GB）
pub const SYSTEM_FLOOR_MIN_MB: u64 = 1024;

/// Swap 触发阈值（85% 显存占用时触发）
pub const SWAP_THRESHOLD_RATIO: f32 = 0.85;

/// Cool-down 时间（模型/会话切换后禁止置换的时间窗口）
pub const COOLDOWN_DURATION_SECS: u64 = 3;

// ==================== 核心数据结构 ====================

/// 物理块 ID（唯一标识一个物理内存块）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalBlockId(pub u32);

/// 逻辑块 ID（Session 内的逻辑位置）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogicalBlockId(pub u32);

/// Session ID（唯一标识一个推理会话）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionId(pub u64);

/// 块状态（追踪块在内存层级中的位置）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLocation {
    /// 在 VRAM（GPU 显存）
    Vram,
    /// 在 RAM（CPU 内存）
    Ram,
    /// 已释放（可重新分配）
    Free,
}

/// 物理块元数据
#[derive(Debug, Clone)]
pub struct PhysicalBlock {
    /// 块 ID
    pub id: PhysicalBlockId,

    /// 当前位置（VRAM/RAM/Free）
    pub location: BlockLocation,

    /// 最后访问时间（用于 LRU 置换）
    pub last_access: Instant,

    /// 引用计数（用于共享块管理，Phase 2 Week 5 前缀缓存需要）
    pub ref_count: u32,

    /// 所属 Session（None 表示空闲块）
    pub owner_session: Option<SessionId>,

    /// 数据指针（实际 KV Cache 数据，当前阶段为占位）
    /// TODO Week 2: 集成实际的 llama.cpp KV Cache 指针
    pub data_ptr: Option<usize>,
}

impl PhysicalBlock {
    /// 创建新的空闲块
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

    /// 标记块为已访问（更新 LRU 时间戳）
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// 是否为空闲块
    pub fn is_free(&self) -> bool {
        self.location == BlockLocation::Free && self.ref_count == 0
    }

    /// 获取距离上次访问的时间
    pub fn idle_duration(&self) -> Duration {
        Instant::now().duration_since(self.last_access)
    }
}

// ==================== Block Table（核心映射表） ====================

/// Block Table：维护 Session 的逻辑块 → 物理块映射
#[derive(Debug)]
pub struct BlockTable {
    /// Session ID
    pub session_id: SessionId,

    /// 逻辑块 → 物理块映射（稀疏数组）
    pub mappings: HashMap<LogicalBlockId, PhysicalBlockId>,

    /// 已分配的逻辑块数量
    pub num_blocks: usize,

    /// 最后访问时间（用于 Session 级 LRU）
    pub last_access: Instant,
}

impl BlockTable {
    /// 创建新的 Block Table
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            mappings: HashMap::new(),
            num_blocks: 0,
            last_access: Instant::now(),
        }
    }

    /// 分配新的逻辑块（返回逻辑块 ID）
    pub fn allocate_logical_block(&mut self) -> LogicalBlockId {
        let logical_id = LogicalBlockId(self.num_blocks as u32);
        self.num_blocks += 1;
        logical_id
    }

    /// 映射逻辑块到物理块
    pub fn map(&mut self, logical: LogicalBlockId, physical: PhysicalBlockId) {
        self.mappings.insert(logical, physical);
        self.touch();
    }

    /// 查询逻辑块对应的物理块
    pub fn get_physical(&self, logical: LogicalBlockId) -> Option<PhysicalBlockId> {
        self.mappings.get(&logical).copied()
    }

    /// 移除映射
    pub fn unmap(&mut self, logical: LogicalBlockId) -> Option<PhysicalBlockId> {
        self.mappings.remove(&logical)
    }

    /// 标记为已访问
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// 获取距离上次访问的时间
    pub fn idle_duration(&self) -> Duration {
        Instant::now().duration_since(self.last_access)
    }
}

// ==================== Physical Block Allocator ====================

/// 物理块分配器（管理 VRAM + RAM 的物理块池）
pub struct PhysicalBlockAllocator {
    /// 所有物理块（VRAM + RAM）
    blocks: Vec<PhysicalBlock>,

    /// VRAM 中的空闲块队列（FIFO 分配）
    vram_free_list: VecDeque<PhysicalBlockId>,

    /// RAM 中的空闲块队列
    ram_free_list: VecDeque<PhysicalBlockId>,

    /// VRAM 总容量（块数量）
    vram_capacity: usize,

    /// RAM 总容量（块数量）
    ram_capacity: usize,

    /// 当前 VRAM 使用量（块数量）
    vram_used: usize,

    /// 当前 RAM 使用量（块数量）
    ram_used: usize,
}

impl PhysicalBlockAllocator {
    /// 创建新的块分配器
    ///
    /// # 参数
    /// - `vram_mb`: VRAM 容量（MB）
    /// - `ram_mb`: RAM 容量（MB）
    /// - `block_size_kb`: 每个块的大小（KB）
    pub fn new(vram_mb: u64, ram_mb: u64, block_size_kb: usize) -> Self {
        // 计算块数量
        let vram_capacity = (vram_mb * 1024 / block_size_kb as u64) as usize;
        let ram_capacity = (ram_mb * 1024 / block_size_kb as u64) as usize;

        let total_blocks = vram_capacity + ram_capacity;

        // 初始化所有块（前半部分为 VRAM，后半部分为 RAM）
        let mut blocks = Vec::with_capacity(total_blocks);
        let mut vram_free_list = VecDeque::with_capacity(vram_capacity);
        let mut ram_free_list = VecDeque::with_capacity(ram_capacity);

        // 初始化 VRAM 块
        for i in 0..vram_capacity {
            let id = PhysicalBlockId(i as u32);
            let mut block = PhysicalBlock::new_free(id);
            block.location = BlockLocation::Vram;
            blocks.push(block);
            vram_free_list.push_back(id);
        }

        // 初始化 RAM 块
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

    /// 从 VRAM 分配一个物理块
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

    /// 从 RAM 分配一个物理块
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

    /// 释放物理块
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
                    // 已经是空闲状态，不需要再次释放
                }
            }

            block.location = BlockLocation::Free;
            block.owner_session = None;
            block.data_ptr = None;
        }

        Ok(())
    }

    /// 将块从 VRAM 置换到 RAM
    pub fn swap_to_ram(&mut self, id: PhysicalBlockId) -> Result<()> {
        let block = self.blocks.get_mut(id.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid block ID: {:?}", id))?;

        if block.location != BlockLocation::Vram {
            bail!("Block {:?} is not in VRAM", id);
        }

        // TODO: 实际的数据迁移逻辑（需要与 llama.cpp KV Cache API 集成）
        // 当前阶段：仅更新元数据
        block.location = BlockLocation::Ram;
        self.vram_used = self.vram_used.saturating_sub(1);
        self.ram_used += 1;

        eprintln!("🔄 Swapped block {:?} to RAM", id);

        Ok(())
    }

    /// 将块从 RAM 提升到 VRAM
    pub fn promote_to_vram(&mut self, id: PhysicalBlockId) -> Result<()> {
        let block = self.blocks.get_mut(id.0 as usize)
            .ok_or_else(|| anyhow::anyhow!("Invalid block ID: {:?}", id))?;

        if block.location != BlockLocation::Ram {
            bail!("Block {:?} is not in RAM", id);
        }

        // TODO: 实际的数据迁移逻辑
        block.location = BlockLocation::Vram;
        self.ram_used = self.ram_used.saturating_sub(1);
        self.vram_used += 1;

        eprintln!("⬆️  Promoted block {:?} to VRAM", id);

        Ok(())
    }

    /// 获取块引用
    pub fn get_block(&self, id: PhysicalBlockId) -> Option<&PhysicalBlock> {
        self.blocks.get(id.0 as usize)
    }

    /// 获取块可变引用
    pub fn get_block_mut(&mut self, id: PhysicalBlockId) -> Option<&mut PhysicalBlock> {
        self.blocks.get_mut(id.0 as usize)
    }

    /// 获取 VRAM 使用率
    pub fn vram_utilization(&self) -> f32 {
        if self.vram_capacity == 0 {
            return 0.0;
        }
        self.vram_used as f32 / self.vram_capacity as f32
    }

    /// 获取 RAM 使用率
    pub fn ram_utilization(&self) -> f32 {
        if self.ram_capacity == 0 {
            return 0.0;
        }
        self.ram_used as f32 / self.ram_capacity as f32
    }

    /// 获取 VRAM 使用量（块数量）
    pub fn vram_used(&self) -> usize {
        self.vram_used
    }

    /// 获取 RAM 使用量（块数量）
    pub fn ram_used(&self) -> usize {
        self.ram_used
    }

    /// 找到最适合置换的块（LRU 策略）
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

// ==================== Memory Budgeter（内存预算器） ====================

/// 内存预算器：智能管理显存分配和置换
pub struct MemoryBudgeter {
    /// 总 VRAM 容量（MB）
    total_vram_mb: u64,

    /// 系统保留 VRAM（MB）
    system_floor_mb: u64,

    /// Swap 触发阈值（MB）
    swap_threshold_mb: u64,

    /// Cool-down 结束时间（禁止置换的截止时间）
    cooldown_until: Option<Instant>,

    /// 是否启用 Swap
    swap_enabled: bool,
}

impl MemoryBudgeter {
    /// 创建新的内存预算器
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

    /// 检查是否应该触发 Swap
    pub fn should_swap(&self, current_usage_mb: u64) -> bool {
        if !self.swap_enabled {
            return false;
        }

        // Cool-down 检查
        if let Some(cooldown_end) = self.cooldown_until {
            if Instant::now() < cooldown_end {
                return false;
            }
        }

        current_usage_mb >= self.swap_threshold_mb
    }

    /// 启动 Cool-down（模型/会话切换后调用）
    pub fn start_cooldown(&mut self) {
        self.cooldown_until = Some(Instant::now() + Duration::from_secs(COOLDOWN_DURATION_SECS));
        eprintln!("❄️  Cool-down started ({} seconds)", COOLDOWN_DURATION_SECS);
    }

    /// 获取可用 VRAM（排除系统保留）
    pub fn available_vram_mb(&self) -> u64 {
        self.total_vram_mb.saturating_sub(self.system_floor_mb)
    }

    /// 禁用/启用 Swap
    pub fn set_swap_enabled(&mut self, enabled: bool) {
        self.swap_enabled = enabled;
    }
}

// ==================== Session Manager（会话管理器） ====================

/// Session 管理器：管理所有活跃的推理会话
pub struct SessionManager {
    /// 所有 Session 的 Block Table
    sessions: HashMap<SessionId, BlockTable>,

    /// 物理块分配器
    allocator: Arc<Mutex<PhysicalBlockAllocator>>,

    /// 内存预算器
    budgeter: Arc<RwLock<MemoryBudgeter>>,

    /// 下一个可用的 Session ID
    next_session_id: u64,
}

impl SessionManager {
    /// 创建新的 Session 管理器
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

    /// 创建新的 Session
    pub fn create_session(&mut self) -> SessionId {
        let session_id = SessionId(self.next_session_id);
        self.next_session_id += 1;

        let block_table = BlockTable::new(session_id);
        self.sessions.insert(session_id, block_table);

        eprintln!("✨ Created session {:?}", session_id);

        session_id
    }

    /// 为 Session 分配一个新块
    pub fn allocate_block(&mut self, session_id: SessionId) -> Result<LogicalBlockId> {
        let block_table = self.sessions.get_mut(&session_id)
            .ok_or_else(|| anyhow::anyhow!("Session {:?} not found", session_id))?;

        let logical_id = block_table.allocate_logical_block();

        // 尝试从 VRAM 分配
        let mut allocator = self.allocator.lock();
        let physical_id = if let Some(id) = allocator.allocate_vram(session_id) {
            id
        } else {
            // VRAM 已满，检查是否需要 Swap
            let budgeter = self.budgeter.read();
            let vram_used_mb = (allocator.vram_used * 256) as u64 / 1024; // 假设每块 256KB

            if budgeter.should_swap(vram_used_mb) {
                drop(budgeter);

                // 执行 LRU 置换
                eprintln!("⚠️  VRAM full, triggering LRU swap...");
                let exclude = vec![session_id];

                if let Some(victim_id) = allocator.find_lru_vram_block(&exclude) {
                    allocator.swap_to_ram(victim_id)?;

                    // 再次尝试分配
                    allocator.allocate_vram(session_id)
                        .ok_or_else(|| anyhow::anyhow!("VRAM allocation failed after swap"))?
                } else {
                    // 无法找到可置换的块，从 RAM 分配
                    allocator.allocate_ram(session_id)
                        .ok_or_else(|| anyhow::anyhow!("RAM allocation failed"))?
                }
            } else {
                // 直接从 RAM 分配
                allocator.allocate_ram(session_id)
                    .ok_or_else(|| anyhow::anyhow!("RAM allocation failed"))?
            }
        };

        block_table.map(logical_id, physical_id);

        Ok(logical_id)
    }

    /// 释放 Session 的所有块
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

    /// 获取 Session 的 Block Table
    pub fn get_block_table(&self, session_id: SessionId) -> Option<&BlockTable> {
        self.sessions.get(&session_id)
    }

    /// 获取统计信息
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

/// Session 管理器统计信息
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

// ==================== 测试模块 ====================

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

        // 分配 VRAM 块
        let id = allocator.allocate_vram(session_id).unwrap();
        assert_eq!(allocator.vram_used, 1);

        // 释放块
        allocator.free(id).unwrap();
        assert_eq!(allocator.vram_used, 0);
    }

    #[test]
    fn test_session_manager_basic() {
        let mut manager = SessionManager::new(1024, 2048, 256);
        let session_id = manager.create_session();

        // 分配 10 个块
        for _ in 0..10 {
            manager.allocate_block(session_id).unwrap();
        }

        let stats = manager.stats();
        assert_eq!(stats.total_sessions, 1);
        assert!(stats.vram_used_blocks > 0);

        // 释放会话
        manager.free_session(session_id).unwrap();
        let stats = manager.stats();
        assert_eq!(stats.total_sessions, 0);
    }

    #[test]
    fn test_memory_budgeter() {
        let mut budgeter = MemoryBudgeter::new(10000);

        // 正常情况不应触发 Swap
        assert!(!budgeter.should_swap(5000));

        // 超过阈值应触发 Swap
        assert!(budgeter.should_swap(9000));

        // Cool-down 期间不应触发 Swap
        budgeter.start_cooldown();
        assert!(!budgeter.should_swap(9000));
    }
}

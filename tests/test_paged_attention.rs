/**
 * Phase 2 Week 1: Paged Attention 综合测试
 *
 * 测试目标：
 * 1. 基础功能：Block Table 映射、分配器、预算器
 * 2. 压力测试：8 个 128k 上下文并发会话
 * 3. Swap 机制：VRAM 溢出时自动置换
 * 4. LRU 策略：正确驱逐最久未访问的块
 * 5. Cool-down：置换冷却期正确工作
 */

use loci::paged_attention::*;
use std::thread;
use std::time::Duration;

// ==================== 基础功能测试 ====================

#[test]
fn test_block_table_lifecycle() {
    let session_id = SessionId(1);
    let mut table = BlockTable::new(session_id);

    // 分配 100 个逻辑块
    let mut mappings = Vec::new();
    for i in 0..100 {
        let logical = table.allocate_logical_block();
        let physical = PhysicalBlockId(i);
        table.map(logical, physical);
        mappings.push((logical, physical));
    }

    // 验证映射
    for (logical, physical) in &mappings {
        assert_eq!(table.get_physical(*logical), Some(*physical));
    }

    // 移除一半映射
    for i in (0..100).step_by(2) {
        let logical = LogicalBlockId(i);
        assert!(table.unmap(logical).is_some());
        assert_eq!(table.get_physical(logical), None);
    }

    println!("✅ Block table lifecycle test passed");
}

#[test]
fn test_physical_allocator_exhaustion() {
    let mut allocator = PhysicalBlockAllocator::new(
        512,  // 512MB VRAM
        1024, // 1GB RAM
        256,  // 256KB per block
    );

    let session_id = SessionId(1);

    // 计算预期块数量
    let vram_blocks = (512 * 1024 / 256) as usize;
    let ram_blocks = (1024 * 1024 / 256) as usize;

    // 填满 VRAM
    let mut vram_allocs = Vec::new();
    for _ in 0..vram_blocks {
        if let Some(id) = allocator.allocate_vram(session_id) {
            vram_allocs.push(id);
        } else {
            break;
        }
    }

    println!("Allocated {} VRAM blocks (expected {})", vram_allocs.len(), vram_blocks);
    assert!(vram_allocs.len() > 0);
    assert_eq!(allocator.vram_utilization(), 1.0);

    // VRAM 已满，下一次分配应该失败
    assert!(allocator.allocate_vram(session_id).is_none());

    // 但 RAM 应该仍然可用
    let mut ram_allocs = Vec::new();
    for _ in 0..ram_blocks {
        if let Some(id) = allocator.allocate_ram(session_id) {
            ram_allocs.push(id);
        } else {
            break;
        }
    }

    println!("Allocated {} RAM blocks (expected {})", ram_allocs.len(), ram_blocks);
    assert!(ram_allocs.len() > 0);

    // 释放所有 VRAM 块
    for id in vram_allocs {
        allocator.free(id).unwrap();
    }

    assert_eq!(allocator.vram_used(), 0);
    println!("✅ Allocator exhaustion test passed");
}

#[test]
fn test_lru_eviction() {
    let mut allocator = PhysicalBlockAllocator::new(
        256,  // 小 VRAM 触发置换
        1024,
        256,
    );

    let session1 = SessionId(1);
    let session2 = SessionId(2);

    // Session 1 分配 2 个块
    let block1 = allocator.allocate_vram(session1).unwrap();
    thread::sleep(Duration::from_millis(10));

    let _block2 = allocator.allocate_vram(session1).unwrap();
    thread::sleep(Duration::from_millis(10));

    // Session 2 分配 1 个块（更新时间戳）
    let block3 = allocator.allocate_vram(session2).unwrap();

    // 找到 LRU 块（应该是 block1，最早分配）
    let lru = allocator.find_lru_vram_block(&[]).unwrap();
    assert_eq!(lru, block1);

    // 排除 session1 后，应该选中 block3
    let lru_exclude = allocator.find_lru_vram_block(&[session1]).unwrap();
    assert_eq!(lru_exclude, block3);

    println!("✅ LRU eviction test passed");
}

#[test]
fn test_memory_budgeter_thresholds() {
    let mut budgeter = MemoryBudgeter::new(10000); // 10GB VRAM

    // 低于阈值：不应触发 Swap
    assert!(!budgeter.should_swap(5000));

    // 达到阈值：应触发 Swap (85% = 8500MB)
    assert!(budgeter.should_swap(8500));

    // 超过阈值：应触发 Swap
    assert!(budgeter.should_swap(9000));

    // 启动 Cool-down
    budgeter.start_cooldown();

    // Cool-down 期间：即使超过阈值也不应触发 Swap
    assert!(!budgeter.should_swap(9000));

    // 等待 Cool-down 结束（3 秒）
    thread::sleep(Duration::from_secs(4));

    // Cool-down 结束后：应恢复正常
    assert!(budgeter.should_swap(9000));

    println!("✅ Memory budgeter thresholds test passed");
}

// ==================== Session Manager 集成测试 ====================

#[test]
fn test_session_manager_basic_operations() {
    let mut manager = SessionManager::new(
        1024, // 1GB VRAM
        2048, // 2GB RAM
        256,  // 256KB per block
    );

    // 创建 3 个 Session
    let session1 = manager.create_session();
    let session2 = manager.create_session();
    let session3 = manager.create_session();

    // 每个 Session 分配 10 个块
    for _ in 0..10 {
        manager.allocate_block(session1).unwrap();
        manager.allocate_block(session2).unwrap();
        manager.allocate_block(session3).unwrap();
    }

    let stats = manager.stats();
    assert_eq!(stats.total_sessions, 3);
    assert_eq!(stats.vram_used_blocks + stats.ram_used_blocks, 30);

    println!("Stats: {:?}", stats);

    // 释放 Session 1
    manager.free_session(session1).unwrap();
    let stats = manager.stats();
    assert_eq!(stats.total_sessions, 2);
    assert_eq!(stats.vram_used_blocks + stats.ram_used_blocks, 20);

    // 释放剩余 Session
    manager.free_session(session2).unwrap();
    manager.free_session(session3).unwrap();

    let stats = manager.stats();
    assert_eq!(stats.total_sessions, 0);
    assert_eq!(stats.vram_used_blocks, 0);
    assert_eq!(stats.ram_used_blocks, 0);

    println!("✅ Session manager basic operations test passed");
}

// ==================== 压力测试 ====================

#[test]
fn test_concurrent_sessions_128k_context() {
    // 模拟 8 个 128k 上下文的并发会话
    // 128k tokens / 32 tokens per block = 4000 blocks per session
    // 8 sessions * 4000 blocks = 32000 blocks total

    const CONTEXT_SIZE: usize = 128 * 1024; // 128k tokens
    const BLOCKS_PER_SESSION: usize = CONTEXT_SIZE / BLOCK_SIZE;
    const NUM_SESSIONS: usize = 8;

    let mut manager = SessionManager::new(
        4096,  // 4GB VRAM
        16384, // 16GB RAM
        256,   // 256KB per block
    );

    println!("🚀 Starting concurrent sessions stress test:");
    println!("   Sessions: {}", NUM_SESSIONS);
    println!("   Context per session: {} tokens ({} blocks)", CONTEXT_SIZE, BLOCKS_PER_SESSION);
    println!("   Total blocks: {}", BLOCKS_PER_SESSION * NUM_SESSIONS);

    // 创建 8 个会话
    let mut sessions = Vec::new();
    for i in 0..NUM_SESSIONS {
        let session_id = manager.create_session();
        sessions.push(session_id);
        println!("   ✨ Created session {} ({:?})", i + 1, session_id);
    }

    // 为每个会话分配所有块
    let start = std::time::Instant::now();

    for (i, session_id) in sessions.iter().enumerate() {
        println!("   📦 Allocating blocks for session {}...", i + 1);

        for block_idx in 0..BLOCKS_PER_SESSION {
            if let Err(e) = manager.allocate_block(*session_id) {
                panic!("❌ Failed to allocate block {} for session {}: {}", block_idx, i + 1, e);
            }

            // 每 1000 个块打印进度
            if block_idx > 0 && block_idx % 1000 == 0 {
                let stats = manager.stats();
                println!(
                    "      Progress: {}/{} blocks | VRAM: {:.1}% | RAM: {:.1}%",
                    block_idx,
                    BLOCKS_PER_SESSION,
                    stats.vram_utilization * 100.0,
                    stats.ram_utilization * 100.0
                );
            }
        }

        let stats = manager.stats();
        println!(
            "   ✅ Session {} allocated | VRAM: {}/{} ({:.1}%) | RAM: {}/{} ({:.1}%)",
            i + 1,
            stats.vram_used_blocks,
            stats.vram_total_blocks,
            stats.vram_utilization * 100.0,
            stats.ram_used_blocks,
            stats.ram_total_blocks,
            stats.ram_utilization * 100.0
        );
    }

    let elapsed = start.elapsed();
    println!("   ⏱️  Total allocation time: {:.2}s", elapsed.as_secs_f64());

    // 验证最终状态
    let final_stats = manager.stats();
    assert_eq!(final_stats.total_sessions, NUM_SESSIONS);

    let total_allocated = final_stats.vram_used_blocks + final_stats.ram_used_blocks;
    println!("   📊 Final stats:");
    println!(
        "      Total allocated blocks: {} (expected {})",
        total_allocated,
        BLOCKS_PER_SESSION * NUM_SESSIONS
    );
    println!(
        "      VRAM: {}/{} ({:.1}%)",
        final_stats.vram_used_blocks,
        final_stats.vram_total_blocks,
        final_stats.vram_utilization * 100.0
    );
    println!(
        "      RAM: {}/{} ({:.1}%)",
        final_stats.ram_used_blocks,
        final_stats.ram_total_blocks,
        final_stats.ram_utilization * 100.0
    );

    // 系统应该没有崩溃，所有分配都成功
    println!("✅ Concurrent sessions stress test passed!");
}

#[test]
fn test_swap_mechanism_under_pressure() {
    // 测试 VRAM 不足时的自动 Swap 机制
    let mut manager = SessionManager::new(
        512,  // 仅 512MB VRAM（故意设置很小以触发 Swap）
        4096, // 4GB RAM
        256,  // 256KB per block
    );

    let vram_blocks = (512 * 1024 / 256) as usize;
    println!("🧪 Testing swap mechanism:");
    println!("   VRAM capacity: {} blocks", vram_blocks);

    let session1 = manager.create_session();
    let session2 = manager.create_session();

    // Session 1 占满 VRAM
    println!("   📦 Filling VRAM with session 1...");
    for i in 0..vram_blocks {
        manager.allocate_block(session1).unwrap();

        if i % 500 == 0 {
            let stats = manager.stats();
            println!(
                "      Progress: {}/{} | VRAM: {:.1}%",
                i,
                vram_blocks,
                stats.vram_utilization * 100.0
            );
        }
    }

    let stats_after_fill = manager.stats();
    println!("   ✅ VRAM filled: {:.1}%", stats_after_fill.vram_utilization * 100.0);

    // Session 2 继续分配（应触发 Swap）
    println!("   🔄 Allocating for session 2 (should trigger swap)...");
    for i in 0..100 {
        manager.allocate_block(session2).unwrap();

        if i % 20 == 0 {
            let stats = manager.stats();
            println!(
                "      Swap progress: {} blocks | VRAM: {:.1}% | RAM: {:.1}%",
                i,
                stats.vram_utilization * 100.0,
                stats.ram_utilization * 100.0
            );
        }
    }

    let final_stats = manager.stats();
    println!("   📊 Final state:");
    println!(
        "      VRAM: {}/{} ({:.1}%)",
        final_stats.vram_used_blocks,
        final_stats.vram_total_blocks,
        final_stats.vram_utilization * 100.0
    );
    println!(
        "      RAM: {}/{} ({:.1}%)",
        final_stats.ram_used_blocks,
        final_stats.ram_total_blocks,
        final_stats.ram_utilization * 100.0
    );

    // 验证 RAM 被使用（说明发生了 Swap）
    assert!(final_stats.ram_used_blocks > 0, "RAM should have been used for swapping");

    println!("✅ Swap mechanism test passed!");
}

// ==================== 性能基准测试 ====================

#[test]
#[ignore] // 使用 `cargo test -- --ignored` 运行
fn benchmark_allocation_performance() {
    let mut manager = SessionManager::new(
        8192,  // 8GB VRAM
        16384, // 16GB RAM
        256,
    );

    let session = manager.create_session();

    const NUM_BLOCKS: usize = 10000;

    println!("⏱️  Benchmarking allocation performance ({} blocks)...", NUM_BLOCKS);

    let start = std::time::Instant::now();

    for _ in 0..NUM_BLOCKS {
        manager.allocate_block(session).unwrap();
    }

    let elapsed = start.elapsed();
    let blocks_per_sec = NUM_BLOCKS as f64 / elapsed.as_secs_f64();

    println!("   ✅ Allocated {} blocks in {:.3}s", NUM_BLOCKS, elapsed.as_secs_f64());
    println!("   📈 Performance: {:.0} blocks/sec", blocks_per_sec);

    // 性能目标：至少 10000 blocks/sec
    assert!(blocks_per_sec > 10000.0, "Allocation performance below target");
}

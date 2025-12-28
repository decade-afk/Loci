/**
 * Phase 2 Week 1: Paged Attention 演示程序
 *
 * 展示功能：
 * 1. 创建多个并发会话
 * 2. 动态内存分配与管理
 * 3. 自动 Swap 机制
 * 4. 实时统计信息
 */

use loci::paged_attention::{SessionManager, BLOCK_SIZE};
use std::time::Instant;

fn main() {
    println!("╔════════════════════════════════════════╗");
    println!("║   Loci Phase 2 Week 1 Demo            ║");
    println!("║   Paged Attention System               ║");
    println!("╚════════════════════════════════════════╗");
    println!();

    // 配置：4GB VRAM + 16GB RAM
    const VRAM_MB: u64 = 4096;
    const RAM_MB: u64 = 16384;
    const BLOCK_SIZE_KB: usize = 256;

    println!("⚙️  Configuration:");
    println!("   VRAM: {} GB", VRAM_MB / 1024);
    println!("   RAM: {} GB", RAM_MB / 1024);
    println!("   Block Size: {} KB ({} tokens)", BLOCK_SIZE_KB, BLOCK_SIZE);
    println!();

    // 创建 Session Manager
    let mut manager = SessionManager::new(VRAM_MB, RAM_MB, BLOCK_SIZE_KB);

    // Demo 1: 基础会话操作
    demo_basic_sessions(&mut manager);

    // Demo 2: 大上下文场景（模拟 128k tokens）
    demo_large_context(&mut manager);

    // Demo 3: 多会话并发（8 个会话）
    demo_concurrent_sessions(&mut manager);

    println!();
    println!("✅ All demos completed successfully!");
}

fn demo_basic_sessions(manager: &mut SessionManager) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 1: 基础会话操作");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 创建 3 个会话
    let session1 = manager.create_session();
    let session2 = manager.create_session();
    let session3 = manager.create_session();

    println!("✨ Created 3 sessions");

    // 每个会话分配 10 个块
    for i in 0..10 {
        manager.allocate_block(session1).unwrap();
        manager.allocate_block(session2).unwrap();
        manager.allocate_block(session3).unwrap();

        if i == 4 || i == 9 {
            print_stats(manager, &format!("After {} blocks per session", i + 1));
        }
    }

    // 释放 session2
    println!("\n🗑️  Freeing session 2...");
    manager.free_session(session2).unwrap();
    print_stats(manager, "After freeing session 2");

    // 清理
    manager.free_session(session1).unwrap();
    manager.free_session(session3).unwrap();

    println!("✅ Demo 1 completed\n");
}

fn demo_large_context(manager: &mut SessionManager) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 2: 大上下文场景 (128k tokens)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    const CONTEXT_TOKENS: usize = 128 * 1024;
    const BLOCKS_NEEDED: usize = CONTEXT_TOKENS / BLOCK_SIZE;

    println!("📊 Target: {} tokens ({} blocks)", CONTEXT_TOKENS, BLOCKS_NEEDED);

    let session = manager.create_session();
    let start = Instant::now();

    for i in 0..BLOCKS_NEEDED {
        manager.allocate_block(session).unwrap();

        // 每 1000 块打印进度
        if i > 0 && i % 1000 == 0 {
            let progress = (i as f32 / BLOCKS_NEEDED as f32) * 100.0;
            print_stats(manager, &format!("Progress: {:.1}% ({}/{})", progress, i, BLOCKS_NEEDED));
        }
    }

    let elapsed = start.elapsed();
    println!("\n⏱️  Allocation time: {:.3}s", elapsed.as_secs_f64());
    println!("   Throughput: {:.0} blocks/sec", BLOCKS_NEEDED as f64 / elapsed.as_secs_f64());

    print_stats(manager, "Final state");

    // 清理
    manager.free_session(session).unwrap();

    println!("✅ Demo 2 completed\n");
}

fn demo_concurrent_sessions(manager: &mut SessionManager) {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Demo 3: 多会话并发 (8 sessions × 64k context)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    const NUM_SESSIONS: usize = 8;
    const CONTEXT_PER_SESSION: usize = 64 * 1024; // 64k tokens per session
    const BLOCKS_PER_SESSION: usize = CONTEXT_PER_SESSION / BLOCK_SIZE;

    println!("📊 Configuration:");
    println!("   Sessions: {}", NUM_SESSIONS);
    println!("   Context per session: {} tokens ({} blocks)", CONTEXT_PER_SESSION, BLOCKS_PER_SESSION);
    println!("   Total: {} tokens ({} blocks)\n",
             CONTEXT_PER_SESSION * NUM_SESSIONS,
             BLOCKS_PER_SESSION * NUM_SESSIONS);

    // 创建所有会话
    let mut sessions = Vec::new();
    for i in 0..NUM_SESSIONS {
        let session = manager.create_session();
        sessions.push(session);
        println!("✨ Created session {}: {:?}", i + 1, session);
    }

    println!();

    // 为每个会话分配块
    let start = Instant::now();

    for (i, session) in sessions.iter().enumerate() {
        println!("📦 Allocating for session {}...", i + 1);

        for block_idx in 0..BLOCKS_PER_SESSION {
            manager.allocate_block(*session).unwrap();

            // 每 500 块打印一次
            if block_idx > 0 && block_idx % 500 == 0 {
                let progress = (block_idx as f32 / BLOCKS_PER_SESSION as f32) * 100.0;
                print!("   Progress: {:.1}% ({}/{})\r", progress, block_idx, BLOCKS_PER_SESSION);
            }
        }

        println!("   ✅ Session {} complete ({} blocks)              ", i + 1, BLOCKS_PER_SESSION);
    }

    let elapsed = start.elapsed();

    println!();
    println!("⏱️  Total time: {:.3}s", elapsed.as_secs_f64());
    println!("   Throughput: {:.0} blocks/sec",
             (BLOCKS_PER_SESSION * NUM_SESSIONS) as f64 / elapsed.as_secs_f64());

    print_stats(manager, "Final concurrent state");

    // 逐个释放会话
    println!("\n🧹 Cleaning up sessions...");
    for (i, session) in sessions.iter().enumerate() {
        manager.free_session(*session).unwrap();
        println!("   🗑️  Freed session {}", i + 1);
    }

    print_stats(manager, "After cleanup");

    println!("✅ Demo 3 completed");
}

fn print_stats(manager: &SessionManager, context: &str) {
    let stats = manager.stats();

    println!("\n📊 Stats [{}]:", context);
    println!("   Sessions: {}", stats.total_sessions);
    println!("   VRAM: {}/{} blocks ({:.1}%)",
             stats.vram_used_blocks,
             stats.vram_total_blocks,
             stats.vram_utilization * 100.0);
    println!("   RAM:  {}/{} blocks ({:.1}%)",
             stats.ram_used_blocks,
             stats.ram_total_blocks,
             stats.ram_utilization * 100.0);
    println!();
}

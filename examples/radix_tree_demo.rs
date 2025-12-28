//! # Radix Tree 前缀缓存演示程序
//!
//! 演示场景：
//! 1. 基础 Radix Tree 操作
//! 2. 前缀共享演示
//! 3. KV Cache Manager 使用
//! 4. 真实场景模拟（多用户共享系统提示）

use loci::radix_tree::*;

fn main() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  Loci Phase 2 Week 4: Radix Tree 前缀缓存演示        ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    demo_1_basic_radix_tree();
    println!("\n{}\n", "=".repeat(60));

    demo_2_prefix_sharing();
    println!("\n{}\n", "=".repeat(60));

    demo_3_kv_cache_manager();
    println!("\n{}\n", "=".repeat(60));

    demo_4_realistic_scenario();
    println!("\n{}\n", "=".repeat(60));

    println!("✅ All demos completed successfully!");
}

// ==================== Demo 1: 基础 Radix Tree 操作 ====================

fn demo_1_basic_radix_tree() {
    println!("📋 Demo 1: 基础 Radix Tree 操作\n");

    let tree = RadixTree::new();

    // 插入几个 Token 序列
    println!("📥 插入 Token 序列:");
    let sequences = vec![
        vec![1, 2, 3, 4, 5],
        vec![1, 2, 3, 6, 7],
        vec![1, 2, 8, 9, 10],
        vec![1, 11, 12],
    ];

    for (i, seq) in sequences.iter().enumerate() {
        let (node_id, shared_len) = tree.insert(seq).unwrap();
        println!(
            "  [{}] 插入 {:?} → NodeID: {}, 共享前缀长度: {}",
            i + 1,
            seq,
            node_id,
            shared_len
        );
    }

    // 搜索
    println!("\n🔍 搜索 Token 序列:");
    let search_queries = vec![
        vec![1, 2, 3],
        vec![1, 2, 8],
        vec![1, 11],
        vec![5, 6, 7], // 不存在
    ];

    for query in &search_queries {
        if let Some((node_id, shared_len)) = tree.search(query) {
            println!(
                "  查询 {:?} → 找到 NodeID: {}, 共享长度: {}",
                query, node_id, shared_len
            );
        } else {
            println!("  查询 {:?} → 未找到", query);
        }
    }

    // 统计信息
    println!("\n📊 Radix Tree 统计:");
    let stats = tree.stats();
    println!("  总节点数: {}", stats.total_nodes);
    println!("  总 Token 数: {}", stats.total_tokens);
    println!("  总引用计数: {}", stats.total_refs);
    println!("  可淘汰节点数: {}", stats.evictable_nodes);
}

// ==================== Demo 2: 前缀共享演示 ====================

fn demo_2_prefix_sharing() {
    println!("📋 Demo 2: 前缀共享演示\n");

    let tree = RadixTree::new();

    println!("场景: 多个用户使用相同的系统提示\n");

    // 系统提示（10 个 tokens）
    let system_prompt = vec![100, 101, 102, 103, 104, 105, 106, 107, 108, 109];
    println!("🤖 系统提示 (10 tokens): {:?}\n", system_prompt);

    // 5 个用户，每个用户添加不同的内容
    println!("👥 用户会话:");
    for i in 0..5 {
        let mut tokens = system_prompt.clone();
        // 用户特定内容（5 tokens）
        tokens.extend(vec![200 + i, 201 + i, 202 + i, 203 + i, 204 + i]);

        let (node_id, shared_len) = tree.insert(&tokens).unwrap();

        println!(
            "  用户 {} → 总长度: {} tokens, 共享前缀: {} tokens (节约 {:.1}%)",
            i + 1,
            tokens.len(),
            shared_len,
            (shared_len as f64 / tokens.len() as f64) * 100.0
        );
        println!("           NodeID: {}, 用户内容: {:?}", node_id, &tokens[10..]);
    }

    // 计算内存节省
    println!("\n💾 内存节省分析:");
    let stats = tree.stats();

    let total_requested = 5 * 15; // 5 个用户 × 15 tokens
    let actual_stored = stats.total_tokens;
    let saved_percent = ((total_requested - actual_stored) as f64 / total_requested as f64) * 100.0;

    println!("  总请求: {} tokens", total_requested);
    println!("  实际存储: {} tokens", actual_stored);
    println!("  节省: {} tokens ({:.1}%)", total_requested - actual_stored, saved_percent);
}

// ==================== Demo 3: KV Cache Manager 使用 ====================

fn demo_3_kv_cache_manager() {
    println!("📋 Demo 3: KV Cache Manager 使用\n");

    let manager = KVCacheManager::new();

    println!("场景: 管理多个会话的 KV Cache\n");

    // 创建 10 个会话
    println!("📦 创建会话:");
    for i in 0..10 {
        let tokens = if i < 5 {
            // 前 5 个会话共享前缀 [1, 2, 3]
            vec![1, 2, 3, 10 + i, 20 + i, 30 + i]
        } else {
            // 后 5 个会话共享前缀 [1, 2, 4]
            vec![1, 2, 4, 40 + i, 50 + i, 60 + i]
        };

        let shared = manager.insert_prefix(&format!("session-{}", i), &tokens).unwrap();
        println!(
            "  session-{}: {} tokens, 共享 {} tokens",
            i,
            tokens.len(),
            shared
        );
    }

    // 查询会话路径
    println!("\n🔍 查询会话路径:");
    for i in [0, 5, 9].iter() {
        if let Some(path) = manager.get_session_path(&format!("session-{}", i)) {
            println!("  session-{}: {:?}", i, path);
        }
    }

    // 统计信息
    println!("\n📊 KV Cache Manager 统计:");
    let stats = manager.stats();
    println!("  总会话数: {}", stats.total_sessions);
    println!("  Radix Tree 节点数: {}", stats.total_nodes);
    println!("  总请求 tokens: {}", stats.total_requested_tokens);
    println!("  总共享 tokens: {}", stats.total_shared_tokens);
    println!("  缓存命中率: {:.2}%", stats.hit_rate * 100.0);
    println!("  内存节省: {:.2}%", stats.memory_saved_percent);

    // 移除会话
    println!("\n🗑️  移除会话:");
    for i in 0..3 {
        manager.remove_session(&format!("session-{}", i)).unwrap();
        println!("  已移除 session-{}", i);
    }

    let stats_after = manager.stats();
    println!("\n📊 移除后统计:");
    println!("  剩余会话数: {}", stats_after.total_sessions);
}

// ==================== Demo 4: 真实场景模拟 ====================

fn demo_4_realistic_scenario() {
    println!("📋 Demo 4: 真实场景模拟\n");

    let manager = KVCacheManager::new();

    println!("场景: AI 助手服务 - 3 种角色，每种角色 10 个用户\n");

    // 定义 3 种系统提示（代表不同角色）
    let system_prompts = vec![
        ("Code Assistant", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), // 10 tokens
        ("Creative Writer", vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]), // 10 tokens
        ("Data Analyst", vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), // 10 tokens
    ];

    println!("🤖 系统角色:");
    for (i, (role, prompt)) in system_prompts.iter().enumerate() {
        println!("  [{}] {}: {} tokens", i + 1, role, prompt.len());
    }

    println!("\n👥 创建用户会话:");

    let mut session_id = 0;
    for (role_idx, (role_name, system_prompt)) in system_prompts.iter().enumerate() {
        println!("\n  {} 角色:", role_name);

        for user_id in 0..10 {
            let mut tokens = system_prompt.clone();

            // 添加用户历史消息（20 tokens）
            for j in 0..20 {
                tokens.push(1000 + role_idx as i32 * 1000 + user_id * 100 + j);
            }

            let shared = manager.insert_prefix(
                &format!("session-{}", session_id),
                &tokens
            ).unwrap();

            if user_id < 3 {
                // 只显示前 3 个
                println!(
                    "    用户 {}: {} tokens (共享 {} tokens, {:.1}%)",
                    user_id + 1,
                    tokens.len(),
                    shared,
                    (shared as f64 / tokens.len() as f64) * 100.0
                );
            }

            session_id += 1;
        }

        println!("    ... (还有 7 个用户)");
    }

    // 最终统计
    println!("\n📊 最终统计:");
    let stats = manager.stats();

    println!("  总会话数: {}", stats.total_sessions);
    println!("  Radix Tree 节点数: {}", stats.total_nodes);
    println!("  总请求 tokens: {} ({:.1} KB)", stats.total_requested_tokens, stats.total_requested_tokens as f64 / 256.0);
    println!("  总共享 tokens: {} ({:.1} KB)", stats.total_shared_tokens, stats.total_shared_tokens as f64 / 256.0);
    println!("  实际存储 tokens: {} ({:.1} KB)",
             stats.total_tokens,
             stats.total_tokens as f64 / 256.0);

    let saved_tokens = stats.total_requested_tokens - stats.total_tokens;
    println!("\n💾 内存节省:");
    println!("  节省 tokens: {} ({:.1} KB)", saved_tokens, saved_tokens as f64 / 256.0);
    println!("  缓存命中率: {:.2}%", stats.hit_rate * 100.0);
    println!("  内存节省比例: {:.2}%", stats.memory_saved_percent);

    // 假设每个 token 的 KV Cache 占用 2KB（典型值）
    let kv_cache_size_per_token_kb = 2.0;
    let total_size_without_sharing = stats.total_requested_tokens as f64 * kv_cache_size_per_token_kb;
    let actual_size_with_sharing = stats.total_tokens as f64 * kv_cache_size_per_token_kb;
    let saved_size_mb = (total_size_without_sharing - actual_size_with_sharing) / 1024.0;

    println!("\n🎯 实际显存节省 (假设每 token KV Cache = 2KB):");
    println!("  无前缀缓存: {:.2} MB", total_size_without_sharing / 1024.0);
    println!("  有前缀缓存: {:.2} MB", actual_size_with_sharing / 1024.0);
    println!("  节省显存: {:.2} MB ({:.1}%)",
             saved_size_mb,
             (saved_size_mb / (total_size_without_sharing / 1024.0)) * 100.0);
}

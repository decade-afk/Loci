//! Radix Tree 前缀缓存系统测试套件
//!
//! 覆盖内容：
//! - RadixNode 基础功能
//! - RadixTree 插入/搜索/删除
//! - 前缀共享测试
//! - KVCacheManager 集成
//! - 性能基准测试

use loci::radix_tree::*;

// ==================== RadixNode 测试 ====================

#[test]
fn test_radix_node_creation() {
    let node = RadixNode::new(1, vec![1, 2, 3], None);
    assert_eq!(node.id, 1);
    assert_eq!(node.tokens, vec![1, 2, 3]);
    assert_eq!(node.ref_count, 0);
    assert!(node.can_evict());
    assert_eq!(node.len(), 3);
}

#[test]
fn test_radix_node_root() {
    let root = RadixNode::root(0);
    assert_eq!(root.id, 0);
    assert!(root.tokens.is_empty());
    assert!(root.is_empty());
    assert_eq!(root.ref_count, 0);
}

#[test]
fn test_radix_node_ref_counting() {
    let mut node = RadixNode::new(1, vec![1, 2, 3], None);

    // 初始状态
    assert_eq!(node.ref_count, 0);
    assert!(node.can_evict());

    // 增加引用
    node.inc_ref();
    assert_eq!(node.ref_count, 1);
    assert!(!node.can_evict());

    node.inc_ref();
    assert_eq!(node.ref_count, 2);

    // 减少引用
    let count = node.dec_ref();
    assert_eq!(count, 1);
    assert_eq!(node.ref_count, 1);

    let count = node.dec_ref();
    assert_eq!(count, 0);
    assert!(node.can_evict());
}

#[test]
fn test_radix_node_children() {
    let mut node = RadixNode::new(1, vec![1, 2, 3], None);

    // 添加子节点
    node.add_child(4, 10);
    node.add_child(5, 11);

    // 获取子节点
    assert_eq!(node.get_child(4), Some(10));
    assert_eq!(node.get_child(5), Some(11));
    assert_eq!(node.get_child(6), None);

    assert_eq!(node.children.len(), 2);
}

// ==================== RadixTree 基础测试 ====================

#[test]
fn test_radix_tree_creation() {
    let tree = RadixTree::new();
    let stats = tree.stats();

    assert_eq!(stats.total_nodes, 1); // 只有根节点
    assert_eq!(stats.total_tokens, 0);
    assert_eq!(stats.total_refs, 0);
}

#[test]
fn test_radix_tree_insert_simple() {
    let tree = RadixTree::new();
    let tokens = vec![1, 2, 3];

    let result = tree.insert(&tokens);
    assert!(result.is_ok());

    let (node_id, shared_len) = result.unwrap();
    assert_eq!(shared_len, 0); // 第一次插入，无共享

    let node = tree.get_node(node_id).unwrap();
    assert_eq!(node.tokens, tokens);
    assert_eq!(node.ref_count, 1);

    let stats = tree.stats();
    assert_eq!(stats.total_nodes, 2); // 根 + 新节点
    assert_eq!(stats.total_refs, 1);
}

#[test]
fn test_radix_tree_insert_empty() {
    let tree = RadixTree::new();
    let result = tree.insert(&[]);
    assert!(result.is_err());
}

#[test]
fn test_radix_tree_prefix_sharing() {
    let tree = RadixTree::new();

    // 插入 [1, 2, 3]
    let (node1_id, shared1) = tree.insert(&[1, 2, 3]).unwrap();
    assert_eq!(shared1, 0);

    // 插入 [1, 2, 4]（共享前缀 [1, 2]）
    let (node2_id, shared2) = tree.insert(&[1, 2, 4]).unwrap();
    assert_eq!(shared2, 2); // 共享 [1, 2]

    assert_ne!(node1_id, node2_id);

    let stats = tree.stats();
    println!("Stats after prefix sharing: {:?}", stats);
    assert_eq!(stats.total_refs, 2);
}

#[test]
fn test_radix_tree_complete_overlap() {
    let tree = RadixTree::new();

    // 插入 [1, 2, 3]
    let (node1_id, _) = tree.insert(&[1, 2, 3]).unwrap();

    // 再次插入 [1, 2, 3]（完全重复）
    let (node2_id, shared2) = tree.insert(&[1, 2, 3]).unwrap();
    assert_eq!(shared2, 3); // 完全共享
    assert_eq!(node1_id, node2_id); // 同一个节点

    let node = tree.get_node(node1_id).unwrap();
    assert_eq!(node.ref_count, 2); // 引用计数增加
}

#[test]
fn test_radix_tree_nested_prefixes() {
    let tree = RadixTree::new();

    // 插入多个有嵌套关系的序列
    tree.insert(&[1, 2, 3, 4, 5]).unwrap();
    tree.insert(&[1, 2, 3, 6, 7]).unwrap();
    tree.insert(&[1, 2, 8, 9]).unwrap();
    tree.insert(&[1, 10]).unwrap();

    let stats = tree.stats();
    println!("Stats after nested prefixes: {:?}", stats);
    assert!(stats.total_nodes > 1);
}

// ==================== RadixTree 搜索测试 ====================

#[test]
fn test_radix_tree_search_exact_match() {
    let tree = RadixTree::new();
    tree.insert(&[1, 2, 3]).unwrap();

    // 完全匹配
    let result = tree.search(&[1, 2, 3]);
    assert!(result.is_some());
    let (_, shared_len) = result.unwrap();
    assert_eq!(shared_len, 3);
}

#[test]
fn test_radix_tree_search_prefix_match() {
    let tree = RadixTree::new();
    tree.insert(&[1, 2, 3, 4]).unwrap();

    // 前缀匹配
    let result = tree.search(&[1, 2]);
    assert!(result.is_some());
    let (_, shared_len) = result.unwrap();
    assert_eq!(shared_len, 2);
}

#[test]
fn test_radix_tree_search_no_match() {
    let tree = RadixTree::new();
    tree.insert(&[1, 2, 3]).unwrap();

    // 无匹配
    let result = tree.search(&[4, 5, 6]);
    assert!(result.is_none());
}

#[test]
fn test_radix_tree_search_partial_match() {
    let tree = RadixTree::new();
    tree.insert(&[1, 2, 3]).unwrap();
    tree.insert(&[1, 2, 4]).unwrap();

    // 搜索 [1, 2, 5]（部分匹配 [1, 2]）
    let result = tree.search(&[1, 2, 5]);
    assert!(result.is_some());
    let (_, shared_len) = result.unwrap();
    assert_eq!(shared_len, 2);
}

#[test]
fn test_radix_tree_search_empty() {
    let tree = RadixTree::new();
    tree.insert(&[1, 2, 3]).unwrap();

    let result = tree.search(&[]);
    assert!(result.is_none());
}

// ==================== RadixTree 删除测试 ====================

#[test]
fn test_radix_tree_remove_node() {
    let tree = RadixTree::new();
    let (node_id, _) = tree.insert(&[1, 2, 3]).unwrap();

    // 尝试删除（仍有引用）
    let result = tree.remove_node(node_id);
    assert!(result.is_err());

    // 减少引用计数
    tree.dec_ref(node_id).unwrap();

    // 再次尝试删除
    let result = tree.remove_node(node_id);
    assert!(result.is_ok());

    // 节点应该已被删除
    let node = tree.get_node(node_id);
    assert!(node.is_none());
}

#[test]
fn test_radix_tree_dec_ref() {
    let tree = RadixTree::new();
    let (node_id, _) = tree.insert(&[1, 2, 3]).unwrap();

    let node = tree.get_node(node_id).unwrap();
    assert_eq!(node.ref_count, 1);

    // 减少引用计数
    let count = tree.dec_ref(node_id).unwrap();
    assert_eq!(count, 0);

    let node = tree.get_node(node_id).unwrap();
    assert_eq!(node.ref_count, 0);
}

// ==================== RadixTree 路径测试 ====================

#[test]
fn test_radix_tree_get_path_tokens() {
    let tree = RadixTree::new();
    let tokens = vec![1, 2, 3, 4, 5];
    let (node_id, _) = tree.insert(&tokens).unwrap();

    let path = tree.get_path_tokens(node_id);
    assert_eq!(path, tokens);
}

#[test]
fn test_radix_tree_get_path_with_sharing() {
    let tree = RadixTree::new();

    // 插入两个共享前缀的序列
    tree.insert(&[1, 2, 3]).unwrap();
    let (node_id, _) = tree.insert(&[1, 2, 4]).unwrap();

    let path = tree.get_path_tokens(node_id);
    assert_eq!(path, vec![1, 2, 4]);
}

// ==================== KVCacheManager 测试 ====================

#[test]
fn test_kv_cache_manager_creation() {
    let manager = KVCacheManager::new();
    let stats = manager.stats();

    assert_eq!(stats.total_sessions, 0);
    assert_eq!(stats.total_requested_tokens, 0);
    assert_eq!(stats.total_shared_tokens, 0);
    assert_eq!(stats.hit_rate, 0.0);
}

#[test]
fn test_kv_cache_manager_insert_prefix() {
    let manager = KVCacheManager::new();

    let shared = manager.insert_prefix("session-1", &[1, 2, 3]).unwrap();
    assert_eq!(shared, 0); // 第一次插入

    let stats = manager.stats();
    assert_eq!(stats.total_sessions, 1);
    assert_eq!(stats.total_requested_tokens, 3);
    assert_eq!(stats.total_shared_tokens, 0);
}

#[test]
fn test_kv_cache_manager_prefix_sharing() {
    let manager = KVCacheManager::new();

    // Session 1: [1, 2, 3]
    let shared1 = manager.insert_prefix("session-1", &[1, 2, 3]).unwrap();
    assert_eq!(shared1, 0);

    // Session 2: [1, 2, 4]（共享前缀 [1, 2]）
    let shared2 = manager.insert_prefix("session-2", &[1, 2, 4]).unwrap();
    assert_eq!(shared2, 2);

    let stats = manager.stats();
    assert_eq!(stats.total_sessions, 2);
    assert_eq!(stats.total_requested_tokens, 6); // 3 + 3
    assert_eq!(stats.total_shared_tokens, 2);    // [1, 2]
    assert!((stats.hit_rate - 0.333).abs() < 0.01); // 2/6 ≈ 0.333
}

#[test]
fn test_kv_cache_manager_hit_rate() {
    let manager = KVCacheManager::new();

    // 插入多个共享前缀的会话
    manager.insert_prefix("session-1", &[1, 2, 3, 4, 5]).unwrap();
    manager.insert_prefix("session-2", &[1, 2, 3, 6, 7]).unwrap();
    manager.insert_prefix("session-3", &[1, 2, 8, 9, 10]).unwrap();

    let stats = manager.stats();
    println!("Hit rate: {:.2}%", stats.hit_rate * 100.0);
    println!("Memory saved: {:.2}%", stats.memory_saved_percent);

    assert!(stats.hit_rate > 0.0);
    assert!(stats.memory_saved_percent > 0.0);
}

#[test]
fn test_kv_cache_manager_remove_session() {
    let manager = KVCacheManager::new();

    manager.insert_prefix("session-1", &[1, 2, 3]).unwrap();
    manager.insert_prefix("session-2", &[1, 2, 4]).unwrap();

    let stats = manager.stats();
    assert_eq!(stats.total_sessions, 2);

    // 移除 session-1
    let result = manager.remove_session("session-1");
    assert!(result.is_ok());

    let stats = manager.stats();
    assert_eq!(stats.total_sessions, 1);
}

#[test]
fn test_kv_cache_manager_get_session_path() {
    let manager = KVCacheManager::new();
    let tokens = vec![1, 2, 3, 4, 5];

    manager.insert_prefix("session-1", &tokens).unwrap();

    let path = manager.get_session_path("session-1");
    assert!(path.is_some());
    assert_eq!(path.unwrap(), tokens);
}

#[test]
fn test_kv_cache_manager_search_prefix() {
    let manager = KVCacheManager::new();

    manager.insert_prefix("session-1", &[1, 2, 3, 4]).unwrap();

    // 搜索完全匹配
    let result = manager.search_prefix(&[1, 2, 3, 4]);
    assert!(result.is_some());
    let (_, shared_len) = result.unwrap();
    assert_eq!(shared_len, 4);

    // 搜索前缀
    let result = manager.search_prefix(&[1, 2]);
    assert!(result.is_some());
}

// ==================== 压力测试 ====================

#[test]
fn test_radix_tree_many_insertions() {
    let tree = RadixTree::new();

    // 插入 100 个序列
    for i in 0..100 {
        let tokens = vec![i % 10, (i / 10) % 10, i];
        tree.insert(&tokens).unwrap();
    }

    let stats = tree.stats();
    println!("Many insertions stats: {:?}", stats);
    assert!(stats.total_nodes > 1);
}

#[test]
fn test_kv_cache_manager_high_sharing() {
    let manager = KVCacheManager::new();

    // 模拟高度共享的场景（多个会话共享同一系统提示）
    let system_prompt = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 10 tokens

    for i in 0..20 {
        let mut tokens = system_prompt.clone();
        tokens.extend(vec![100 + i, 200 + i, 300 + i]); // 用户特定内容

        manager.insert_prefix(&format!("session-{}", i), &tokens).unwrap();
    }

    let stats = manager.stats();
    println!("\nHigh sharing scenario:");
    println!("  Total sessions: {}", stats.total_sessions);
    println!("  Total requested: {} tokens", stats.total_requested_tokens);
    println!("  Total shared: {} tokens", stats.total_shared_tokens);
    println!("  Hit rate: {:.2}%", stats.hit_rate * 100.0);
    println!("  Memory saved: {:.2}%", stats.memory_saved_percent);

    // 期望高命中率（系统提示被重复使用）
    assert!(stats.hit_rate > 0.5); // >50% 命中率
}

#[test]
fn test_kv_cache_manager_realistic_scenario() {
    let manager = KVCacheManager::new();

    // 模拟真实场景：
    // - 3 个不同的系统提示（每个 20 tokens）
    // - 每个系统提示下有 10 个用户会话（每个用户内容 30 tokens）

    let system_prompts = vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        vec![41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    ];

    let mut session_id = 0;
    for system_prompt in &system_prompts {
        for user_id in 0..10 {
            let mut tokens = system_prompt.clone();
            // 添加用户特定内容（30 tokens）
            for j in 0..30 {
                tokens.push(1000 + user_id * 100 + j);
            }

            manager.insert_prefix(
                &format!("session-{}", session_id),
                &tokens
            ).unwrap();
            session_id += 1;
        }
    }

    let stats = manager.stats();
    println!("\nRealistic scenario:");
    println!("  Total sessions: {}", stats.total_sessions);
    println!("  Total requested: {} tokens", stats.total_requested_tokens);
    println!("  Total shared: {} tokens", stats.total_shared_tokens);
    println!("  Hit rate: {:.2}%", stats.hit_rate * 100.0);
    println!("  Memory saved: {:.2}%", stats.memory_saved_percent);

    assert_eq!(stats.total_sessions, 30);
    // 期望较高的命中率（系统提示被重复使用）
    assert!(stats.hit_rate > 0.3); // >30% 命中率
}

// ==================== 性能基准测试 ====================

#[test]
#[ignore] // 需要手动运行
fn benchmark_radix_tree_insertion() {
    use std::time::Instant;

    let tree = RadixTree::new();
    let n = 10000;

    let start = Instant::now();
    for i in 0..n {
        let tokens = vec![i % 100, (i / 100) % 100, i % 50];
        tree.insert(&tokens).unwrap();
    }
    let duration = start.elapsed();

    let ops_per_sec = n as f64 / duration.as_secs_f64();
    println!("\nInsertion benchmark:");
    println!("  Operations: {}", n);
    println!("  Duration: {:?}", duration);
    println!("  Ops/sec: {:.0}", ops_per_sec);

    assert!(ops_per_sec > 1000.0); // >1000 ops/sec
}

#[test]
#[ignore]
fn benchmark_radix_tree_search() {
    use std::time::Instant;

    let tree = RadixTree::new();

    // 先插入数据
    for i in 0..1000 {
        let tokens = vec![i % 100, (i / 100) % 100, i % 50];
        tree.insert(&tokens).unwrap();
    }

    let n = 10000;
    let start = Instant::now();
    for i in 0..n {
        let tokens = vec![i % 100, (i / 100) % 100];
        tree.search(&tokens);
    }
    let duration = start.elapsed();

    let ops_per_sec = n as f64 / duration.as_secs_f64();
    println!("\nSearch benchmark:");
    println!("  Operations: {}", n);
    println!("  Duration: {:?}", duration);
    println!("  Ops/sec: {:.0}", ops_per_sec);

    assert!(ops_per_sec > 5000.0); // >5000 ops/sec
}

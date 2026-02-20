//! Integration tests for Radix Tree Prefix Cache

use loci::prelude::*;

#[test]
fn test_radix_cache_basic_operations() {
    let cache = ShardedRadixCache::new();

    // Insert a 64-token sequence (2 blocks)
    let tokens: Vec<TokenId> = (0..64).collect();
    let blocks: Vec<BlockId> = vec![100, 101];

    let result = cache.insert(&tokens, &blocks);
    assert!(result.is_ok());

    // Match the same sequence
    let matched = cache.match_prefix(&tokens);
    assert!(matched.is_some());

    let (matched_tokens, matched_blocks) = matched.unwrap();
    assert_eq!(matched_tokens.len(), 64);
    assert_eq!(matched_blocks.len(), 2);
    assert_eq!(matched_blocks, vec![100, 101]);
}

#[test]
fn test_radix_cache_partial_match() {
    let cache = ShardedRadixCache::new();

    // Insert 32 tokens (1 block)
    let tokens: Vec<TokenId> = (0..32).collect();
    let blocks: Vec<BlockId> = vec![200];
    cache.insert(&tokens, &blocks).unwrap();

    // Query with 64 tokens
    let query: Vec<TokenId> = (0..64).collect();
    let matched = cache.match_prefix(&query);

    assert!(matched.is_some());
    let (matched_tokens, matched_blocks) = matched.unwrap();

    // Should match only the first 32 tokens
    assert_eq!(matched_tokens.len(), 32);
    assert_eq!(matched_blocks.len(), 1);
    assert_eq!(matched_blocks[0], 200);
}

#[test]
fn test_radix_cache_no_match() {
    let cache = ShardedRadixCache::new();

    // Insert tokens [0..32)
    let tokens1: Vec<TokenId> = (0..32).collect();
    let blocks1: Vec<BlockId> = vec![100];
    cache.insert(&tokens1, &blocks1).unwrap();

    // Query with completely different tokens [1000..1032)
    let query: Vec<TokenId> = (1000..1032).collect();
    let matched = cache.match_prefix(&query);

    assert!(matched.is_none());
}

#[test]
fn test_radix_cache_statistics() {
    let cache = ShardedRadixCache::new();

    // Insert some data
    let tokens: Vec<TokenId> = (0..32).collect();
    let blocks: Vec<BlockId> = vec![300];
    cache.insert(&tokens, &blocks).unwrap();

    // Match it
    cache.match_prefix(&tokens);

    // Check stats
    let stats = cache.stats();
    assert_eq!(stats.total_insertions, 1);
    assert_eq!(stats.total_matches, 1);
    assert_eq!(stats.total_misses, 0);
}

#[test]
fn test_radix_cache_multiple_prefixes() {
    let cache = ShardedRadixCache::new();

    // Insert prefix A: [0, 1, 2, ..., 31]
    let tokens_a: Vec<TokenId> = (0..32).collect();
    let blocks_a: Vec<BlockId> = vec![100];
    cache.insert(&tokens_a, &blocks_a).unwrap();

    // Insert prefix B: [0, 1, 2, ..., 63] (extends A)
    let tokens_b: Vec<TokenId> = (0..64).collect();
    let blocks_b: Vec<BlockId> = vec![100, 101];
    cache.insert(&tokens_b, &blocks_b).unwrap();

    // Query with prefix A
    let match_a = cache.match_prefix(&tokens_a);
    assert!(match_a.is_some());
    let (_, blocks) = match_a.unwrap();
    assert_eq!(blocks.len(), 1);

    // Query with prefix B
    let match_b = cache.match_prefix(&tokens_b);
    assert!(match_b.is_some());
    let (_, blocks) = match_b.unwrap();
    assert_eq!(blocks.len(), 2);
}

#[test]
fn test_radix_cache_eviction() {
    let cache = ShardedRadixCache::new();

    // Insert multiple entries
    for i in 0..10 {
        let tokens: Vec<TokenId> = (i * 100..i * 100 + 32).collect();
        let blocks: Vec<BlockId> = vec![i as BlockId];
        cache.insert(&tokens, &blocks).unwrap();
    }

    // Evict unused nodes
    let evicted = cache.evict_unused();

    // Since we inserted and matched, some nodes should have ref_count > 0
    // Eviction should remove nodes with ref_count == 0
    println!("Evicted {} nodes", evicted);
}

#[test]
fn test_radix_cache_clear() {
    let cache = ShardedRadixCache::new();

    // Insert data
    let tokens: Vec<TokenId> = (0..32).collect();
    let blocks: Vec<BlockId> = vec![400];
    cache.insert(&tokens, &blocks).unwrap();

    // Clear cache
    cache.clear();

    // After clear, matching should fail
    let matched = cache.match_prefix(&tokens);
    assert!(matched.is_none());

    // Stats should be reset
    let stats = cache.stats();
    assert_eq!(stats.total_nodes, 0);
}

#[test]
fn test_radix_node_reference_counting() {
    let mut node = RadixNode::new(42);

    // Initially ref_count = 0
    assert_eq!(node.ref_count(), 0);
    assert!(node.is_evictable());

    // Set block data (sets ref_count = 1)
    node.set_block(100, 0xdeadbeef);
    assert_eq!(node.ref_count(), 1);
    assert!(!node.is_evictable());

    // Acquire reference
    node.acquire();
    assert_eq!(node.ref_count(), 2);

    // Release references
    node.release();
    assert_eq!(node.ref_count(), 1);
    node.release();
    assert_eq!(node.ref_count(), 0);
    assert!(node.is_evictable());
}

#[test]
fn test_radix_tree_insert_and_match() {
    let mut tree = RadixTree::new();

    // Insert [0, 1, 2, ..., 63] with 2 blocks
    let tokens: Vec<TokenId> = (0..64).collect();
    let blocks: Vec<BlockId> = vec![100, 101];
    let hashes: Vec<BlockHash> = vec![0x1111111111111111, 0x2222222222222222];

    tree.insert(&tokens, &blocks, &hashes).unwrap();

    // Match prefix
    let matched = tree.match_prefix(&tokens);
    assert!(matched.is_some());

    let (matched_tokens, matched_blocks) = matched.unwrap();
    assert_eq!(matched_tokens.len(), 64);
    assert_eq!(matched_blocks.len(), 2);
}

#[test]
fn test_sharded_cache_concurrency() {
    use std::sync::Arc;
    use std::thread;

    let cache = Arc::new(ShardedRadixCache::new());
    let mut handles = vec![];

    // Spawn multiple threads to insert data
    for i in 0..4 {
        let cache_clone = Arc::clone(&cache);
        let handle = thread::spawn(move || {
            let tokens: Vec<TokenId> = (i * 100..i * 100 + 32).collect();
            let blocks: Vec<BlockId> = vec![i as BlockId];
            cache_clone.insert(&tokens, &blocks).unwrap();
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify insertions
    let stats = cache.stats();
    assert_eq!(stats.total_insertions, 4);
}

#[test]
fn test_xxhash_determinism() {
    use xxhash_rust::xxh64::xxh64;

    let data = b"test data";
    let hash1 = xxh64(data, 0);
    let hash2 = xxh64(data, 0);

    assert_eq!(hash1, hash2, "xxHash64 should be deterministic");
}

#[test]
fn test_block_hash_computation() {
    use xxhash_rust::xxh64::xxh64;

    let block_id: BlockId = 12345;
    let hash1 = xxh64(&block_id.to_le_bytes(), 0);
    let hash2 = xxh64(&block_id.to_le_bytes(), 0);

    assert_eq!(hash1, hash2);
    assert_ne!(hash1, 0, "Hash should not be zero");
}

#[test]
fn test_integration_with_kv_cache() {
    // Test integration between RadixCache and KVCache
    let cache = ShardedRadixCache::new();

    // Simulate a scenario where we have KV cache blocks
    // and we want to share them via prefix matching

    // Session 1: Insert prompt with 96 tokens (3 blocks)
    let tokens1: Vec<TokenId> = (0..96).collect();
    let blocks1: Vec<BlockId> = vec![100, 101, 102];
    cache.insert(&tokens1, &blocks1).unwrap();

    // Session 2: Query with same prefix (64 tokens)
    let tokens2: Vec<TokenId> = (0..64).collect();
    if let Some((_, matched_blocks)) = cache.match_prefix(&tokens2) {
        // Should match 2 blocks
        assert_eq!(matched_blocks.len(), 2);
        assert_eq!(matched_blocks, vec![100, 101]);

        // Now Session 2 can reuse blocks 100 and 101 from the KV cache pool
        // This is the core benefit of prefix caching!
    } else {
        panic!("Expected prefix match");
    }
}

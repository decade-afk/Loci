//! Radix Tree Prefix Cache Demonstration
//!
//! This example demonstrates the usage of the Radix Tree prefix caching system:
//!
//! 1. **Block Hash Fingerprinting**: xxHash64 for fast block identification
//! 2. **Prefix Matching**: Longest-prefix matching for KV cache sharing
//! 3. **Reference Counting**: Automatic memory management
//! 4. **Sharded Concurrency**: 16-way RwLock sharding for high throughput
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example radix_cache_demo
//! ```

use loci::prelude::*;
use xxhash_rust::xxh64::xxh64;

fn main() {
    println!("=== Radix Tree Prefix Cache Demo ===\n");

    // Create a sharded radix cache
    let cache = ShardedRadixCache::new();
    println!("✓ Created ShardedRadixCache with 16 shards\n");

    // ===== Example 1: Basic Insertion and Matching =====
    println!("--- Example 1: Basic Insertion and Matching ---");

    // Simulate a prompt: "The quick brown fox jumps"
    // Token IDs: [1, 2, 3, 4, 5, ...]
    let tokens1: Vec<TokenId> = (1..=64).collect();
    let blocks1: Vec<BlockId> = vec![100, 101]; // 2 blocks (32 tokens each)

    cache.insert(&tokens1, &blocks1).unwrap();
    println!("Inserted 64 tokens with 2 blocks");

    // Query with same prefix
    let query1: Vec<TokenId> = (1..=64).collect();
    if let Some((matched_tokens, matched_blocks)) = cache.match_prefix(&query1) {
        println!(
            "✓ Matched {} tokens, {} blocks: {:?}",
            matched_tokens.len(),
            matched_blocks.len(),
            matched_blocks
        );
    }
    println!();

    // ===== Example 2: Partial Prefix Matching =====
    println!("--- Example 2: Partial Prefix Matching ---");

    // Query with longer sequence
    let query2: Vec<TokenId> = (1..=96).collect();
    if let Some((matched_tokens, matched_blocks)) = cache.match_prefix(&query2) {
        println!(
            "Query: 96 tokens → Matched {} tokens, {} blocks",
            matched_tokens.len(),
            matched_blocks.len()
        );
        println!(
            "✓ Prefix cache hit! Reusing {} blocks",
            matched_blocks.len()
        );
    }
    println!();

    // ===== Example 3: Multiple Prompts with Common Prefix =====
    println!("--- Example 3: Multiple Prompts with Common Prefix ---");

    // Prompt A: "Hello world, how are"
    let tokens_a: Vec<TokenId> = (100..=131).collect();
    let blocks_a: Vec<BlockId> = vec![200];
    cache.insert(&tokens_a, &blocks_a).unwrap();
    println!("Inserted Prompt A: 32 tokens → 1 block");

    // Prompt B: "Hello world, how are you today?" (extends A)
    let tokens_b: Vec<TokenId> = (100..=163).collect();
    let blocks_b: Vec<BlockId> = vec![200, 201];
    cache.insert(&tokens_b, &blocks_b).unwrap();
    println!("Inserted Prompt B: 64 tokens → 2 blocks");

    // Query with Prompt A
    if let Some((matched_tokens, matched_blocks)) = cache.match_prefix(&tokens_a) {
        println!("Query Prompt A → Matched {} blocks", matched_blocks.len());
    }

    // Query with Prompt B
    if let Some((matched_tokens, matched_blocks)) = cache.match_prefix(&tokens_b) {
        println!("Query Prompt B → Matched {} blocks", matched_blocks.len());
    }
    println!();

    // ===== Example 4: Cache Statistics =====
    println!("--- Example 4: Cache Statistics ---");

    let stats = cache.stats();
    println!("Total Insertions: {}", stats.total_insertions);
    println!("Total Matches:    {}", stats.total_matches);
    println!("Total Misses:     {}", stats.total_misses);
    println!("Total Evictions:  {}", stats.total_evictions);
    println!("Total Nodes:      {}", stats.total_nodes);
    println!();

    // ===== Example 5: Eviction Test =====
    println!("--- Example 5: Eviction Test ---");

    // Insert some temporary data
    for i in 0..5_u64 {
        let tokens: Vec<TokenId> = (1000 + i * 32..1032 + i * 32).map(|x| x as u32).collect();
        let blocks: Vec<BlockId> = vec![1000 + i];
        cache.insert(&tokens, &blocks).unwrap();
    }
    println!("Inserted 5 temporary token sequences");

    let evicted = cache.evict_unused();
    println!("✓ Evicted {} unused nodes\n", evicted);

    // ===== Example 6: Concurrent Access Simulation =====
    println!("--- Example 6: Concurrent Access (Single-threaded Demo) ---");

    use std::sync::Arc;

    let shared_cache = Arc::new(cache);

    // Simulate multiple "sessions" accessing the cache
    for session_id in 0..3_u64 {
        let tokens: Vec<TokenId> = (2000 + session_id * 10..2032 + session_id * 10)
            .map(|x| x as u32)
            .collect();

        // Try to match prefix
        if let Some((_matched_tokens, matched_blocks)) = shared_cache.match_prefix(&tokens) {
            println!(
                "Session {}: Cache HIT - {} blocks",
                session_id,
                matched_blocks.len()
            );
        } else {
            // Cache miss, insert new data
            let blocks: Vec<BlockId> = vec![2000 + session_id];
            shared_cache.insert(&tokens, &blocks).unwrap();
            println!("Session {}: Cache MISS - inserted new entry", session_id);
        }
    }
    println!();

    // ===== Final Statistics =====
    println!("--- Final Statistics ---");
    let final_stats = shared_cache.stats();
    println!("Total Insertions: {}", final_stats.total_insertions);
    println!("Total Matches:    {}", final_stats.total_matches);
    println!("Total Misses:     {}", final_stats.total_misses);
    println!("Cache Hit Rate:   {:.2}%", {
        let total = final_stats.total_matches + final_stats.total_misses;
        if total > 0 {
            (final_stats.total_matches as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    });
    println!();

    // ===== Example 7: Hash Consistency Check =====
    println!("--- Example 7: Block Hash Consistency ---");

    // In a real scenario, block hashes are computed from actual KV cache data
    // Here we demonstrate that xxHash64 is deterministic
    let block_id: BlockId = 999;
    let hash1 = xxh64(&block_id.to_le_bytes(), 0);
    let hash2 = xxh64(&block_id.to_le_bytes(), 0);

    println!("Block ID: {}", block_id);
    println!("Hash 1:   0x{:016x}", hash1);
    println!("Hash 2:   0x{:016x}", hash2);
    println!(
        "✓ Hash consistency: {}",
        if hash1 == hash2 { "PASS" } else { "FAIL" }
    );
    println!();

    println!("=== Demo Complete ===");
}

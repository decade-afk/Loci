//! Example demonstrating enhanced inference engine features
//!
//! This example shows how to use:
//! - Result caching
//! - Timeout control
//! - Resource management
//! - Concurrency control
//! - Batch inference

use loci::prelude::*;
use std::time::Duration;

fn main() -> Result<()> {
    // Example 1: Basic caching configuration
    println!("=== Example 1: Caching ===");
    {
        let cache_config = CacheConfig::new()
            .with_max_entries(1000)
            .with_ttl(Duration::from_secs(3600))
            .with_stats(true);

        let mut engine = InferenceEngine::builder()
            .model_path("path/to/model.gguf")
            .with_cache_config(cache_config)
            .with_cache(true)
            .build()?;

        // First call will compute
        let result1 = engine.generate_with_params(
            "What is Rust?",
            &InferenceParams::default(),
        )?;

        // Second call will use cache
        let result2 = engine.generate_with_params(
            "What is Rust?",
            &InferenceParams::default(),
        )?;

        // Check cache statistics
        let stats = engine.cache_stats();
        println!("Cache hits: {}, misses: {}, hit rate: {:.2}%",
                 stats.hits, stats.misses, stats.hit_rate() * 100.0);
    }

    // Example 2: Timeout control
    println!("\n=== Example 2: Timeout Control ===");
    {
        let timeout_config = TimeoutConfig::new()
            .with_default_timeout(Duration::from_secs(30))
            .with_min_timeout(Duration::from_millis(100))
            .with_max_timeout(Duration::from_secs(300))
            .with_enabled(true);

        let mut engine = InferenceEngine::builder()
            .model_path("path/to/model.gguf")
            .with_timeout_config(timeout_config)
            .with_timeout(true)
            .build()?;

        // Generate with timeout
        let result = engine.generate_with_timeout(
            "Tell me a short story",
            &InferenceParams::default(),
            Duration::from_secs(10),
        )?;

        println!("Generated text: {}", result);

        // Check timeout statistics
        let timeout_stats = engine.timeout_stats();
        println!("Timeouts: {}, completed: {}",
                 timeout_stats.timeouts, timeout_stats.completed);
    }

    // Example 3: Resource management
    println!("\n=== Example 3: Resource Management ===");
    {
        let resource_limits = ResourceLimits::new()
            .with_max_memory_bytes(8_000_000_000) // 8GB
            .with_max_memory_percent(80.0)
            .with_max_cpu_percent(90.0)
            .with_max_concurrent_ops(4)
            .with_grace_period(1000);

        let mut engine = InferenceEngine::builder()
            .model_path("path/to/model.gguf")
            .with_resource_limits(resource_limits)
            .build()?;

        // Check resource usage
        let stats = engine.resource_stats();
        println!("Resource summary: {}", engine.resource_summary());

        // Generate with resource monitoring
        let result = engine.generate_with_params(
            "What is machine learning?",
            &InferenceParams::default(),
        )?;

        println!("Generated text: {}", result);
    }

    // Example 4: Concurrency control
    println!("\n=== Example 4: Concurrency Control ===");
    {
        let concurrency_config = ConcurrencyConfig::new()
            .with_max_concurrent(2)
            .with_queue_size(100)
            .with_rate_limit(true, 10)
            .with_request_timeout(Duration::from_secs(60));

        let mut engine = InferenceEngine::builder()
            .model_path("path/to/model.gguf")
            .with_concurrency_config(concurrency_config)
            .build()?;

        // Check concurrency status
        let stats = engine.concurrency_stats();
        println!("Active: {}, queued: {}, peak: {}",
                 stats.active_ops, stats.queued_ops, stats.peak_concurrent);

        // Generate with concurrency control
        let result = engine.generate_with_params(
            "What is artificial intelligence?",
            &InferenceParams::default(),
        )?;

        println!("Generated text: {}", result);
    }

    // Example 5: Batch inference
    println!("\n=== Example 5: Batch Inference ===");
    {
        let mut engine = InferenceEngine::builder()
            .model_path("path/to/model.gguf")
            .with_cache(true)
            .with_concurrency_config(
                ConcurrencyConfig::new().with_max_concurrent(4)
            )
            .build()?;

        let prompts = vec![
            "What is Python?".to_string(),
            "What is JavaScript?".to_string(),
            "What is C++?".to_string(),
            "What is Java?".to_string(),
        ];

        // Sequential batch inference
        let results = engine.generate_batch(
            &prompts,
            &InferenceParams::default(),
        )?;

        for (prompt, result) in prompts.iter().zip(results.iter()) {
            match result {
                Ok(text) => println!("Q: {}\nA: {}\n", prompt, text),
                Err(e) => println!("Q: {} - Error: {}\n", prompt, e),
            }
        }
    }

    // Example 6: Streaming with timeout
    println!("\n=== Example 6: Streaming with Timeout ===");
    {
        let mut engine = InferenceEngine::builder()
            .model_path("path/to/model.gguf")
            .with_timeout(true)
            .build()?;

        println!("Streaming response: ");

        engine.generate_stream_with_timeout(
            "Tell me about programming languages",
            &InferenceParams::default(),
            Duration::from_secs(15),
            |token| {
                print!("{}", token);
                true // Continue streaming
            },
        )?;

        println!("\n");
    }

    // Example 7: Full configuration
    println!("\n=== Example 7: Full Configuration ===");
    {
        let mut engine = InferenceEngine::builder()
            .model_path("path/to/model.gguf")
            .backend("llama.cpp")
            .context_size(4096)
            .threads(8)
            .batch_size(512)
            .gpu_layers(-1)
            // Enhanced features
            .with_cache(true)
            .with_cache_config(
                CacheConfig::new()
                    .with_max_entries(1000)
                    .with_ttl(Duration::from_secs(3600))
            )
            .with_timeout(true)
            .with_timeout_config(
                TimeoutConfig::new()
                    .with_default_timeout(Duration::from_secs(60))
            )
            .with_resource_limits(
                ResourceLimits::new()
                    .with_max_memory_bytes(8_000_000_000)
                    .with_max_concurrent_ops(4)
            )
            .with_concurrency_config(
                ConcurrencyConfig::new()
                    .with_max_concurrent(4)
                    .with_queue_size(100)
            )
            .build()?;

        // Get comprehensive statistics
        println!("Cache stats: hits={}, misses={}",
                 engine.cache_stats().hits,
                 engine.cache_stats().misses);
        println!("Timeout stats: completed={}, timeouts={}",
                 engine.timeout_stats().completed,
                 engine.timeout_stats().timeouts);
        println!("Resource: {}", engine.resource_summary());
        println!("Concurrency: active={}, queued={}",
                 engine.concurrency_stats().active_ops,
                 engine.concurrency_stats().queued_ops);
    }

    Ok(())
}

/// Example showing manual cache management
fn manual_cache_management() -> Result<()> {
    let mut engine = InferenceEngine::builder()
        .model_path("path/to/model.gguf")
        .with_cache(true)
        .build()?;

    // Generate some content
    engine.generate_with_params(
        "Test prompt",
        &InferenceParams::default(),
    )?;

    // Manually cleanup expired cache entries
    engine.cleanup_cache();

    // Clear entire cache
    engine.clear_cache();

    // Disable caching temporarily
    engine.set_cache_enabled(false);

    // Re-enable caching
    engine.set_cache_enabled(true);

    Ok(())
}

/// Example showing error recovery with timeouts
fn error_recovery_with_timeout() -> Result<()> {
    let mut engine = InferenceEngine::builder()
        .model_path("path/to/model.gguf")
        .with_timeout(true)
        .with_timeout_config(
            TimeoutConfig::new()
                .with_default_timeout(Duration::from_secs(30))
        )
        .build()?;

    // Try with short timeout
    match engine.generate_with_timeout(
        "Long prompt that might timeout",
        &InferenceParams::default(),
        Duration::from_secs(5),
    ) {
        Ok(result) => println!("Success: {}", result),
        Err(e) => {
            if e.to_string().contains("timed out") {
                println!("Operation timed out, retrying with longer timeout...");
                
                // Retry with longer timeout
                match engine.generate_with_timeout(
                    "Long prompt that might timeout",
                    &InferenceParams::default(),
                    Duration::from_secs(60),
                ) {
                    Ok(result) => println!("Success on retry: {}", result),
                    Err(e) => println!("Failed again: {}", e),
                }
            } else {
                println!("Error: {}", e);
            }
        }
    }

    Ok(())
}

/// Example showing resource-aware inference
fn resource_aware_inference() -> Result<()> {
    let mut engine = InferenceEngine::builder()
        .model_path("path/to/model.gguf")
        .with_resource_limits(
            ResourceLimits::new()
                .with_max_memory_percent(75.0)
                .with_max_cpu_percent(85.0)
        )
        .build()?;

    // Check if system is under load before starting
    if engine.is_under_load() {
        println!("System under load, waiting...");
        std::thread::sleep(Duration::from_secs(5));
    }

    // Generate
    let result = engine.generate_with_params(
        "What is cloud computing?",
        &InferenceParams::default(),
    )?;

    println!("Result: {}", result);

    // Check resource usage after generation
    let stats = engine.resource_stats();
    println!("Memory usage: {} / {} ({:.1}%)",
             stats.memory.current_usage,
             stats.memory.total_allocated,
             stats.memory.usage_percent);

    Ok(())
}
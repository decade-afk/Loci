//! Comprehensive Loci Framework Demonstration
//!
//! This example showcases the full capabilities of the Loci framework:
//!
//! 1. **Radix Tree Prefix Caching** - Intelligent KV cache sharing
//! 2. **Model Hot-Swap** - Seamless model switching
//! 3. **Deep Programmable Adapters** - LoRA/QLoRA/AdapterFusion
//! 4. **Multi-Session Management** - Concurrent inference
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example comprehensive_demo
//! ```

use loci::prelude::*;

fn main() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   Loci Framework - Comprehensive Demonstration         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // ===== Phase 1: Radix Tree Prefix Caching =====
    demo_radix_cache();

    // ===== Phase 2: Model Hot-Swap =====
    demo_model_hot_swap();

    // ===== Phase 3: Deep Programmable Adapters =====
    demo_adapter_system();

    // ===== Phase 4: Integration Demo =====
    demo_full_integration();

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   Demo Complete - All Systems Operational!            ║");
    println!("╚════════════════════════════════════════════════════════╝");
}

fn demo_radix_cache() {
    println!("┌────────────────────────────────────────────────────────┐");
    println!("│  Phase 1: Radix Tree Prefix Caching                   │");
    println!("└────────────────────────────────────────────────────────┘\n");

    let cache = ShardedRadixCache::new();

    // Scenario: Three users with similar prompts
    println!("📝 Scenario: Chat-based Q&A system\n");

    // User 1: "Explain quantum computing in simple terms"
    let prompt1: Vec<TokenId> = (1..=64).collect();
    let blocks1: Vec<BlockId> = vec![100, 101];
    cache.insert(&prompt1, &blocks1).unwrap();
    println!("User 1: Inserted 64 tokens (2 blocks)");

    // User 2: "Explain quantum computing in simple terms, focusing on qubits"
    let prompt2: Vec<TokenId> = (1..=96).collect();
    if let Some((matched_tokens, matched_blocks)) = cache.match_prefix(&prompt2) {
        println!(
            "User 2: ✓ Matched {} tokens ({} blocks) - Saved computation!",
            matched_tokens.len(),
            matched_blocks.len()
        );

        // Release when done
        cache.release_blocks(&matched_tokens, &matched_blocks).unwrap();
    }

    // User 3: "Explain quantum computing applications"
    let prompt3: Vec<TokenId> = (1..=48).collect();
    if let Some((matched_tokens, matched_blocks)) = cache.match_prefix(&prompt3) {
        println!(
            "User 3: ✓ Matched {} tokens ({} blocks)",
            matched_tokens.len(),
            matched_blocks.len()
        );

        cache.release_blocks(&matched_tokens, &matched_blocks).unwrap();
    }

    let stats = cache.stats();
    println!("\n📊 Cache Statistics:");
    println!("   Insertions: {}", stats.total_insertions);
    println!("   Matches:    {}", stats.total_matches);
    println!("   Misses:     {}", stats.total_misses);
    println!(
        "   Hit Rate:   {:.1}%\n",
        (stats.total_matches as f64 / (stats.total_matches + stats.total_misses) as f64) * 100.0
    );
}

fn demo_model_hot_swap() {
    println!("┌────────────────────────────────────────────────────────┐");
    println!("│  Phase 2: Model Hot-Swap                              │");
    println!("└────────────────────────────────────────────────────────┘\n");

    let registry = HotSwapModelRegistry::new();

    println!("📦 Loading models...\n");

    // Load different models
    let qwen = registry
        .load_model("qwen-0.5b.gguf", 2048)
        .expect("Failed to load Qwen");
    println!("✓ Loaded Qwen 0.5B (ID: {})", qwen.as_u64());

    let llama = registry
        .load_model("llama-3-8b.gguf", 4096)
        .expect("Failed to load Llama");
    println!("✓ Loaded Llama 3 8B (ID: {})", llama.as_u64());

    let mistral = registry
        .load_model("mistral-7b.gguf", 8192)
        .expect("Failed to load Mistral");
    println!("✓ Loaded Mistral 7B (ID: {})\n", mistral.as_u64());

    // Demonstrate hot-swap
    println!("🔄 Performing hot-swap: Qwen → Llama\n");

    match registry.switch_model(qwen, llama) {
        Ok(()) => println!("✓ Hot-swap successful! (Zero downtime)\n"),
        Err(e) => println!("✗ Hot-swap failed: {}\n", e),
    }

    // Add LoRA to Llama
    println!("🎯 Merging LoRA adapter to Llama...\n");

    let lora = LoRAConfig {
        path: "llama-lora-math.gguf".to_string(),
        scale: 1.0,
    };

    registry.merge_lora(llama, lora).expect("LoRA merge failed");
    println!("✓ LoRA merged successfully\n");

    // List all models
    println!("📋 Model Registry Status:\n");
    for (idx, model) in registry.list_models().iter().enumerate() {
        println!("   [{}] {}", idx + 1, model.id);
        println!("       Path:       {}", model.path);
        println!("       Ref Count:  {}", model.ref_count);
        println!("       LoRAs:      {:?}", model.active_loras);
        println!("       State:      {}\n", model.state);
    }
}

fn demo_adapter_system() {
    println!("┌────────────────────────────────────────────────────────┐");
    println!("│  Phase 3: Deep Programmable Adapters                  │");
    println!("└────────────────────────────────────────────────────────┘\n");

    let mut adapter_registry = AdapterRegistry::new();

    println!("🔧 Registering adapters...\n");

    // Register LoRA
    let lora_id = adapter_registry
        .register_lora(LoRAAdapterConfig {
            path: "math-lora.gguf".to_string(),
            rank: 32,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        })
        .unwrap();
    println!("✓ Registered LoRA (ID: {:?})", lora_id);

    // Register QLoRA
    let qlora_id = adapter_registry
        .register_qlora(QLoRAAdapterConfig {
            path: "code-qlora.gguf".to_string(),
            rank: 16,
            alpha: 16.0,
            quantization: QuantizationType::NF4,
            double_quantization: true,
        })
        .unwrap();
    println!("✓ Registered QLoRA (ID: {:?})", qlora_id);
    println!("   Quantization: NF4 (4-bit)");
    println!("   Memory savings: ~87.5%\n");

    // Apply individual adapter
    println!("🎯 Applying LoRA adapter...\n");
    let input = vec![1.0; 128];
    let output = adapter_registry.apply_adapter(lora_id, &input).unwrap();
    println!("   Input size:  {}", input.len());
    println!("   Output size: {}\n", output.len());

    // Demonstrate AdapterFusion
    println!("🔀 Demonstrating AdapterFusion...\n");

    let fusion_config = AdapterFusionConfig {
        adapter_ids: vec![lora_id, qlora_id],
        fusion_weights: vec![0.7, 0.3],
        strategy: FusionStrategy::Weighted,
    };

    let fused_output = adapter_registry
        .apply_fusion(fusion_config, &input)
        .unwrap();
    println!("   Strategy:    Weighted (70% LoRA + 30% QLoRA)");
    println!("   Output size: {}\n", fused_output.len());

    // Statistics
    let stats = adapter_registry.get_stats();
    println!("📊 Adapter Registry Statistics:\n");
    println!("   Total Adapters:  {}", stats.total_adapters);
    println!("   LoRA Count:      {}", stats.lora_count);
    println!("   QLoRA Count:     {}", stats.qlora_count);
    println!("   Fusion Count:    {}", stats.fusion_count);
    println!("   Total Memory:    {} bytes\n", stats.total_memory);
}

fn demo_full_integration() {
    println!("┌────────────────────────────────────────────────────────┐");
    println!("│  Phase 4: Full Integration                            │");
    println!("└────────────────────────────────────────────────────────┘\n");

    println!("🚀 Simulating a complete inference pipeline:\n");

    // Step 1: Initialize all components
    let radix_cache = ShardedRadixCache::new();
    let model_registry = HotSwapModelRegistry::new();
    let mut adapter_registry = AdapterRegistry::new();

    println!("Step 1: Initialize components");
    println!("   ✓ Radix Cache (16 shards)");
    println!("   ✓ Model Registry");
    println!("   ✓ Adapter Registry\n");

    // Step 2: Load model with adapters
    println!("Step 2: Load base model + adapters");

    let base_model = model_registry
        .load_model("qwen-0.5b.gguf", 2048)
        .unwrap();
    println!("   ✓ Base Model: Qwen 0.5B");

    let math_lora = adapter_registry
        .register_lora(LoRAAdapterConfig {
            path: "math-lora.gguf".to_string(),
            rank: 32,
            alpha: 32.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string()],
        })
        .unwrap();
    println!("   ✓ Math LoRA (rank=32)");

    let code_lora = adapter_registry
        .register_qlora(QLoRAAdapterConfig {
            path: "code-qlora.gguf".to_string(),
            rank: 16,
            alpha: 16.0,
            quantization: QuantizationType::NF4,
            double_quantization: true,
        })
        .unwrap();
    println!("   ✓ Code QLoRA (rank=16, 4-bit)\n");

    // Step 3: Simulate 3 concurrent sessions
    println!("Step 3: Simulate 3 concurrent sessions\n");

    for session_id in 0..3 {
        println!("   Session {}:", session_id + 1);

        // Generate prompt
        let prompt_tokens: Vec<TokenId> = (1 + session_id * 10..65 + session_id * 10)
            .map(|x| x as u32)
            .collect();

        // Check prefix cache
        if let Some((matched_tokens, matched_blocks)) = radix_cache.match_prefix(&prompt_tokens) {
            println!("      ✓ Prefix match: {} blocks", matched_blocks.len());

            // Release blocks when done
            radix_cache
                .release_blocks(&matched_tokens, &matched_blocks)
                .unwrap();
        } else {
            println!("      ✗ No prefix match, inserting...");
            let blocks: Vec<BlockId> = vec![200 + session_id];
            radix_cache.insert(&prompt_tokens, &blocks).unwrap();
        }

        // Apply adapter (simulated)
        let input = vec![1.0; 64];
        let _output = adapter_registry.apply_adapter(math_lora, &input).unwrap();
        println!("      ✓ Applied Math LoRA");

        println!();
    }

    // Step 4: Final statistics
    println!("Step 4: Final Statistics\n");

    let cache_stats = radix_cache.stats();
    println!("   Radix Cache:");
    println!("      Matches:    {}", cache_stats.total_matches);
    println!("      Hit Rate:   {:.1}%", {
        let total = cache_stats.total_matches + cache_stats.total_misses;
        if total > 0 {
            (cache_stats.total_matches as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    });

    let model_count = model_registry.model_count();
    println!("\n   Model Registry:");
    println!("      Loaded:     {}", model_count);

    let adapter_stats = adapter_registry.get_stats();
    println!("\n   Adapter Registry:");
    println!("      Total:      {}", adapter_stats.total_adapters);
    println!("      Memory:     {} KB", adapter_stats.total_memory / 1024);

    println!("\n✨ Integration successful!\n");
}

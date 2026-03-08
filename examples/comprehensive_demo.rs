//! Comprehensive Loci demonstration aligned with the current public API.

use loci::prelude::*;

fn main() {
    println!("=== Loci Comprehensive Demo ===\n");

    demo_radix_cache();
    demo_model_hot_swap();
    demo_adapter_system();
    demo_full_integration();

    println!("\n=== Demo Complete ===");
}

fn demo_radix_cache() {
    println!("[1] Radix Prefix Cache");

    let cache = ShardedRadixCache::new();

    let prompt1: Vec<TokenId> = (1..=64).collect();
    let blocks1: Vec<BlockId> = vec![100, 101];
    cache.insert(&prompt1, &blocks1).unwrap();

    let prompt2: Vec<TokenId> = (1..=96).collect();
    if let Some((matched_tokens, matched_blocks)) = cache.match_prefix(&prompt2) {
        println!(
            "  matched tokens={} blocks={}",
            matched_tokens.len(),
            matched_blocks.len()
        );
        cache
            .release_blocks(&matched_tokens, &matched_blocks)
            .unwrap();
    }

    let stats = cache.stats();
    println!(
        "  stats: insertions={} matches={} misses={}\n",
        stats.total_insertions, stats.total_matches, stats.total_misses
    );
}

fn demo_model_hot_swap() {
    println!("[2] Model Hot-Swap");

    let registry = HotSwapModelRegistry::new();

    let qwen = registry.load_model("qwen-0.5b.gguf", 2048).unwrap();
    let llama = registry.load_model("llama-3-8b.gguf", 4096).unwrap();

    let _ = registry.switch_model(qwen, llama);

    let lora = LoRAConfig {
        path: "llama-lora-math.gguf".to_string(),
        scale: 1.0,
    };
    let _ = registry.merge_lora(llama, lora);

    println!("  model count={}\n", registry.model_count());
}

fn demo_adapter_system() {
    println!("[3] Adapter System");

    let mut adapter_registry = AdapterRegistry::new();

    let lora_id = adapter_registry
        .register_lora(LoRAAdapterConfig {
            path: "math-lora.gguf".to_string(),
            rank: 32,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            use_bias: false,
        })
        .unwrap();

    let qlora_id = adapter_registry
        .register_qlora(QLoRAAdapterConfig {
            path: "code-qlora.gguf".to_string(),
            rank: 16,
            alpha: 16.0,
            quantization: QuantizationType::NF4,
            double_quantization: true,
            compute_dtype: "float16".to_string(),
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        })
        .unwrap();

    let input = vec![1.0; 128];
    let out1 = adapter_registry.apply_adapter(lora_id, &input).unwrap();
    let out2 = adapter_registry.apply_adapter(qlora_id, &out1).unwrap();

    println!(
        "  adapters={} memory={} bytes out={}\n",
        adapter_registry.list_adapters().len(),
        adapter_registry.get_total_memory_footprint(),
        out2.len()
    );
}

fn demo_full_integration() {
    println!("[4] Integrated Flow");

    let radix_cache = ShardedRadixCache::new();
    let model_registry = HotSwapModelRegistry::new();
    let mut adapter_registry = AdapterRegistry::new();

    let _base_model = model_registry.load_model("qwen-0.5b.gguf", 2048).unwrap();

    let math_lora = adapter_registry
        .register_lora(LoRAAdapterConfig {
            path: "math-lora.gguf".to_string(),
            rank: 32,
            alpha: 32.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string()],
            use_bias: false,
        })
        .unwrap();

    let _code_lora = adapter_registry
        .register_qlora(QLoRAAdapterConfig {
            path: "code-qlora.gguf".to_string(),
            rank: 16,
            alpha: 16.0,
            quantization: QuantizationType::NF4,
            double_quantization: true,
            compute_dtype: "float16".to_string(),
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        })
        .unwrap();

    for session_id in 0..3 {
        let prompt_tokens: Vec<TokenId> = (1 + session_id * 10..65 + session_id * 10)
            .map(|x| x as u32)
            .collect();

        if radix_cache.match_prefix(&prompt_tokens).is_none() {
            let blocks: Vec<BlockId> = vec![200 + session_id];
            radix_cache.insert(&prompt_tokens, &blocks).unwrap();
        }

        let input = vec![1.0; 64];
        let _ = adapter_registry.apply_adapter(math_lora, &input).unwrap();
    }

    let cache_stats = radix_cache.stats();
    println!(
        "  cache matches={} misses={}",
        cache_stats.total_matches, cache_stats.total_misses
    );
    println!("  loaded models={}", model_registry.model_count());
    println!(
        "  adapters={} memory={} KB\n",
        adapter_registry.list_adapters().len(),
        adapter_registry.get_total_memory_footprint() / 1024
    );
}

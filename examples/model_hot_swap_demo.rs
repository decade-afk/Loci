//! Model Hot-Swap and LoRA Merging Demonstration
//!
//! This example demonstrates:
//!
//! 1. **Model Hot-Swap**: Seamless switching between models
//! 2. **LoRA Dynamic Merging**: Runtime merging of LoRA adapters
//! 3. **Shared Model Registry**: Efficient memory usage across sessions
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example model_hot_swap_demo
//! ```

use loci::prelude::*;

fn main() {
    println!("=== Model Hot-Swap & LoRA Merging Demo ===\n");

    // ===== Example 1: Model Registry Basics =====
    println!("--- Example 1: Model Registry Basics ---");

    let registry = HotSwapModelRegistry::new();
    println!("✓ Created HotSwapModelRegistry\n");

    // Load first model
    let model1 = registry
        .load_model("qwen-0.5b.gguf", 2048)
        .expect("Failed to load model 1");
    println!("Loaded Model 1 (Qwen 0.5B): ID={}", model1.as_u64());

    // Load second model
    let model2 = registry
        .load_model("llama-3-8b.gguf", 4096)
        .expect("Failed to load model 2");
    println!("Loaded Model 2 (Llama 3 8B): ID={}", model2.as_u64());

    println!("Total models loaded: {}\n", registry.model_count());

    // ===== Example 2: Reference Counting =====
    println!("--- Example 2: Reference Counting ---");

    // Load same model twice (should reuse)
    let model1_dup = registry
        .load_model("qwen-0.5b.gguf", 2048)
        .expect("Failed to load duplicate");

    println!("Model 1 ID:          {}", model1.as_u64());
    println!("Model 1 Duplicate ID: {}", model1_dup.as_u64());
    println!("IDs match: {}", model1 == model1_dup);

    if let Some(m) = registry.get_model(model1) {
        println!("Model 1 ref_count: {}", m.ref_count());
    }

    println!();

    // ===== Example 3: Model Hot-Swap =====
    println!("--- Example 3: Model Hot-Swap ---");

    println!("Current model: Model 1 (Qwen)");
    println!("Switching to: Model 2 (Llama 3)");

    // Perform hot-swap
    match registry.switch_model(model1, model2) {
        Ok(()) => println!("✓ Model switch successful!"),
        Err(e) => println!("✗ Model switch failed: {}", e),
    }

    // Check new ref counts
    if let Some(m1) = registry.get_model(model1) {
        println!("Model 1 ref_count after switch: {}", m1.ref_count());
    }

    if let Some(m2) = registry.get_model(model2) {
        println!("Model 2 ref_count after switch: {}", m2.ref_count());
    }

    println!();

    // ===== Example 4: LoRA Merging =====
    println!("--- Example 4: LoRA Dynamic Merging ---");

    // Create LoRA configuration
    let lora_config = LoRAConfig {
        path: "qwen-lora-math.gguf".to_string(),
        scale: 1.0,
    };

    println!("Merging LoRA adapter: {}", lora_config.path);
    println!("Scaling factor: {}", lora_config.scale);

    match registry.merge_lora(model2, lora_config.clone()) {
        Ok(()) => println!("✓ LoRA merge successful!"),
        Err(e) => println!("✗ LoRA merge failed: {}", e),
    }

    // Check active LoRAs
    if let Some(m) = registry.get_model(model2) {
        let loras = m.active_loras();
        println!("Active LoRAs on Model 2: {:?}", loras);
    }

    println!();

    // ===== Example 5: Multiple LoRA Adapters =====
    println!("--- Example 5: Multiple LoRA Adapters ---");

    let lora_code = LoRAConfig {
        path: "qwen-lora-code.gguf".to_string(),
        scale: 0.8,
    };

    registry.merge_lora(model2, lora_code).expect("Failed to merge code LoRA");

    if let Some(m) = registry.get_model(model2) {
        let loras = m.active_loras();
        println!("Total LoRAs on Model 2: {}", loras.len());
        for (idx, lora) in loras.iter().enumerate() {
            println!("  [{}] {}", idx + 1, lora);
        }
    }

    println!();

    // ===== Example 6: Clearing LoRAs =====
    println!("--- Example 6: Clearing LoRAs ---");

    println!("Clearing all LoRAs from Model 2...");
    registry.clear_loras(model2).expect("Failed to clear LoRAs");

    if let Some(m) = registry.get_model(model2) {
        let loras = m.active_loras();
        println!("✓ LoRAs cleared. Remaining: {}", loras.len());
    }

    println!();

    // ===== Example 7: Listing All Models =====
    println!("--- Example 7: Listing All Models ---");

    let models = registry.list_models();
    println!("Total models in registry: {}", models.len());

    for (idx, info) in models.iter().enumerate() {
        println!("\nModel {}:", idx + 1);
        println!("  ID:        {}", info.id);
        println!("  Path:      {}", info.path);
        println!("  Ref Count: {}", info.ref_count);
        println!("  State:     {}", info.state);
        println!("  LoRAs:     {:?}", info.active_loras);
    }

    println!();

    // ===== Example 8: Model Unloading =====
    println!("--- Example 8: Model Unloading ---");

    println!("Models before unload: {}", registry.model_count());

    // Unload duplicate reference
    registry.unload_model(model1_dup).expect("Failed to unload");
    println!("Unloaded duplicate reference");

    // Unload original reference
    registry.unload_model(model1).expect("Failed to unload");
    println!("Unloaded original reference");

    println!("Models after unload: {}", registry.model_count());

    // Model 1 should be removed (ref_count reached 0)
    if registry.has_model(model1) {
        println!("⚠ Model 1 still in registry");
    } else {
        println!("✓ Model 1 successfully removed");
    }

    println!();

    // ===== Example 9: Concurrent Access Simulation =====
    println!("--- Example 9: Concurrent Access (Single-threaded Demo) ---");

    use std::sync::Arc;

    let shared_registry = Arc::new(registry);

    // Simulate 3 "sessions" sharing the same model
    for session_id in 0..3 {
        let model_ref = shared_registry
            .get_model(model2)
            .expect("Model 2 not found");

        println!(
            "Session {}: Using Model 2 (ref_count: {})",
            session_id,
            model_ref.ref_count()
        );
    }

    println!();

    // ===== Final Statistics =====
    println!("--- Final Statistics ---");

    let final_models = shared_registry.list_models();
    println!("Total models loaded: {}", final_models.len());

    for model in &final_models {
        println!("  {}: ref_count={}", model.id, model.ref_count);
    }

    println!();
    println!("=== Demo Complete ===");
}

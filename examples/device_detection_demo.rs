//! Device Detection and Auto-Selection Demo
//!
//! This example demonstrates:
//! - Automatic device detection (CUDA, Metal, Vulkan, ROCm, OpenCL, CPU)
//! - Device enumeration and information query
//! - Automatic selection of best device
//! - Model size-based device recommendation
//! - Manual device selection

use loci::prelude::*;

fn main() -> Result<()> {
    println!("=== Loci Device Detection Demo ===\n");

    // Create device selector
    let selector = DeviceSelector::new();

    // 1. List all detected devices
    println!("1. Detected Devices:");
    println!("   Total devices: {}\n", selector.device_count());

    for (i, device) in selector.devices().iter().enumerate() {
        println!("   Device {}:", i);
        println!("     Name: {}", device.name);
        println!("     Type: {}", device.device_type);
        println!("     Memory: {:.2} GB", device.memory_gb());
        println!("     Available: {}", device.available);
        if device.compute_capability > 0.0 {
            println!("     Compute Capability: {:.1}", device.compute_capability);
        }
        println!();
    }

    // 2. Check specific backends
    println!("2. Backend Availability:");
    println!("   CUDA: {}", selector.has_backend(DeviceType::CUDA));
    println!("   Metal: {}", selector.has_backend(DeviceType::Metal));
    println!("   Vulkan: {}", selector.has_backend(DeviceType::Vulkan));
    println!("   ROCm: {}", selector.has_backend(DeviceType::ROCm));
    println!("   OpenCL: {}", selector.has_backend(DeviceType::OpenCL));
    println!("   CPU: {}\n", selector.has_backend(DeviceType::CPU));

    // 3. Automatic device selection
    println!("3. Automatic Device Selection:");
    let auto_config = selector.auto_select();
    println!(
        "   Selected: {} (Device ID: {})",
        auto_config.device_type, auto_config.device_id
    );
    println!("   GPU Layers: {}\n", auto_config.n_gpu_layers);

    // 4. Model size-based recommendation
    println!("4. Model Size Recommendations:");

    let model_sizes = vec![
        ("Qwen-0.5B", 0.6),
        ("Qwen-1.5B", 1.5),
        ("Llama-7B", 7.0),
        ("Llama-13B", 13.0),
        ("Llama-70B", 70.0),
    ];

    for (name, size) in model_sizes {
        let config = selector.recommend_for_model(size);
        println!(
            "   {} ({} GB) -> {} (Device {}, {} GPU layers)",
            name, size, config.device_type, config.device_id, config.n_gpu_layers
        );
    }

    println!("\n5. Usage Examples:\n");

    // Example 1: Auto mode
    println!("   Example 1: Automatic device selection");
    println!("   ```rust");
    println!("   let selector = DeviceSelector::new();");
    println!("   let config = selector.auto_select();");
    println!("   // config automatically uses best available device");
    println!("   ```\n");

    // Example 2: Manual selection
    if let Some(device) = selector.device(0) {
        println!("   Example 2: Manual device selection");
        println!("   ```rust");
        println!("   let selector = DeviceSelector::new();");
        println!("   if let Some(device) = selector.device(0) {{");
        println!("       let config = DeviceConfig::gpu_all(");
        println!("           DeviceType::{:?},", device.device_type);
        println!("           device.id");
        println!("       );");
        println!("   }}");
        println!("   ```\n");
    }

    // Example 3: Memory-aware selection
    println!("   Example 3: Memory-aware selection for 7B model");
    println!("   ```rust");
    println!("   let selector = DeviceSelector::new();");
    println!("   let config = selector.recommend_for_model(7.0); // 7GB model");
    println!("   // Automatically handles partial GPU offloading if needed");
    println!("   ```\n");

    println!("=== Demo Complete ===");

    Ok(())
}

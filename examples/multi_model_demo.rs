/**
 * Loci Phase 3 Week 2: 多模型热切换示例
 *
 * 演示功能：
 * 1. 加载多个模型
 * 2. 在 Session 之间切换模型
 * 3. 加载和卸载 LoRA 适配器
 * 4. 内存预算管理
 */

use loci::{LoRAConfig, LoRAAdapter, MODEL_REGISTRY};
use std::path::Path;
use anyhow::Result;

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║   Loci Multi-Model Hot-Switching Demo       ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

    // 1. 加载第一个模型（基础模型）
    println!("📦 加载基础模型...");
    let model1_path = Path::new("models/llama-7b-q4_0.gguf");
    let model1_id = MODEL_REGISTRY.load_model(model1_path)?;
    println!("✅ 模型 1 ID: {}", model1_id);
    println!();

    // 2. 加载第二个模型（代码模型）
    println!("📦 加载代码模型...");
    let model2_path = Path::new("models/codellama-7b-q4_0.gguf");
    let model2_id = MODEL_REGISTRY.load_model(model2_path)?;
    println!("✅ 模型 2 ID: {}", model2_id);
    println!();

    // 3. 创建两个会话
    let session1_id = "session-conversation";
    let session2_id = "session-coding";

    println!("🔧 创建会话...");
    MODEL_REGISTRY.switch_model(session1_id, &model1_id)?;
    MODEL_REGISTRY.switch_model(session2_id, &model2_id)?;
    println!("✅ {} → {}", session1_id, model1_id);
    println!("✅ {} → {}", session2_id, model2_id);
    println!();

    // 4. 演示模型切换
    println!("🔄 演示热切换：Session 1 切换到代码模型");
    MODEL_REGISTRY.switch_model(session1_id, &model2_id)?;
    println!("✅ {} 现在使用 {}", session1_id, model2_id);
    println!();

    // 5. 加载 LoRA 适配器
    println!("🎨 加载 LoRA 适配器（风格插件）...");
    let lora_config = LoRAConfig {
        path: Path::new("loras/creative-writing.gguf").to_path_buf(),
        scale: 0.8,
        priority: 1,
    };

    match LoRAAdapter::new(lora_config) {
        Ok(lora) => {
            let lora_id = lora.id.clone();
            println!("✅ LoRA 已加载: {}", lora_id);

            // 附加 LoRA 到模型 1
            let model1 = MODEL_REGISTRY.get_model(session2_id)?;
            let mut model1 = model1.write().unwrap();

            match model1.add_lora(std::sync::Arc::new(std::sync::RwLock::new(lora))) {
                Ok(_) => println!("✅ LoRA 已附加到模型"),
                Err(e) => println!("⚠️  LoRA 附加失败: {}", e),
            }
        }
        Err(e) => println!("⚠️  LoRA 加载失败（可能文件不存在）: {}", e),
    }
    println!();

    // 6. 显示内存统计
    println!("📊 内存使用统计:");
    let (used, budget, percent) = MODEL_REGISTRY.memory_stats();
    println!("  已用: {} MB", used / 1024 / 1024);
    println!("  预算: {} MB", budget / 1024 / 1024);
    println!("  使用率: {:.2}%", percent);
    println!();

    // 7. 列出所有已加载的模型
    println!("📋 已加载的模型:");
    for model_id in MODEL_REGISTRY.list_models() {
        println!("  - {}", model_id);
    }
    println!();

    // 8. 卸载模型
    println!("🗑️  卸载模型 1 (将失败，因为仍在使用)");
    match MODEL_REGISTRY.unload_model(&model1_id) {
        Ok(_) => println!("✅ 模型 1 已卸载"),
        Err(e) => println!("⚠️  无法卸载: {}", e),
    }
    println!();

    // 9. 释放会话引用后卸载
    println!("🔓 释放所有会话对模型 1 的引用...");
    // 在实际应用中，这里会调用 session.close() 等方法
    println!("（在实际应用中需要实现 Session 管理）");
    println!();

    println!("╔══════════════════════════════════════════════╗");
    println!("║              演示完成！                      ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();
    println!("💡 关键特性:");
    println!("  ✓ 多模型同时加载");
    println!("  ✓ Session 级别模型热切换");
    println!("  ✓ LoRA 动态附加/分离");
    println!("  ✓ 内存预算自动管理");
    println!("  ✓ 引用计数防误删");

    Ok(())
}

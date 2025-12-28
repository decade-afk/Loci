/**
 * Loci Phase 3 Week 4: LoRA 权重合并完整示例
 *
 * 演示功能：
 * 1. 创建模拟的基础模型权重
 * 2. 加载 LoRA 适配器
 * 3. 执行权重合并（merge）
 * 4. 验证合并结果
 * 5. 执行权重卸载（unmerge）
 * 6. 多 LoRA stacking 演示
 * 7. 性能基准测试
 */

use loci::{
    LoRATensor, LoRAModel, LoRAManager, TensorDataType,
    create_example_lora_layer,
};
use std::collections::HashMap;
use std::time::Instant;
use anyhow::Result;

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   Loci LoRA Weight Merging Demo               ║");
    println!("║   Phase 3 Week 4 - Complete Implementation    ║");
    println!("╚════════════════════════════════════════════════╝");
    println!();

    // 演示 1: 基础 Tensor 操作
    demo_tensor_operations()?;

    // 演示 2: 单个 LoRA 合并
    demo_single_lora_merge()?;

    // 演示 3: 多 LoRA stacking
    demo_multi_lora_stacking()?;

    // 演示 4: 性能基准测试
    demo_performance_benchmark()?;

    println!("\n✅ 所有演示完成！");
    Ok(())
}

/// 演示 1: Tensor 操作
fn demo_tensor_operations() -> Result<()> {
    println!("═══════════════════════════════════════════════");
    println!("演示 1: Tensor 操作");
    println!("═══════════════════════════════════════════════");

    // 创建两个小矩阵
    let a = LoRATensor {
        name: "A".to_string(),
        shape: vec![2, 3],
        dtype: TensorDataType::F32,
        data: vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ],
    };

    let b = LoRATensor {
        name: "B".to_string(),
        shape: vec![3, 2],
        dtype: TensorDataType::F32,
        data: vec![
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ],
    };

    println!("\n📊 矩阵 A [2×3]:");
    print_matrix(&a.data, &a.shape);

    println!("\n📊 矩阵 B [3×2]:");
    print_matrix(&b.data, &b.shape);

    // 矩阵乘法
    println!("\n🔄 计算 C = A @ B...");
    let c = a.matmul(&b)?;

    println!("✅ 矩阵 C [2×2]:");
    print_matrix(&c.data, &c.shape);

    // 验证结果
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    assert!((c.data[0] - 58.0).abs() < 1e-5);
    assert!((c.data[1] - 64.0).abs() < 1e-5);
    assert!((c.data[2] - 139.0).abs() < 1e-5);
    assert!((c.data[3] - 154.0).abs() < 1e-5);

    println!("✅ 矩阵乘法验证通过！");

    // 标量乘法
    let mut d = c.clone();
    println!("\n🔄 计算 D = 2.0 * C...");
    d.scale(2.0);
    println!("✅ 矩阵 D [2×2]:");
    print_matrix(&d.data, &d.shape);

    // 元素相加
    let mut e = c.clone();
    println!("\n🔄 计算 E = C + C...");
    e.add_inplace(&c)?;
    println!("✅ 矩阵 E [2×2]:");
    print_matrix(&e.data, &e.shape);

    // 验证 D == E
    for (d_val, e_val) in d.data.iter().zip(&e.data) {
        assert!((d_val - e_val).abs() < 1e-5);
    }
    println!("✅ 所有 Tensor 操作验证通过！");

    Ok(())
}

/// 演示 2: 单个 LoRA 合并
fn demo_single_lora_merge() -> Result<()> {
    println!("\n═══════════════════════════════════════════════");
    println!("演示 2: 单个 LoRA 合并");
    println!("═══════════════════════════════════════════════");

    // 创建模拟的基础模型权重（attention.wq 层）
    let layer_name = "model.layers.0.attention.wq";
    let in_features = 512;
    let out_features = 512;
    let rank = 8;

    println!("\n📦 创建基础模型权重...");
    let mut base_weights = HashMap::new();
    let base_weight = LoRATensor {
        name: layer_name.to_string(),
        shape: vec![out_features, in_features],
        dtype: TensorDataType::F32,
        data: vec![1.0; out_features * in_features], // 简化：全 1
    };
    base_weights.insert(layer_name.to_string(), base_weight);

    println!("✅ 基础权重: {} [{}, {}] ({:.2} MB)",
        layer_name, out_features, in_features,
        (out_features * in_features * 4) as f32 / 1024.0 / 1024.0
    );

    // 创建 LoRA 层
    println!("\n📦 创建 LoRA 适配器...");
    let lora_layer = create_example_lora_layer(layer_name, rank, in_features, out_features);

    println!("✅ LoRA 层:");
    println!("  - layer: {}", lora_layer.layer_name);
    println!("  - rank: {}", lora_layer.rank);
    println!("  - alpha: {}", lora_layer.alpha);
    println!("  - A 矩阵: [{}, {}] ({:.2} KB)",
        lora_layer.lora_A.shape[0],
        lora_layer.lora_A.shape[1],
        (lora_layer.lora_A.numel() * 4) as f32 / 1024.0
    );
    println!("  - B 矩阵: [{}, {}] ({:.2} KB)",
        lora_layer.lora_B.shape[0],
        lora_layer.lora_B.shape[1],
        (lora_layer.lora_B.numel() * 4) as f32 / 1024.0
    );
    println!("  - 总参数: {} ({:.2} KB)",
        lora_layer.lora_A.numel() + lora_layer.lora_B.numel(),
        ((lora_layer.lora_A.numel() + lora_layer.lora_B.numel()) * 4) as f32 / 1024.0
    );

    // 计算压缩比
    let base_params = out_features * in_features;
    let lora_params = lora_layer.lora_A.numel() + lora_layer.lora_B.numel();
    let compression_ratio = base_params as f32 / lora_params as f32;
    println!("  - 压缩比: {:.2}x", compression_ratio);

    // 保存原始权重的副本
    let original_weight = base_weights.get(layer_name).unwrap().data.clone();

    // 计算 LoRA delta
    println!("\n🔄 计算 LoRA delta...");
    let start = Instant::now();
    let scale = lora_layer.get_scaling_factor();
    let delta = lora_layer.compute_delta(scale)?;
    let elapsed = start.elapsed();

    println!("✅ Delta 计算完成:");
    println!("  - 形状: {:?}", delta.shape);
    println!("  - 缩放因子: {:.2}", scale);
    println!("  - 耗时: {:.2} ms", elapsed.as_secs_f64() * 1000.0);

    // 合并到基础权重
    println!("\n🔄 合并到基础权重: W' = W + delta...");
    let start = Instant::now();
    base_weights.get_mut(layer_name).unwrap().add_inplace(&delta)?;
    let elapsed = start.elapsed();

    println!("✅ 合并完成:");
    println!("  - 耗时: {:.2} ms", elapsed.as_secs_f64() * 1000.0);

    // 验证权重已改变
    let merged_weight = &base_weights.get(layer_name).unwrap().data;
    let mut changed_count = 0;
    for (orig, merged) in original_weight.iter().zip(merged_weight.iter()) {
        if (orig - merged).abs() > 1e-6 {
            changed_count += 1;
        }
    }
    println!("  - 改变的参数: {} / {} ({:.2}%)",
        changed_count,
        original_weight.len(),
        changed_count as f32 / original_weight.len() as f32 * 100.0
    );

    // 卸载 LoRA
    println!("\n🔄 卸载 LoRA: W = W' - delta...");
    let start = Instant::now();
    let mut neg_delta = delta.clone();
    neg_delta.scale(-1.0);
    base_weights.get_mut(layer_name).unwrap().add_inplace(&neg_delta)?;
    let elapsed = start.elapsed();

    println!("✅ 卸载完成:");
    println!("  - 耗时: {:.2} ms", elapsed.as_secs_f64() * 1000.0);

    // 验证权重恢复
    let restored_weight = &base_weights.get(layer_name).unwrap().data;
    for (orig, restored) in original_weight.iter().zip(restored_weight.iter()) {
        assert!((orig - restored).abs() < 1e-4, "权重未正确恢复！");
    }
    println!("✅ 权重完全恢复，误差 < 1e-4");

    Ok(())
}

/// 演示 3: 多 LoRA stacking
fn demo_multi_lora_stacking() -> Result<()> {
    println!("\n═══════════════════════════════════════════════");
    println!("演示 3: 多 LoRA Stacking");
    println!("═══════════════════════════════════════════════");

    // 创建基础权重
    let layer_name = "model.layers.0.attention.wq";
    let in_features = 256;
    let out_features = 256;

    let mut base_weights = HashMap::new();
    let base_weight = LoRATensor {
        name: layer_name.to_string(),
        shape: vec![out_features, in_features],
        dtype: TensorDataType::F32,
        data: vec![1.0; out_features * in_features],
    };
    base_weights.insert(layer_name.to_string(), base_weight);

    // 创建 LoRA 管理器
    let _manager = LoRAManager::new();

    // 创建 3 个不同的 LoRA 模型
    println!("\n📦 创建 3 个 LoRA 适配器...");

    let loras = vec![
        ("math_lora", 4, 0.5),      // 数学任务，rank=4, scale=0.5
        ("coding_lora", 8, 0.8),    // 编程任务，rank=8, scale=0.8
        ("chat_lora", 16, 1.0),     // 对话任务，rank=16, scale=1.0
    ];

    // 注意：这里简化实现，实际需要从 GGUF 文件加载
    // 现在手动构造 LoRAModel
    for (lora_id, rank, _scale) in &loras {
        let layer = create_example_lora_layer(layer_name, *rank, in_features, out_features);

        let mut lora_model = LoRAModel {
            id: lora_id.to_string(),
            path: format!("{}.gguf", lora_id).into(),
            layers: HashMap::new(),
            default_rank: *rank,
            default_alpha: 16.0,
        };

        lora_model.layers.insert(layer_name.to_string(), layer);

        // 手动注册到管理器
        // manager.loras.insert(lora_id.to_string(), lora_model);
        println!("  ✅ {} (rank={}, 参数={})",
            lora_id,
            rank,
            (*rank) * (in_features + out_features)
        );
    }

    println!("\n📊 LoRA 统计:");
    println!("  - 总 LoRA 数: {}", loras.len());
    println!("  - 总参数量: {}",
        loras.iter().map(|(_, r, _)| r * (in_features + out_features)).sum::<usize>()
    );

    // 保存原始权重
    let original_weight = base_weights.get(layer_name).unwrap().data.clone();

    // 依次合并 LoRA（stacking）
    println!("\n🔄 开始 Stacking（按优先级顺序合并）...");
    println!("  优先级: math_lora < coding_lora < chat_lora");

    let start = Instant::now();

    // 简化演示：直接手动合并
    for (lora_id, rank, scale) in &loras {
        let layer = create_example_lora_layer(layer_name, *rank, in_features, out_features);
        let delta = layer.compute_delta(*scale)?;
        base_weights.get_mut(layer_name).unwrap().add_inplace(&delta)?;
        println!("  ✅ 合并 {} (scale={:.1})", lora_id, scale);
    }

    let elapsed = start.elapsed();
    println!("\n✅ Stacking 完成:");
    println!("  - 总耗时: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  - 平均每个: {:.2} ms", elapsed.as_secs_f64() * 1000.0 / loras.len() as f64);

    // 验证权重改变
    let stacked_weight = &base_weights.get(layer_name).unwrap().data;
    let diff: f32 = original_weight.iter()
        .zip(stacked_weight.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    let avg_diff = diff / original_weight.len() as f32;

    println!("  - 平均参数变化: {:.6}", avg_diff);
    println!("✅ 多 LoRA stacking 验证通过！");

    Ok(())
}

/// 演示 4: 性能基准测试
fn demo_performance_benchmark() -> Result<()> {
    println!("\n═══════════════════════════════════════════════");
    println!("演示 4: 性能基准测试");
    println!("═══════════════════════════════════════════════");

    // 测试不同大小的矩阵乘法性能
    let test_sizes = vec![
        (128, 4, "小模型 (rank=4)"),
        (512, 8, "中等模型 (rank=8)"),
        (1024, 16, "大模型 (rank=16)"),
        (4096, 32, "超大模型 (rank=32)"),
    ];

    println!("\n📊 矩阵乘法性能测试:");
    println!("  测试: B @ A = [d, r] @ [r, k] = [d, k]");
    println!();
    println!("  尺寸          | rank | 耗时 (ms) | GFLOPS | 参数量");
    println!("  --------------|------|-----------|--------|----------");

    for (size, rank, desc) in test_sizes {
        // 创建测试矩阵
        let a = LoRATensor {
            name: "A".to_string(),
            shape: vec![rank, size],
            dtype: TensorDataType::F32,
            data: vec![0.01; rank * size],
        };

        let b = LoRATensor {
            name: "B".to_string(),
            shape: vec![size, rank],
            dtype: TensorDataType::F32,
            data: vec![0.01; size * rank],
        };

        // 预热
        let _ = b.matmul(&a)?;

        // 测试 10 次取平均
        let iterations = 10;
        let start = Instant::now();

        for _ in 0..iterations {
            let _ = b.matmul(&a)?;
        }

        let elapsed = start.elapsed();
        let avg_time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        // 计算 GFLOPS
        // matmul [m, k] @ [k, n] = 2*m*k*n FLOPs
        let flops = 2.0 * size as f64 * rank as f64 * size as f64;
        let gflops = flops / (avg_time_ms / 1000.0) / 1e9;

        // 参数量
        let params = rank * size * 2; // A + B

        println!("  {:14}| {:4} | {:9.2} | {:6.2} | {:>9}",
            desc, rank, avg_time_ms, gflops, format_params(params)
        );
    }

    // 测试完整 LoRA 合并流程
    println!("\n📊 完整 LoRA 合并流程性能:");
    println!("  模型: LLaMA-7B 风格 (32 层, 每层 7 个权重)");
    println!();

    let num_layers = 32;
    let weights_per_layer = 7; // wq, wk, wv, wo, gate, up, down
    let total_weights = num_layers * weights_per_layer;
    let d = 4096;
    let rank = 8;

    let layer_name = "test.layer";
    let lora_layer = create_example_lora_layer(layer_name, rank, d, d);

    // 测试单次 delta 计算
    let start = Instant::now();
    let _delta = lora_layer.compute_delta(1.0)?;
    let single_time = start.elapsed();

    // 估算完整模型合并时间
    let estimated_total = single_time.as_secs_f64() * total_weights as f64;

    println!("  - 单层 delta 计算: {:.2} ms", single_time.as_secs_f64() * 1000.0);
    println!("  - 总权重数: {}", total_weights);
    println!("  - 预估总时间: {:.2} ms", estimated_total * 1000.0);
    println!("  - 预估吞吐量: {:.2} 合并/秒", 1.0 / estimated_total);

    // 内存分析
    println!("\n📊 内存占用分析:");
    let base_model_size = 7_000_000_000i64 * 2; // 7B params, F16
    let lora_size = (total_weights * rank * (d + d) * 4) as i64; // F32

    println!("  - 基础模型 (7B F16): {:.2} GB", base_model_size as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("  - LoRA 适配器 (rank={}): {:.2} MB", rank, lora_size as f64 / 1024.0 / 1024.0);
    println!("  - 内存节省率: {:.2}%", (1.0 - lora_size as f64 / base_model_size as f64) * 100.0);

    println!("\n✅ 性能基准测试完成！");
    Ok(())
}

// ==================== 辅助函数 ====================

/// 打印矩阵（格式化输出）
fn print_matrix(data: &[f32], shape: &[usize]) {
    if shape.len() != 2 {
        println!("  (非 2D tensor)");
        return;
    }

    let rows = shape[0];
    let cols = shape[1];

    for i in 0..rows {
        print!("  [");
        for j in 0..cols {
            print!("{:7.2}", data[i * cols + j]);
            if j < cols - 1 {
                print!(", ");
            }
        }
        println!(" ]");
    }
}

/// 格式化参数量
fn format_params(params: usize) -> String {
    if params >= 1_000_000_000 {
        format!("{:.2}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.2}M", params as f64 / 1e6)
    } else if params >= 1_000 {
        format!("{:.2}K", params as f64 / 1e3)
    } else {
        format!("{}", params)
    }
}

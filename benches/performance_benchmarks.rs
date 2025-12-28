//! # Loci 性能基准测试套件
//!
//! 测试各模块的性能指标，验证设计目标。

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use loci::*;
use std::time::Duration;

// ==================== Paged Attention 基准测试 ====================

fn bench_paged_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("paged_attention");

    // 创建会话管理器
    let budget = paged_attention::MemoryBudget {
        total_vram_mb: 8192,
        total_ram_mb: 32768,
    };
    let manager = paged_attention::SessionManager::new(budget);

    // 测试 1: 创建会话
    group.bench_function("create_session", |b| {
        b.iter(|| {
            let session_id = manager.create_session().unwrap();
            black_box(session_id);
        });
    });

    // 测试 2: 分配块
    let session_id = manager.create_session().unwrap();
    group.bench_function("allocate_block", |b| {
        b.iter(|| {
            let block_id = manager.allocate_block(session_id).unwrap();
            black_box(block_id);
        });
    });

    // 测试 3: Swap 到 RAM
    let block_id = manager.allocate_block(session_id).unwrap();
    group.bench_function("swap_to_ram", |b| {
        b.iter(|| {
            let mut mgr = manager.clone();
            mgr.swap_to_ram(block_id).unwrap();
        });
    });

    // 测试 4: Swap 回 VRAM
    group.bench_function("swap_to_vram", |b| {
        b.iter(|| {
            let mut mgr = manager.clone();
            mgr.swap_to_vram(block_id).unwrap();
        });
    });

    group.finish();
}

// ==================== Constraint Sampling 基准测试 ====================

fn bench_constraints(c: &mut Criterion) {
    let mut group = c.benchmark_group("constraints");

    // 测试数据
    let vocab_size = 50000;
    let generated_tokens: Vec<i32> = (0..100).collect();

    // 测试 1: Regex 约束检查
    let regex_constraint = constraints::RegexConstraint::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    let ctx = constraints::ConstraintContext {
        generated_tokens: &generated_tokens,
        generated_text: Some("2024-01-01"),
        candidate_token: 2024,
        candidate_text: Some("2024"),
        vocab_size,
    };

    group.bench_function("regex_is_allowed", |b| {
        b.iter(|| {
            let allowed = regex_constraint.is_allowed(black_box(2024), black_box(&ctx));
            black_box(allowed);
        });
    });

    // 测试 2: JSON Schema 约束检查
    let json_constraint = constraints::JsonSchemaConstraint::new(r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"}
        }
    }"#).unwrap();

    group.bench_function("json_is_allowed", |b| {
        b.iter(|| {
            let allowed = json_constraint.is_allowed(black_box(123), black_box(&ctx));
            black_box(allowed);
        });
    });

    // 测试 3: TokenMask 创建
    group.bench_function("token_mask_create", |b| {
        b.iter(|| {
            let mask = constraints::TokenMask::new_allow_all(black_box(vocab_size));
            black_box(mask);
        });
    });

    // 测试 4: TokenMask 查询
    let mask = constraints::TokenMask::new_allow_all(vocab_size);
    group.bench_function("token_mask_is_allowed", |b| {
        b.iter(|| {
            let allowed = mask.is_allowed(black_box(1234));
            black_box(allowed);
        });
    });

    // 测试 5: 批量过滤
    group.bench_function("token_mask_batch_filter", |b| {
        b.iter(|| {
            let mut mask = constraints::TokenMask::new_allow_all(vocab_size);
            for i in 0..100 {
                mask.disallow(i);
            }
            black_box(mask);
        });
    });

    group.finish();
}

// ==================== Radix Tree 基准测试 ====================

fn bench_radix_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("radix_tree");

    let manager = radix_tree::KVCacheManager::new();

    // 测试 1: 插入单个 prompt
    let tokens: Vec<radix_tree::TokenId> = (0..100).collect();
    group.bench_function("insert_prompt", |b| {
        b.iter(|| {
            let node_id = manager.insert_prompt(black_box(tokens.clone())).unwrap();
            black_box(node_id);
        });
    });

    // 测试 2: 搜索前缀（命中）
    manager.insert_prompt(tokens.clone()).unwrap();
    let query: Vec<radix_tree::TokenId> = (0..50).collect();
    group.bench_function("search_prefix_hit", |b| {
        b.iter(|| {
            let result = manager.search_prefix(black_box(&query));
            black_box(result);
        });
    });

    // 测试 3: 搜索前缀（未命中）
    let query_miss: Vec<radix_tree::TokenId> = (1000..1050).collect();
    group.bench_function("search_prefix_miss", |b| {
        b.iter(|| {
            let result = manager.search_prefix(black_box(&query_miss));
            black_box(result);
        });
    });

    // 测试 4: LCP 计算
    let seq1: Vec<radix_tree::TokenId> = (0..100).collect();
    let seq2: Vec<radix_tree::TokenId> = (0..80).chain(200..220).collect();
    group.bench_function("compute_lcp", |b| {
        b.iter(|| {
            // 内部 LCP 算法
            let lcp_len = seq1.iter()
                .zip(seq2.iter())
                .take_while(|(a, b)| a == b)
                .count();
            black_box(lcp_len);
        });
    });

    // 测试 5: 批量插入（测试前缀共享效率）
    group.bench_function("batch_insert_similar", |b| {
        b.iter(|| {
            let mgr = radix_tree::KVCacheManager::new();
            let prefix: Vec<radix_tree::TokenId> = (0..50).collect();

            // 插入 100 个相似 prompt
            for i in 0..100 {
                let mut prompt = prefix.clone();
                prompt.extend(1000 + i..1010 + i);
                mgr.insert_prompt(prompt).unwrap();
            }
            black_box(mgr);
        });
    });

    group.finish();
}

// ==================== 插件系统基准测试 ====================

fn bench_plugin_system(c: &mut Criterion) {
    let mut group = c.benchmark_group("plugin_system");

    // 测试 1: LogitsView 创建（零拷贝）
    let mut logits_data: Vec<f32> = (0..50000).map(|i| i as f32 * 0.001).collect();
    group.bench_function("logits_view_create", |b| {
        b.iter(|| {
            let view = plugin_system::LogitsView::new(black_box(&mut logits_data));
            black_box(view);
        });
    });

    // 测试 2: LogitsView 读取
    let mut view = plugin_system::LogitsView::new(&mut logits_data);
    group.bench_function("logits_view_get", |b| {
        b.iter(|| {
            let val = view.get(black_box(1234));
            black_box(val);
        });
    });

    // 测试 3: LogitsView 写入
    group.bench_function("logits_view_set", |b| {
        b.iter(|| {
            view.set(black_box(1234), black_box(0.5));
        });
    });

    // 测试 4: Watchdog 超时检测（模拟）
    let quota = plugin_system::ResourceQuota {
        timeout: Duration::from_millis(50),
        max_memory_mb: 100,
    };
    let watchdog = plugin_system::Watchdog::new(quota);

    group.bench_function("watchdog_check", |b| {
        b.iter(|| {
            let result = watchdog.execute_with_timeout(|| {
                // 模拟插件工作
                std::thread::sleep(Duration::from_micros(10));
                Ok(42)
            });
            black_box(result);
        });
    });

    group.finish();
}

// ==================== Suspend/Resume 基准测试 ====================

fn bench_suspend_resume(c: &mut Criterion) {
    let mut group = c.benchmark_group("suspend_resume");

    // 测试 1: ControlFlow 匹配
    let control_flow = suspend::ControlFlow::Continue;
    group.bench_function("control_flow_match", |b| {
        b.iter(|| {
            match black_box(&control_flow) {
                suspend::ControlFlow::Continue => 0,
                suspend::ControlFlow::Suspend(_) => 1,
                suspend::ControlFlow::Stop(_) => 2,
            }
        });
    });

    // 测试 2: ResumeContext 创建
    group.bench_function("resume_context_create", |b| {
        b.iter(|| {
            let ctx = suspend::ResumeContext {
                injection_type: suspend::InjectionType::ToolResult,
                content: black_box("Tool result".to_string()),
                metadata: std::collections::HashMap::new(),
            };
            black_box(ctx);
        });
    });

    group.finish();
}

// ==================== 综合性能测试 ====================

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");

    // 模拟完整推理流程（不含实际模型）
    group.bench_function("full_pipeline_simulation", |b| {
        b.iter(|| {
            // 1. Paged Attention: 创建会话
            let manager = paged_attention::SessionManager::new(paged_attention::MemoryBudget {
                total_vram_mb: 8192,
                total_ram_mb: 32768,
            });
            let session = manager.create_session().unwrap();

            // 2. Radix Tree: 插入 prompt
            let cache_manager = radix_tree::KVCacheManager::new();
            let tokens: Vec<radix_tree::TokenId> = (0..100).collect();
            cache_manager.insert_prompt(tokens).unwrap();

            // 3. Constraint: 应用约束
            let constraint = constraints::TokenMask::new_allow_all(50000);
            let allowed = constraint.is_allowed(1234);

            // 4. Plugin: 转换 logits
            let mut logits = vec![0.1; 1000];
            let mut view = plugin_system::LogitsView::new(&mut logits);
            view.set(0, 1.5);

            black_box((session, allowed, view));
        });
    });

    group.finish();
}

// ==================== 内存基准测试 ====================

fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory");

    // 测试 1: Radix Tree 内存节省
    group.bench_function("radix_tree_memory_savings", |b| {
        b.iter(|| {
            let manager = radix_tree::KVCacheManager::new();
            let prefix: Vec<radix_tree::TokenId> = (0..50).collect();

            // 插入 100 个相似 prompt
            for i in 0..100 {
                let mut prompt = prefix.clone();
                prompt.extend(1000 + i..1010 + i);
                manager.insert_prompt(prompt).unwrap();
            }

            let stats = manager.get_stats();
            black_box(stats.memory_saved_percent);
        });
    });

    group.finish();
}

// ==================== Criterion 配置 ====================

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)                     // 每个测试 100 个样本
        .measurement_time(Duration::from_secs(5))  // 每个测试运行 5 秒
        .warm_up_time(Duration::from_secs(1));     // 预热 1 秒

    targets =
        bench_paged_attention,
        bench_constraints,
        bench_radix_tree,
        bench_plugin_system,
        bench_suspend_resume,
        bench_end_to_end,
        bench_memory
}

criterion_main!(benches);

/**
 * Loci 流式输出示例
 *
 * 演示如何使用 Loci 的流式生成功能：
 * 1. 控制台实时输出
 * 2. 自定义回调
 * 3. 提前停止
 * 4. 批量回调
 * 5. 累积器回调
 *
 * 运行方式：
 * cargo run --example streaming_demo
 */

use loci::{
    LociEngine, EngineConfig,
    ConsoleCallback, AccumulatorCallback, ClosureCallback, BatchedCallback,
    StreamControlFlow, StreamCallback, StreamToken,
};

fn main() -> anyhow::Result<()> {
    println!("=== Loci 流式输出示例 ===\n");

    // 注意：这个示例需要真实的 GGUF 模型文件
    // 如果没有模型，请修改路径或注释掉实际的生成调用

    let model_path = std::env::var("LOCI_MODEL_PATH")
        .unwrap_or_else(|_| "models/llama-3-8b-q4_k_m.gguf".to_string());

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("⚠️  模型文件不存在: {}", model_path);
        eprintln!("请设置环境变量 LOCI_MODEL_PATH 指向有效的 GGUF 模型");
        eprintln!("\n运行测试（不需要真实模型）...\n");
        run_tests();
        return Ok(());
    }

    // 初始化引擎
    println!("📦 正在加载模型...");
    let config = EngineConfig {
        model_path: model_path.clone(),
        n_gpu_layers: -1,  // 自动探测
        n_ctx: 2048,
        ..Default::default()
    };

    let engine = LociEngine::new(config)?;
    println!("✅ 模型加载完成\n");

    // ==================== 示例 1: 控制台实时输出 ====================
    println!("--- 示例 1: 控制台实时输出 ---");
    let mut console_callback = ConsoleCallback::new(true);  // 立即刷新

    let stats = engine.generate_stream(
        "Write a short poem about AI",
        50,
        &mut console_callback,
    )?;

    println!("\n统计信息:");
    println!("  生成 tokens: {}", stats.generated_tokens);
    println!("  吞吐量: {:.2} tokens/s\n", stats.tokens_per_second);

    // ==================== 示例 2: 自定义回调 ====================
    println!("\n--- 示例 2: 自定义回调（添加计数器） ---");
    let mut token_count = 0;
    let mut callback = ClosureCallback::new(|token, _id, _pos| {
        token_count += 1;
        print!("[{}] {}", token_count, token);
        std::io::Write::flush(&mut std::io::stdout()).ok();
        StreamControlFlow::Continue
    });

    let stats = engine.generate_stream(
        "Explain quantum computing in one sentence",
        30,
        &mut callback,
    )?;

    println!("\n\n总计生成 {} tokens\n", token_count);

    // ==================== 示例 3: 提前停止 ====================
    println!("\n--- 示例 3: 提前停止（生成 10 个 token 后停止） ---");
    let mut callback = ClosureCallback::new(|token, _id, pos| {
        print!("{}", token);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        if pos >= 9 {  // 生成 10 个 token 后停止
            StreamControlFlow::Stop
        } else {
            StreamControlFlow::Continue
        }
    });

    let stats = engine.generate_stream(
        "Tell me a long story about",
        100,  // 虽然请求 100 个，但会提前停止
        &mut callback,
    )?;

    println!("\n\n实际生成: {} tokens（请求 100，提前停止）\n", stats.generated_tokens);

    // ==================== 示例 4: 累积器回调 ====================
    println!("\n--- 示例 4: 累积器回调（收集完整输出） ---");
    let mut accumulator = AccumulatorCallback::new();

    let stats = engine.generate_stream(
        "What is Rust?",
        40,
        &mut accumulator,
    )?;

    println!("完整输出:\n{}\n", accumulator.get_content());
    println!("Token 详情:");
    for (i, token) in accumulator.get_tokens().iter().enumerate() {
        println!("  [{}] '{}' (id={})", i, token.content, token.token_id);
        if i >= 5 {
            println!("  ...");
            break;
        }
    }

    // ==================== 示例 5: 批量回调 ====================
    println!("\n\n--- 示例 5: 批量回调（每 5 个 token 一次） ---");
    let mut batch_count = 0;
    let mut callback = BatchedCallback::new(
        |tokens: &[StreamToken]| {
            batch_count += 1;
            println!("批次 {}: 收到 {} tokens", batch_count, tokens.len());
            for token in tokens {
                print!("{}", token.content);
            }
            println!();
            StreamControlFlow::Continue
        },
        5,  // 批量大小 = 5
    );

    let stats = engine.generate_stream(
        "Count from 1 to 20",
        30,
        &mut callback,
    )?;

    println!("\n总批次数: {}", batch_count);

    println!("\n✅ 所有示例完成！");
    Ok(())
}

/// 运行不需要真实模型的测试
fn run_tests() {
    println!("--- 测试回调接口 ---\n");

    // 测试 ClosureCallback
    println!("1. 测试 ClosureCallback:");
    let mut count = 0;
    let mut callback = ClosureCallback::new(|token, id, pos| {
        count += 1;
        println!("  Token #{}: '{}' (id={}, pos={})", count, token, id, pos);
        StreamControlFlow::Continue
    });

    callback.on_token("Hello", 1, 0);
    callback.on_token(" ", 2, 1);
    callback.on_token("World", 3, 2);
    println!("  累计: {} tokens\n", count);

    // 测试 AccumulatorCallback
    println!("2. 测试 AccumulatorCallback:");
    let mut accumulator = AccumulatorCallback::new();
    accumulator.on_token("Rust", 10, 0);
    accumulator.on_token(" ", 11, 1);
    accumulator.on_token("is", 12, 2);
    accumulator.on_token(" ", 13, 3);
    accumulator.on_token("awesome", 14, 4);

    println!("  完整内容: '{}'", accumulator.get_content());
    println!("  Token 数量: {}\n", accumulator.get_tokens().len());

    // 测试提前停止
    println!("3. 测试提前停止:");
    let mut stopped = false;
    let mut callback = ClosureCallback::new(|_token, _id, pos| {
        if pos >= 2 {
            stopped = true;
            StreamControlFlow::Stop
        } else {
            StreamControlFlow::Continue
        }
    });

    callback.on_token("a", 1, 0);
    callback.on_token("b", 2, 1);
    callback.on_token("c", 3, 2);
    println!("  在位置 2 停止: {}\n", stopped);

    // 测试 BatchedCallback
    println!("4. 测试 BatchedCallback:");
    let mut batch_sizes = Vec::new();
    let mut callback = BatchedCallback::new(
        |tokens: &[StreamToken]| {
            batch_sizes.push(tokens.len());
            println!("  批次: {} tokens", tokens.len());
            StreamControlFlow::Continue
        },
        3,  // 批量大小 = 3
    );

    for i in 0..7 {
        callback.on_token(&format!("t{}", i), i, i as usize);
    }

    use loci::streaming::StreamStats;
    callback.on_complete(&StreamStats::default());

    println!("  批次大小: {:?}", batch_sizes);
    println!("  总批次: {}\n", batch_sizes.len());

    println!("✅ 所有测试通过！");
}

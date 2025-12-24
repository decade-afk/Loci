/**
 * loci-cli: Phase 1验证工具
 *
 * 用途：
 * 1. 验证Phase 1实现的正确性
 * 2. 性能基准测试（Load Time, Eval Speed）
 * 3. Backend探测验证
 * 4. GGUF加载验证
 *
 * 使用示例：
 * ```
 * loci-cli --model path/to/model.gguf --prompt "Hello, world!"
 * loci-cli --model model.gguf --benchmark
 * loci-cli --backend-info
 * ```
 */

use std::path::PathBuf;
use anyhow::{Result, Context};

// Phase 1依赖
use loci::backend::detect_backend;
use loci::gguf::GGUFModel;
use loci::engine::{LociEngine, EngineConfig};

#[derive(Debug)]
struct CliArgs {
    model_path: Option<PathBuf>,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: f32,
    n_gpu_layers: i32,
    benchmark: bool,
    backend_info: bool,
    gguf_info: bool,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            model_path: None,
            prompt: Some("Once upon a time".to_string()),
            max_tokens: 50,
            temperature: 0.8,
            n_gpu_layers: -1,  // -1表示自动
            benchmark: false,
            backend_info: false,
            gguf_info: false,
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();

    println!("╔════════════════════════════════════════╗");
    println!("║  Loci CLI - Phase 1 Verification Tool ║");
    println!("╚════════════════════════════════════════╝");
    println!();

    let args = parse_args()?;

    // 模式1：Backend信息
    if args.backend_info {
        return show_backend_info();
    }

    // 模式2：GGUF信息
    if args.gguf_info {
        if let Some(model_path) = args.model_path {
            return show_gguf_info(&model_path);
        } else {
            eprintln!("❌ --gguf-info requires --model");
            std::process::exit(1);
        }
    }

    // 模式3：Benchmark
    if args.benchmark {
        if let Some(model_path) = args.model_path {
            return run_benchmark(&model_path, &args);
        } else {
            eprintln!("❌ --benchmark requires --model");
            std::process::exit(1);
        }
    }

    // 模式4：交互式生成
    if let Some(model_path) = args.model_path {
        return run_interactive(&model_path, &args);
    }

    // 默认：显示帮助
    show_help();
    Ok(())
}

/// 解析命令行参数
fn parse_args() -> Result<CliArgs> {
    let mut args = CliArgs::default();
    let mut iter = std::env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--model" | "-m" => {
                args.model_path = Some(PathBuf::from(
                    iter.next().context("--model requires a path")?
                ));
            }
            "--prompt" | "-p" => {
                args.prompt = Some(iter.next().context("--prompt requires a string")?);
            }
            "--max-tokens" | "-n" => {
                args.max_tokens = iter.next()
                    .context("--max-tokens requires a number")?
                    .parse()
                    .context("Invalid number for --max-tokens")?;
            }
            "--temperature" | "-t" => {
                args.temperature = iter.next()
                    .context("--temperature requires a number")?
                    .parse()
                    .context("Invalid number for --temperature")?;
            }
            "--gpu-layers" | "-ngl" => {
                args.n_gpu_layers = iter.next()
                    .context("--gpu-layers requires a number")?
                    .parse()
                    .context("Invalid number for --gpu-layers")?;
            }
            "--benchmark" | "-b" => {
                args.benchmark = true;
            }
            "--backend-info" => {
                args.backend_info = true;
            }
            "--gguf-info" => {
                args.gguf_info = true;
            }
            "--help" | "-h" => {
                show_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("❌ Unknown argument: {}", arg);
                show_help();
                std::process::exit(1);
            }
        }
    }

    Ok(args)
}

/// 显示Backend信息
fn show_backend_info() -> Result<()> {
    println!("🔍 Detecting compute backend...");
    println!();

    let backend = detect_backend();
    let device_info = backend.device_info();

    println!("Backend Information:");
    println!("  Type: {}", device_info.backend_type);
    println!("  Name: {}", device_info.name);
    println!("  Compute: {}", device_info.compute_capability);

    if device_info.memory_total_mb > 0 {
        println!("  Memory: {:.2} GB total, {:.2} GB available",
                 device_info.memory_total_mb as f64 / 1024.0,
                 device_info.memory_available_mb as f64 / 1024.0);
    }

    println!("  Recommended GPU Layers: {}", backend.recommended_gpu_layers());
    println!();

    println!("✅ Backend detection successful!");
    Ok(())
}

/// 显示GGUF模型信息
fn show_gguf_info(model_path: &PathBuf) -> Result<()> {
    println!("📂 Loading GGUF model...");
    println!();

    let model = GGUFModel::load(model_path)
        .context("Failed to load GGUF model")?;

    let metadata = model.metadata();

    println!("GGUF Model Information:");
    println!("  Version: {}", metadata.version);
    println!("  Tensor Count: {}", metadata.tensor_count);
    println!("  Metadata KV Count: {}", metadata.metadata_kv_count);
    println!();

    if let Some(name) = &metadata.model_name {
        println!("  Model Name: {}", name);
    }
    if let Some(arch) = &metadata.architecture {
        println!("  Architecture: {}", arch);
    }
    if let Some(emb_len) = metadata.embedding_length {
        println!("  Embedding Length: {}", emb_len);
    }
    if let Some(block_count) = metadata.block_count {
        println!("  Block Count: {}", block_count);
    }
    if let Some(ctx_len) = metadata.context_length {
        println!("  Context Length: {}", ctx_len);
    }

    println!();
    println!("  File Size: {:.2} GB", model.total_size_gb());
    println!();

    println!("Tensor Information (first 10):");
    for (i, name) in model.tensor_names().iter().take(10).enumerate() {
        println!("  {}. {}", i + 1, name);
    }

    if model.tensor_names().len() > 10 {
        println!("  ... and {} more tensors", model.tensor_names().len() - 10);
    }

    println!();
    println!("✅ GGUF loaded successfully (zero-copy)!");
    Ok(())
}

/// 运行性能基准测试
fn run_benchmark(model_path: &PathBuf, args: &CliArgs) -> Result<()> {
    println!("🎯 Running Phase 1 Benchmark...");
    println!();

    let config = EngineConfig {
        model_path: model_path.to_string_lossy().to_string(),
        n_gpu_layers: args.n_gpu_layers,
        temperature: args.temperature,
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Model: {}", config.model_path);
    println!("  Context Size: {}", config.n_ctx);
    println!("  Batch Size: {}", config.n_batch);
    println!("  GPU Layers: {}", config.n_gpu_layers);
    println!("  Temperature: {}", config.temperature);
    println!();

    // 测试1：加载时间
    println!("Test 1: Model Load Time");
    let load_start = std::time::Instant::now();
    let engine = LociEngine::new(config)?;
    let load_time = load_start.elapsed();

    println!("  ✅ Load Time: {:.2}ms", load_time.as_millis());
    println!("  Target: < 500ms ... {}",
             if load_time.as_millis() < 500 { "PASS 🎯" } else { "FAIL ⚠️" });
    println!();

    // 测试2：推理速度
    println!("Test 2: Inference Speed");
    let test_prompts = vec![
        "The quick brown fox",
        "In a galaxy far far away",
        "Once upon a time",
    ];

    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("  Prompt {}: \"{}\"", i + 1, prompt);
        engine.generate(prompt, 20)?;

        let stats = engine.stats();
        println!("  Eval Speed: {:.2} tokens/s", stats.eval_tokens_per_second());
        println!();
    }

    let final_stats = engine.stats();
    let avg_speed = final_stats.eval_tokens_per_second();

    println!("Benchmark Results:");
    println!("  Backend: {}", engine.backend_name());
    println!("  Load Time: {:.2}ms", final_stats.load_time_ms);
    println!("  Avg Eval Speed: {:.2} tokens/s", avg_speed);

    // Phase 1目标检查
    let cpu_target = 10.0;
    let gpu_target = 20.0;

    if engine.backend_name().contains("CUDA") || engine.backend_name().contains("Metal") {
        println!("  GPU Target: > {} tokens/s ... {}",
                 gpu_target,
                 if avg_speed > gpu_target { "PASS 🎯" } else { "FAIL ⚠️" });
    } else {
        println!("  CPU Target: > {} tokens/s ... {}",
                 cpu_target,
                 if avg_speed > cpu_target { "PASS 🎯" } else { "FAIL ⚠️" });
    }

    println!();
    println!("✅ Benchmark completed!");
    Ok(())
}

/// 运行交互式生成
fn run_interactive(model_path: &PathBuf, args: &CliArgs) -> Result<()> {
    println!("🚀 Initializing engine...");
    println!();

    let config = EngineConfig {
        model_path: model_path.to_string_lossy().to_string(),
        n_gpu_layers: args.n_gpu_layers,
        temperature: args.temperature,
        ..Default::default()
    };

    let engine = LociEngine::new(config)?;

    println!();
    println!("📝 Generating text...");
    println!("─────────────────────────────────────");

    let prompt = args.prompt.as_ref().unwrap();
    print!("{}", prompt);
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let generated = engine.generate(prompt, args.max_tokens)?;

    println!();
    println!("─────────────────────────────────────");
    println!();

    let stats = engine.stats();
    println!("Performance:");
    println!("  Backend: {}", engine.backend_name());
    println!("  Load Time: {:.2}ms", stats.load_time_ms);
    println!("  Prompt Eval: {} tokens in {:.2}ms ({:.2} t/s)",
             stats.prompt_eval_count,
             stats.prompt_eval_time_ms,
             stats.prompt_tokens_per_second());
    println!("  Generation: {} tokens in {:.2}ms ({:.2} t/s)",
             stats.eval_count,
             stats.eval_time_ms,
             stats.eval_tokens_per_second());

    println!();
    println!("✅ Generation completed!");
    Ok(())
}

/// 显示帮助信息
fn show_help() {
    println!("Usage: loci-cli [OPTIONS]");
    println!();
    println!("Options:");
    println!("  -m, --model <PATH>          Path to GGUF model file");
    println!("  -p, --prompt <TEXT>         Prompt text (default: \"Once upon a time\")");
    println!("  -n, --max-tokens <N>        Maximum tokens to generate (default: 50)");
    println!("  -t, --temperature <F>       Temperature for sampling (default: 0.8)");
    println!("  -ngl, --gpu-layers <N>      GPU layers (-1 for auto, default: -1)");
    println!("  -b, --benchmark             Run performance benchmark");
    println!("  --backend-info              Show compute backend information");
    println!("  --gguf-info                 Show GGUF model information");
    println!("  -h, --help                  Show this help message");
    println!();
    println!("Examples:");
    println!("  loci-cli --backend-info");
    println!("  loci-cli -m model.gguf --gguf-info");
    println!("  loci-cli -m model.gguf -p \"Hello, world!\" -n 100");
    println!("  loci-cli -m model.gguf --benchmark");
}

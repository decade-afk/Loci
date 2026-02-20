use clap::Parser;
use loci::prelude::*;
use loci::inference::GenerationParams;
use std::path::PathBuf;
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(name = "loci")]
#[command(about = "A cross-platform local LLM inference tool", long_about = None)]
struct Args {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: PathBuf,

    /// Prompt text (if not provided, enters interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,

    /// Backend name (llama.cpp, candle, or dynamically registered backend)
    #[arg(long, default_value = "llama.cpp")]
    backend: String,

    /// Context size
    #[arg(short = 'c', long, default_value_t = 4096)]
    context_size: u32,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 512)]
    max_tokens: u32,

    /// Temperature (0.0 = greedy, higher = more random)
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-p sampling threshold
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Top-k sampling threshold
    #[arg(long, default_value_t = 40)]
    top_k: u32,

    /// Number of threads (default: auto-detect)
    #[arg(long)]
    threads: Option<u32>,

    /// Disable GPU acceleration
    #[arg(long)]
    cpu_only: bool,

    /// Number of GPU layers to offload (-1 = all)
    #[arg(long, default_value_t = -1)]
    gpu_layers: i32,

    /// Enable streaming output
    #[arg(short, long)]
    stream: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Loading model from: {}", args.model.display());
    println!("Backend: {}", args.backend);
    println!("Context size: {}", args.context_size);
    println!("GPU layers: {}", args.gpu_layers);

    let mut builder = InferenceEngine::builder()
        .model_path(&args.model)
        .backend(&args.backend)
        .context_size(args.context_size)
        .batch_size(512)
        .gpu_layers(args.gpu_layers);

    if args.cpu_only {
        builder = builder.cpu_only();
    }

    if let Some(threads) = args.threads {
        builder = builder.threads(threads);
    }

    // Create inference engine
    let mut engine = builder.build()?;

    // Get model info
    let info = engine.model_info();
    println!("Model loaded successfully!");
    println!("  Vocabulary size: {}", info.n_vocab);
    println!("  Training context: {}", info.n_ctx_train);
    println!("  Embedding dimension: {}", info.n_embd);
    println!();

    // Create generation parameters
    let gen_params = GenerationParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        ..Default::default()
    };

    if let Some(prompt) = args.prompt {
        // Single prompt mode
        run_single_prompt(&mut engine, &prompt, gen_params, args.stream)?;
    } else {
        // Interactive mode
        run_interactive(&mut engine, gen_params, args.stream)?;
    }

    Ok(())
}

fn run_single_prompt(
    engine: &mut InferenceEngine,
    prompt: &str,
    params: GenerationParams,
    stream: bool,
) -> anyhow::Result<()> {
    println!("Prompt: {}", prompt);
    println!("\nResponse:");
    println!("---");

    if stream {
        engine.generate_stream(prompt, params, |token| {
            print!("{}", token);
            io::stdout().flush().unwrap();
            true
        })?;
        println!();
    } else {
        let response = engine.generate(prompt, params)?;
        println!("{}", response);
    }

    println!("---");
    Ok(())
}

fn run_interactive(
    engine: &mut InferenceEngine,
    params: GenerationParams,
    stream: bool,
) -> anyhow::Result<()> {
    println!("Interactive mode. Type 'exit' or 'quit' to exit.");
    println!();

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            println!("Goodbye!");
            break;
        }

        println!("\nResponse:");
        println!("---");

        if stream {
            engine.generate_stream(input, params.clone(), |token| {
                print!("{}", token);
                io::stdout().flush().unwrap();
                true
            })?;
            println!();
        } else {
            let response = engine.generate(input, params.clone())?;
            println!("{}", response);
        }

        println!("---");
        println!();
    }

    Ok(())
}

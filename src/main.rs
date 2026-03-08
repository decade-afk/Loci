use clap::{Args as ClapArgs, Parser, Subcommand};
use loci::image_kernel::{load_dynamic_image_plugin, ImageGenerationRequest};
use loci::inference::GenerationParams;
use loci::plugin_registry::PluginRegistry;
use loci::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_HTTP_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_HTTP_LINE_BYTES: usize = 8 * 1024;
const MIN_PROMPT_BYTES_LIMIT: usize = 1024;

fn parse_prompt_bytes(raw: &str) -> std::result::Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("invalid integer: `{raw}`"))?;
    if value < MIN_PROMPT_BYTES_LIMIT {
        return Err(format!(
            "max prompt bytes must be >= {}",
            MIN_PROMPT_BYTES_LIMIT
        ));
    }
    Ok(value)
}

#[derive(Parser, Debug)]
#[command(name = "loci")]
#[command(about = "A cross-platform local LLM inference tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    // Legacy mode (kept for backward compatibility): loci -m ... -p ...
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Prompt text (if not provided, enters interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,
    /// Backend name (llama.cpp, candle, or dynamically registered backend)
    #[arg(long)]
    backend: Option<String>,
    /// Dynamic backend library to register before engine build (.dll/.so/.dylib)
    #[arg(long = "backend-lib")]
    backend_lib: Option<PathBuf>,
    /// Dynamic backend registration name (defaults to backend name or `dynamic.plugin`)
    #[arg(long = "backend-register-name")]
    backend_register_name: Option<String>,
    /// Context length
    #[arg(short = 'c', long = "context-length", visible_alias = "context-size")]
    context_size: Option<u32>,
    /// Max prompt bytes safety limit (minimum 1024)
    #[arg(long = "max-prompt-bytes", value_parser = parse_prompt_bytes)]
    max_prompt_bytes: Option<usize>,
    /// Maximum tokens to generate
    #[arg(short = 'n', long)]
    max_tokens: Option<u32>,
    /// Temperature (0.0 = greedy, higher = more random)
    #[arg(short, long)]
    temperature: Option<f32>,
    /// Top-p sampling threshold
    #[arg(long)]
    top_p: Option<f32>,
    /// Min-p sampling threshold
    #[arg(long)]
    min_p: Option<f32>,
    /// Top-k sampling threshold
    #[arg(long)]
    top_k: Option<u32>,
    /// Repetition penalty
    #[arg(long = "repetition-penalty", visible_alias = "repeat-penalty")]
    repetition_penalty: Option<f32>,
    /// Number of threads (default: auto-detect)
    #[arg(long)]
    threads: Option<u32>,
    /// Disable GPU acceleration
    #[arg(long)]
    cpu_only: bool,
    /// Number of GPU layers to offload (-1 = all)
    #[arg(long)]
    gpu_layers: Option<i32>,
    /// LoRA adapter path(s). Accepted for compatibility; backend merge support is build-dependent.
    #[arg(long = "lora-path")]
    lora_paths: Vec<PathBuf>,
    /// Load plugin(s) for this run (.wasm => WASM plugin, otherwise dynamic plugin)
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    /// Enable streaming output
    #[arg(short, long)]
    stream: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate text from a prompt
    Generate(GenerateCmd),
    /// Generate image from a prompt (text-to-image)
    Image(ImageCmd),
    /// Start REST server
    Serve(ServeCmd),
    /// Run agent mode with a selected tool
    Agent(AgentCmd),
    /// Manage plugins
    Plugin(PluginCmd),
}

#[derive(ClapArgs, Debug, Clone)]
struct EngineArgs {
    /// Backend name (llama.cpp, candle, ...)
    #[arg(long, default_value = "llama.cpp")]
    backend: String,
    /// Dynamic backend library to register before engine build (.dll/.so/.dylib)
    #[arg(long = "backend-lib")]
    backend_lib: Option<PathBuf>,
    /// Dynamic backend registration name (defaults to backend name or `dynamic.plugin`)
    #[arg(long = "backend-register-name")]
    backend_register_name: Option<String>,
    /// Context length
    #[arg(
        short = 'c',
        long = "context-length",
        visible_alias = "context-size",
        default_value_t = 4096
    )]
    context_size: u32,
    /// Max prompt bytes safety limit (minimum 1024)
    #[arg(long = "max-prompt-bytes", value_parser = parse_prompt_bytes)]
    max_prompt_bytes: Option<usize>,
    /// Number of threads (default: auto-detect)
    #[arg(long)]
    threads: Option<u32>,
    /// Disable GPU acceleration
    #[arg(long)]
    cpu_only: bool,
    /// Number of GPU layers to offload (-1 = all)
    #[arg(long, default_value_t = -1)]
    gpu_layers: i32,
    /// LoRA adapter path(s). Accepted for compatibility; backend merge support is build-dependent.
    #[arg(long = "lora-path")]
    lora_paths: Vec<PathBuf>,
}

#[derive(ClapArgs, Debug, Clone)]
struct SamplingArgs {
    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 512)]
    max_tokens: u32,
    /// Temperature (0.0 = greedy, higher = more random)
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f32,
    /// Top-p sampling threshold
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    /// Min-p sampling threshold
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// Top-k sampling threshold
    #[arg(long, default_value_t = 40)]
    top_k: u32,
    /// Repetition penalty
    #[arg(
        long = "repetition-penalty",
        visible_alias = "repeat-penalty",
        default_value_t = 1.1
    )]
    repetition_penalty: f32,
}

#[derive(ClapArgs, Debug, Clone)]
struct GenerateCmd {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: PathBuf,
    /// Prompt text (if not provided, enters interactive mode)
    #[arg(short, long)]
    prompt: Option<String>,
    /// Enable streaming output
    #[arg(short, long)]
    stream: bool,
    /// Load plugin(s) for this run (.wasm => WASM plugin, otherwise dynamic plugin)
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    sampling: SamplingArgs,
}

#[derive(ClapArgs, Debug, Clone)]
struct ServeCmd {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: PathBuf,
    /// Listen host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Listen port
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// API type (currently supports: rest)
    #[arg(long, default_value = "rest")]
    api_type: String,
    /// Load plugin(s) globally for this server (.wasm => WASM plugin, otherwise dynamic plugin)
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    sampling: SamplingArgs,
}

#[derive(ClapArgs, Debug, Clone)]
struct AgentCmd {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: PathBuf,
    /// Tool name (e.g., web_search)
    #[arg(long, default_value = "none")]
    tool: String,
    /// Agent prompt
    #[arg(short, long)]
    prompt: String,
    /// Enable streaming output
    #[arg(short, long)]
    stream: bool,
    /// Load plugin(s) for this run (.wasm => WASM plugin, otherwise dynamic plugin)
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    sampling: SamplingArgs,
}

#[derive(ClapArgs, Debug, Clone)]
struct PluginCmd {
    /// Registry config file path
    #[arg(long, default_value = "loci_plugins.toml")]
    registry: PathBuf,
    #[command(subcommand)]
    command: PluginAction,
}

#[derive(ClapArgs, Debug, Clone)]
struct ImageCmd {
    /// Text prompt for image generation
    #[arg(short, long)]
    prompt: String,
    /// Diffusion model identifier (Hugging Face model ID or local path)
    #[arg(long, default_value = "hf-internal-testing/tiny-stable-diffusion-pipe")]
    model_id: String,
    /// Output image path
    #[arg(short, long, default_value = "outputs/t2i.png")]
    output: PathBuf,
    /// Number of denoising steps
    #[arg(long, default_value_t = 4)]
    steps: u32,
    /// Guidance scale (CFG)
    #[arg(long, default_value_t = 0.0)]
    guidance_scale: f32,
    /// Optional image width (if omitted, model default is used)
    #[arg(long)]
    width: Option<u32>,
    /// Optional image height (if omitted, model default is used)
    #[arg(long)]
    height: Option<u32>,
    /// Optional random seed for reproducibility
    #[arg(long)]
    seed: Option<u64>,
    /// Dynamic image kernel plugin path (.dll/.so/.dylib). If provided, use plugin kernel for inference.
    #[arg(long = "kernel-plugin")]
    kernel_plugin: Option<PathBuf>,
    /// Python executable to run the image backend script
    #[arg(long, default_value = "python")]
    python: String,
    /// Use CUDA if available (falls back to CPU automatically)
    #[arg(long)]
    use_cuda: bool,
}

#[derive(Subcommand, Debug, Clone)]
enum PluginAction {
    /// Load a plugin from path (.wasm => wasm plugin, else dynamic plugin)
    Load { path: PathBuf },
    /// List all registered plugins
    List,
    /// Show detailed info for one plugin
    Info { name: String },
    /// Unload a hot-swappable plugin by name
    Unload { name: String },
    /// Reload a hot-swappable plugin by name
    Reload { name: String },
    /// Enable plugin by name
    Enable { name: String },
    /// Disable plugin by name
    Disable { name: String },
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

#[derive(Debug)]
enum HttpRequestParseError {
    Io(io::Error),
    EmptyRequest,
    InvalidRequestLine,
    HeaderLineTooLong(usize),
    HeadersTooLarge(usize),
    InvalidContentLength(String),
    BodyTooLarge { content_length: usize, limit: usize },
}

impl std::fmt::Display for HttpRequestParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error while reading request: {err}"),
            Self::EmptyRequest => write!(f, "empty request"),
            Self::InvalidRequestLine => write!(f, "invalid HTTP request line"),
            Self::HeaderLineTooLong(line_bytes) => {
                write!(
                    f,
                    "single HTTP header line too large: {line_bytes} bytes (limit: {MAX_HTTP_LINE_BYTES} bytes)"
                )
            }
            Self::HeadersTooLarge(header_bytes) => write!(
                f,
                "HTTP headers too large: {header_bytes} bytes (limit: {MAX_HTTP_HEADER_BYTES} bytes)"
            ),
            Self::InvalidContentLength(raw) => {
                write!(f, "invalid Content-Length header value: `{raw}`")
            }
            Self::BodyTooLarge {
                content_length,
                limit,
            } => write!(
                f,
                "request body too large: {content_length} bytes (limit: {limit} bytes)"
            ),
        }
    }
}

impl std::error::Error for HttpRequestParseError {}

impl HttpRequestParseError {
    fn status_code(&self) -> &'static str {
        match self {
            Self::BodyTooLarge { .. } => "413 Payload Too Large",
            _ => "400 Bad Request",
        }
    }
}

#[derive(Debug, Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    top_k: Option<u32>,
    repetition_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
struct GenerateResponse {
    response: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Serialize)]
struct ModelInfoResponse {
    status: &'static str,
    version: &'static str,
    n_vocab: u32,
    n_ctx_train: u32,
    n_embd: u32,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Some(Commands::Generate(cmd)) => run_generate_command(cmd),
        Some(Commands::Image(cmd)) => run_image_command(cmd),
        Some(Commands::Serve(cmd)) => run_serve_command(cmd),
        Some(Commands::Agent(cmd)) => run_agent_command(cmd),
        Some(Commands::Plugin(cmd)) => run_plugin_command(cmd),
        None => run_legacy_mode(cli),
    }
}

fn run_legacy_mode(cli: Cli) -> anyhow::Result<()> {
    let model = cli.model.ok_or_else(|| {
        anyhow::anyhow!("missing required argument --model in legacy mode (or use subcommands)")
    })?;

    let engine_args = EngineArgs {
        backend: cli.backend.unwrap_or_else(|| "llama.cpp".to_string()),
        backend_lib: cli.backend_lib,
        backend_register_name: cli.backend_register_name,
        context_size: cli.context_size.unwrap_or(4096),
        max_prompt_bytes: cli.max_prompt_bytes,
        threads: cli.threads,
        cpu_only: cli.cpu_only,
        gpu_layers: cli.gpu_layers.unwrap_or(-1),
        lora_paths: cli.lora_paths,
    };
    let sampling = SamplingArgs {
        max_tokens: cli.max_tokens.unwrap_or(512),
        temperature: cli.temperature.unwrap_or(0.8),
        top_p: cli.top_p.unwrap_or(0.95),
        min_p: cli.min_p.unwrap_or(0.0),
        top_k: cli.top_k.unwrap_or(40),
        repetition_penalty: cli.repetition_penalty.unwrap_or(1.1),
    };
    let plugins = load_plugins(&cli.plugins)?;

    let mut engine = build_engine(&model, &engine_args)?;
    let gen_params = to_generation_params(&sampling);
    if let Some(prompt) = cli.prompt {
        run_single_prompt(
            &mut engine,
            &prompt,
            gen_params,
            cli.stream,
            plugins.as_ref(),
        )?;
    } else {
        run_interactive(&mut engine, gen_params, cli.stream, plugins.as_ref())?;
    }
    Ok(())
}

fn run_generate_command(cmd: GenerateCmd) -> anyhow::Result<()> {
    let plugins = load_plugins(&cmd.plugins)?;
    let mut engine = build_engine(&cmd.model, &cmd.engine)?;
    let gen_params = to_generation_params(&cmd.sampling);
    if let Some(prompt) = cmd.prompt {
        run_single_prompt(
            &mut engine,
            &prompt,
            gen_params,
            cmd.stream,
            plugins.as_ref(),
        )?;
    } else {
        run_interactive(&mut engine, gen_params, cmd.stream, plugins.as_ref())?;
    }
    Ok(())
}

fn run_image_command(cmd: ImageCmd) -> anyhow::Result<()> {
    if let Some(parent) = cmd.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    println!("Text-to-image mode");
    println!("Prompt: {}", cmd.prompt);
    println!("Model: {}", cmd.model_id);
    println!("Output: {}", cmd.output.display());

    let device = if cmd.use_cuda { "cuda" } else { "cpu" }.to_string();

    if let Some(kernel_plugin) = &cmd.kernel_plugin {
        println!("Kernel plugin: {}", kernel_plugin.display());
        let kernel = load_dynamic_image_plugin(kernel_plugin).map_err(|e| anyhow::anyhow!(e))?;
        let request = ImageGenerationRequest {
            prompt: cmd.prompt.clone(),
            model_id: cmd.model_id.clone(),
            steps: cmd.steps,
            guidance_scale: cmd.guidance_scale,
            width: cmd.width,
            height: cmd.height,
            seed: cmd.seed,
            device,
        };
        let result = kernel
            .plugin()
            .generate(&request)
            .map_err(|e| anyhow::anyhow!(e))?;
        if result.image_bytes.is_empty() {
            return Err(anyhow::anyhow!(
                "image kernel returned empty image bytes (format={})",
                result.format
            ));
        }
        fs::write(&cmd.output, &result.image_bytes)?;
        println!(
            "Image generation completed via plugin kernel (format={}): {}",
            result.format,
            cmd.output.display()
        );
        return Ok(());
    }

    let script_path = PathBuf::from("scripts").join("t2i_generate.py");
    if !script_path.exists() {
        return Err(anyhow::anyhow!(
            "missing image generation script: {}",
            script_path.display()
        ));
    }

    let mut child = Command::new(&cmd.python);
    child
        .arg(&script_path)
        .arg("--prompt")
        .arg(&cmd.prompt)
        .arg("--model-id")
        .arg(&cmd.model_id)
        .arg("--output")
        .arg(&cmd.output)
        .arg("--steps")
        .arg(cmd.steps.to_string())
        .arg("--guidance-scale")
        .arg(cmd.guidance_scale.to_string());

    if let Some(width) = cmd.width {
        child.arg("--width").arg(width.to_string());
    }
    if let Some(height) = cmd.height {
        child.arg("--height").arg(height.to_string());
    }
    if let Some(seed) = cmd.seed {
        child.arg("--seed").arg(seed.to_string());
    }
    child.arg("--device").arg(device);

    let status = child.status()?;
    if !status.success() {
        return Err(anyhow::anyhow!(
            "text-to-image script failed with status: {status}"
        ));
    }

    println!("Image generation completed: {}", cmd.output.display());
    Ok(())
}

fn run_agent_command(cmd: AgentCmd) -> anyhow::Result<()> {
    println!("Agent mode");
    println!("Tool: {}", cmd.tool);
    let plugins = load_plugins(&cmd.plugins)?;
    let mut engine = build_engine(&cmd.model, &cmd.engine)?;
    let mut prompt = String::new();
    if cmd.tool.eq_ignore_ascii_case("web_search") {
        prompt.push_str(
            "Tool `web_search` requested. Since offline tool execution is not wired here, infer based on model knowledge.\n\n",
        );
    } else if !cmd.tool.eq_ignore_ascii_case("none") {
        prompt.push_str(&format!(
            "Tool `{}` requested. Execute in reasoning space and answer directly.\n\n",
            cmd.tool
        ));
    }
    prompt.push_str(&cmd.prompt);
    run_single_prompt(
        &mut engine,
        &prompt,
        to_generation_params(&cmd.sampling),
        cmd.stream,
        plugins.as_ref(),
    )?;
    Ok(())
}

fn run_plugin_command(cmd: PluginCmd) -> anyhow::Result<()> {
    let mut registry = load_registry(&cmd.registry)?;
    match cmd.command {
        PluginAction::Load { path } => {
            if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("wasm"))
                .unwrap_or(false)
            {
                registry.load_wasm_plugin(&path)?;
            } else {
                registry.load_dynamic_plugin(&path)?;
            }
            registry.save_to_file(&cmd.registry)?;
            println!("Loaded plugin: {}", path.display());
        }
        PluginAction::List => {
            let plugins = registry.list_detailed();
            if plugins.is_empty() {
                println!("No plugins loaded.");
            } else {
                println!("name\tversion\tenabled\ttype\thot_reloadable\tsource");
                for p in plugins {
                    let source = p.source.unwrap_or_else(|| "-".to_string());
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}",
                        p.name, p.version, p.enabled, p.plugin_type, p.hot_reloadable, source
                    );
                }
            }
        }
        PluginAction::Info { name } => {
            if let Some(p) = registry.get_info(&name) {
                println!("name: {}", p.name);
                println!("version: {}", p.version);
                println!("enabled: {}", p.enabled);
                println!("type: {}", p.plugin_type);
                println!("hot_reloadable: {}", p.hot_reloadable);
                println!("source: {}", p.source.unwrap_or_else(|| "-".to_string()));
            } else {
                return Err(anyhow::anyhow!("Plugin not found: {}", name));
            }
        }
        PluginAction::Unload { name } => {
            registry.unload(&name)?;
            registry.save_to_file(&cmd.registry)?;
            println!("Unloaded plugin: {name}");
        }
        PluginAction::Reload { name } => {
            registry.reload(&name)?;
            registry.save_to_file(&cmd.registry)?;
            println!("Reloaded plugin: {name}");
        }
        PluginAction::Enable { name } => {
            registry.enable(&name)?;
            registry.save_to_file(&cmd.registry)?;
            println!("Enabled plugin: {name}");
        }
        PluginAction::Disable { name } => {
            registry.disable(&name)?;
            registry.save_to_file(&cmd.registry)?;
            println!("Disabled plugin: {name}");
        }
    }
    Ok(())
}

fn run_serve_command(cmd: ServeCmd) -> anyhow::Result<()> {
    if !cmd.api_type.eq_ignore_ascii_case("rest") {
        println!(
            "Unsupported --api-type `{}`; falling back to REST.",
            cmd.api_type
        );
    }

    let plugins = load_plugins(&cmd.plugins)?;
    let mut engine = build_engine(&cmd.model, &cmd.engine)?;
    let default_sampling = cmd.sampling.clone();

    let addr = format!("{}:{}", cmd.host, cmd.port);
    let listener = TcpListener::bind(&addr)?;
    println!("Loci REST server listening on http://{addr}");
    println!(
        "Endpoints: GET /health, GET /v1/health, GET /info, GET /v1/info, POST /generate, POST /v1/generate"
    );

    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                if let Err(err) = handle_connection(
                    &mut stream,
                    &mut engine,
                    &default_sampling,
                    plugins.as_ref(),
                ) {
                    let _ = write_json_response(
                        &mut stream,
                        "500 Internal Server Error",
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    );
                }
            }
            Err(err) => {
                eprintln!("accept error: {err}");
            }
        }
    }

    Ok(())
}

fn build_engine(model: &Path, engine_args: &EngineArgs) -> anyhow::Result<InferenceEngine> {
    println!("Loading model from: {}", model.display());
    println!("Backend: {}", engine_args.backend);
    println!("Context size: {}", engine_args.context_size);
    println!("GPU layers: {}", engine_args.gpu_layers);
    if let Some(limit) = engine_args.max_prompt_bytes {
        std::env::set_var("LOCI_MAX_PROMPT_BYTES", limit.to_string());
        println!("Max prompt bytes: {} (from --max-prompt-bytes)", limit);
    } else if let Ok(raw) = std::env::var("LOCI_MAX_PROMPT_BYTES") {
        match raw.parse::<usize>() {
            Ok(limit) if limit >= MIN_PROMPT_BYTES_LIMIT => {
                println!("Max prompt bytes: {} (from LOCI_MAX_PROMPT_BYTES)", limit);
            }
            Ok(limit) => {
                println!(
                    "Warning: LOCI_MAX_PROMPT_BYTES={} is below minimum {}, backend will use default",
                    limit, MIN_PROMPT_BYTES_LIMIT
                );
            }
            Err(_) => {
                println!(
                    "Warning: LOCI_MAX_PROMPT_BYTES={} is invalid, backend will use default",
                    raw
                );
            }
        }
    }
    if !engine_args.lora_paths.is_empty() {
        println!("LoRA adapters requested:");
        for lora in &engine_args.lora_paths {
            println!("  - {}", lora.display());
            if !lora.exists() {
                return Err(anyhow::anyhow!(
                    "LoRA path does not exist: {}",
                    lora.display()
                ));
            }
        }
    }

    let mut selected_backend = engine_args.backend.clone();

    let mut builder = InferenceEngine::builder()
        .model_path(model)
        .backend(&selected_backend)
        .context_size(engine_args.context_size)
        .batch_size(512)
        .gpu_layers(engine_args.gpu_layers);

    if let Some(backend_lib) = &engine_args.backend_lib {
        if !backend_lib.exists() {
            return Err(anyhow::anyhow!(
                "Dynamic backend library does not exist: {}",
                backend_lib.display()
            ));
        }

        let registration_name = engine_args
            .backend_register_name
            .clone()
            .or_else(|| {
                if engine_args.backend.trim().is_empty()
                    || engine_args.backend.eq_ignore_ascii_case("llama.cpp")
                {
                    None
                } else {
                    Some(engine_args.backend.clone())
                }
            })
            .unwrap_or_else(|| "dynamic.plugin".to_string());

        println!(
            "Registering dynamic backend `{}` from {}",
            registration_name,
            backend_lib.display()
        );
        builder = builder.load_dynamic_backend(registration_name.clone(), backend_lib.clone());

        if engine_args.backend.trim().is_empty()
            || engine_args.backend.eq_ignore_ascii_case("llama.cpp")
        {
            selected_backend = registration_name;
            builder = builder.backend(&selected_backend);
        }
    }

    if engine_args.cpu_only {
        builder = builder.cpu_only();
    }
    if let Some(threads) = engine_args.threads {
        builder = builder.threads(threads);
    }

    let engine = builder.build()?;
    let info = engine.model_info();
    println!("Model loaded successfully!");
    println!("  Vocabulary size: {}", info.n_vocab);
    println!("  Training context: {}", info.n_ctx_train);
    println!("  Embedding dimension: {}", info.n_embd);
    if !engine_args.lora_paths.is_empty() {
        println!(
            "Warning: --lora-path is accepted, but runtime LoRA merge is backend-dependent and not enabled in this CLI path yet."
        );
    }
    println!();

    Ok(engine)
}

fn to_generation_params(sampling: &SamplingArgs) -> GenerationParams {
    GenerationParams {
        max_tokens: sampling.max_tokens,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        min_p: sampling.min_p,
        top_k: sampling.top_k,
        repeat_penalty: sampling.repetition_penalty,
        ..Default::default()
    }
}

fn load_registry(path: &Path) -> anyhow::Result<PluginRegistry> {
    let mut registry = PluginRegistry::with_config_path(path);
    if path.exists() {
        registry.load_from_file(path)?;
    }
    Ok(registry)
}

fn run_single_prompt(
    engine: &mut InferenceEngine,
    prompt: &str,
    params: GenerationParams,
    stream: bool,
    plugins: Option<&PluginRegistry>,
) -> anyhow::Result<()> {
    let prompt = apply_pre_generate(prompt, plugins)?;
    println!("Prompt: {}", prompt);
    println!("\nResponse:");
    println!("---");

    if stream {
        engine.generate_stream(&prompt, params, |token| {
            let rendered = match apply_on_token(token, plugins) {
                Ok(t) => t,
                Err(err) => {
                    eprintln!("plugin on_token failed: {err}");
                    return false;
                }
            };
            print!("{rendered}");
            io::stdout().flush().ok();
            true
        })?;
        println!();
    } else {
        let response = engine.generate(&prompt, params)?;
        let response = apply_post_generate(&response, plugins)?;
        println!("{response}");
    }

    println!("---");
    Ok(())
}

fn run_interactive(
    engine: &mut InferenceEngine,
    params: GenerationParams,
    stream: bool,
    plugins: Option<&PluginRegistry>,
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
        let processed_input = apply_pre_generate(input, plugins)?;

        if stream {
            engine.generate_stream(&processed_input, params.clone(), |token| {
                let rendered = match apply_on_token(token, plugins) {
                    Ok(t) => t,
                    Err(err) => {
                        eprintln!("plugin on_token failed: {err}");
                        return false;
                    }
                };
                print!("{rendered}");
                io::stdout().flush().ok();
                true
            })?;
            println!();
        } else {
            let response = engine.generate(&processed_input, params.clone())?;
            let response = apply_post_generate(&response, plugins)?;
            println!("{response}");
        }

        println!("---");
        println!();
    }

    Ok(())
}

fn handle_connection(
    stream: &mut TcpStream,
    engine: &mut InferenceEngine,
    default_sampling: &SamplingArgs,
    plugins: Option<&PluginRegistry>,
) -> anyhow::Result<()> {
    let request = match read_http_request(stream) {
        Ok(request) => request,
        Err(parse_error) => {
            write_json_response(
                stream,
                parse_error.status_code(),
                &ErrorResponse {
                    error: parse_error.to_string(),
                },
            )?;
            return Ok(());
        }
    };

    if request.method == "GET" && (request.path == "/health" || request.path == "/v1/health") {
        write_plain_response(stream, "200 OK", "application/json", r#"{"status":"ok"}"#)?;
        return Ok(());
    }

    if request.method == "GET" && (request.path == "/info" || request.path == "/v1/info") {
        let info = engine.model_info();
        write_json_response(
            stream,
            "200 OK",
            &ModelInfoResponse {
                status: "ok",
                version: env!("CARGO_PKG_VERSION"),
                n_vocab: info.n_vocab,
                n_ctx_train: info.n_ctx_train,
                n_embd: info.n_embd,
            },
        )?;
        return Ok(());
    }

    if request.method == "POST" && (request.path == "/generate" || request.path == "/v1/generate") {
        let payload: GenerateRequest = match serde_json::from_slice(&request.body) {
            Ok(payload) => payload,
            Err(err) => {
                write_json_response(
                    stream,
                    "400 Bad Request",
                    &ErrorResponse {
                        error: format!("invalid JSON payload for /generate: {err}"),
                    },
                )?;
                return Ok(());
            }
        };

        let params = GenerationParams {
            max_tokens: payload.max_tokens.unwrap_or(default_sampling.max_tokens),
            temperature: payload.temperature.unwrap_or(default_sampling.temperature),
            top_p: payload.top_p.unwrap_or(default_sampling.top_p),
            min_p: payload.min_p.unwrap_or(default_sampling.min_p),
            top_k: payload.top_k.unwrap_or(default_sampling.top_k),
            repeat_penalty: payload
                .repetition_penalty
                .unwrap_or(default_sampling.repetition_penalty),
            ..Default::default()
        };
        let prompt = apply_pre_generate(&payload.prompt, plugins)?;

        match engine.generate(&prompt, params) {
            Ok(response) => {
                let response = apply_post_generate(&response, plugins)?;
                write_json_response(stream, "200 OK", &GenerateResponse { response })?;
            }
            Err(err) => {
                write_json_response(
                    stream,
                    "500 Internal Server Error",
                    &ErrorResponse {
                        error: err.to_string(),
                    },
                )?;
            }
        }
        return Ok(());
    }

    write_json_response(
        stream,
        "404 Not Found",
        &ErrorResponse {
            error: "endpoint not found".to_string(),
        },
    )?;
    Ok(())
}

fn read_http_request(
    stream: &mut TcpStream,
) -> std::result::Result<HttpRequest, HttpRequestParseError> {
    let mut reader = BufReader::new(stream.try_clone().map_err(HttpRequestParseError::Io)?);
    parse_http_request_from_reader(&mut reader)
}

fn parse_http_request_from_reader<R: BufRead + Read>(
    reader: &mut R,
) -> std::result::Result<HttpRequest, HttpRequestParseError> {
    let mut first_line = String::new();
    let first_line_bytes = reader
        .read_line(&mut first_line)
        .map_err(HttpRequestParseError::Io)?;
    if first_line_bytes > MAX_HTTP_LINE_BYTES {
        return Err(HttpRequestParseError::HeaderLineTooLong(first_line_bytes));
    }
    if first_line.trim().is_empty() {
        return Err(HttpRequestParseError::EmptyRequest);
    }

    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    if method.is_empty() || path.is_empty() {
        return Err(HttpRequestParseError::InvalidRequestLine);
    }

    let mut total_header_bytes = first_line_bytes;
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        let line_bytes = reader
            .read_line(&mut line)
            .map_err(HttpRequestParseError::Io)?;
        if line_bytes > MAX_HTTP_LINE_BYTES {
            return Err(HttpRequestParseError::HeaderLineTooLong(line_bytes));
        }
        total_header_bytes += line_bytes;
        if total_header_bytes > MAX_HTTP_HEADER_BYTES {
            return Err(HttpRequestParseError::HeadersTooLarge(total_header_bytes));
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse::<usize>().map_err(|_| {
                    HttpRequestParseError::InvalidContentLength(value.trim().to_string())
                })?;
            }
        }
    }
    if content_length > MAX_HTTP_BODY_BYTES {
        return Err(HttpRequestParseError::BodyTooLarge {
            content_length,
            limit: MAX_HTTP_BODY_BYTES,
        });
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(HttpRequestParseError::Io)?;
    }

    Ok(HttpRequest { method, path, body })
}

fn write_json_response<T: Serialize>(
    stream: &mut TcpStream,
    status: &str,
    payload: &T,
) -> anyhow::Result<()> {
    let body = serde_json::to_string(payload)?;
    write_plain_response(stream, status, "application/json", &body)
}

fn write_plain_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn load_plugins(paths: &[PathBuf]) -> anyhow::Result<Option<PluginRegistry>> {
    if paths.is_empty() {
        return Ok(None);
    }

    let mut registry = PluginRegistry::new();
    for path in paths {
        load_plugin_file(&mut registry, path)?;
        println!("Loaded runtime plugin: {}", path.display());
    }
    Ok(Some(registry))
}

fn load_plugin_file(registry: &mut PluginRegistry, path: &Path) -> anyhow::Result<()> {
    if path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("wasm"))
        .unwrap_or(false)
    {
        registry.load_wasm_plugin(path)?;
    } else {
        registry.load_dynamic_plugin(path)?;
    }
    Ok(())
}

fn apply_pre_generate(prompt: &str, plugins: Option<&PluginRegistry>) -> anyhow::Result<String> {
    match plugins {
        Some(registry) => Ok(registry.apply_pre_generate(prompt)?),
        None => Ok(prompt.to_string()),
    }
}

fn apply_post_generate(response: &str, plugins: Option<&PluginRegistry>) -> anyhow::Result<String> {
    match plugins {
        Some(registry) => Ok(registry.apply_post_generate(response)?),
        None => Ok(response.to_string()),
    }
}

fn apply_on_token(token: &str, plugins: Option<&PluginRegistry>) -> anyhow::Result<String> {
    match plugins {
        Some(registry) => Ok(registry.apply_on_token(token)?),
        None => Ok(token.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_generate_context_length_alias() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--context-length",
            "2048",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => assert_eq!(cmd.engine.context_size, 2048),
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_repetition_penalty() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--repetition-penalty",
            "1.2",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert!((cmd.sampling.repetition_penalty - 1.2).abs() < 1e-6);
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_multiple_plugins() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--plugin",
            "a.dll",
            "--plugin",
            "b.wasm",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => assert_eq!(cmd.plugins.len(), 2),
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_image_command_defaults() {
        let cli = Cli::try_parse_from(["loci", "image", "--prompt", "a tiny robot reading a book"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Image(cmd)) => {
                assert_eq!(
                    cmd.model_id,
                    "hf-internal-testing/tiny-stable-diffusion-pipe"
                );
                assert_eq!(cmd.steps, 4);
                assert_eq!(cmd.guidance_scale, 0.0);
                assert!(cmd.kernel_plugin.is_none());
            }
            _ => panic!("expected image command"),
        }
    }

    #[test]
    fn parse_generate_dynamic_backend_args() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--backend-lib",
            "plugin_backend.dll",
            "--backend-register-name",
            "my.plugin.backend",
            "--backend",
            "my.plugin.backend",
            "--prompt",
            "hello",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert_eq!(cmd.engine.backend, "my.plugin.backend");
                assert_eq!(
                    cmd.engine
                        .backend_lib
                        .as_deref()
                        .map(|p| p.to_string_lossy().to_string()),
                    Some("plugin_backend.dll".to_string())
                );
                assert_eq!(
                    cmd.engine.backend_register_name.as_deref(),
                    Some("my.plugin.backend")
                );
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_plugin_reload_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "plugin",
            "--registry",
            "loci_plugins.toml",
            "reload",
            "demo_plugin",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Plugin(cmd)) => match cmd.command {
                PluginAction::Reload { name } => assert_eq!(name, "demo_plugin"),
                _ => panic!("expected reload action"),
            },
            _ => panic!("expected plugin command"),
        }
    }

    #[test]
    fn parse_plugin_info_command() {
        let cli = Cli::try_parse_from(["loci", "plugin", "info", "demo_plugin"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Plugin(cmd)) => match cmd.command {
                PluginAction::Info { name } => assert_eq!(name, "demo_plugin"),
                _ => panic!("expected info action"),
            },
            _ => panic!("expected plugin command"),
        }
    }

    #[test]
    fn parse_generate_max_prompt_bytes() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--max-prompt-bytes",
            "65536",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert_eq!(cmd.engine.max_prompt_bytes, Some(65_536));
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_max_prompt_bytes_too_small() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--max-prompt-bytes",
            "512",
        ]);
        assert!(cli.is_err(), "value below minimum should be rejected");
    }

    #[test]
    fn parse_http_request_from_reader_success() {
        let raw = b"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: 18\r\n\r\n{\"prompt\":\"hello\"}";
        let mut reader = BufReader::new(Cursor::new(raw.as_slice()));
        let request = parse_http_request_from_reader(&mut reader).expect("parse should succeed");

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/generate");
        assert_eq!(request.body, b"{\"prompt\":\"hello\"}");
    }

    #[test]
    fn parse_http_request_body_too_large() {
        let raw = format!(
            "POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            MAX_HTTP_BODY_BYTES + 1
        );
        let mut reader = BufReader::new(Cursor::new(raw.into_bytes()));
        let err = parse_http_request_from_reader(&mut reader).expect_err("should reject");

        assert!(matches!(err, HttpRequestParseError::BodyTooLarge { .. }));
        assert_eq!(err.status_code(), "413 Payload Too Large");
    }

    #[test]
    fn parse_http_request_invalid_request_line() {
        let raw = b"INVALID_ONLY\r\nHost: localhost\r\n\r\n";
        let mut reader = BufReader::new(Cursor::new(raw.as_slice()));
        let err = parse_http_request_from_reader(&mut reader).expect_err("should reject");

        assert!(matches!(err, HttpRequestParseError::InvalidRequestLine));
        assert_eq!(err.status_code(), "400 Bad Request");
    }

    #[test]
    fn parse_http_request_invalid_content_length() {
        let raw = b"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Length: abc\r\n\r\n";
        let mut reader = BufReader::new(Cursor::new(raw.as_slice()));
        let err = parse_http_request_from_reader(&mut reader).expect_err("should reject");

        assert!(matches!(
            err,
            HttpRequestParseError::InvalidContentLength(_)
        ));
        assert_eq!(err.status_code(), "400 Bad Request");
    }
}

use clap::{ArgAction, Args as ClapArgs, Parser, Subcommand, ValueEnum};
use crossbeam::channel::{self, Sender, TrySendError};
use loci::execution_policy_plugin::{ExecutionPolicyDescriptor, ExecutionPolicyRegistry};
use loci::http_compat::{
    chunk_text_for_streaming, compatibility_created_at, estimate_token_count,
    normalize_openai_embedding_input, ollama_stream_event, openai_chat_messages_to_prompt,
    openai_chat_stream_chunk, OllamaGenerateRequest, OllamaGenerateResponse, OllamaModelTag,
    OllamaTagsResponse, OpenAiChatChoice, OpenAiChatCompletionsRequest,
    OpenAiChatCompletionsResponse, OpenAiChatMessage, OpenAiEmbeddingData, OpenAiEmbeddingsRequest,
    OpenAiEmbeddingsResponse, OpenAiModelDescriptor, OpenAiModelListResponse, OpenAiUsage,
};
use loci::image_kernel::{load_dynamic_image_plugin, ImageGenerationRequest};
use loci::inference::GenerationParams;
use loci::management_auth::{
    ManagementAuthContext, ManagementAuthDecision, ManagementAuthPolicyPlugin,
    ManagementAuthPolicyRegistry,
};
use loci::model_pull_jobs::{
    ModelPullJobEvent, ModelPullJobManager, ModelPullJobRequest, ModelPullJobSnapshot,
};
use loci::model_pull_policy::{
    authorize_model_pull_request, ModelPullPolicyContext, ModelPullPolicyDescriptor,
    ModelPullPolicyPlugin, ModelPullPolicyRegistry,
};
use loci::model_pull_verifier::{
    ModelPullVerifierDescriptor, ModelPullVerifierPlugin, ModelPullVerifierRegistry,
};
use loci::model_store::{ModelPullOptions, ModelPullProgress, ModelStore, StoredModel};
use loci::plugin_registry::PluginRegistry;
use loci::policy_registry::DynamicPolicyRegistry;
use loci::prelude::*;
use loci::resource_planner::{ModelResourceEstimate, ResourcePlan, ResourcePlanner};
use loci::runtime_events::{
    RuntimeEvent, RuntimeEventBus, RuntimeEventCategory, RuntimeEventOutcome,
};
use loci::serve_dispatch::{
    QueueFullAction, QueuePressureContext, ServeDispatchPolicyDescriptor,
    ServeDispatchPolicyPlugin, ServeDispatchPolicyRegistry,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_HTTP_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_HTTP_HEADER_BYTES: usize = 64 * 1024;
const MAX_HTTP_LINE_BYTES: usize = 8 * 1024;
const MIN_PROMPT_BYTES_LIMIT: usize = 1024;
const OPENAPI_SPEC_YAML_PATH: &str = "/openapi.yaml";
const OPENAPI_SPEC_JSON_PATH: &str = "/openapi.json";
const OPENAPI_SPEC_YAML: &str = include_str!("../docs/openapi/loci-rest-v1.yaml");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenApiSpecFormat {
    Yaml,
    Json,
}

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

fn parse_positive_u32(raw: &str) -> std::result::Result<u32, String> {
    let value = raw
        .parse::<u32>()
        .map_err(|_| format!("invalid integer: `{raw}`"))?;
    if value == 0 {
        return Err("value must be greater than 0".to_string());
    }
    Ok(value)
}

fn parse_tensor_split(raw: &str) -> std::result::Result<Vec<f32>, String> {
    let values = raw
        .split(',')
        .map(str::trim)
        .map(|part| {
            if part.is_empty() {
                return Err("tensor split contains an empty segment".to_string());
            }
            let value = part
                .parse::<f32>()
                .map_err(|_| format!("invalid tensor split value: `{part}`"))?;
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "tensor split value must be finite and non-negative: `{part}`"
                ));
            }
            Ok(value)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if values.is_empty() {
        return Err("tensor split must contain at least one value".to_string());
    }
    if !values.iter().any(|value| *value > 0.0) {
        return Err("tensor split must contain at least one positive value".to_string());
    }

    Ok(values)
}

fn format_tensor_split(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Debug, Clone, PartialEq)]
struct TensorSplitArg(Vec<f32>);

impl TensorSplitArg {
    fn to_vec(&self) -> Vec<f32> {
        self.0.clone()
    }
}

impl std::str::FromStr for TensorSplitArg {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        parse_tensor_split(raw).map(Self)
    }
}

fn resolve_toggle(enabled: bool, disabled: bool, default: bool) -> bool {
    if disabled {
        false
    } else if enabled {
        true
    } else {
        default
    }
}

fn split_shell_words(input: &str) -> std::result::Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut active_quote: Option<char> = None;

    while let Some(ch) = chars.next() {
        match active_quote {
            Some(quote) => {
                if ch == quote {
                    active_quote = None;
                } else if ch == '\\' && quote == '"' {
                    if let Some(next) = chars.next() {
                        if next == '"' || next == '\\' {
                            current.push(next);
                        } else {
                            current.push('\\');
                            current.push(next);
                        }
                    } else {
                        current.push('\\');
                    }
                } else {
                    current.push(ch);
                }
            }
            None => match ch {
                '"' | '\'' => active_quote = Some(ch),
                c if c.is_whitespace() => {
                    if !current.is_empty() {
                        out.push(std::mem::take(&mut current));
                    }
                }
                _ => current.push(ch),
            },
        }
    }

    if active_quote.is_some() {
        return Err("unclosed quote in command".to_string());
    }
    if !current.is_empty() {
        out.push(current);
    }
    Ok(out)
}

fn parse_mcp_stdio_spec(raw: &str) -> std::result::Result<McpStdioServerConfig, String> {
    let Some((name, commandline)) = raw.split_once('=') else {
        return Err("mcp spec must be NAME=COMMAND [ARGS...]".to_string());
    };
    let name = name.trim();
    if name.is_empty() {
        return Err("mcp spec has empty server name".to_string());
    }
    let argv = split_shell_words(commandline.trim())?;
    if argv.is_empty() {
        return Err("mcp spec must include executable command".to_string());
    }

    let mut config = McpStdioServerConfig::new(name.to_string(), argv[0].clone());
    config.args = argv.into_iter().skip(1).collect();
    Ok(config)
}

fn merge_tool_allowlists(
    base: Option<Vec<String>>,
    extra: Option<Vec<String>>,
) -> Option<Vec<String>> {
    match (base, extra) {
        (None, None) => None,
        (Some(mut one), None) | (None, Some(mut one)) => {
            one.sort();
            one.dedup();
            Some(one)
        }
        (Some(left), Some(right)) => {
            let right_set = right.into_iter().collect::<HashSet<_>>();
            let mut out = left
                .into_iter()
                .filter(|name| right_set.contains(name))
                .collect::<Vec<_>>();
            out.sort();
            out.dedup();
            Some(out)
        }
    }
}

#[derive(Debug, Deserialize)]
struct ConfigCliSpecObject {
    #[serde(default)]
    args: Option<Vec<String>>,
    #[serde(default)]
    commandline: Option<String>,
    #[serde(default, alias = "command_line")]
    command_line: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ConfigCliSpec {
    Args(Vec<String>),
    Commandline(String),
    Object(ConfigCliSpecObject),
}

fn spec_to_cli_args(spec: ConfigCliSpec) -> anyhow::Result<Vec<String>> {
    let mut args = match spec {
        ConfigCliSpec::Args(args) => args,
        ConfigCliSpec::Commandline(line) => split_shell_words(line.trim())
            .map_err(|e| anyhow::anyhow!("invalid commandline in config: {e}"))?,
        ConfigCliSpec::Object(obj) => {
            let line = obj.commandline.or(obj.command_line);
            match (obj.args, line) {
                (Some(args), None) => args,
                (None, Some(line)) => split_shell_words(line.trim())
                    .map_err(|e| anyhow::anyhow!("invalid commandline in config: {e}"))?,
                (Some(_), Some(_)) => {
                    return Err(anyhow::anyhow!(
                        "config must provide either `args` or `commandline`, not both"
                    ));
                }
                (None, None) => {
                    return Err(anyhow::anyhow!(
                        "config must provide `args` array or `commandline` string"
                    ));
                }
            }
        }
    };

    if args.first().map(|s| s == "loci").unwrap_or(false) {
        args.remove(0);
    }
    if args.is_empty() {
        return Err(anyhow::anyhow!("config command is empty"));
    }
    Ok(args)
}

fn read_cli_args_from_config(path: &Path) -> anyhow::Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let spec = match ext.as_str() {
        "json" => serde_json::from_str::<ConfigCliSpec>(&content).map_err(|e| {
            anyhow::anyhow!("failed to parse JSON config '{}': {}", path.display(), e)
        })?,
        "toml" => toml::from_str::<ConfigCliSpec>(&content).map_err(|e| {
            anyhow::anyhow!("failed to parse TOML config '{}': {}", path.display(), e)
        })?,
        _ => ConfigCliSpec::Commandline(content),
    };

    spec_to_cli_args(spec)
}

#[derive(Parser, Debug)]
#[command(name = "loci")]
#[command(about = "A cross-platform local LLM inference tool", long_about = None)]
struct Cli {
    /// Load command from configuration file (.json/.toml or plain commandline text)
    #[arg(long = "config")]
    config: Option<PathBuf>,

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
    /// Multi-GPU split strategy: none, layer, or row
    #[arg(long = "gpu-split-mode", value_enum)]
    gpu_split_mode: Option<GpuSplitModeArg>,
    /// Primary GPU index used when --gpu-split-mode=none
    #[arg(long = "main-gpu")]
    main_gpu: Option<u32>,
    /// Relative split weights per GPU, comma-separated, e.g. 3,2,1
    #[arg(long = "tensor-split")]
    tensor_split: Option<TensorSplitArg>,
    /// Retry model loading with fewer GPU layers when the requested placement does not fit.
    #[arg(long = "auto-gpu-fallback")]
    auto_gpu_fallback: bool,
    /// Layer decrement used by --auto-gpu-fallback for each retry.
    #[arg(long = "gpu-fallback-step", value_parser = parse_positive_u32)]
    gpu_fallback_step: Option<u32>,
    /// Derive GPU/CPU placement automatically from model size and detected hardware.
    #[arg(long = "auto-resource-plan")]
    auto_resource_plan: bool,
    /// Enable memory-mapped model loading (disk-backed paging)
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_mmap")]
    mmap: bool,
    /// Disable memory-mapped model loading
    #[arg(long = "no-mmap", action = ArgAction::SetTrue, overrides_with = "mmap")]
    no_mmap: bool,
    /// Lock model pages into RAM when supported by the OS
    #[arg(long)]
    mlock: bool,
    /// Enable K/Q/V and KV cache offload to device
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_kv_offload")]
    kv_offload: bool,
    /// Disable K/Q/V and KV cache offload to device
    #[arg(
        long = "no-kv-offload",
        action = ArgAction::SetTrue,
        overrides_with = "kv_offload"
    )]
    no_kv_offload: bool,
    /// Enable host op offload to device
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_op_offload")]
    op_offload: bool,
    /// Disable host op offload to device
    #[arg(
        long = "no-op-offload",
        action = ArgAction::SetTrue,
        overrides_with = "op_offload"
    )]
    no_op_offload: bool,
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
    /// Manage management-auth policy registry
    AuthPolicy(AuthPolicyCmd),
    /// Manage serve dispatch policy registry
    DispatchPolicy(DispatchPolicyCmd),
    /// Manage execution policy registry
    ExecutionPolicy(ExecutionPolicyCmd),
    /// Manage persisted/active inference sessions
    Session(SessionCmd),
    /// Generate image from a prompt (text-to-image)
    Image(ImageCmd),
    /// Run multimodal input/output pipeline with multimodal I/O plugin.
    Multimodal(MultimodalCmd),
    /// Orchestrate multiple models with routing or ensemble plus multimodal I/O.
    Orchestrate(OrchestrateCmd),
    /// Start REST server
    Serve(ServeCmd),
    /// Run agent mode with a selected tool
    Agent(AgentCmd),
    /// Manage plugins
    Plugin(PluginCmd),
    /// Manage MCP server registry and connectivity
    Mcp(McpCmd),
    /// Manage built-in model asset store
    Model(ModelCmd),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum GpuSplitModeArg {
    None,
    Layer,
    Row,
}

impl From<GpuSplitModeArg> for GpuSplitMode {
    fn from(value: GpuSplitModeArg) -> Self {
        match value {
            GpuSplitModeArg::None => GpuSplitMode::None,
            GpuSplitModeArg::Layer => GpuSplitMode::Layer,
            GpuSplitModeArg::Row => GpuSplitMode::Row,
        }
    }
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
    /// Multi-GPU split strategy: none, layer, or row
    #[arg(
        long = "gpu-split-mode",
        value_enum,
        default_value_t = GpuSplitModeArg::Layer
    )]
    gpu_split_mode: GpuSplitModeArg,
    /// Primary GPU index used when --gpu-split-mode=none
    #[arg(long = "main-gpu", default_value_t = 0)]
    main_gpu: u32,
    /// Relative split weights per GPU, comma-separated, e.g. 3,2,1
    #[arg(long = "tensor-split")]
    tensor_split: Option<TensorSplitArg>,
    /// Retry model loading with fewer GPU layers when the requested placement does not fit.
    #[arg(long = "auto-gpu-fallback")]
    auto_gpu_fallback: bool,
    /// Layer decrement used by --auto-gpu-fallback for each retry.
    #[arg(
        long = "gpu-fallback-step",
        default_value_t = 8,
        value_parser = parse_positive_u32
    )]
    gpu_fallback_step: u32,
    /// Derive GPU/CPU placement automatically from model size and detected hardware.
    #[arg(long = "auto-resource-plan")]
    auto_resource_plan: bool,
    /// Enable memory-mapped model loading (disk-backed paging)
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_mmap")]
    mmap: bool,
    /// Disable memory-mapped model loading
    #[arg(long = "no-mmap", action = ArgAction::SetTrue, overrides_with = "mmap")]
    no_mmap: bool,
    /// Lock model pages into RAM when supported by the OS
    #[arg(long)]
    mlock: bool,
    /// Enable K/Q/V and KV cache offload to device
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_kv_offload")]
    kv_offload: bool,
    /// Disable K/Q/V and KV cache offload to device
    #[arg(
        long = "no-kv-offload",
        action = ArgAction::SetTrue,
        overrides_with = "kv_offload"
    )]
    no_kv_offload: bool,
    /// Enable host op offload to device
    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_op_offload")]
    op_offload: bool,
    /// Disable host op offload to device
    #[arg(
        long = "no-op-offload",
        action = ArgAction::SetTrue,
        overrides_with = "op_offload"
    )]
    no_op_offload: bool,
    /// LoRA adapter path(s). Accepted for compatibility; backend merge support is build-dependent.
    #[arg(long = "lora-path")]
    lora_paths: Vec<PathBuf>,
    /// Dynamic execution policy plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "execution-policy-plugin")]
    execution_policy_plugins: Vec<PathBuf>,
    /// Execution policy registry file
    #[arg(
        long = "execution-policy-registry",
        default_value = "loci_execution_policies.toml"
    )]
    execution_policy_registry: PathBuf,
    /// Execution policy name from builtin/dynamic registry
    #[arg(long = "execution-policy-name")]
    execution_policy_name: Option<String>,
}

impl EngineArgs {
    fn resolved_gpu_split_mode(&self) -> GpuSplitMode {
        self.gpu_split_mode.into()
    }

    fn resolved_tensor_split(&self) -> Option<Vec<f32>> {
        self.tensor_split.as_ref().map(TensorSplitArg::to_vec)
    }

    fn resolved_gpu_fallback_step(&self) -> Option<u32> {
        if self.auto_gpu_fallback {
            Some(self.gpu_fallback_step)
        } else {
            None
        }
    }

    fn resolved_auto_resource_plan(&self) -> bool {
        self.auto_resource_plan && !self.cpu_only
    }

    fn resolved_mmap(&self) -> bool {
        resolve_toggle(self.mmap, self.no_mmap, true)
    }

    fn resolved_mlock(&self) -> bool {
        self.mlock
    }

    fn resolved_kv_offload(&self) -> bool {
        resolve_toggle(self.kv_offload, self.no_kv_offload, true)
    }

    fn resolved_op_offload(&self) -> bool {
        resolve_toggle(self.op_offload, self.no_op_offload, true)
    }
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
    model: Option<PathBuf>,
    /// Model asset id from built-in model store
    #[arg(long = "model-id")]
    model_id: Option<String>,
    /// Model store root (used by --model-id)
    #[arg(long = "model-store", default_value = "models")]
    model_store: PathBuf,
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
    model: Option<PathBuf>,
    /// Model asset id from built-in model store
    #[arg(long = "model-id")]
    model_id: Option<String>,
    /// Model store root (used by --model-id)
    #[arg(long = "model-store", default_value = "models")]
    model_store: PathBuf,
    /// Listen host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Listen port
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// API type (currently supports: rest)
    #[arg(long, default_value = "rest")]
    api_type: String,
    /// Number of serve worker threads
    #[arg(long = "workers", default_value_t = 4)]
    workers: usize,
    /// Bounded request queue size shared by workers
    #[arg(long = "queue-size", default_value_t = 128)]
    queue_size: usize,
    /// Backpressure policy when queue is full
    #[arg(long = "backpressure", value_enum, default_value_t = ServeBackpressureArg::Reject)]
    backpressure: ServeBackpressureArg,
    /// Dynamic backpressure policy plugin library (.dll/.so/.dylib)
    #[arg(long = "backpressure-plugin")]
    backpressure_plugins: Vec<PathBuf>,
    /// Backpressure policy registry file
    #[arg(
        long = "backpressure-registry",
        default_value = "loci_dispatch_policies.toml"
    )]
    backpressure_registry: PathBuf,
    /// Backpressure policy name from builtin/dynamic registry
    #[arg(long = "backpressure-policy-name")]
    backpressure_policy_name: Option<String>,
    /// Management auth registry file
    #[arg(
        long = "management-auth-registry",
        default_value = "loci_management_auth.toml"
    )]
    management_auth_registry: PathBuf,
    /// Dynamic management auth policy plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "management-auth-plugin")]
    management_auth_plugins: Vec<PathBuf>,
    /// Management auth policy name from builtin/dynamic registry
    #[arg(long = "management-auth-policy-name")]
    management_auth_policy_name: Option<String>,
    /// Model pull policy registry file
    #[arg(
        long = "model-pull-policy-registry",
        default_value = "loci_model_pull_policies.toml"
    )]
    model_pull_policy_registry: PathBuf,
    /// Dynamic model pull policy plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "model-pull-policy-plugin")]
    model_pull_policy_plugins: Vec<PathBuf>,
    /// Model pull policy name from builtin/dynamic registry
    #[arg(long = "model-pull-policy-name")]
    model_pull_policy_name: Option<String>,
    /// Model pull verifier registry file
    #[arg(
        long = "model-pull-verifier-registry",
        default_value = "loci_model_pull_verifiers.toml"
    )]
    model_pull_verifier_registry: PathBuf,
    /// Dynamic model pull verifier plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "model-pull-verifier-plugin")]
    model_pull_verifier_plugins: Vec<PathBuf>,
    /// Model pull verifier name from builtin/dynamic registry
    #[arg(long = "model-pull-verifier-name")]
    model_pull_verifier_name: Option<String>,
    /// Optional bearer token used to enable builtin bearer-token management auth policy
    #[arg(long = "management-auth-bearer-token")]
    management_auth_bearer_token: Option<String>,
    /// Request scope protected by management auth policy
    #[arg(long = "management-auth-scope", value_enum)]
    management_auth_scope: Option<ManagementAuthScopeArg>,
    /// Protected request path prefix, repeatable; used when --management-auth-scope=custom
    #[arg(long = "management-auth-prefix")]
    management_auth_prefixes: Vec<String>,
    /// MCP stdio server spec(s), format: NAME=COMMAND [ARGS...]
    #[arg(long = "mcp-stdio")]
    mcp_stdio: Vec<String>,
    /// MCP registry file path for loading saved servers
    #[arg(long = "mcp-registry")]
    mcp_registry: Option<PathBuf>,
    /// MCP server name(s) from registry to load; empty means all enabled
    #[arg(long = "mcp-server")]
    mcp_servers: Vec<String>,
    /// Dynamic tool plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "tool-plugin")]
    tool_plugins: Vec<PathBuf>,
    /// Tool plugin registry file
    #[arg(
        long = "tool-plugin-registry",
        default_value = "loci_tool_plugins.toml"
    )]
    tool_plugin_registry: PathBuf,
    /// Session store plugin kind (builtin: memory/sqlite[/redis])
    #[arg(long = "session-store-kind", default_value = "sqlite")]
    session_store_kind: String,
    /// Session store option key-value, repeat: --session-store-option key=value
    #[arg(long = "session-store-option")]
    session_store_options: Vec<String>,
    /// Optional dynamic session store plugin library (.dll/.so/.dylib)
    #[arg(long = "session-store-plugin")]
    session_store_plugin: Option<PathBuf>,
    /// Load plugin(s) globally for this server (.wasm => WASM plugin, otherwise dynamic plugin)
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    sampling: SamplingArgs,
}

#[derive(ClapArgs, Debug, Clone)]
struct DispatchPolicyCmd {
    /// Dispatch policy registry file
    #[arg(long, default_value = "loci_dispatch_policies.toml")]
    registry: PathBuf,
    /// Dynamic serve dispatch policy plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    #[command(subcommand)]
    command: DispatchPolicyAction,
}

#[derive(Subcommand, Debug, Clone)]
enum DispatchPolicyAction {
    /// List builtin and dynamically loaded dispatch policies
    List,
    /// Show one dispatch policy entry
    Info { name: String },
    /// Set active dispatch policy in registry
    Activate { name: String },
    /// Validate and load one dispatch policy plugin library
    Load { path: PathBuf },
    /// Unload one dynamic dispatch policy from current registry view
    Unload { name: String },
    /// Reload one dynamic dispatch policy from current registry view
    Reload { name: String },
}

#[derive(ClapArgs, Debug, Clone)]
struct AuthPolicyCmd {
    /// Management auth policy registry file
    #[arg(long, default_value = "loci_management_auth.toml")]
    registry: PathBuf,
    /// Dynamic management auth policy plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    /// Optional bearer token used to enable builtin bearer-token management auth policy
    #[arg(long = "bearer-token")]
    bearer_token: Option<String>,
    #[command(subcommand)]
    command: AuthPolicyAction,
}

#[derive(Subcommand, Debug, Clone)]
enum AuthPolicyAction {
    /// List builtin and dynamically loaded management auth policies
    List,
    /// Show one management auth policy entry
    Info { name: String },
    /// Set active management auth policy in registry
    Activate { name: String },
    /// Validate and load one management auth policy plugin library
    Load { path: PathBuf },
    /// Unload one dynamic management auth policy from current registry view
    Unload { name: String },
    /// Reload one dynamic management auth policy from current registry view
    Reload { name: String },
}

#[derive(ClapArgs, Debug, Clone)]
struct ExecutionPolicyCmd {
    /// Execution policy registry file
    #[arg(long, default_value = "loci_execution_policies.toml")]
    registry: PathBuf,
    /// Dynamic execution policy plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    #[command(subcommand)]
    command: ExecutionPolicyAction,
}

#[derive(Subcommand, Debug, Clone)]
enum ExecutionPolicyAction {
    /// List builtin and dynamically loaded execution policies
    List,
    /// Show one execution policy entry
    Info { name: String },
    /// Set active execution policy in registry
    Activate { name: String },
    /// Validate and load one execution policy plugin library
    Load { path: PathBuf },
    /// Unload one dynamic execution policy from current registry view
    Unload { name: String },
    /// Reload one dynamic execution policy from current registry view
    Reload { name: String },
}

#[derive(ClapArgs, Debug, Clone)]
struct AgentCmd {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Model asset id from built-in model store
    #[arg(long = "model-id")]
    model_id: Option<String>,
    /// Model store root (used by --model-id)
    #[arg(long = "model-store", default_value = "models")]
    model_store: PathBuf,
    /// Tool name (e.g., calculator, read_text_file, list_directory, all, none)
    #[arg(long, default_value = "none")]
    tool: String,
    /// Optional skill name from builtins or loaded skill packs.
    #[arg(long)]
    skill: Option<String>,
    /// Skill pack file(s), supports .json/.toml.
    #[arg(long = "skill-pack")]
    skill_packs: Vec<PathBuf>,
    /// MCP stdio server spec(s), format: NAME=COMMAND [ARGS...]
    #[arg(long = "mcp-stdio")]
    mcp_stdio: Vec<String>,
    /// MCP registry file path for loading saved servers
    #[arg(long = "mcp-registry")]
    mcp_registry: Option<PathBuf>,
    /// MCP server name(s) from registry to load; empty means all enabled
    #[arg(long = "mcp-server")]
    mcp_servers: Vec<String>,
    /// Dynamic tool plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "tool-plugin")]
    tool_plugins: Vec<PathBuf>,
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
struct ModelCmd {
    /// Model store root directory
    #[arg(long, default_value = "models")]
    store: PathBuf,
    #[command(subcommand)]
    command: ModelAction,
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

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ModalOutputArg {
    Text,
    Image,
    Audio,
}

impl From<ModalOutputArg> for OutputModality {
    fn from(value: ModalOutputArg) -> Self {
        match value {
            ModalOutputArg::Text => OutputModality::Text,
            ModalOutputArg::Image => OutputModality::Image,
            ModalOutputArg::Audio => OutputModality::Audio,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum OrchestrationModeArg {
    Route,
    Ensemble,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum RoutingStrategyArg {
    FirstHealthy,
    RoundRobin,
    FastestProbe,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum EnsembleMergeArg {
    Concatenate,
    Longest,
    Judge,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ServeBackpressureArg {
    Reject,
    Block,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ManagementAuthScopeArg {
    ControlPlane,
    All,
    Custom,
}

#[derive(ClapArgs, Debug, Clone)]
struct MultimodalCmd {
    /// Path to the GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,
    /// Model asset id from built-in model store
    #[arg(long = "model-id")]
    model_id: Option<String>,
    /// Model store root (used by --model-id)
    #[arg(long = "model-store", default_value = "models")]
    model_store: PathBuf,
    /// User prompt
    #[arg(short, long)]
    prompt: String,
    /// Image input path(s)
    #[arg(long = "image-input")]
    image_inputs: Vec<PathBuf>,
    /// Audio input path(s)
    #[arg(long = "audio-input")]
    audio_inputs: Vec<PathBuf>,
    /// Requested output modality (repeatable: text/image/audio)
    #[arg(long = "output-modality", value_enum, default_values_t = [ModalOutputArg::Text])]
    output_modalities: Vec<ModalOutputArg>,
    /// Multimodal I/O plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "mm-plugin")]
    mm_plugins: Vec<PathBuf>,
    /// Multimodal I/O plugin name to use
    #[arg(long = "mm-plugin-name", default_value = "descriptor")]
    mm_plugin_name: String,
    /// Artifact output directory for image/audio plans
    #[arg(long = "artifact-dir", default_value = "outputs/multimodal")]
    artifact_dir: PathBuf,
    /// Diffusion model for generated image outputs
    #[arg(long, default_value = "hf-internal-testing/tiny-stable-diffusion-pipe")]
    image_model_id: String,
    /// Number of denoising steps for generated image outputs
    #[arg(long, default_value_t = 4)]
    image_steps: u32,
    /// Guidance scale (CFG) for generated image outputs
    #[arg(long, default_value_t = 0.0)]
    image_guidance_scale: f32,
    /// Optional image width
    #[arg(long)]
    image_width: Option<u32>,
    /// Optional image height
    #[arg(long)]
    image_height: Option<u32>,
    /// Optional image random seed
    #[arg(long)]
    image_seed: Option<u64>,
    /// Optional image kernel plugin for generated image outputs
    #[arg(long = "image-kernel-plugin")]
    image_kernel_plugin: Option<PathBuf>,
    /// Python executable for image backend script
    #[arg(long, default_value = "python")]
    python: String,
    /// Use CUDA for generated image outputs if available
    #[arg(long)]
    use_cuda: bool,
    /// Load text plugin(s) for this run (.wasm => WASM plugin, otherwise dynamic plugin)
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
    #[command(flatten)]
    engine: EngineArgs,
    #[command(flatten)]
    sampling: SamplingArgs,
}

#[derive(ClapArgs, Debug, Clone)]
struct OrchestrateCmd {
    /// Path(s) to GGUF model files, repeat --model for multiple candidates.
    #[arg(long = "model", num_args = 1..)]
    models: Vec<PathBuf>,
    /// Model asset id(s), repeat --model-id for multiple candidates.
    #[arg(long = "model-id", num_args = 1..)]
    model_ids: Vec<String>,
    /// Model store root (used by --model-id and --judge-model-id)
    #[arg(long = "model-store", default_value = "models")]
    model_store: PathBuf,
    /// User prompt
    #[arg(short, long)]
    prompt: String,
    /// Orchestration mode: route (single winner) or ensemble (combine candidates)
    #[arg(long, value_enum, default_value_t = OrchestrationModeArg::Route)]
    mode: OrchestrationModeArg,
    /// Routing strategy when --mode route
    #[arg(
        long = "routing-strategy",
        value_enum,
        default_value_t = RoutingStrategyArg::FirstHealthy
    )]
    routing_strategy: RoutingStrategyArg,
    /// Probe prompt used by fastest-probe routing strategy
    #[arg(long, default_value = "reply with a short health signal")]
    probe_prompt: String,
    /// Probe max tokens used by fastest-probe routing strategy
    #[arg(long, default_value_t = 16)]
    probe_max_tokens: u32,
    /// Merge strategy when --mode ensemble
    #[arg(
        long = "ensemble-merge",
        value_enum,
        default_value_t = EnsembleMergeArg::Concatenate
    )]
    ensemble_merge: EnsembleMergeArg,
    /// Optional judge model path (used only when --ensemble-merge judge)
    #[arg(long = "judge-model")]
    judge_model: Option<PathBuf>,
    /// Optional judge model id from model store (used only when --ensemble-merge judge)
    #[arg(long = "judge-model-id")]
    judge_model_id: Option<String>,
    /// Context length for loaded orchestration models
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
    /// Maximum tokens to generate per model call
    #[arg(short = 'n', long, default_value_t = 512)]
    max_tokens: u32,
    /// Image input path(s)
    #[arg(long = "image-input")]
    image_inputs: Vec<PathBuf>,
    /// Audio input path(s)
    #[arg(long = "audio-input")]
    audio_inputs: Vec<PathBuf>,
    /// Requested output modality (repeatable: text/image/audio)
    #[arg(long = "output-modality", value_enum, default_values_t = [ModalOutputArg::Text])]
    output_modalities: Vec<ModalOutputArg>,
    /// Multimodal I/O plugin library (.dll/.so/.dylib), repeatable
    #[arg(long = "mm-plugin")]
    mm_plugins: Vec<PathBuf>,
    /// Multimodal I/O plugin name to use
    #[arg(long = "mm-plugin-name", default_value = "descriptor")]
    mm_plugin_name: String,
    /// Artifact output directory for image/audio plans
    #[arg(long = "artifact-dir", default_value = "outputs/multimodal")]
    artifact_dir: PathBuf,
    /// Diffusion model for generated image outputs
    #[arg(long, default_value = "hf-internal-testing/tiny-stable-diffusion-pipe")]
    image_model_id: String,
    /// Number of denoising steps for generated image outputs
    #[arg(long, default_value_t = 4)]
    image_steps: u32,
    /// Guidance scale (CFG) for generated image outputs
    #[arg(long, default_value_t = 0.0)]
    image_guidance_scale: f32,
    /// Optional image width
    #[arg(long)]
    image_width: Option<u32>,
    /// Optional image height
    #[arg(long)]
    image_height: Option<u32>,
    /// Optional image random seed
    #[arg(long)]
    image_seed: Option<u64>,
    /// Optional image kernel plugin for generated image outputs
    #[arg(long = "image-kernel-plugin")]
    image_kernel_plugin: Option<PathBuf>,
    /// Python executable for image backend script
    #[arg(long, default_value = "python")]
    python: String,
    /// Use CUDA for generated image outputs if available
    #[arg(long)]
    use_cuda: bool,
    /// Load text plugin(s) for this run (.wasm => WASM plugin, otherwise dynamic plugin)
    #[arg(long = "plugin")]
    plugins: Vec<PathBuf>,
}

#[derive(ClapArgs, Debug, Clone)]
struct McpCmd {
    /// MCP registry config file path
    #[arg(long, default_value = "loci_mcp.toml")]
    registry: PathBuf,
    #[command(subcommand)]
    command: McpAction,
}

#[derive(ClapArgs, Debug, Clone)]
struct SessionCmd {
    /// Session store plugin kind (builtin: memory, sqlite[, redis if enabled], or dynamic kind)
    #[arg(long = "store-kind", default_value = "sqlite")]
    store_kind: String,
    /// Session store option key-value, repeat: --store-option key=value
    #[arg(long = "store-option")]
    store_options: Vec<String>,
    /// Optional dynamic session store plugin library (.dll/.so/.dylib)
    #[arg(long = "store-plugin")]
    store_plugin: Option<PathBuf>,
    /// Model store root (used by --model-id in create)
    #[arg(long = "model-store", default_value = "models")]
    model_store: PathBuf,
    #[command(subcommand)]
    command: SessionAction,
}

#[derive(Subcommand, Debug, Clone)]
enum SessionAction {
    /// Create a new session and persist snapshot
    Create {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// Model asset id from built-in model store
        #[arg(long = "model-id")]
        model_id: Option<String>,
        /// Context size for loaded model
        #[arg(
            short = 'c',
            long = "context-length",
            visible_alias = "context-size",
            default_value_t = 4096
        )]
        context_size: u32,
        /// Skip persisting immediately after create
        #[arg(long, default_value_t = false)]
        no_save: bool,
    },
    /// Generate response in one session and persist updated snapshot
    Generate {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
        /// Prompt text
        #[arg(short, long)]
        prompt: String,
        /// Maximum tokens
        #[arg(short = 'n', long, default_value_t = 512)]
        max_tokens: u32,
        /// Skip persisting after generation
        #[arg(long, default_value_t = false)]
        no_save: bool,
    },
    /// Suspend one session for external wait
    Suspend {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
        /// Suspend reason (e.g. tool_call, user_input)
        #[arg(long)]
        reason: String,
        /// Optional suspend payload
        #[arg(long)]
        data: Option<String>,
        /// Skip persisting after suspend
        #[arg(long, default_value_t = false)]
        no_save: bool,
    },
    /// Resume one suspended session with external data
    Resume {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
        /// External data injected as tool/user result
        #[arg(long = "external-data")]
        external_data: String,
        /// Skip persisting after resume
        #[arg(long, default_value_t = false)]
        no_save: bool,
    },
    /// Show session info (optionally including conversation records)
    Info {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
        /// Include conversation records in output
        #[arg(long, default_value_t = false)]
        with_records: bool,
    },
    /// List active and/or persisted sessions
    List {
        /// Include active in-memory sessions
        #[arg(long, default_value_t = false)]
        active: bool,
        /// Include persisted snapshots from store
        #[arg(long, default_value_t = true)]
        persisted: bool,
    },
    /// Restore one persisted session into memory
    Restore {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
    },
    /// Restore all persisted sessions into memory
    RestoreAll,
    /// Save one active session into store
    Save {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
    },
    /// Save all active sessions into store
    SaveAll,
    /// Delete one persisted session snapshot
    Delete {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
    },
    /// Destroy one active session and delete its persisted snapshot
    Destroy {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
    },
    /// Clear in-memory context/history for one session
    Clear {
        /// Session id
        #[arg(long = "session-id")]
        session_id: u64,
        /// Skip persisting after clear
        #[arg(long, default_value_t = false)]
        no_save: bool,
    },
}

#[derive(Subcommand, Debug, Clone)]
enum McpAction {
    /// Add/update one MCP server and optionally probe connectivity
    Connect {
        /// MCP server spec: NAME=COMMAND [ARGS...]
        spec: String,
        /// Optional local tool prefix (e.g. mcp.fs.)
        #[arg(long)]
        tool_prefix: Option<String>,
        /// Probe server connectivity before saving
        #[arg(long, default_value_t = true)]
        probe: bool,
        /// Save/update this server in registry file
        #[arg(long, default_value_t = true)]
        save: bool,
    },
    /// Remove one MCP server from registry
    Disconnect { name: String },
    /// Enable one MCP server in registry
    Enable { name: String },
    /// Disable one MCP server in registry
    Disable { name: String },
    /// List all MCP servers in registry
    List,
    /// Probe registry MCP servers and show status
    Status,
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

#[derive(Subcommand, Debug, Clone)]
enum ModelAction {
    /// Inspect model resource placement without loading the model into the inference engine.
    Plan {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: Option<PathBuf>,
        /// Model asset id from built-in model store
        #[arg(long = "model-id")]
        model_id: Option<String>,
        /// Requested runtime context size used for KV cache sizing
        #[arg(
            short = 'c',
            long = "context-length",
            visible_alias = "context-size",
            default_value_t = 4096
        )]
        context_size: u32,
        /// Render the plan as JSON
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Register an existing local model file without copying it.
    Add {
        /// Local model file path
        path: PathBuf,
        /// Optional model id (defaults to sanitized file stem)
        #[arg(long)]
        id: Option<String>,
        /// Optional display name
        #[arg(long)]
        name: Option<String>,
        /// Tag(s) for filtering/grouping
        #[arg(long = "tag")]
        tags: Vec<String>,
    },
    /// Import/copy a local model file into managed store.
    Pull {
        /// Source model location: local file path or http(s) URL
        source: String,
        /// Mirror source(s) with higher priority than primary source
        #[arg(long = "mirror")]
        mirrors: Vec<String>,
        /// Optional model id (defaults to sanitized name)
        #[arg(long)]
        id: Option<String>,
        /// Optional display name
        #[arg(long)]
        name: Option<String>,
        /// Expected sha256 checksum (64 hex chars, optional `sha256:` prefix)
        #[arg(long = "sha256")]
        sha256: Option<String>,
        /// Disable HTTP range resume (resume is enabled by default)
        #[arg(long = "no-resume", default_value_t = false)]
        no_resume: bool,
        /// Tag(s) for filtering/grouping
        #[arg(long = "tag")]
        tags: Vec<String>,
    },
    /// List registered model assets.
    List {
        /// Render output as JSON.
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Show one model asset by id.
    Info { id: String },
    /// Remove one model asset record.
    Remove {
        id: String,
        /// Also delete the referenced local file.
        #[arg(long, default_value_t = false)]
        delete_file: bool,
    },
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
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
struct ResourcePlanReport {
    model: String,
    estimate: ModelResourceEstimate,
    plan: ResourcePlan,
    devices: Vec<DeviceInfoResponse>,
}

#[derive(Debug, Serialize)]
struct AuthErrorResponse {
    error: String,
    policy: String,
}

#[derive(Debug, Deserialize)]
struct SessionCreateRequest {
    model: Option<String>,
    model_id: Option<String>,
    context_size: Option<u32>,
    save: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SessionGenerateRequest {
    prompt: String,
    max_tokens: Option<u32>,
    save: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SessionSuspendRequest {
    reason: String,
    data: Option<String>,
    save: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SessionResumeRequest {
    external_data: String,
    save: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct SessionClearRequest {
    save: Option<bool>,
}

#[derive(Debug, Serialize)]
struct SessionSummaryResponse {
    session_id: u64,
    model_id: u64,
    context_length: usize,
    max_context: u32,
    state: String,
    message_count: usize,
}

#[derive(Debug, Serialize)]
struct SessionDetailResponse {
    session: SessionSummaryResponse,
    records: Vec<SessionRecord>,
}

#[derive(Debug, Serialize)]
struct SessionListResponse {
    active: Vec<SessionSummaryResponse>,
    persisted: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct SessionMutationResponse {
    session_id: u64,
    persisted: bool,
    state: String,
}

#[derive(Debug, Serialize)]
struct SessionCreateResponse {
    session_id: u64,
    model_path: String,
    model_id: u64,
    context_size: u32,
    persisted: bool,
}

#[derive(Debug, Serialize)]
struct SessionGenerateResponse {
    session_id: u64,
    response: String,
    persisted: bool,
    state: String,
}

#[derive(Debug, Serialize)]
struct ToolListResponse {
    tools: Vec<loci::function_calling::FunctionDefinition>,
}

#[derive(Debug, Serialize)]
struct ToolInvokeResponse {
    tool: String,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolInvokeRequest {
    #[serde(alias = "tool")]
    name: String,
    #[serde(default)]
    arguments: HashMap<String, Value>,
}

#[derive(Debug, Serialize)]
struct ToolPluginRegistryEntryResponse {
    name: String,
    version: String,
    dynamic: bool,
    source: Option<String>,
    functions: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ToolPluginRegistryListResponse {
    plugins: Vec<ToolPluginRegistryEntryResponse>,
}

#[derive(Debug, Serialize)]
struct ToolPluginRegistryMutationResponse {
    name: String,
    version: String,
    dynamic: bool,
    source: Option<String>,
    functions: Vec<String>,
}

#[derive(Debug, Serialize)]
struct RuntimeEventListResponse {
    events: Vec<RuntimeEvent>,
}

enum SessionApiRoute {
    Collection,
    Item { session_id: SessionId },
    Generate { session_id: SessionId },
    Suspend { session_id: SessionId },
    Resume { session_id: SessionId },
    Save { session_id: SessionId },
    Restore { session_id: SessionId },
    Clear { session_id: SessionId },
    Destroy { session_id: SessionId },
}

enum ModelStoreApiRoute {
    Collection,
    Pull,
    PullJobsCollection,
    PullJobItem { id: String },
    PullJobCancel { id: String },
    PullJobEvents { id: String },
    Item { id: String },
}

#[derive(Debug, Deserialize)]
struct RuntimePluginLoadRequest {
    path: String,
    activate: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ModelAssetRegisterRequest {
    path: String,
    id: Option<String>,
    name: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ModelAssetPullRequest {
    source: String,
    #[serde(default)]
    mirrors: Vec<String>,
    id: Option<String>,
    name: Option<String>,
    sha256: Option<String>,
    no_resume: Option<bool>,
    stream: Option<bool>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ModelAssetListResponse {
    models: Vec<StoredModel>,
}

#[derive(Debug, Serialize)]
struct ModelAssetRemoveResponse {
    model: StoredModel,
    deleted_file: bool,
}

#[derive(Debug, Serialize)]
struct ModelAssetPullJobListResponse {
    jobs: Vec<ModelPullJobSnapshot>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ModelAssetPullStreamEvent {
    Progress { progress: ModelPullProgress },
    Complete { model: StoredModel },
    Error { error: String },
}

#[derive(Debug, Serialize)]
struct PolicyRegistryEntryResponse {
    name: String,
    dynamic: bool,
    source: Option<String>,
    active: bool,
}

#[derive(Debug, Serialize)]
struct PolicyRegistryListResponse {
    active: Option<String>,
    policies: Vec<PolicyRegistryEntryResponse>,
}

#[derive(Debug, Serialize)]
struct PolicyRegistryMutationResponse {
    name: String,
    active: bool,
    source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PolicyApiRoute {
    Collection,
    Item { name: String },
    Load,
    Activate { name: String },
    Reload { name: String },
    Unload { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolApiRoute {
    Collection,
    Item { name: String },
    Invoke,
    PluginCollection,
    PluginItem { name: String },
    PluginLoad,
    PluginReload { name: String },
    PluginUnload { name: String },
}

enum RuntimeEventsApiRoute {
    Collection,
    Stream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ManagementAuthScopeConfig {
    ControlPlane,
    All,
    Custom(Vec<String>),
}

impl ManagementAuthScopeConfig {
    fn from_args(scope: ManagementAuthScopeArg, prefixes: &[String]) -> anyhow::Result<Self> {
        match scope {
            ManagementAuthScopeArg::ControlPlane | ManagementAuthScopeArg::All
                if !prefixes.is_empty() =>
            {
                Err(anyhow::anyhow!(
                    "--management-auth-prefix requires --management-auth-scope=custom"
                ))
            }
            ManagementAuthScopeArg::ControlPlane => Ok(Self::ControlPlane),
            ManagementAuthScopeArg::All => Ok(Self::All),
            ManagementAuthScopeArg::Custom => {
                if prefixes.is_empty() {
                    return Err(anyhow::anyhow!(
                        "--management-auth-scope=custom requires at least one --management-auth-prefix"
                    ));
                }

                let mut normalized = prefixes
                    .iter()
                    .map(|prefix| normalize_management_auth_prefix(prefix))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                normalized.sort();
                normalized.dedup();
                Ok(Self::Custom(normalized))
            }
        }
    }

    fn requires_auth(&self, path: &str) -> bool {
        match self {
            Self::ControlPlane => is_control_plane_api_request(path),
            Self::All => true,
            Self::Custom(prefixes) => {
                let path = request_path_for_matching(path);
                let canonical = strip_v1_path_prefix(path);
                prefixes.iter().any(|prefix| {
                    path_matches_management_prefix(path, prefix)
                        || path_matches_management_prefix(canonical, prefix)
                })
            }
        }
    }

    fn display_label(&self) -> String {
        match self {
            Self::ControlPlane => "control-plane".to_string(),
            Self::All => "all".to_string(),
            Self::Custom(prefixes) => format!("custom({})", prefixes.join(",")),
        }
    }

    fn registry_scope_name(&self) -> String {
        match self {
            Self::ControlPlane => "control-plane".to_string(),
            Self::All => "all".to_string(),
            Self::Custom(_) => "custom".to_string(),
        }
    }

    fn registry_prefixes(&self) -> Vec<String> {
        match self {
            Self::Custom(prefixes) => prefixes.clone(),
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct ModelInfoResponse {
    status: &'static str,
    version: &'static str,
    engine: &'static str,
    positioning: &'static str,
    runtime_model: String,
    backend: BackendInfoResponse,
    capabilities: EngineCapabilityResponse,
    plugins: EnginePluginSummaryResponse,
    tools: EngineToolSummaryResponse,
    integrations: EngineIntegrationResponse,
    n_vocab: u32,
    n_ctx_train: u32,
    n_embd: u32,
    n_layer: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    param_count: Option<u64>,
    architecture: String,
}

#[derive(Debug, Serialize)]
struct BackendInfoResponse {
    name: String,
    version: String,
    has_gpu_support: bool,
    supported_formats: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EngineCapabilityResponse {
    streaming: bool,
    embeddings: bool,
    multimodal: bool,
    resource_planning: bool,
    openai_compat: bool,
    ollama_compat: bool,
    plugin_hot_reload: bool,
}

#[derive(Debug, Serialize)]
struct EnginePluginSummaryResponse {
    text_plugins_total: usize,
    text_plugins_enabled: usize,
    text_plugins_dynamic: usize,
    text_plugins_wasm: usize,
    tool_plugins_total: usize,
}

#[derive(Debug, Serialize)]
struct EngineToolSummaryResponse {
    registered_functions: usize,
    tool_plugin_functions: usize,
}

#[derive(Debug, Serialize)]
struct EngineIntegrationResponse {
    rest: bool,
    c_api: bool,
    plugin_upgradeable: bool,
    openapi_spec_path: &'static str,
    openapi_spec_json_path: &'static str,
}

#[derive(Debug, Deserialize)]
struct ModelPlanRequest {
    model: Option<String>,
    model_id: Option<String>,
    context_size: Option<u32>,
}

#[derive(Debug, Serialize)]
struct DeviceInfoResponse {
    id: i32,
    name: String,
    memory_bytes: u64,
    device_type: String,
    compute_capability: f32,
    available: bool,
}

#[derive(Debug, Default)]
struct ServerMetricsCounters {
    total_requests: u64,
    total_client_errors: u64,
    total_server_errors: u64,
    total_latency_ms: u128,
    endpoint_hits: HashMap<String, u64>,
}

#[derive(Debug)]
struct ServerMetrics {
    started_at_unix_ms: u64,
    counters: Mutex<ServerMetricsCounters>,
}

#[derive(Debug, Serialize)]
struct ServerMetricsResponse {
    status: &'static str,
    started_at_unix_ms: u64,
    uptime_ms: u64,
    total_requests: u64,
    total_client_errors: u64,
    total_server_errors: u64,
    average_latency_ms: f64,
    endpoint_hits: BTreeMap<String, u64>,
}

impl ServerMetrics {
    fn new() -> Self {
        Self {
            started_at_unix_ms: unix_ms_now(),
            counters: Mutex::new(ServerMetricsCounters::default()),
        }
    }

    fn record(&self, endpoint: &str, status_code: u16, latency: Duration) {
        let mut counters = self
            .counters
            .lock()
            .expect("server metrics mutex should not be poisoned");
        counters.total_requests += 1;
        if (400..500).contains(&status_code) {
            counters.total_client_errors += 1;
        }
        if status_code >= 500 {
            counters.total_server_errors += 1;
        }
        counters.total_latency_ms += latency.as_millis();
        *counters
            .endpoint_hits
            .entry(endpoint.to_string())
            .or_insert(0) += 1;
    }

    fn snapshot(&self) -> ServerMetricsResponse {
        let counters = self
            .counters
            .lock()
            .expect("server metrics mutex should not be poisoned");
        let average_latency_ms = if counters.total_requests == 0 {
            0.0
        } else {
            (counters.total_latency_ms as f64) / (counters.total_requests as f64)
        };

        let mut endpoint_hits = BTreeMap::new();
        for (endpoint, count) in &counters.endpoint_hits {
            endpoint_hits.insert(endpoint.clone(), *count);
        }

        ServerMetricsResponse {
            status: "ok",
            started_at_unix_ms: self.started_at_unix_ms,
            uptime_ms: unix_ms_now().saturating_sub(self.started_at_unix_ms),
            total_requests: counters.total_requests,
            total_client_errors: counters.total_client_errors,
            total_server_errors: counters.total_server_errors,
            average_latency_ms,
            endpoint_hits,
        }
    }
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn status_code_value(status: &str) -> u16 {
    status
        .split_whitespace()
        .next()
        .and_then(|x| x.parse::<u16>().ok())
        .unwrap_or(500)
}

fn gib(bytes: u64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0 * 1024.0)
}

fn device_info_response(device: &DeviceInfo) -> DeviceInfoResponse {
    DeviceInfoResponse {
        id: device.id,
        name: device.name.clone(),
        memory_bytes: device.memory_bytes,
        device_type: device.device_type.to_string(),
        compute_capability: device.compute_capability,
        available: device.available,
    }
}

fn build_resource_plan_report(
    model: &Path,
    estimate: ModelResourceEstimate,
    plan: ResourcePlan,
    devices: &[DeviceInfo],
) -> ResourcePlanReport {
    ResourcePlanReport {
        model: model.display().to_string(),
        estimate,
        plan,
        devices: devices.iter().map(device_info_response).collect(),
    }
}

fn build_model_info_response(
    engine: &InferenceEngine,
    plugins: Option<&PluginRegistry>,
) -> ModelInfoResponse {
    let metadata = engine.model_metadata();
    let backend = engine.backend_capabilities();
    let tool_plugins = engine.list_tool_plugins();
    let tool_plugin_functions = tool_plugins
        .iter()
        .map(|plugin| plugin.function_names.len())
        .sum();

    ModelInfoResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        engine: "loci",
        positioning: "embeddable-ai-inference-engine",
        runtime_model: format!("loci/{}:{}", engine.backend_name(), metadata.architecture),
        backend: BackendInfoResponse {
            name: engine.backend_name().to_string(),
            version: backend
                .as_ref()
                .map(|capabilities| capabilities.version.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            has_gpu_support: backend
                .as_ref()
                .map(|capabilities| capabilities.has_gpu_support)
                .unwrap_or(false),
            supported_formats: backend
                .as_ref()
                .map(|capabilities| capabilities.supported_formats.clone())
                .unwrap_or_default(),
        },
        capabilities: EngineCapabilityResponse {
            streaming: engine.supports_streaming(),
            embeddings: engine.supports_embeddings(),
            multimodal: engine.supports_multimodal(),
            resource_planning: true,
            openai_compat: true,
            ollama_compat: true,
            plugin_hot_reload: true,
        },
        plugins: EnginePluginSummaryResponse {
            text_plugins_total: plugins.map_or(0, PluginRegistry::count),
            text_plugins_enabled: plugins.map_or(0, PluginRegistry::count_enabled),
            text_plugins_dynamic: plugins.map_or(0, PluginRegistry::count_dynamic),
            text_plugins_wasm: plugins.map_or(0, PluginRegistry::count_wasm),
            tool_plugins_total: tool_plugins.len(),
        },
        tools: EngineToolSummaryResponse {
            registered_functions: engine.function_calling_manager().list_functions().len(),
            tool_plugin_functions,
        },
        integrations: EngineIntegrationResponse {
            rest: true,
            c_api: true,
            plugin_upgradeable: true,
            openapi_spec_path: OPENAPI_SPEC_YAML_PATH,
            openapi_spec_json_path: OPENAPI_SPEC_JSON_PATH,
        },
        n_vocab: metadata.n_vocab,
        n_ctx_train: metadata.n_ctx_train,
        n_embd: metadata.n_embd,
        n_layer: metadata.n_layer,
        param_count: metadata.param_count,
        architecture: metadata.architecture,
    }
}

fn print_resource_plan_summary(estimate: &ModelResourceEstimate, plan: &ResourcePlan) {
    println!("Resource metadata source: {:?}", estimate.metadata_source);
    if let Some(metadata) = &estimate.gguf_metadata {
        println!(
            "GGUF metadata: arch={} version={} tensors={} train_ctx={} embd={} layers={} heads={} kv_heads={} file_type={}",
            metadata
                .architecture
                .as_deref()
                .unwrap_or("unknown"),
            metadata.version,
            metadata.tensor_count,
            metadata
                .context_length
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            metadata
                .embedding_length
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            metadata
                .block_count
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            metadata
                .attention_head_count
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            metadata
                .attention_head_count_kv
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            metadata
                .file_type
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
    }
    println!(
        "Resource estimate: model={:.2} GiB kv_cache={:.2} GiB working_set={:.2} GiB total={:.2} GiB",
        gib(estimate.model_bytes),
        gib(estimate.kv_cache_bytes),
        gib(estimate.working_set_bytes),
        gib(estimate.total_bytes)
    );
    println!(
        "Resource plan: use_gpu={} gpu_layers={} mmap={} mlock={} kv_offload={} op_offload={} split_mode={:?} main_gpu={} tensor_split={} rationale={}",
        plan.use_gpu,
        plan.n_gpu_layers,
        plan.use_mmap,
        plan.use_mlock,
        plan.kv_offload,
        plan.op_offload,
        plan.split_mode,
        plan.main_gpu,
        plan.tensor_split
            .as_ref()
            .map(|values| format_tensor_split(values))
            .unwrap_or_else(|| "auto".to_string()),
        plan.rationale
    );
}

trait ServeDispatchPolicy: Send + Sync {
    fn dispatch(
        &self,
        sender: &Sender<TcpStream>,
        stream: TcpStream,
    ) -> std::result::Result<(), TcpStream>;
}

struct PluginBackpressureDispatchPolicy {
    plugin: Arc<dyn ServeDispatchPolicyPlugin>,
}

impl PluginBackpressureDispatchPolicy {
    fn new(plugin: Arc<dyn ServeDispatchPolicyPlugin>) -> Self {
        Self { plugin }
    }
}

impl ServeDispatchPolicy for PluginBackpressureDispatchPolicy {
    fn dispatch(
        &self,
        sender: &Sender<TcpStream>,
        mut stream: TcpStream,
    ) -> std::result::Result<(), TcpStream> {
        let policy = &self.plugin;
        let mut attempt: u32 = 0;
        loop {
            match sender.try_send(stream) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Disconnected(stream)) => return Err(stream),
                Err(TrySendError::Full(stream_full)) => {
                    let action = policy.on_queue_full(&QueuePressureContext {
                        attempt,
                        queue_len: sender.len(),
                        queue_capacity: sender.capacity().unwrap_or(0),
                    });
                    match action {
                        QueueFullAction::Reject => return Err(stream_full),
                        QueueFullAction::Block => {
                            return sender.send(stream_full).map_err(|e| e.0);
                        }
                        QueueFullAction::RetryAfterMillis(delay_ms) => {
                            let max_retries = policy.max_retries();
                            if attempt >= max_retries {
                                return Err(stream_full);
                            }
                            if delay_ms > 0 {
                                thread::sleep(Duration::from_millis(delay_ms));
                            }
                            attempt = attempt.saturating_add(1);
                            stream = stream_full;
                        }
                    }
                }
            }
        }
    }
}

struct ActiveServeDispatchPolicy {
    name: RwLock<String>,
    policy: RwLock<Arc<dyn ServeDispatchPolicy>>,
}

impl ActiveServeDispatchPolicy {
    fn new(name: String, policy: Arc<dyn ServeDispatchPolicy>) -> Self {
        Self {
            name: RwLock::new(name),
            policy: RwLock::new(policy),
        }
    }

    fn name(&self) -> String {
        self.name.read().clone()
    }

    fn dispatch(
        &self,
        sender: &Sender<TcpStream>,
        stream: TcpStream,
    ) -> std::result::Result<(), TcpStream> {
        Arc::clone(&self.policy.read()).dispatch(sender, stream)
    }

    fn activate_plugin(&self, name: String, plugin: Arc<dyn ServeDispatchPolicyPlugin>) {
        *self.policy.write() = Arc::new(PluginBackpressureDispatchPolicy::new(plugin));
        *self.name.write() = name;
    }
}

struct ActiveManagementAuthPolicy {
    name: RwLock<String>,
    policy: RwLock<Arc<dyn ManagementAuthPolicyPlugin>>,
}

impl ActiveManagementAuthPolicy {
    fn new(name: String, policy: Arc<dyn ManagementAuthPolicyPlugin>) -> Self {
        Self {
            name: RwLock::new(name),
            policy: RwLock::new(policy),
        }
    }

    fn name(&self) -> String {
        self.name.read().clone()
    }

    fn snapshot(&self) -> (String, Arc<dyn ManagementAuthPolicyPlugin>) {
        (self.name.read().clone(), Arc::clone(&self.policy.read()))
    }

    fn activate(&self, name: String, policy: Arc<dyn ManagementAuthPolicyPlugin>) {
        *self.policy.write() = policy;
        *self.name.write() = name;
    }
}

struct ActiveModelPullPolicy {
    name: RwLock<String>,
    policy: RwLock<Arc<dyn ModelPullPolicyPlugin>>,
}

impl ActiveModelPullPolicy {
    fn new(name: String, policy: Arc<dyn ModelPullPolicyPlugin>) -> Self {
        Self {
            name: RwLock::new(name),
            policy: RwLock::new(policy),
        }
    }

    fn name(&self) -> String {
        self.name.read().clone()
    }

    fn snapshot(&self) -> (String, Arc<dyn ModelPullPolicyPlugin>) {
        (self.name.read().clone(), Arc::clone(&self.policy.read()))
    }

    fn activate(&self, name: String, policy: Arc<dyn ModelPullPolicyPlugin>) {
        *self.policy.write() = policy;
        *self.name.write() = name;
    }
}

struct ActiveModelPullVerifier {
    name: RwLock<String>,
    verifier: RwLock<Arc<dyn ModelPullVerifierPlugin>>,
}

impl ActiveModelPullVerifier {
    fn new(name: String, verifier: Arc<dyn ModelPullVerifierPlugin>) -> Self {
        Self {
            name: RwLock::new(name),
            verifier: RwLock::new(verifier),
        }
    }

    fn name(&self) -> String {
        self.name.read().clone()
    }

    fn snapshot(&self) -> (String, Arc<dyn ModelPullVerifierPlugin>) {
        (self.name.read().clone(), Arc::clone(&self.verifier.read()))
    }

    fn activate(&self, name: String, verifier: Arc<dyn ModelPullVerifierPlugin>) {
        *self.verifier.write() = verifier;
        *self.name.write() = name;
    }
}

fn default_backpressure_policy_name(mode: ServeBackpressureArg) -> &'static str {
    match mode {
        ServeBackpressureArg::Reject => "reject",
        ServeBackpressureArg::Block => "block",
    }
}

fn session_info_to_summary(info: &SessionInfo) -> SessionSummaryResponse {
    SessionSummaryResponse {
        session_id: info.session_id.as_u64(),
        model_id: info.model_id.as_u64(),
        context_length: info.context_length,
        max_context: info.max_context,
        state: format!("{:?}", info.state),
        message_count: info.message_count,
    }
}

fn parse_session_api_route(path: &str) -> Option<SessionApiRoute> {
    let path = path.split('?').next().unwrap_or(path);
    let normalized = path.strip_prefix("/v1").unwrap_or(path);
    if normalized == "/sessions" {
        return Some(SessionApiRoute::Collection);
    }
    let rest = normalized.strip_prefix("/sessions/")?;
    let parts = rest.split('/').collect::<Vec<_>>();
    if parts.is_empty() || parts[0].is_empty() {
        return None;
    }
    let session_id = parts[0].parse::<u64>().ok().map(SessionId::from)?;
    if parts.len() == 1 {
        return Some(SessionApiRoute::Item { session_id });
    }
    if parts.len() != 2 {
        return None;
    }
    match parts[1] {
        "generate" => Some(SessionApiRoute::Generate { session_id }),
        "suspend" => Some(SessionApiRoute::Suspend { session_id }),
        "resume" => Some(SessionApiRoute::Resume { session_id }),
        "save" => Some(SessionApiRoute::Save { session_id }),
        "restore" => Some(SessionApiRoute::Restore { session_id }),
        "clear" => Some(SessionApiRoute::Clear { session_id }),
        "destroy" => Some(SessionApiRoute::Destroy { session_id }),
        _ => None,
    }
}

fn parse_model_store_api_route(path: &str) -> Option<ModelStoreApiRoute> {
    let path = path.split('?').next().unwrap_or(path);
    let normalized = path.strip_prefix("/v1").unwrap_or(path);
    if normalized == "/models/assets" {
        return Some(ModelStoreApiRoute::Collection);
    }
    if normalized == "/models/assets/pull" {
        return Some(ModelStoreApiRoute::Pull);
    }
    if normalized == "/models/assets/pulls" {
        return Some(ModelStoreApiRoute::PullJobsCollection);
    }
    if let Some(rest) = normalized.strip_prefix("/models/assets/pulls/") {
        let parts = rest.split('/').collect::<Vec<_>>();
        if parts.is_empty() || parts[0].is_empty() {
            return None;
        }
        let id = parts[0].to_string();
        return match parts.len() {
            1 => Some(ModelStoreApiRoute::PullJobItem { id }),
            2 => match parts[1] {
                "cancel" => Some(ModelStoreApiRoute::PullJobCancel { id }),
                "events" => Some(ModelStoreApiRoute::PullJobEvents { id }),
                _ => None,
            },
            _ => None,
        };
    }
    let id = normalized.strip_prefix("/models/assets/")?;
    if id.is_empty() || id.contains('/') {
        return None;
    }
    Some(ModelStoreApiRoute::Item { id: id.to_string() })
}

fn parse_policy_api_route(path: &str, prefix: &str) -> Option<PolicyApiRoute> {
    let path = path.split('?').next().unwrap_or(path);
    let normalized = path.strip_prefix("/v1").unwrap_or(path);
    if normalized == prefix {
        return Some(PolicyApiRoute::Collection);
    }
    if normalized == format!("{prefix}/load") {
        return Some(PolicyApiRoute::Load);
    }
    let rest = normalized.strip_prefix(&(prefix.to_string() + "/"))?;
    let parts = rest.split('/').collect::<Vec<_>>();
    if parts.is_empty() || parts[0].is_empty() {
        return None;
    }
    let name = parts[0].to_string();
    if parts.len() == 1 {
        return Some(PolicyApiRoute::Item { name });
    }
    if parts.len() != 2 {
        return None;
    }
    match parts[1] {
        "activate" => Some(PolicyApiRoute::Activate { name }),
        "reload" => Some(PolicyApiRoute::Reload { name }),
        "unload" => Some(PolicyApiRoute::Unload { name }),
        _ => None,
    }
}

fn parse_dispatch_policy_api_route(path: &str) -> Option<PolicyApiRoute> {
    parse_policy_api_route(path, "/dispatch-policies")
}

fn parse_execution_policy_api_route(path: &str) -> Option<PolicyApiRoute> {
    parse_policy_api_route(path, "/execution-policies")
}

fn parse_auth_policy_api_route(path: &str) -> Option<PolicyApiRoute> {
    parse_policy_api_route(path, "/auth-policies")
}

fn parse_model_pull_policy_api_route(path: &str) -> Option<PolicyApiRoute> {
    parse_policy_api_route(path, "/model-pull-policies")
}

fn parse_model_pull_verifier_api_route(path: &str) -> Option<PolicyApiRoute> {
    parse_policy_api_route(path, "/model-pull-verifiers")
}

fn parse_tool_api_route(path: &str) -> Option<ToolApiRoute> {
    let path = path.split('?').next().unwrap_or(path);
    let normalized = path.strip_prefix("/v1").unwrap_or(path);
    if normalized == "/tools/plugins" {
        return Some(ToolApiRoute::PluginCollection);
    }
    if normalized == "/tools/plugins/load" {
        return Some(ToolApiRoute::PluginLoad);
    }
    if let Some(rest) = normalized.strip_prefix("/tools/plugins/") {
        let parts = rest.split('/').collect::<Vec<_>>();
        if parts.is_empty() || parts[0].is_empty() {
            return None;
        }
        let name = parts[0].to_string();
        return match parts.len() {
            1 => Some(ToolApiRoute::PluginItem { name }),
            2 => match parts[1] {
                "reload" => Some(ToolApiRoute::PluginReload { name }),
                "unload" => Some(ToolApiRoute::PluginUnload { name }),
                _ => None,
            },
            _ => None,
        };
    }
    if normalized == "/tools" {
        return Some(ToolApiRoute::Collection);
    }
    if normalized == "/tools/invoke" {
        return Some(ToolApiRoute::Invoke);
    }
    let name = normalized.strip_prefix("/tools/")?;
    if name.is_empty() || name.contains('/') {
        return None;
    }
    Some(ToolApiRoute::Item {
        name: name.to_string(),
    })
}

fn parse_runtime_events_api_route(path: &str) -> Option<RuntimeEventsApiRoute> {
    let path = path.split('?').next().unwrap_or(path);
    let normalized = path.strip_prefix("/v1").unwrap_or(path);
    match normalized {
        "/events" => Some(RuntimeEventsApiRoute::Collection),
        "/events/stream" => Some(RuntimeEventsApiRoute::Stream),
        _ => None,
    }
}

fn request_path_for_matching(path: &str) -> &str {
    path.split('?').next().unwrap_or(path)
}

fn request_query_value<'a>(path: &'a str, key: &str) -> Option<&'a str> {
    let query = path.split_once('?')?.1;
    for pair in query.split('&') {
        let (candidate_key, candidate_value) = match pair.split_once('=') {
            Some((candidate_key, candidate_value)) => (candidate_key, candidate_value),
            None => (pair, ""),
        };
        if candidate_key == key {
            return Some(candidate_value);
        }
    }
    None
}

fn request_query_bool(path: &str, key: &str) -> Option<bool> {
    let value = request_query_value(path, key)?;
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn request_query_usize(path: &str, key: &str) -> Option<usize> {
    request_query_value(path, key)?.trim().parse::<usize>().ok()
}

fn strip_v1_path_prefix(path: &str) -> &str {
    match path.strip_prefix("/v1") {
        Some("") => "/",
        Some(stripped) if stripped.starts_with('/') => stripped,
        _ => path,
    }
}

fn path_matches_management_prefix(path: &str, prefix: &str) -> bool {
    if prefix == "/" {
        return path.starts_with('/');
    }

    path == prefix
        || path
            .strip_prefix(prefix)
            .map(|remainder| remainder.starts_with('/'))
            .unwrap_or(false)
}

fn normalize_management_auth_prefix(prefix: &str) -> anyhow::Result<String> {
    let prefix = prefix.trim();
    if prefix.is_empty() {
        return Err(anyhow::anyhow!("management auth prefix cannot be empty"));
    }
    if !prefix.starts_with('/') {
        return Err(anyhow::anyhow!(
            "management auth prefix '{}' must start with '/'",
            prefix
        ));
    }
    if prefix == "/" {
        return Ok("/".to_string());
    }

    Ok(prefix.trim_end_matches('/').to_string())
}

fn parse_management_auth_scope_name(raw: &str) -> anyhow::Result<ManagementAuthScopeArg> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "control-plane" | "control_plane" | "controlplane" => {
            Ok(ManagementAuthScopeArg::ControlPlane)
        }
        "all" => Ok(ManagementAuthScopeArg::All),
        "custom" => Ok(ManagementAuthScopeArg::Custom),
        other => Err(anyhow::anyhow!(
            "invalid stored management auth scope '{}'",
            other
        )),
    }
}

fn is_high_risk_automation_api_request(path: &str) -> bool {
    ["/tools", "/browser", "/device"]
        .iter()
        .any(|prefix| path_matches_management_prefix(path, prefix))
}

fn is_control_plane_api_request(path: &str) -> bool {
    let path = request_path_for_matching(path);
    let canonical = strip_v1_path_prefix(path);
    let model_plan_request = path == "/models/plan" || canonical == "/models/plan";
    parse_session_api_route(path).is_some()
        || parse_model_store_api_route(path).is_some()
        || parse_dispatch_policy_api_route(path).is_some()
        || parse_execution_policy_api_route(path).is_some()
        || parse_auth_policy_api_route(path).is_some()
        || parse_model_pull_policy_api_route(path).is_some()
        || parse_model_pull_verifier_api_route(path).is_some()
        || parse_runtime_events_api_route(path).is_some()
        || model_plan_request
        || is_high_risk_automation_api_request(path)
        || is_high_risk_automation_api_request(canonical)
}

fn requested_openapi_spec_format(path: &str) -> Option<OpenApiSpecFormat> {
    let path = request_path_for_matching(path);
    let canonical = strip_v1_path_prefix(path);
    if path == OPENAPI_SPEC_YAML_PATH || canonical == OPENAPI_SPEC_YAML_PATH {
        return Some(OpenApiSpecFormat::Yaml);
    }
    if path == OPENAPI_SPEC_JSON_PATH || canonical == OPENAPI_SPEC_JSON_PATH {
        return Some(OpenApiSpecFormat::Json);
    }
    None
}

fn build_openapi_json_spec() -> anyhow::Result<String> {
    let spec: Value = serde_yaml::from_str(OPENAPI_SPEC_YAML)?;
    Ok(serde_json::to_string_pretty(&spec)?)
}

fn resolve_management_auth_scope(
    requested_scope: Option<ManagementAuthScopeArg>,
    requested_prefixes: &[String],
    store: &DynamicPolicyRegistry,
) -> anyhow::Result<(ManagementAuthScopeConfig, bool)> {
    if !requested_prefixes.is_empty() && requested_scope.is_none() {
        return Err(anyhow::anyhow!(
            "--management-auth-prefix requires --management-auth-scope=custom"
        ));
    }

    if let Some(scope) = requested_scope {
        return Ok((
            ManagementAuthScopeConfig::from_args(scope, requested_prefixes)?,
            true,
        ));
    }

    if !requested_prefixes.is_empty() {
        return Err(anyhow::anyhow!(
            "--management-auth-prefix requires --management-auth-scope=custom"
        ));
    }

    if let Some(scope_name) = store.scope() {
        let scope = parse_management_auth_scope_name(scope_name)?;
        return Ok((
            ManagementAuthScopeConfig::from_args(scope, store.prefixes())?,
            false,
        ));
    }

    Ok((ManagementAuthScopeConfig::ControlPlane, false))
}

fn persist_management_auth_scope(
    store: &mut DynamicPolicyRegistry,
    scope: &ManagementAuthScopeConfig,
) -> anyhow::Result<()> {
    store.set_scope(Some(scope.registry_scope_name()));
    store.set_prefixes(scope.registry_prefixes());
    store
        .persist()
        .map_err(|e| anyhow::anyhow!("failed persisting management auth registry: {}", e))
}

fn build_management_auth_context(
    stream: &TcpStream,
    request: &HttpRequest,
) -> ManagementAuthContext {
    ManagementAuthContext {
        method: request.method.clone(),
        path: request.path.clone(),
        headers: request.headers.clone(),
        remote_addr: stream.peer_addr().ok().map(|addr| addr.to_string()),
    }
}

fn ensure_candidate_management_policy_authorizes_request(
    context: &ManagementAuthContext,
    policy_name: &str,
    policy: &dyn ManagementAuthPolicyPlugin,
) -> anyhow::Result<()> {
    match policy.authorize(context) {
        ManagementAuthDecision::Allow => Ok(()),
        ManagementAuthDecision::Deny(error) => Err(anyhow::anyhow!(
            "refusing to activate management auth policy '{}': current request would be denied: {}",
            policy_name,
            error
        )),
    }
}

fn authorize_management_request(
    context: &ManagementAuthContext,
    scope: &ManagementAuthScopeConfig,
    active_policy: &ActiveManagementAuthPolicy,
) -> anyhow::Result<Option<(&'static str, AuthErrorResponse)>> {
    if !scope.requires_auth(&context.path) {
        return Ok(None);
    }

    let (policy_name, policy) = active_policy.snapshot();
    match policy.authorize(context) {
        ManagementAuthDecision::Allow => Ok(None),
        ManagementAuthDecision::Deny(error) => Ok(Some((
            "401 Unauthorized",
            AuthErrorResponse {
                error,
                policy: policy_name,
            },
        ))),
    }
}

fn dispatch_descriptor_to_response(
    descriptor: ServeDispatchPolicyDescriptor,
    active_name: Option<&str>,
) -> PolicyRegistryEntryResponse {
    let active = active_name
        .map(|name| name == descriptor.name.as_str())
        .unwrap_or(false);
    PolicyRegistryEntryResponse {
        name: descriptor.name,
        dynamic: descriptor.dynamic,
        source: descriptor.source.map(|path| path.display().to_string()),
        active,
    }
}

fn execution_descriptor_to_response(
    descriptor: ExecutionPolicyDescriptor,
    active_name: Option<&str>,
) -> PolicyRegistryEntryResponse {
    let active = active_name
        .map(|name| name == descriptor.name.as_str())
        .unwrap_or(false);
    PolicyRegistryEntryResponse {
        name: descriptor.name,
        dynamic: descriptor.dynamic,
        source: descriptor.source.map(|path| path.display().to_string()),
        active,
    }
}

fn auth_descriptor_to_response(
    descriptor: loci::management_auth::ManagementAuthPolicyDescriptor,
    active_name: Option<&str>,
) -> PolicyRegistryEntryResponse {
    let active = active_name
        .map(|name| name == descriptor.name.as_str())
        .unwrap_or(false);
    PolicyRegistryEntryResponse {
        name: descriptor.name,
        dynamic: descriptor.dynamic,
        source: descriptor.source.map(|path| path.display().to_string()),
        active,
    }
}

fn model_pull_descriptor_to_response(
    descriptor: ModelPullPolicyDescriptor,
    active_name: Option<&str>,
) -> PolicyRegistryEntryResponse {
    let active = active_name
        .map(|name| name == descriptor.name.as_str())
        .unwrap_or(false);
    PolicyRegistryEntryResponse {
        name: descriptor.name,
        dynamic: descriptor.dynamic,
        source: descriptor.source.map(|path| path.display().to_string()),
        active,
    }
}

fn model_pull_verifier_descriptor_to_response(
    descriptor: ModelPullVerifierDescriptor,
    active_name: Option<&str>,
) -> PolicyRegistryEntryResponse {
    let active = active_name
        .map(|name| name == descriptor.name.as_str())
        .unwrap_or(false);
    PolicyRegistryEntryResponse {
        name: descriptor.name,
        dynamic: descriptor.dynamic,
        source: descriptor.source.map(|path| path.display().to_string()),
        active,
    }
}

fn build_model_pull_policy_context(payload: &ModelAssetPullRequest) -> ModelPullPolicyContext {
    ModelPullPolicyContext::new(
        payload.source.clone(),
        payload.mirrors.clone(),
        payload.id.clone(),
        payload.name.clone(),
        payload.sha256.clone(),
        !payload.no_resume.unwrap_or(false),
        payload.tags.clone(),
    )
}

#[allow(clippy::too_many_arguments)]
fn publish_runtime_event(
    runtime_events: &RuntimeEventBus,
    category: RuntimeEventCategory,
    action: impl Into<String>,
    outcome: RuntimeEventOutcome,
    endpoint: Option<&str>,
    request: Option<&HttpRequest>,
    status_code: Option<u16>,
    subject: Option<String>,
    details: Option<Value>,
) {
    let mut event = RuntimeEvent::new(category, action, outcome);
    event.endpoint = endpoint.map(str::to_string);
    event.method = request.map(|request| request.method.clone());
    event.path = request.map(|request| request.path.clone());
    event.status_code = status_code;
    event.subject = subject;
    event.details = details;
    runtime_events.emit(event);
}

fn tool_plugin_descriptor_to_response(
    descriptor: loci::LoadedToolPluginDescriptor,
) -> ToolPluginRegistryEntryResponse {
    ToolPluginRegistryEntryResponse {
        name: descriptor.name,
        version: descriptor.version,
        dynamic: descriptor.dynamic,
        source: descriptor.source.map(|path| path.display().to_string()),
        functions: descriptor.function_names,
    }
}

fn load_dynamic_policy_registry(path: &Path) -> anyhow::Result<DynamicPolicyRegistry> {
    let mut registry = DynamicPolicyRegistry::with_config_path(path);
    if path.exists() {
        registry.load_from_file(path)?;
    }
    Ok(registry)
}

fn merge_plugin_paths(primary: &[PathBuf], overlay: &[PathBuf]) -> Vec<PathBuf> {
    let mut merged = primary.to_vec();
    for path in overlay {
        if !merged.iter().any(|existing| existing == path) {
            merged.push(path.clone());
        }
    }
    merged
}

fn build_dispatch_policy_registry(
    plugin_paths: &[PathBuf],
) -> anyhow::Result<ServeDispatchPolicyRegistry> {
    let registry = ServeDispatchPolicyRegistry::with_builtin_policies();
    for plugin_path in plugin_paths {
        registry.load_dynamic_policy(plugin_path).map_err(|e| {
            anyhow::anyhow!(
                "failed loading backpressure plugin '{}': {}",
                plugin_path.display(),
                e
            )
        })?;
    }
    Ok(registry)
}

fn build_execution_policy_registry(
    plugin_paths: &[PathBuf],
) -> anyhow::Result<ExecutionPolicyRegistry> {
    let registry = ExecutionPolicyRegistry::with_builtin_policies();
    for plugin_path in plugin_paths {
        registry.load_dynamic_policy(plugin_path).map_err(|e| {
            anyhow::anyhow!(
                "failed loading execution policy plugin '{}': {}",
                plugin_path.display(),
                e
            )
        })?;
    }
    Ok(registry)
}

fn build_management_auth_policy_registry(
    plugin_paths: &[PathBuf],
    bearer_token: Option<&str>,
) -> anyhow::Result<ManagementAuthPolicyRegistry> {
    let registry = ManagementAuthPolicyRegistry::with_builtin_and_bearer_token(bearer_token)?;
    for plugin_path in plugin_paths {
        registry.load_dynamic_policy(plugin_path).map_err(|e| {
            anyhow::anyhow!(
                "failed loading management auth policy plugin '{}': {}",
                plugin_path.display(),
                e
            )
        })?;
    }
    Ok(registry)
}

fn build_model_pull_policy_registry(
    plugin_paths: &[PathBuf],
) -> anyhow::Result<ModelPullPolicyRegistry> {
    let registry = ModelPullPolicyRegistry::with_builtin_policies();
    for plugin_path in plugin_paths {
        registry.load_dynamic_policy(plugin_path).map_err(|e| {
            anyhow::anyhow!(
                "failed loading model pull policy plugin '{}': {}",
                plugin_path.display(),
                e
            )
        })?;
    }
    Ok(registry)
}

fn build_model_pull_verifier_registry(
    plugin_paths: &[PathBuf],
) -> anyhow::Result<ModelPullVerifierRegistry> {
    let registry = ModelPullVerifierRegistry::with_builtin_verifiers();
    for plugin_path in plugin_paths {
        registry.load_dynamic_verifier(plugin_path).map_err(|e| {
            anyhow::anyhow!(
                "failed loading model pull verifier plugin '{}': {}",
                plugin_path.display(),
                e
            )
        })?;
    }
    Ok(registry)
}

fn resolve_dispatch_policy_from_registry(
    registry: &ServeDispatchPolicyRegistry,
    requested_name: Option<&str>,
    fallback: ServeBackpressureArg,
) -> anyhow::Result<(String, Arc<dyn ServeDispatchPolicyPlugin>)> {
    let selected_policy_name = requested_name
        .map(|name| name.to_string())
        .unwrap_or_else(|| default_backpressure_policy_name(fallback).to_string());
    let policy = registry.get(&selected_policy_name).ok_or_else(|| {
        let available = registry.list_names().join(", ");
        anyhow::anyhow!(
            "unknown backpressure policy '{}'. available: {}",
            selected_policy_name,
            available
        )
    })?;
    Ok((selected_policy_name, policy))
}

fn build_engine_with_execution_registry(
    model: &Path,
    engine_args: &EngineArgs,
) -> anyhow::Result<(InferenceEngine, Arc<ExecutionPolicyRegistry>, String)> {
    let resource_plan = if engine_args.resolved_auto_resource_plan() {
        let estimate =
            ResourcePlanner::estimate_model_requirements(model, engine_args.context_size)
                .map_err(|e| anyhow::anyhow!(e))?;
        let plan = ResourcePlanner::plan_for_model(model, engine_args.context_size)
            .map_err(|e| anyhow::anyhow!(e))?;
        print_resource_plan_summary(&estimate, &plan);
        Some(plan)
    } else {
        None
    };
    let effective_gpu_layers = if engine_args.cpu_only {
        0
    } else if let Some(plan) = &resource_plan {
        plan.n_gpu_layers
    } else {
        engine_args.gpu_layers
    };
    let effective_split_mode = if engine_args.cpu_only {
        GpuSplitMode::None
    } else if let Some(plan) = &resource_plan {
        plan.split_mode
    } else {
        engine_args.resolved_gpu_split_mode()
    };
    let effective_main_gpu = if engine_args.cpu_only {
        0
    } else if let Some(plan) = &resource_plan {
        plan.main_gpu
    } else {
        engine_args.main_gpu
    };
    let effective_tensor_split = if engine_args.cpu_only {
        None
    } else if let Some(plan) = &resource_plan {
        plan.tensor_split.clone()
    } else {
        engine_args.resolved_tensor_split()
    };
    let effective_mmap = resource_plan
        .as_ref()
        .map(|plan| plan.use_mmap)
        .unwrap_or_else(|| engine_args.resolved_mmap());
    let effective_mlock = resource_plan
        .as_ref()
        .map(|plan| plan.use_mlock)
        .unwrap_or_else(|| engine_args.resolved_mlock());
    let effective_kv_offload = if engine_args.cpu_only {
        false
    } else if let Some(plan) = &resource_plan {
        plan.kv_offload
    } else {
        engine_args.resolved_kv_offload()
    };
    let effective_op_offload = if engine_args.cpu_only {
        false
    } else if let Some(plan) = &resource_plan {
        plan.op_offload
    } else {
        engine_args.resolved_op_offload()
    };

    println!("Loading model from: {}", model.display());
    println!("Backend: {}", engine_args.backend);
    println!("Context size: {}", engine_args.context_size);
    println!("GPU layers: {}", effective_gpu_layers);
    println!(
        "GPU split: mode={:?} main_gpu={} tensor_split={}",
        effective_split_mode,
        effective_main_gpu,
        effective_tensor_split
            .as_ref()
            .map(|values| format_tensor_split(values))
            .unwrap_or_else(|| "auto".to_string())
    );
    println!(
        "Tiered loading: mmap={} mlock={} kv_offload={} op_offload={}",
        effective_mmap, effective_mlock, effective_kv_offload, effective_op_offload
    );
    println!(
        "GPU load fallback: {}",
        engine_args
            .resolved_gpu_fallback_step()
            .map(|step| format!("enabled (step={step})"))
            .unwrap_or_else(|| "disabled".to_string())
    );
    println!(
        "Auto resource plan: {}",
        if engine_args.resolved_auto_resource_plan() {
            "enabled"
        } else {
            "disabled"
        }
    );
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

    let execution_policy_store =
        load_dynamic_policy_registry(&engine_args.execution_policy_registry)?;
    let merged_execution_policy_plugins = merge_plugin_paths(
        execution_policy_store.plugins(),
        &engine_args.execution_policy_plugins,
    );
    let execution_policy_registry = Arc::new(build_execution_policy_registry(
        &merged_execution_policy_plugins,
    )?);
    let selected_execution_policy_name = engine_args
        .execution_policy_name
        .clone()
        .or_else(|| execution_policy_store.active().map(|name| name.to_string()))
        .unwrap_or_else(|| "default.execution.policy".to_string());
    let execution_policy = execution_policy_registry
        .get(&selected_execution_policy_name)
        .ok_or_else(|| {
            let available = execution_policy_registry.list_names().join(", ");
            anyhow::anyhow!(
                "unknown execution policy '{}'. available: {}",
                selected_execution_policy_name,
                available
            )
        })?;

    let mut selected_backend = engine_args.backend.clone();

    let mut builder = InferenceEngine::builder()
        .model_path(model)
        .backend(&selected_backend)
        .context_size(engine_args.context_size)
        .batch_size(512)
        .gpu_layers(engine_args.gpu_layers)
        .with_gpu_split_mode(engine_args.resolved_gpu_split_mode())
        .with_main_gpu(engine_args.main_gpu)
        .with_mmap(engine_args.resolved_mmap())
        .with_mlock(engine_args.resolved_mlock())
        .with_kv_offload(engine_args.resolved_kv_offload())
        .with_op_offload(engine_args.resolved_op_offload())
        .with_execution_policy_arc(execution_policy);

    if let Some(tensor_split) = engine_args.resolved_tensor_split() {
        builder = builder.with_tensor_split(tensor_split);
    }
    if let Some(resource_plan) = resource_plan {
        builder = builder.with_resource_plan(resource_plan);
    }
    if let Some(step) = engine_args.resolved_gpu_fallback_step() {
        builder = builder.with_auto_gpu_layer_fallback(step);
    }

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
    println!("  Execution policy: {}", selected_execution_policy_name);
    if !engine_args.lora_paths.is_empty() {
        println!(
            "Warning: --lora-path is accepted, but runtime LoRA merge is backend-dependent and not enabled in this CLI path yet."
        );
    }
    println!();

    Ok((
        engine,
        execution_policy_registry,
        selected_execution_policy_name,
    ))
}

fn parse_cli_from_config_file(path: &Path) -> anyhow::Result<Cli> {
    let args = read_cli_args_from_config(path)?;
    let mut argv = Vec::with_capacity(args.len() + 1);
    argv.push("loci".to_string());
    argv.extend(args);
    Cli::try_parse_from(argv).map_err(|e| {
        anyhow::anyhow!(
            "failed to parse CLI args from config '{}': {}",
            path.display(),
            e
        )
    })
}

fn dispatch_cli(cli: Cli, allow_config: bool) -> anyhow::Result<()> {
    if let Some(config_path) = cli.config.clone() {
        if !allow_config {
            return Err(anyhow::anyhow!(
                "nested --config is not allowed inside a config file"
            ));
        }

        let nested = parse_cli_from_config_file(&config_path)?;
        if nested.config.is_some() {
            return Err(anyhow::anyhow!(
                "config '{}' resolves to another --config; nesting is not allowed",
                config_path.display()
            ));
        }
        return dispatch_cli(nested, false);
    }

    match cli.command {
        Some(Commands::Generate(cmd)) => run_generate_command(cmd),
        Some(Commands::AuthPolicy(cmd)) => run_auth_policy_command(cmd),
        Some(Commands::DispatchPolicy(cmd)) => run_dispatch_policy_command(cmd),
        Some(Commands::ExecutionPolicy(cmd)) => run_execution_policy_command(cmd),
        Some(Commands::Session(cmd)) => run_session_command(cmd),
        Some(Commands::Image(cmd)) => run_image_command(cmd),
        Some(Commands::Multimodal(cmd)) => run_multimodal_command(cmd),
        Some(Commands::Orchestrate(cmd)) => run_orchestrate_command(cmd),
        Some(Commands::Serve(cmd)) => run_serve_command(cmd),
        Some(Commands::Agent(cmd)) => run_agent_command(cmd),
        Some(Commands::Plugin(cmd)) => run_plugin_command(cmd),
        Some(Commands::Mcp(cmd)) => run_mcp_command(cmd),
        Some(Commands::Model(cmd)) => run_model_command(cmd),
        None => run_legacy_mode(cli),
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    dispatch_cli(cli, true)
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
        gpu_split_mode: cli.gpu_split_mode.unwrap_or(GpuSplitModeArg::Layer),
        main_gpu: cli.main_gpu.unwrap_or(0),
        tensor_split: cli.tensor_split,
        auto_gpu_fallback: cli.auto_gpu_fallback,
        gpu_fallback_step: cli.gpu_fallback_step.unwrap_or(8),
        auto_resource_plan: cli.auto_resource_plan,
        mmap: cli.mmap,
        no_mmap: cli.no_mmap,
        mlock: cli.mlock,
        kv_offload: cli.kv_offload,
        no_kv_offload: cli.no_kv_offload,
        op_offload: cli.op_offload,
        no_op_offload: cli.no_op_offload,
        lora_paths: cli.lora_paths,
        execution_policy_plugins: Vec::new(),
        execution_policy_registry: PathBuf::from("loci_execution_policies.toml"),
        execution_policy_name: None,
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

    let (mut engine, _execution_policy_registry, _) =
        build_engine_with_execution_registry(&model, &engine_args)?;
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
    let model = resolve_model_reference(
        cmd.model.as_deref(),
        cmd.model_id.as_deref(),
        &cmd.model_store,
    )?;
    let (mut engine, _execution_policy_registry, _) =
        build_engine_with_execution_registry(&model, &cmd.engine)?;
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

fn run_auth_policy_command(cmd: AuthPolicyCmd) -> anyhow::Result<()> {
    let mut store = load_dynamic_policy_registry(&cmd.registry)?;
    let merged_plugins = merge_plugin_paths(store.plugins(), &cmd.plugins);
    let registry =
        build_management_auth_policy_registry(&merged_plugins, cmd.bearer_token.as_deref())?;

    match cmd.command {
        AuthPolicyAction::List => {
            for item in registry.descriptors() {
                println!(
                    "{}\tdynamic={}\tactive={}\tsource={}",
                    item.name,
                    item.dynamic,
                    store.active() == Some(item.name.as_str()),
                    item.source
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "-".to_string())
                );
            }
        }
        AuthPolicyAction::Info { name } => {
            let item = registry
                .describe(&name)
                .ok_or_else(|| anyhow::anyhow!("management auth policy '{}' not found", name))?;
            println!("name={}", item.name);
            println!("dynamic={}", item.dynamic);
            println!("active={}", store.active() == Some(item.name.as_str()));
            println!(
                "source={}",
                item.source
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
        AuthPolicyAction::Activate { name } => {
            if registry.get(&name).is_none() {
                return Err(anyhow::anyhow!(
                    "management auth policy '{}' not found",
                    name
                ));
            }
            store.set_active(Some(name.clone()));
            store.persist()?;
            println!("Active management auth policy: {}", name);
            println!("Registry: {}", cmd.registry.display());
        }
        AuthPolicyAction::Load { path } => {
            let name = registry.load_dynamic_policy(&path).map_err(|e| {
                anyhow::anyhow!(
                    "failed loading management auth policy plugin '{}': {}",
                    path.display(),
                    e
                )
            })?;
            store.add_plugin_path(path.clone());
            store.persist()?;
            let descriptor = registry.describe(&name).ok_or_else(|| {
                anyhow::anyhow!("loaded management auth policy '{}' missing", name)
            })?;
            println!("Loaded management auth policy: {}", descriptor.name);
            println!("Dynamic: {}", descriptor.dynamic);
            if let Some(source) = descriptor.source {
                println!("Source: {}", source.display());
            }
            println!("Registry: {}", cmd.registry.display());
        }
        AuthPolicyAction::Unload { name } => {
            if store.active() == Some(name.as_str()) {
                return Err(anyhow::anyhow!(
                    "management auth policy '{}' is active; activate another policy before unload",
                    name
                ));
            }
            let source = registry
                .describe(&name)
                .and_then(|item| item.source)
                .ok_or_else(|| anyhow::anyhow!("management auth policy '{}' not found", name))?;
            registry.unload_dynamic_policy(&name)?;
            store.remove_plugin_path(&source);
            store.persist()?;
            println!("Unloaded management auth policy: {}", name);
            println!("Registry: {}", cmd.registry.display());
        }
        AuthPolicyAction::Reload { name } => {
            registry.reload_dynamic_policy(&name)?;
            let item = registry.describe(&name).ok_or_else(|| {
                anyhow::anyhow!("management auth policy '{}' not found after reload", name)
            })?;
            println!("Reloaded management auth policy: {}", item.name);
            println!(
                "source={}",
                item.source
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
    }
    Ok(())
}

fn run_dispatch_policy_command(cmd: DispatchPolicyCmd) -> anyhow::Result<()> {
    let mut store = load_dynamic_policy_registry(&cmd.registry)?;
    let merged_plugins = merge_plugin_paths(store.plugins(), &cmd.plugins);
    let registry = build_dispatch_policy_registry(&merged_plugins)?;

    match cmd.command {
        DispatchPolicyAction::List => {
            for item in registry.descriptors() {
                println!(
                    "{}\tdynamic={}\tactive={}\tsource={}",
                    item.name,
                    item.dynamic,
                    store.active() == Some(item.name.as_str()),
                    item.source
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "-".to_string())
                );
            }
        }
        DispatchPolicyAction::Info { name } => {
            let item = registry
                .describe(&name)
                .ok_or_else(|| anyhow::anyhow!("dispatch policy '{}' not found", name))?;
            println!("name={}", item.name);
            println!("dynamic={}", item.dynamic);
            println!("active={}", store.active() == Some(item.name.as_str()));
            println!(
                "source={}",
                item.source
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
        DispatchPolicyAction::Activate { name } => {
            if registry.get(&name).is_none() {
                return Err(anyhow::anyhow!("dispatch policy '{}' not found", name));
            }
            store.set_active(Some(name.clone()));
            store.persist()?;
            println!("Active dispatch policy: {}", name);
            println!("Registry: {}", cmd.registry.display());
        }
        DispatchPolicyAction::Load { path } => {
            let name = registry.load_dynamic_policy(&path).map_err(|e| {
                anyhow::anyhow!(
                    "failed loading dispatch policy plugin '{}': {}",
                    path.display(),
                    e
                )
            })?;
            store.add_plugin_path(path.clone());
            store.persist()?;
            let descriptor = registry
                .describe(&name)
                .ok_or_else(|| anyhow::anyhow!("loaded dispatch policy '{}' missing", name))?;
            println!("Loaded dispatch policy: {}", descriptor.name);
            println!("Dynamic: {}", descriptor.dynamic);
            if let Some(source) = descriptor.source {
                println!("Source: {}", source.display());
            }
            println!("Registry: {}", cmd.registry.display());
        }
        DispatchPolicyAction::Unload { name } => {
            if store.active() == Some(name.as_str()) {
                return Err(anyhow::anyhow!(
                    "dispatch policy '{}' is active; activate another policy before unload",
                    name
                ));
            }
            let source = registry
                .describe(&name)
                .and_then(|item| item.source)
                .ok_or_else(|| anyhow::anyhow!("dispatch policy '{}' not found", name))?;
            registry.unload_dynamic_policy(&name)?;
            store.remove_plugin_path(&source);
            store.persist()?;
            println!("Unloaded dispatch policy: {}", name);
            println!("Registry: {}", cmd.registry.display());
        }
        DispatchPolicyAction::Reload { name } => {
            registry.reload_dynamic_policy(&name)?;
            let item = registry.describe(&name).ok_or_else(|| {
                anyhow::anyhow!("dispatch policy '{}' not found after reload", name)
            })?;
            println!("Reloaded dispatch policy: {}", item.name);
            println!(
                "source={}",
                item.source
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
    }
    Ok(())
}

fn run_execution_policy_command(cmd: ExecutionPolicyCmd) -> anyhow::Result<()> {
    let mut store = load_dynamic_policy_registry(&cmd.registry)?;
    let merged_plugins = merge_plugin_paths(store.plugins(), &cmd.plugins);
    let registry = build_execution_policy_registry(&merged_plugins)?;

    match cmd.command {
        ExecutionPolicyAction::List => {
            for item in registry.descriptors() {
                println!(
                    "{}\tdynamic={}\tactive={}\tsource={}",
                    item.name,
                    item.dynamic,
                    store.active() == Some(item.name.as_str()),
                    item.source
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "-".to_string())
                );
            }
        }
        ExecutionPolicyAction::Info { name } => {
            let item = registry
                .describe(&name)
                .ok_or_else(|| anyhow::anyhow!("execution policy '{}' not found", name))?;
            println!("name={}", item.name);
            println!("dynamic={}", item.dynamic);
            println!("active={}", store.active() == Some(item.name.as_str()));
            println!(
                "source={}",
                item.source
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
        ExecutionPolicyAction::Activate { name } => {
            if registry.get(&name).is_none() {
                return Err(anyhow::anyhow!("execution policy '{}' not found", name));
            }
            store.set_active(Some(name.clone()));
            store.persist()?;
            println!("Active execution policy: {}", name);
            println!("Registry: {}", cmd.registry.display());
        }
        ExecutionPolicyAction::Load { path } => {
            let name = registry.load_dynamic_policy(&path).map_err(|e| {
                anyhow::anyhow!(
                    "failed loading execution policy plugin '{}': {}",
                    path.display(),
                    e
                )
            })?;
            store.add_plugin_path(path.clone());
            store.persist()?;
            let descriptor = registry
                .describe(&name)
                .ok_or_else(|| anyhow::anyhow!("loaded execution policy '{}' missing", name))?;
            println!("Loaded execution policy: {}", descriptor.name);
            println!("Dynamic: {}", descriptor.dynamic);
            if let Some(source) = descriptor.source {
                println!("Source: {}", source.display());
            }
            println!("Registry: {}", cmd.registry.display());
        }
        ExecutionPolicyAction::Unload { name } => {
            if store.active() == Some(name.as_str()) {
                return Err(anyhow::anyhow!(
                    "execution policy '{}' is active; activate another policy before unload",
                    name
                ));
            }
            let source = registry
                .describe(&name)
                .and_then(|item| item.source)
                .ok_or_else(|| anyhow::anyhow!("execution policy '{}' not found", name))?;
            registry.unload_dynamic_policy(&name)?;
            store.remove_plugin_path(&source);
            store.persist()?;
            println!("Unloaded execution policy: {}", name);
            println!("Registry: {}", cmd.registry.display());
        }
        ExecutionPolicyAction::Reload { name } => {
            registry.reload_dynamic_policy(&name)?;
            let item = registry.describe(&name).ok_or_else(|| {
                anyhow::anyhow!("execution policy '{}' not found after reload", name)
            })?;
            println!("Reloaded execution policy: {}", item.name);
            println!(
                "source={}",
                item.source
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "-".to_string())
            );
        }
    }
    Ok(())
}

fn resolve_model_reference(
    model: Option<&Path>,
    model_id: Option<&str>,
    model_store_root: &Path,
) -> anyhow::Result<PathBuf> {
    match (model, model_id) {
        (Some(path), None) => Ok(path.to_path_buf()),
        (None, Some(id)) => {
            let store = ModelStore::new(model_store_root);
            let item = store
                .get(id)
                .map_err(|e| anyhow::anyhow!("failed resolving model id '{}': {}", id, e))?;
            if !item.path.exists() {
                return Err(anyhow::anyhow!(
                    "model '{}' points to missing file: {}",
                    id,
                    item.path.display()
                ));
            }
            Ok(item.path)
        }
        (Some(_), Some(_)) => Err(anyhow::anyhow!(
            "provide either --model or --model-id, not both"
        )),
        (None, None) => Err(anyhow::anyhow!(
            "missing model reference: provide --model <PATH> or --model-id <ID>"
        )),
    }
}

fn resolve_model_candidates(
    models: &[PathBuf],
    model_ids: &[String],
    model_store_root: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for model in models {
        let key = model.to_string_lossy().to_string();
        if seen.insert(key) {
            out.push(model.clone());
        }
    }

    for model_id in model_ids {
        let resolved = resolve_model_reference(None, Some(model_id), model_store_root)?;
        let key = resolved.to_string_lossy().to_string();
        if seen.insert(key) {
            out.push(resolved);
        }
    }

    if out.is_empty() {
        return Err(anyhow::anyhow!(
            "no candidate model provided: use --model and/or --model-id"
        ));
    }
    Ok(out)
}

struct ImageGenerationRuntimeOptions<'a> {
    prompt: &'a str,
    model_id: &'a str,
    output: &'a Path,
    steps: u32,
    guidance_scale: f32,
    width: Option<u32>,
    height: Option<u32>,
    seed: Option<u64>,
    kernel_plugin: Option<&'a Path>,
    python: &'a str,
    use_cuda: bool,
}

fn run_image_generation(options: &ImageGenerationRuntimeOptions<'_>) -> anyhow::Result<()> {
    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let device = if options.use_cuda { "cuda" } else { "cpu" }.to_string();
    if let Some(kernel_plugin) = options.kernel_plugin {
        let kernel = load_dynamic_image_plugin(kernel_plugin).map_err(|e| anyhow::anyhow!(e))?;
        let request = ImageGenerationRequest {
            prompt: options.prompt.to_string(),
            model_id: options.model_id.to_string(),
            steps: options.steps,
            guidance_scale: options.guidance_scale,
            width: options.width,
            height: options.height,
            seed: options.seed,
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
        fs::write(options.output, &result.image_bytes)?;
        return Ok(());
    }

    let script_path = PathBuf::from("scripts").join("t2i_generate.py");
    if !script_path.exists() {
        return Err(anyhow::anyhow!(
            "missing image generation script: {}",
            script_path.display()
        ));
    }

    let mut child = Command::new(options.python);
    child
        .arg(&script_path)
        .arg("--prompt")
        .arg(options.prompt)
        .arg("--model-id")
        .arg(options.model_id)
        .arg("--output")
        .arg(options.output)
        .arg("--steps")
        .arg(options.steps.to_string())
        .arg("--guidance-scale")
        .arg(options.guidance_scale.to_string());

    if let Some(width) = options.width {
        child.arg("--width").arg(width.to_string());
    }
    if let Some(height) = options.height {
        child.arg("--height").arg(height.to_string());
    }
    if let Some(seed) = options.seed {
        child.arg("--seed").arg(seed.to_string());
    }
    child.arg("--device").arg(device);

    let status = child.status()?;
    if !status.success() {
        return Err(anyhow::anyhow!(
            "text-to-image script failed with status: {status}"
        ));
    }
    Ok(())
}

fn run_image_command(cmd: ImageCmd) -> anyhow::Result<()> {
    println!("Text-to-image mode");
    println!("Prompt: {}", cmd.prompt);
    println!("Model: {}", cmd.model_id);
    println!("Output: {}", cmd.output.display());
    if let Some(kernel_plugin) = &cmd.kernel_plugin {
        println!("Kernel plugin: {}", kernel_plugin.display());
    }

    run_image_generation(&ImageGenerationRuntimeOptions {
        prompt: &cmd.prompt,
        model_id: &cmd.model_id,
        output: &cmd.output,
        steps: cmd.steps,
        guidance_scale: cmd.guidance_scale,
        width: cmd.width,
        height: cmd.height,
        seed: cmd.seed,
        kernel_plugin: cmd.kernel_plugin.as_deref(),
        python: &cmd.python,
        use_cuda: cmd.use_cuda,
    })?;
    println!("Image generation completed: {}", cmd.output.display());
    Ok(())
}

struct MultimodalArtifactRuntimeOptions<'a> {
    artifact_dir: &'a Path,
    image_model_id: &'a str,
    image_steps: u32,
    image_guidance_scale: f32,
    image_width: Option<u32>,
    image_height: Option<u32>,
    image_seed: Option<u64>,
    image_kernel_plugin: Option<&'a Path>,
    python: &'a str,
    use_cuda: bool,
}

fn get_mm_plugin<'a>(
    mm_registry: &'a MultimodalIoRegistry,
    plugin_name: &str,
) -> anyhow::Result<&'a dyn MultimodalIoPlugin> {
    mm_registry.get(plugin_name).ok_or_else(|| {
        let available = mm_registry
            .list()
            .into_iter()
            .map(|(name, _, enabled, _)| {
                if enabled {
                    name
                } else {
                    format!("{name}(disabled)")
                }
            })
            .collect::<Vec<_>>();
        anyhow::anyhow!(
            "multimodal I/O plugin '{}' not found or disabled. available: {}",
            plugin_name,
            available.join(", ")
        )
    })
}

fn materialize_multimodal_outputs(
    request: &MultimodalRequest,
    plan: &MultimodalOutputPlan,
    options: &MultimodalArtifactRuntimeOptions<'_>,
) -> anyhow::Result<()> {
    let wants_image = request.wants_image_output();
    let wants_audio = request.wants_audio_output();
    if wants_image || wants_audio {
        fs::create_dir_all(options.artifact_dir)?;
    }

    if wants_image {
        let image_prompts = if plan.image_prompts.is_empty() {
            vec![plan.text_response.clone()]
        } else {
            plan.image_prompts.clone()
        };
        for (idx, image_prompt) in image_prompts.iter().enumerate() {
            let path = options.artifact_dir.join(format!("image_{idx:02}.png"));
            run_image_generation(&ImageGenerationRuntimeOptions {
                prompt: image_prompt,
                model_id: options.image_model_id,
                output: &path,
                steps: options.image_steps,
                guidance_scale: options.image_guidance_scale,
                width: options.image_width,
                height: options.image_height,
                seed: options.image_seed,
                kernel_plugin: options.image_kernel_plugin,
                python: options.python,
                use_cuda: options.use_cuda,
            })?;
            println!("Image output generated: {}", path.display());
        }
    }

    if wants_audio {
        let audio_prompts = if plan.audio_prompts.is_empty() {
            vec![plan.text_response.clone()]
        } else {
            plan.audio_prompts.clone()
        };
        for (idx, audio_prompt) in audio_prompts.iter().enumerate() {
            let path = options.artifact_dir.join(format!("audio_{idx:02}.txt"));
            fs::write(&path, audio_prompt)?;
            println!(
                "Audio output plan saved: {} (text prompt for downstream TTS plugin)",
                path.display()
            );
        }
    }

    Ok(())
}

fn run_multimodal_command(cmd: MultimodalCmd) -> anyhow::Result<()> {
    let model = resolve_model_reference(
        cmd.model.as_deref(),
        cmd.model_id.as_deref(),
        &cmd.model_store,
    )?;
    println!("Multimodal mode");
    println!("Model: {}", model.display());
    println!("MM plugin: {}", cmd.mm_plugin_name);

    let plugins = load_plugins(&cmd.plugins)?;
    let (mut engine, _execution_policy_registry, _) =
        build_engine_with_execution_registry(&model, &cmd.engine)?;
    let mut mm_registry = MultimodalIoRegistry::with_builtin_plugins();

    for plugin_path in &cmd.mm_plugins {
        let name = mm_registry.load_dynamic_plugin(plugin_path)?;
        println!(
            "Loaded multimodal I/O plugin: {} ({})",
            name,
            plugin_path.display()
        );
    }

    let plugin = get_mm_plugin(&mm_registry, &cmd.mm_plugin_name)?;

    let request = MultimodalRequest {
        prompt: cmd.prompt.clone(),
        image_inputs: cmd.image_inputs.clone(),
        audio_inputs: cmd.audio_inputs.clone(),
        output_modalities: cmd.output_modalities.iter().map(|m| (*m).into()).collect(),
    };

    let prompt = plugin.prepare_prompt(&request)?;
    let prompt = apply_pre_generate(&prompt, plugins.as_ref())?;
    let response = engine.generate(&prompt, to_generation_params(&cmd.sampling))?;
    let response = apply_post_generate(&response, plugins.as_ref())?;
    let mut plan = plugin.interpret_response(&request, &response)?;

    if plan.text_response.trim().is_empty() {
        plan.text_response = response.trim().to_string();
    }

    println!("Prompt: {}", prompt);
    println!("\nText Output:");
    println!("---");
    println!("{}", plan.text_response);
    println!("---");

    materialize_multimodal_outputs(
        &request,
        &plan,
        &MultimodalArtifactRuntimeOptions {
            artifact_dir: &cmd.artifact_dir,
            image_model_id: &cmd.image_model_id,
            image_steps: cmd.image_steps,
            image_guidance_scale: cmd.image_guidance_scale,
            image_width: cmd.image_width,
            image_height: cmd.image_height,
            image_seed: cmd.image_seed,
            image_kernel_plugin: cmd.image_kernel_plugin.as_deref(),
            python: &cmd.python,
            use_cuda: cmd.use_cuda,
        },
    )?;

    Ok(())
}

fn run_orchestrate_command(cmd: OrchestrateCmd) -> anyhow::Result<()> {
    let candidate_models = resolve_model_candidates(&cmd.models, &cmd.model_ids, &cmd.model_store)?;
    println!("Orchestrate mode");
    println!("Mode: {:?}", cmd.mode);
    println!("MM plugin: {}", cmd.mm_plugin_name);
    println!("Model candidates: {}", candidate_models.len());

    if let Some(limit) = cmd.max_prompt_bytes {
        std::env::set_var("LOCI_MAX_PROMPT_BYTES", limit.to_string());
        println!("Max prompt bytes: {} (from --max-prompt-bytes)", limit);
    }

    let plugins = load_plugins(&cmd.plugins)?;
    let mut mm_registry = MultimodalIoRegistry::with_builtin_plugins();
    for plugin_path in &cmd.mm_plugins {
        let name = mm_registry.load_dynamic_plugin(plugin_path)?;
        println!(
            "Loaded multimodal I/O plugin: {} ({})",
            name,
            plugin_path.display()
        );
    }
    let plugin = get_mm_plugin(&mm_registry, &cmd.mm_plugin_name)?;

    let request = MultimodalRequest {
        prompt: cmd.prompt.clone(),
        image_inputs: cmd.image_inputs.clone(),
        audio_inputs: cmd.audio_inputs.clone(),
        output_modalities: cmd.output_modalities.iter().map(|m| (*m).into()).collect(),
    };

    let prompt = plugin.prepare_prompt(&request)?;
    let prompt = apply_pre_generate(&prompt, plugins.as_ref())?;

    let model_registry = ModelRegistry::new();
    let mut model_ids = Vec::with_capacity(candidate_models.len());
    for model_path in &candidate_models {
        let model_id = model_registry.load_model(model_path, cmd.context_size)?;
        println!(
            "Registered candidate model {}: {}",
            model_id,
            model_path.display()
        );
        model_ids.push(model_id);
    }

    let raw_response = match cmd.mode {
        OrchestrationModeArg::Route => {
            let strategy = match cmd.routing_strategy {
                RoutingStrategyArg::FirstHealthy => ModelRoutingStrategy::FirstHealthy,
                RoutingStrategyArg::RoundRobin => ModelRoutingStrategy::RoundRobin,
                RoutingStrategyArg::FastestProbe => ModelRoutingStrategy::FastestProbe {
                    probe_prompt: cmd.probe_prompt.clone(),
                    probe_max_tokens: cmd.probe_max_tokens as usize,
                },
            };

            let routed = model_registry.generate_routed(
                &model_ids,
                &prompt,
                cmd.max_tokens as usize,
                strategy,
            )?;
            println!(
                "Routed selected model: {} (attempts={})",
                routed.selected_model,
                routed.attempts.len()
            );
            for attempt in &routed.attempts {
                if attempt.success {
                    println!(
                        "  attempt model={} status=ok latency={}ms",
                        attempt.model_id, attempt.latency_ms
                    );
                } else {
                    println!(
                        "  attempt model={} status=fail latency={}ms error={}",
                        attempt.model_id,
                        attempt.latency_ms,
                        attempt
                            .error
                            .clone()
                            .unwrap_or_else(|| "unknown".to_string())
                    );
                }
            }
            routed.response
        }
        OrchestrationModeArg::Ensemble => {
            let merge_strategy = match cmd.ensemble_merge {
                EnsembleMergeArg::Concatenate => EnsembleMergeStrategy::Concatenate,
                EnsembleMergeArg::Longest => EnsembleMergeStrategy::Longest,
                EnsembleMergeArg::Judge => EnsembleMergeStrategy::Judge,
            };

            let judge_model_id = if merge_strategy == EnsembleMergeStrategy::Judge {
                let judge_path = if let Some(path) = cmd.judge_model.as_ref() {
                    Some(path.clone())
                } else if let Some(id) = cmd.judge_model_id.as_deref() {
                    Some(resolve_model_reference(None, Some(id), &cmd.model_store)?)
                } else {
                    None
                };

                if let Some(judge_path) = judge_path.as_ref() {
                    if let Some((idx, _)) = candidate_models
                        .iter()
                        .enumerate()
                        .find(|(_, model_path)| *model_path == judge_path)
                    {
                        Some(model_ids[idx])
                    } else {
                        Some(model_registry.load_model(judge_path, cmd.context_size)?)
                    }
                } else {
                    None
                }
            } else {
                if cmd.judge_model.is_some() || cmd.judge_model_id.is_some() {
                    println!(
                        "Warning: --judge-model/--judge-model-id is ignored unless --ensemble-merge judge is selected."
                    );
                }
                None
            };

            let ensemble = model_registry.generate_ensemble(
                &model_ids,
                &prompt,
                cmd.max_tokens as usize,
                merge_strategy,
                judge_model_id,
            )?;
            println!(
                "Ensemble completed: candidates_ok={}, candidates_failed={}",
                ensemble.candidates.len(),
                ensemble.failures.len()
            );
            if let Some(judge_model) = ensemble.judge_model {
                println!("Ensemble judge model: {}", judge_model);
            }
            ensemble.final_response
        }
    };

    let response = apply_post_generate(&raw_response, plugins.as_ref())?;
    let mut plan = plugin.interpret_response(&request, &response)?;
    if plan.text_response.trim().is_empty() {
        plan.text_response = response.trim().to_string();
    }

    println!("Prompt: {}", prompt);
    println!("\nText Output:");
    println!("---");
    println!("{}", plan.text_response);
    println!("---");

    materialize_multimodal_outputs(
        &request,
        &plan,
        &MultimodalArtifactRuntimeOptions {
            artifact_dir: &cmd.artifact_dir,
            image_model_id: &cmd.image_model_id,
            image_steps: cmd.image_steps,
            image_guidance_scale: cmd.image_guidance_scale,
            image_width: cmd.image_width,
            image_height: cmd.image_height,
            image_seed: cmd.image_seed,
            image_kernel_plugin: cmd.image_kernel_plugin.as_deref(),
            python: &cmd.python,
            use_cuda: cmd.use_cuda,
        },
    )?;

    Ok(())
}

fn load_mcp_registry(path: &Path) -> anyhow::Result<McpServerRegistry> {
    let mut registry = McpServerRegistry::with_config_path(path);
    if path.exists() {
        registry.load_from_file(path)?;
    }
    Ok(registry)
}

fn run_mcp_command(cmd: McpCmd) -> anyhow::Result<()> {
    let mut registry = load_mcp_registry(&cmd.registry)?;
    match cmd.command {
        McpAction::Connect {
            spec,
            tool_prefix,
            probe,
            save,
        } => {
            let mut stdio_cfg = parse_mcp_stdio_spec(&spec)
                .map_err(|e| anyhow::anyhow!("invalid MCP spec '{}': {}", spec, e))?;
            stdio_cfg.tool_prefix = tool_prefix;

            if probe {
                let mut client = StdioMcpClient::connect(stdio_cfg.clone())?;
                let tools = McpClient::list_tools(&mut client)?;
                println!(
                    "MCP probe ok: server='{}', tools={}",
                    stdio_cfg.server_name,
                    tools.len()
                );
                if !tools.is_empty() {
                    println!(
                        "Tool names: {}",
                        tools
                            .into_iter()
                            .map(|t| t.name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            } else {
                println!("MCP probe skipped for '{}'", stdio_cfg.server_name);
            }

            if save {
                registry.upsert(McpServerConfig::from(stdio_cfg.clone()))?;
                registry.save_to_file(&cmd.registry)?;
                println!(
                    "Saved MCP server '{}' to {}",
                    stdio_cfg.server_name,
                    cmd.registry.display()
                );
            }
        }
        McpAction::Disconnect { name } => {
            registry.remove(&name)?;
            registry.save_to_file(&cmd.registry)?;
            println!("Removed MCP server: {name}");
        }
        McpAction::Enable { name } => {
            registry.enable(&name)?;
            registry.save_to_file(&cmd.registry)?;
            println!("Enabled MCP server: {name}");
        }
        McpAction::Disable { name } => {
            registry.disable(&name)?;
            registry.save_to_file(&cmd.registry)?;
            println!("Disabled MCP server: {name}");
        }
        McpAction::List => {
            let items = registry.list();
            if items.is_empty() {
                println!("No MCP servers configured.");
            } else {
                println!("name\tenabled\tcommand\targs\ttool_prefix");
                for item in items {
                    println!(
                        "{}\t{}\t{}\t{}\t{}",
                        item.name,
                        item.enabled,
                        item.command,
                        item.args.join(" "),
                        item.tool_prefix.clone().unwrap_or_else(|| "-".to_string())
                    );
                }
            }
        }
        McpAction::Status => {
            let items = registry.list();
            if items.is_empty() {
                println!("No MCP servers configured.");
            } else {
                println!("name\tenabled\tstatus\ttools\terror");
                for item in items {
                    if !item.enabled {
                        println!("{}\tfalse\tskipped\t-\t-", item.name);
                        continue;
                    }
                    let stdio_cfg = item.to_stdio_config();
                    match StdioMcpClient::connect(stdio_cfg.clone()) {
                        Ok(mut client) => match McpClient::list_tools(&mut client) {
                            Ok(tools) => println!("{}\ttrue\tok\t{}\t-", item.name, tools.len()),
                            Err(err) => println!("{}\ttrue\tdegraded\t-\t{}", item.name, err),
                        },
                        Err(err) => println!("{}\ttrue\terror\t-\t{}", item.name, err),
                    }
                }
            }
        }
    }
    Ok(())
}

fn run_agent_command(cmd: AgentCmd) -> anyhow::Result<()> {
    let model = resolve_model_reference(
        cmd.model.as_deref(),
        cmd.model_id.as_deref(),
        &cmd.model_store,
    )?;
    println!("Agent mode");
    println!("Model: {}", model.display());
    println!("Tool: {}", cmd.tool);
    if let Some(skill) = &cmd.skill {
        println!("Skill: {}", skill);
    }
    let plugins = load_plugins(&cmd.plugins)?;
    let (mut engine, _execution_policy_registry, _) =
        build_engine_with_execution_registry(&model, &cmd.engine)?;

    for tool_plugin in &cmd.tool_plugins {
        let (name, functions) = engine.load_dynamic_tool_plugin(tool_plugin)?;
        println!("Loaded tool plugin: {} ({})", name, tool_plugin.display());
        if !functions.is_empty() {
            println!("Plugin tools: {}", functions.join(", "));
        }
    }

    for skill_pack in &cmd.skill_packs {
        let loaded = engine
            .skill_registry_mut()
            .load_pack_from_file(skill_pack)?;
        println!(
            "Loaded skill pack: {} (skills: {})",
            skill_pack.display(),
            loaded.join(", ")
        );
    }

    if let Some(registry_path) = &cmd.mcp_registry {
        let registry = load_mcp_registry(registry_path)?;
        let mut selected = Vec::new();
        if cmd.mcp_servers.is_empty() {
            selected.extend(registry.list_enabled().into_iter().cloned());
        } else {
            for name in &cmd.mcp_servers {
                let server = registry.get(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "MCP server '{}' not found in {}",
                        name,
                        registry_path.display()
                    )
                })?;
                if !server.enabled {
                    return Err(anyhow::anyhow!(
                        "MCP server '{}' is disabled in {}",
                        name,
                        registry_path.display()
                    ));
                }
                selected.push(server.clone());
            }
        }

        for server in selected {
            let report = engine.connect_mcp_stdio_server(server.to_stdio_config())?;
            println!(
                "Connected MCP server '{}' from registry; registered {} tool(s)",
                report.server_name,
                report.registered_tools.len()
            );
        }
    }

    for spec in &cmd.mcp_stdio {
        let config = parse_mcp_stdio_spec(spec)
            .map_err(|e| anyhow::anyhow!("invalid --mcp-stdio '{}': {}", spec, e))?;
        let command = config.command.clone();
        let report = engine.connect_mcp_stdio_server(config)?;
        println!(
            "Connected MCP server '{}' via '{}'; registered {} tool(s)",
            report.server_name,
            command,
            report.registered_tools.len()
        );
        if !report.registered_tools.is_empty() {
            println!("MCP tools: {}", report.registered_tools.join(", "));
        }
    }

    let selected_skill = if let Some(skill_name) = cmd.skill.as_deref() {
        Some(
            engine
                .skill_registry()
                .get(skill_name)
                .cloned()
                .ok_or_else(|| {
                    let available = engine.skill_registry().list_names();
                    let available = if available.is_empty() {
                        "<none>".to_string()
                    } else {
                        available.join(", ")
                    };
                    anyhow::anyhow!(
                        "unknown skill '{}'. available skills: {}",
                        skill_name,
                        available
                    )
                })?,
        )
    } else {
        None
    };

    let base_prompt = if let Some(skill) = &selected_skill {
        skill.compose_prompt(&cmd.prompt)
    } else {
        cmd.prompt.clone()
    };

    if cmd.tool.eq_ignore_ascii_case("none") {
        run_single_prompt(
            &mut engine,
            &base_prompt,
            to_generation_params(&cmd.sampling),
            cmd.stream,
            plugins.as_ref(),
        )?;
        return Ok(());
    }

    let tools = engine
        .function_calling_manager()
        .list_functions()
        .iter()
        .map(|f| f.name.clone())
        .collect::<Vec<_>>();

    if !cmd.tool.eq_ignore_ascii_case("all")
        && engine
            .function_calling_manager()
            .get_function(&cmd.tool)
            .is_none()
    {
        return Err(anyhow::anyhow!(
            "unknown tool `{}`. available tools: {}",
            cmd.tool,
            tools.join(", ")
        ));
    }

    let cli_allowlist = if cmd.tool.eq_ignore_ascii_case("all") {
        None
    } else {
        Some(vec![cmd.tool.clone()])
    };
    let skill_allowlist = selected_skill.as_ref().and_then(|skill| {
        if skill.tool_policy.allowed.is_empty() {
            None
        } else {
            Some(skill.tool_policy.allowed.clone())
        }
    });
    let blocked_tools = selected_skill.as_ref().and_then(|skill| {
        if skill.tool_policy.blocked.is_empty() {
            None
        } else {
            Some(skill.tool_policy.blocked.clone())
        }
    });
    let effective_allowlist = merge_tool_allowlists(cli_allowlist, skill_allowlist);

    if matches!(effective_allowlist.as_ref(), Some(v) if v.is_empty()) {
        println!("Tool allowlist intersection is empty; running without tool execution.");
    }

    let mut tool_prompt = String::new();
    if cmd.tool.eq_ignore_ascii_case("all") {
        tool_prompt.push_str("You may use any available tool allowed by active policy.\n\n");
    } else {
        tool_prompt.push_str(&format!(
            "Prefer using tool `{}` when it is relevant.\n\n",
            cmd.tool
        ));
    }
    tool_prompt.push_str(&base_prompt);
    let tool_prompt = apply_pre_generate(&tool_prompt, plugins.as_ref())?;

    if cmd.stream {
        println!(
            "Streaming is disabled in tool-calling mode; falling back to non-streaming output."
        );
    }

    let rounds = selected_skill
        .as_ref()
        .and_then(|skill| skill.max_tool_rounds)
        .unwrap_or(4);
    let inference_params: InferenceParams = to_generation_params(&cmd.sampling).into();
    let response = engine.generate_with_tools_policy(
        &tool_prompt,
        &inference_params,
        rounds,
        effective_allowlist.as_deref(),
        blocked_tools.as_deref(),
    )?;
    let response = apply_post_generate(&response, plugins.as_ref())?;

    println!("Prompt: {}", tool_prompt);
    println!("\nResponse:");
    println!("---");
    println!("{response}");
    println!("---");
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

fn run_model_command(cmd: ModelCmd) -> anyhow::Result<()> {
    let store = ModelStore::new(&cmd.store);
    match cmd.command {
        ModelAction::Plan {
            model,
            model_id,
            context_size,
            json,
        } => {
            let resolved =
                resolve_model_reference(model.as_deref(), model_id.as_deref(), &cmd.store)?;
            let estimate = ResourcePlanner::estimate_model_requirements(&resolved, context_size)
                .map_err(|e| anyhow::anyhow!(e))?;
            let selector = DeviceSelector::new();
            let plan = ResourcePlanner::plan_for_estimate(&estimate, selector.devices());
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&build_resource_plan_report(
                        &resolved,
                        estimate,
                        plan,
                        selector.devices(),
                    ))?
                );
            } else {
                println!("Model: {}", resolved.display());
                print_resource_plan_summary(&estimate, &plan);
            }
        }
        ModelAction::Add {
            path,
            id,
            name,
            tags,
        } => {
            let item = store.add_external(path, id, name, tags)?;
            println!("Registered model: {}", item.id);
            println!("  Name: {}", item.name);
            println!("  Path: {}", item.path.display());
            println!("  Managed: {}", item.managed);
            println!("  Size: {} bytes", item.size_bytes);
            println!("  Checksum(xxh64): {}", item.checksum_xxh64);
            if let Some(sha256) = &item.checksum_sha256 {
                println!("  Checksum(sha256): {}", sha256);
            }
        }
        ModelAction::Pull {
            source,
            mirrors,
            id,
            name,
            sha256,
            no_resume,
            tags,
        } => {
            let options = ModelPullOptions {
                mirrors,
                expected_sha256: sha256,
                resume: !no_resume,
            };
            let item = store.pull_from_source_with_options(&source, id, name, tags, options)?;
            println!("Imported model: {}", item.id);
            println!("  Name: {}", item.name);
            println!("  Path: {}", item.path.display());
            println!("  Managed: {}", item.managed);
            println!("  Size: {} bytes", item.size_bytes);
            println!("  Checksum(xxh64): {}", item.checksum_xxh64);
            if let Some(sha256) = &item.checksum_sha256 {
                println!("  Checksum(sha256): {}", sha256);
            }
        }
        ModelAction::List { json } => {
            let list = store.list()?;
            if json {
                println!("{}", serde_json::to_string_pretty(&list)?);
            } else if list.is_empty() {
                println!("No models in store: {}", store.root().display());
            } else {
                println!("id\tmanaged\tsize_bytes\tname\tpath");
                for item in list {
                    println!(
                        "{}\t{}\t{}\t{}\t{}",
                        item.id,
                        item.managed,
                        item.size_bytes,
                        item.name,
                        item.path.display()
                    );
                }
            }
        }
        ModelAction::Info { id } => {
            let item = store.get(&id)?;
            println!("{}", serde_json::to_string_pretty(&item)?);
        }
        ModelAction::Remove { id, delete_file } => {
            let removed = store.remove(&id, delete_file)?;
            println!("Removed model: {}", removed.id);
            println!("  Deleted file: {}", delete_file);
            println!("  Path: {}", removed.path.display());
        }
    }
    Ok(())
}

fn parse_store_options(raw_options: &[String]) -> anyhow::Result<HashMap<String, String>> {
    let mut options = HashMap::new();
    for item in raw_options {
        let Some((key, value)) = item.split_once('=') else {
            return Err(anyhow::anyhow!(
                "invalid --store-option '{}': expected key=value",
                item
            ));
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(anyhow::anyhow!(
                "invalid --store-option '{}': empty key",
                item
            ));
        }
        options.insert(key.to_string(), value.trim().to_string());
    }
    Ok(options)
}

fn build_session_manager_from_options(
    store_kind: &str,
    raw_store_options: &[String],
    store_plugin: Option<&Path>,
) -> anyhow::Result<SessionManager> {
    let mut options = parse_store_options(raw_store_options)?;
    if store_kind.eq_ignore_ascii_case("sqlite")
        && !options.contains_key("path")
        && !options.contains_key("db_path")
    {
        options.insert("path".to_string(), "sessions/loci_sessions.db".to_string());
    }

    if let Some(plugin_path) = store_plugin {
        let registry = SessionStoreRegistry::with_builtin_factories();
        let dynamic_kind = registry.load_dynamic_factory(plugin_path).map_err(|e| {
            anyhow::anyhow!(
                "failed loading session store plugin '{}': {}",
                plugin_path.display(),
                e
            )
        })?;
        let target_kind = if store_kind.eq_ignore_ascii_case("auto") {
            dynamic_kind
        } else {
            store_kind.to_string()
        };
        return SessionManager::with_store_plugin_from_registry(&registry, &target_kind, options)
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed creating session manager with store kind '{}': {}",
                    target_kind,
                    e
                )
            });
    }

    SessionManager::with_store_plugin(store_kind, options).map_err(|e| {
        anyhow::anyhow!(
            "failed creating session manager with store kind '{}': {}",
            store_kind,
            e
        )
    })
}

fn build_session_manager(cmd: &SessionCmd) -> anyhow::Result<SessionManager> {
    build_session_manager_from_options(
        &cmd.store_kind,
        &cmd.store_options,
        cmd.store_plugin.as_deref(),
    )
}

fn ensure_session_loaded(manager: &SessionManager, session_id: SessionId) -> anyhow::Result<()> {
    if manager.has_session(session_id) {
        return Ok(());
    }
    manager.restore_session(session_id).map_err(|e| {
        anyhow::anyhow!(
            "failed restoring session {} from store before operation: {}",
            session_id.as_u64(),
            e
        )
    })
}

fn run_session_command(cmd: SessionCmd) -> anyhow::Result<()> {
    let manager = build_session_manager(&cmd)?;
    match cmd.command {
        SessionAction::Create {
            model,
            model_id,
            context_size,
            no_save,
        } => {
            let resolved =
                resolve_model_reference(model.as_deref(), model_id.as_deref(), &cmd.model_store)?;
            let model_id = manager.load_model(resolved.to_string_lossy().as_ref(), context_size)?;
            let session_id = manager.create_session(model_id)?;
            if !no_save {
                manager.save_session(session_id)?;
            }
            println!("Created session: {}", session_id.as_u64());
            println!("  Model path: {}", resolved.display());
            println!("  Model id: {}", model_id);
            println!("  Context size: {}", context_size);
            println!("  Persisted: {}", !no_save);
        }
        SessionAction::Generate {
            session_id,
            prompt,
            max_tokens,
            no_save,
        } => {
            let session_id = SessionId::from(session_id);
            ensure_session_loaded(&manager, session_id)?;
            let handle = manager
                .get_session(session_id)
                .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
            let response = handle.generate(&prompt, max_tokens as usize)?;
            if !no_save {
                manager.save_session(session_id)?;
            }
            println!("Session: {}", session_id.as_u64());
            println!("Prompt: {}", prompt);
            println!("\nResponse:");
            println!("---");
            println!("{response}");
            println!("---");
            println!("Persisted: {}", !no_save);
        }
        SessionAction::Suspend {
            session_id,
            reason,
            data,
            no_save,
        } => {
            let session_id = SessionId::from(session_id);
            ensure_session_loaded(&manager, session_id)?;
            let handle = manager
                .get_session(session_id)
                .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
            handle.suspend(reason.clone(), data.clone())?;
            if !no_save {
                manager.save_session(session_id)?;
            }
            println!("Suspended session: {}", session_id.as_u64());
            println!("  Reason: {}", reason);
            println!("  Data: {}", data.unwrap_or_else(|| "-".to_string()));
            println!("  Persisted: {}", !no_save);
        }
        SessionAction::Resume {
            session_id,
            external_data,
            no_save,
        } => {
            let session_id = SessionId::from(session_id);
            ensure_session_loaded(&manager, session_id)?;
            let handle = manager
                .get_session(session_id)
                .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
            handle.resume(external_data.clone())?;
            if !no_save {
                manager.save_session(session_id)?;
            }
            println!("Resumed session: {}", session_id.as_u64());
            println!("  External data: {}", external_data);
            println!("  Persisted: {}", !no_save);
        }
        SessionAction::Info {
            session_id,
            with_records,
        } => {
            let session_id = SessionId::from(session_id);
            ensure_session_loaded(&manager, session_id)?;
            let handle = manager
                .get_session(session_id)
                .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
            let info = handle.info()?;
            println!("session_id: {}", info.session_id.as_u64());
            println!("model_id: {}", info.model_id);
            println!("context_length: {}", info.context_length);
            println!("max_context: {}", info.max_context);
            println!("state: {:?}", info.state);
            println!("message_count: {}", info.message_count);
            if with_records {
                let records = handle.records()?;
                if records.is_empty() {
                    println!("records: <empty>");
                } else {
                    println!("records:");
                    for (idx, record) in records.iter().enumerate() {
                        println!(
                            "  {}. {}: {}",
                            idx + 1,
                            format!("{:?}", record.role).to_lowercase(),
                            record.content
                        );
                    }
                }
            }
        }
        SessionAction::List { active, persisted } => {
            if active {
                let active_sessions = manager.list_sessions();
                if active_sessions.is_empty() {
                    println!("active sessions: <empty>");
                } else {
                    println!("active sessions:");
                    for session in active_sessions {
                        println!(
                            "  id={} model_id={} state={:?} messages={} ctx={}/{}",
                            session.session_id.as_u64(),
                            session.model_id,
                            session.state,
                            session.message_count,
                            session.context_length,
                            session.max_context
                        );
                    }
                }
            }
            if persisted {
                let mut ids = manager.list_persisted_sessions()?;
                ids.sort_by_key(|id| id.as_u64());
                if ids.is_empty() {
                    println!("persisted sessions: <empty>");
                } else {
                    println!(
                        "persisted sessions: {}",
                        ids.into_iter()
                            .map(|id| id.as_u64().to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
            if !active && !persisted {
                println!("Nothing requested: use --active and/or --persisted");
            }
        }
        SessionAction::Restore { session_id } => {
            let session_id = SessionId::from(session_id);
            manager.restore_session(session_id)?;
            println!("Restored session: {}", session_id.as_u64());
        }
        SessionAction::RestoreAll => {
            let ids = manager.restore_all_sessions()?;
            if ids.is_empty() {
                println!("No sessions restored.");
            } else {
                println!(
                    "Restored sessions: {}",
                    ids.iter()
                        .map(|id| id.as_u64().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
        SessionAction::Save { session_id } => {
            let session_id = SessionId::from(session_id);
            ensure_session_loaded(&manager, session_id)?;
            manager.save_session(session_id)?;
            println!("Saved session: {}", session_id.as_u64());
        }
        SessionAction::SaveAll => {
            let count = manager.save_all_sessions()?;
            println!("Saved active sessions: {}", count);
        }
        SessionAction::Delete { session_id } => {
            manager.delete_persisted_session(SessionId::from(session_id))?;
            println!("Deleted persisted session: {}", session_id);
        }
        SessionAction::Destroy { session_id } => {
            let session_id = SessionId::from(session_id);
            ensure_session_loaded(&manager, session_id)?;
            manager.destroy_session(session_id)?;
            println!("Destroyed session: {}", session_id.as_u64());
        }
        SessionAction::Clear {
            session_id,
            no_save,
        } => {
            let session_id = SessionId::from(session_id);
            ensure_session_loaded(&manager, session_id)?;
            let handle = manager
                .get_session(session_id)
                .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
            handle.clear_context()?;
            if !no_save {
                manager.save_session(session_id)?;
            }
            println!("Cleared session context: {}", session_id.as_u64());
            println!("Persisted: {}", !no_save);
        }
    }
    Ok(())
}

fn run_serve_command(cmd: ServeCmd) -> anyhow::Result<()> {
    let model = resolve_model_reference(
        cmd.model.as_deref(),
        cmd.model_id.as_deref(),
        &cmd.model_store,
    )?;
    if !cmd.api_type.eq_ignore_ascii_case("rest") {
        println!(
            "Unsupported --api-type `{}`; falling back to REST.",
            cmd.api_type
        );
    }

    let plugins = Arc::new(Mutex::new(load_plugins(&cmd.plugins)?));
    let (mut engine_instance, execution_policy_registry, selected_execution_policy_name) =
        build_engine_with_execution_registry(&model, &cmd.engine)?;

    let tool_plugin_store_loaded = load_dynamic_policy_registry(&cmd.tool_plugin_registry)?;
    let merged_tool_plugins =
        merge_plugin_paths(tool_plugin_store_loaded.plugins(), &cmd.tool_plugins);

    for tool_plugin in &merged_tool_plugins {
        let (name, functions) = engine_instance.load_dynamic_tool_plugin(tool_plugin)?;
        println!("Loaded tool plugin: {} ({})", name, tool_plugin.display());
        if !functions.is_empty() {
            println!("Plugin tools: {}", functions.join(", "));
        }
    }

    if let Some(registry_path) = &cmd.mcp_registry {
        let registry = load_mcp_registry(registry_path)?;
        let mut selected = Vec::new();
        if cmd.mcp_servers.is_empty() {
            selected.extend(registry.list_enabled().into_iter().cloned());
        } else {
            for name in &cmd.mcp_servers {
                let server = registry.get(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "MCP server '{}' not found in {}",
                        name,
                        registry_path.display()
                    )
                })?;
                if !server.enabled {
                    return Err(anyhow::anyhow!(
                        "MCP server '{}' is disabled in {}",
                        name,
                        registry_path.display()
                    ));
                }
                selected.push(server.clone());
            }
        }

        for server in selected {
            let report = engine_instance.connect_mcp_stdio_server(server.to_stdio_config())?;
            println!(
                "Connected MCP server '{}' from registry; registered {} tool(s)",
                report.server_name,
                report.registered_tools.len()
            );
            if !report.registered_tools.is_empty() {
                println!("MCP tools: {}", report.registered_tools.join(", "));
            }
        }
    }

    for spec in &cmd.mcp_stdio {
        let config = parse_mcp_stdio_spec(spec)
            .map_err(|e| anyhow::anyhow!("invalid --mcp-stdio '{}': {}", spec, e))?;
        let command = config.command.clone();
        let report = engine_instance.connect_mcp_stdio_server(config)?;
        println!(
            "Connected MCP server '{}' via '{}'; registered {} tool(s)",
            report.server_name,
            command,
            report.registered_tools.len()
        );
        if !report.registered_tools.is_empty() {
            println!("MCP tools: {}", report.registered_tools.join(", "));
        }
    }

    let engine = Arc::new(Mutex::new(engine_instance));
    let tool_plugin_store = Arc::new(Mutex::new(tool_plugin_store_loaded));
    let session_manager = Arc::new(build_session_manager_from_options(
        &cmd.session_store_kind,
        &cmd.session_store_options,
        cmd.session_store_plugin.as_deref(),
    )?);
    let served_model_path = Arc::new(model.clone());
    let model_store_root = Arc::new(cmd.model_store.clone());
    let model_store = Arc::new(Mutex::new(ModelStore::new(cmd.model_store.clone())));
    let model_pull_jobs = Arc::new(ModelPullJobManager::new(Arc::clone(&model_store)));
    let default_sampling = Arc::new(cmd.sampling.clone());
    let metrics = Arc::new(ServerMetrics::new());
    let runtime_events = Arc::new(RuntimeEventBus::new(1024));
    let worker_count = cmd.workers.max(1);
    let queue_size = cmd.queue_size.max(1);
    let dispatch_policy_store_loaded = load_dynamic_policy_registry(&cmd.backpressure_registry)?;
    let stored_dispatch_active = dispatch_policy_store_loaded
        .active()
        .map(|name| name.to_string());
    let merged_backpressure_plugins = merge_plugin_paths(
        dispatch_policy_store_loaded.plugins(),
        &cmd.backpressure_plugins,
    );
    let dispatch_registry = Arc::new(build_dispatch_policy_registry(
        &merged_backpressure_plugins,
    )?);
    let dispatch_policy_store = Arc::new(Mutex::new(dispatch_policy_store_loaded));
    let execution_policy_store = Arc::new(Mutex::new(load_dynamic_policy_registry(
        &cmd.engine.execution_policy_registry,
    )?));
    let mut management_auth_store_loaded =
        load_dynamic_policy_registry(&cmd.management_auth_registry)?;
    let stored_management_auth_active = management_auth_store_loaded
        .active()
        .map(|name| name.to_string());
    let management_auth_plugins = merge_plugin_paths(
        management_auth_store_loaded.plugins(),
        &cmd.management_auth_plugins,
    );
    let management_auth_registry = Arc::new(build_management_auth_policy_registry(
        &management_auth_plugins,
        cmd.management_auth_bearer_token.as_deref(),
    )?);
    let selected_management_auth_name = cmd
        .management_auth_policy_name
        .clone()
        .or(stored_management_auth_active)
        .unwrap_or_else(|| "allow-all.management.auth".to_string());
    let management_auth_policy = management_auth_registry
        .get(&selected_management_auth_name)
        .ok_or_else(|| {
            let available = management_auth_registry.list_names().join(", ");
            anyhow::anyhow!(
                "unknown management auth policy '{}'. available: {}",
                selected_management_auth_name,
                available
            )
        })?;
    let active_management_auth_policy = Arc::new(ActiveManagementAuthPolicy::new(
        selected_management_auth_name.clone(),
        management_auth_policy,
    ));
    let (resolved_management_auth_scope, persist_scope) = resolve_management_auth_scope(
        cmd.management_auth_scope,
        &cmd.management_auth_prefixes,
        &management_auth_store_loaded,
    )?;
    if persist_scope {
        persist_management_auth_scope(
            &mut management_auth_store_loaded,
            &resolved_management_auth_scope,
        )?;
    }
    let management_auth_store = Arc::new(Mutex::new(management_auth_store_loaded));
    let management_auth_scope = Arc::new(resolved_management_auth_scope);
    let model_pull_policy_store_loaded =
        load_dynamic_policy_registry(&cmd.model_pull_policy_registry)?;
    let stored_model_pull_policy_active = model_pull_policy_store_loaded
        .active()
        .map(|name| name.to_string());
    let model_pull_policy_plugins = merge_plugin_paths(
        model_pull_policy_store_loaded.plugins(),
        &cmd.model_pull_policy_plugins,
    );
    let model_pull_policy_registry = Arc::new(build_model_pull_policy_registry(
        &model_pull_policy_plugins,
    )?);
    let selected_model_pull_policy_name = cmd
        .model_pull_policy_name
        .clone()
        .or(stored_model_pull_policy_active)
        .unwrap_or_else(|| "allow-all.model.pull".to_string());
    let model_pull_policy = model_pull_policy_registry
        .get(&selected_model_pull_policy_name)
        .ok_or_else(|| {
            let available = model_pull_policy_registry.list_names().join(", ");
            anyhow::anyhow!(
                "unknown model pull policy '{}'. available: {}",
                selected_model_pull_policy_name,
                available
            )
        })?;
    let active_model_pull_policy = Arc::new(ActiveModelPullPolicy::new(
        selected_model_pull_policy_name.clone(),
        model_pull_policy,
    ));
    let model_pull_policy_store = Arc::new(Mutex::new(model_pull_policy_store_loaded));
    let model_pull_verifier_store_loaded =
        load_dynamic_policy_registry(&cmd.model_pull_verifier_registry)?;
    let stored_model_pull_verifier_active = model_pull_verifier_store_loaded
        .active()
        .map(|name| name.to_string());
    let model_pull_verifier_plugins = merge_plugin_paths(
        model_pull_verifier_store_loaded.plugins(),
        &cmd.model_pull_verifier_plugins,
    );
    let model_pull_verifier_registry = Arc::new(build_model_pull_verifier_registry(
        &model_pull_verifier_plugins,
    )?);
    let selected_model_pull_verifier_name = cmd
        .model_pull_verifier_name
        .clone()
        .or(stored_model_pull_verifier_active)
        .unwrap_or_else(|| "allow-all.model.verify".to_string());
    let model_pull_verifier = model_pull_verifier_registry
        .get(&selected_model_pull_verifier_name)
        .ok_or_else(|| {
            let available = model_pull_verifier_registry.list_names().join(", ");
            anyhow::anyhow!(
                "unknown model pull verifier '{}'. available: {}",
                selected_model_pull_verifier_name,
                available
            )
        })?;
    let active_model_pull_verifier = Arc::new(ActiveModelPullVerifier::new(
        selected_model_pull_verifier_name.clone(),
        model_pull_verifier,
    ));
    let model_pull_verifier_store = Arc::new(Mutex::new(model_pull_verifier_store_loaded));
    let (selected_policy_name, dispatch_policy_plugin) = resolve_dispatch_policy_from_registry(
        dispatch_registry.as_ref(),
        cmd.backpressure_policy_name
            .as_deref()
            .or(stored_dispatch_active.as_deref()),
        cmd.backpressure,
    )?;
    let active_dispatch_policy = Arc::new(ActiveServeDispatchPolicy::new(
        selected_policy_name.clone(),
        Arc::new(PluginBackpressureDispatchPolicy::new(
            dispatch_policy_plugin,
        )),
    ));
    for plugin_path in &cmd.backpressure_plugins {
        println!("Backpressure plugin: {}", plugin_path.display());
    }

    let addr = format!("{}:{}", cmd.host, cmd.port);
    let listener = TcpListener::bind(&addr)?;
    let (request_tx, request_rx) = channel::bounded::<TcpStream>(queue_size);

    println!("Loci REST server listening on http://{addr}");
    println!(
        "Endpoints: GET /health,/info,/metrics,/openapi.yaml,/openapi.json,/events,/events/stream,/v1/models,/api/tags,/models/assets,/models/assets/{{id}},/models/assets/pulls,/models/assets/pulls/{{job_id}},/models/assets/pulls/{{job_id}}/events,/tools,/tools/{{name}},/tools/plugins,/tools/plugins/{{name}},/sessions,/sessions/{{id}},/dispatch-policies,/execution-policies,/auth-policies,/model-pull-policies,/model-pull-verifiers; POST /generate,/v1/chat/completions,/v1/embeddings,/api/generate,/models/plan,/models/assets,/models/assets/pull,/models/assets/pulls,/models/assets/pulls/{{job_id}}/cancel,/tools/invoke,/tools/plugins/load,/tools/plugins/{{name}}/(reload|unload),/sessions,/sessions/{{id}}/generate,/suspend,/resume,/save,/restore,/clear,/destroy,/dispatch-policies/load,/dispatch-policies/{{name}}/(activate|reload|unload),/execution-policies/load,/execution-policies/{{name}}/(activate|reload|unload),/auth-policies/load,/auth-policies/{{name}}/(activate|reload|unload),/model-pull-policies/load,/model-pull-policies/{{name}}/(activate|reload|unload),/model-pull-verifiers/load,/model-pull-verifiers/{{name}}/(activate|reload|unload); DELETE /models/assets/{{id}} (+ /v1/* aliases)"
    );
    println!(
        "Request handling: worker_pool={} queue_size={} dispatch_policy={} execution_policy={} management_auth={} management_auth_scope={} model_pull_policy={} model_pull_verifier={}",
        worker_count,
        queue_size,
        selected_policy_name,
        selected_execution_policy_name,
        selected_management_auth_name,
        management_auth_scope.display_label(),
        selected_model_pull_policy_name,
        selected_model_pull_verifier_name,
    );

    for worker_id in 0..worker_count {
        let rx = request_rx.clone();
        let engine = Arc::clone(&engine);
        let default_sampling = Arc::clone(&default_sampling);
        let plugins = Arc::clone(&plugins);
        let metrics = Arc::clone(&metrics);
        let runtime_events = Arc::clone(&runtime_events);
        let session_manager = Arc::clone(&session_manager);
        let served_model_path = Arc::clone(&served_model_path);
        let model_store_root = Arc::clone(&model_store_root);
        let model_store = Arc::clone(&model_store);
        let model_pull_jobs = Arc::clone(&model_pull_jobs);
        let dispatch_registry = Arc::clone(&dispatch_registry);
        let active_dispatch_policy = Arc::clone(&active_dispatch_policy);
        let dispatch_policy_store = Arc::clone(&dispatch_policy_store);
        let execution_policy_registry = Arc::clone(&execution_policy_registry);
        let execution_policy_store = Arc::clone(&execution_policy_store);
        let tool_plugin_store = Arc::clone(&tool_plugin_store);
        let management_auth_registry = Arc::clone(&management_auth_registry);
        let active_management_auth_policy = Arc::clone(&active_management_auth_policy);
        let management_auth_store = Arc::clone(&management_auth_store);
        let management_auth_scope = Arc::clone(&management_auth_scope);
        let model_pull_policy_registry = Arc::clone(&model_pull_policy_registry);
        let model_pull_policy_store = Arc::clone(&model_pull_policy_store);
        let active_model_pull_policy = Arc::clone(&active_model_pull_policy);
        let model_pull_verifier_registry = Arc::clone(&model_pull_verifier_registry);
        let model_pull_verifier_store = Arc::clone(&model_pull_verifier_store);
        let active_model_pull_verifier = Arc::clone(&active_model_pull_verifier);
        thread::spawn(move || loop {
            let mut stream = match rx.recv() {
                Ok(stream) => stream,
                Err(_) => break,
            };
            let request_started = Instant::now();
            if let Err(err) = handle_connection(
                &mut stream,
                &engine,
                default_sampling.as_ref(),
                &plugins,
                metrics.as_ref(),
                runtime_events.as_ref(),
                session_manager.as_ref(),
                served_model_path.as_path(),
                model_store_root.as_path(),
                model_store.as_ref(),
                model_pull_jobs.as_ref(),
                dispatch_registry.as_ref(),
                active_dispatch_policy.as_ref(),
                dispatch_policy_store.as_ref(),
                execution_policy_registry.as_ref(),
                execution_policy_store.as_ref(),
                tool_plugin_store.as_ref(),
                management_auth_registry.as_ref(),
                management_auth_store.as_ref(),
                management_auth_scope.as_ref(),
                active_management_auth_policy.as_ref(),
                model_pull_policy_registry.as_ref(),
                model_pull_policy_store.as_ref(),
                active_model_pull_policy.as_ref(),
                model_pull_verifier_registry.as_ref(),
                model_pull_verifier_store.as_ref(),
                active_model_pull_verifier.as_ref(),
            ) {
                let _ = write_json_response(
                    &mut stream,
                    "500 Internal Server Error",
                    &ErrorResponse {
                        error: err.to_string(),
                    },
                );
                metrics.record("internal", 500, request_started.elapsed());
                eprintln!("serve worker-{worker_id} request failed: {err}");
            }
        });
    }

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let queued_at = Instant::now();
                if let Err(mut stream) = active_dispatch_policy.dispatch(&request_tx, stream) {
                    let status = "503 Service Unavailable";
                    let _ = write_json_response(
                        &mut stream,
                        status,
                        &ErrorResponse {
                            error: "server busy: request queue is full".to_string(),
                        },
                    );
                    metrics.record(
                        "backpressure",
                        status_code_value(status),
                        queued_at.elapsed(),
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
    engine: &Arc<Mutex<InferenceEngine>>,
    default_sampling: &SamplingArgs,
    plugins: &Arc<Mutex<Option<PluginRegistry>>>,
    metrics: &ServerMetrics,
    runtime_events: &RuntimeEventBus,
    session_manager: &SessionManager,
    served_model_path: &Path,
    model_store_root: &Path,
    model_store: &Mutex<ModelStore>,
    model_pull_jobs: &ModelPullJobManager,
    dispatch_registry: &ServeDispatchPolicyRegistry,
    active_dispatch_policy: &ActiveServeDispatchPolicy,
    dispatch_policy_store: &Mutex<DynamicPolicyRegistry>,
    execution_policy_registry: &ExecutionPolicyRegistry,
    execution_policy_store: &Mutex<DynamicPolicyRegistry>,
    tool_plugin_store: &Mutex<DynamicPolicyRegistry>,
    management_auth_registry: &ManagementAuthPolicyRegistry,
    management_auth_store: &Mutex<DynamicPolicyRegistry>,
    management_auth_scope: &ManagementAuthScopeConfig,
    active_management_auth_policy: &ActiveManagementAuthPolicy,
    model_pull_policy_registry: &ModelPullPolicyRegistry,
    model_pull_policy_store: &Mutex<DynamicPolicyRegistry>,
    active_model_pull_policy: &ActiveModelPullPolicy,
    model_pull_verifier_registry: &ModelPullVerifierRegistry,
    model_pull_verifier_store: &Mutex<DynamicPolicyRegistry>,
    active_model_pull_verifier: &ActiveModelPullVerifier,
) -> anyhow::Result<()> {
    let request_started = Instant::now();
    let request = match read_http_request(stream) {
        Ok(request) => request,
        Err(parse_error) => {
            let status = parse_error.status_code();
            write_json_response(
                stream,
                status,
                &ErrorResponse {
                    error: parse_error.to_string(),
                },
            )?;
            metrics.record(
                "parse_error",
                status_code_value(status),
                request_started.elapsed(),
            );
            return Ok(());
        }
    };

    let management_auth_context = build_management_auth_context(stream, &request);
    if let Some((status, payload)) = authorize_management_request(
        &management_auth_context,
        management_auth_scope,
        active_management_auth_policy,
    )? {
        publish_runtime_event(
            runtime_events,
            RuntimeEventCategory::Auth,
            "management_auth.authorize",
            RuntimeEventOutcome::Denied,
            Some("management_auth"),
            Some(&request),
            Some(status_code_value(status)),
            Some(payload.policy.clone()),
            Some(serde_json::json!({
                "error": payload.error,
            })),
        );
        write_json_response(stream, status, &payload)?;
        metrics.record(
            "management_auth",
            status_code_value(status),
            request_started.elapsed(),
        );
        return Ok(());
    }

    if request.method == "GET" && (request.path == "/health" || request.path == "/v1/health") {
        let status = "200 OK";
        write_plain_response(stream, status, "application/json", r#"{"status":"ok"}"#)?;
        metrics.record(
            "health",
            status_code_value(status),
            request_started.elapsed(),
        );
        return Ok(());
    }

    if request.method == "GET" && (request.path == "/info" || request.path == "/v1/info") {
        let payload = {
            let plugins_guard = plugins
                .lock()
                .expect("plugin registry mutex should not be poisoned");
            let guard = engine
                .lock()
                .expect("inference engine mutex should not be poisoned");
            build_model_info_response(&guard, plugins_guard.as_ref())
        };
        let status = "200 OK";
        write_json_response(stream, status, &payload)?;
        metrics.record("info", status_code_value(status), request_started.elapsed());
        return Ok(());
    }

    if let Some(route) = parse_runtime_events_api_route(&request.path) {
        let endpoint = "events";
        match (request.method.as_str(), route) {
            ("GET", RuntimeEventsApiRoute::Collection) => {
                let limit = request_query_usize(&request.path, "limit");
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &RuntimeEventListResponse {
                        events: runtime_events.recent_events(limit),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", RuntimeEventsApiRoute::Stream) => {
                let replay = request_query_usize(&request.path, "replay");
                let follow = request_query_bool(&request.path, "follow").unwrap_or(true);
                let status = "200 OK";
                write_streaming_response_headers(stream, status, "application/x-ndjson")?;
                for event in runtime_events.recent_events(replay) {
                    write_ndjson_event(stream, &event)?;
                }
                if follow {
                    let receiver = runtime_events.subscribe();
                    loop {
                        match receiver.recv_timeout(Duration::from_millis(250)) {
                            Ok(event) => {
                                if write_ndjson_event(stream, &event).is_err() {
                                    break;
                                }
                            }
                            Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
                            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => break,
                        }
                    }
                }
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if request.method == "GET" {
        if let Some(format) = requested_openapi_spec_format(&request.path) {
            let status = "200 OK";
            match format {
                OpenApiSpecFormat::Yaml => {
                    write_plain_response(stream, status, "application/yaml", OPENAPI_SPEC_YAML)?;
                }
                OpenApiSpecFormat::Json => {
                    let body = build_openapi_json_spec()?;
                    write_plain_response(stream, status, "application/json", &body)?;
                }
            }
            metrics.record(
                "openapi",
                status_code_value(status),
                request_started.elapsed(),
            );
            return Ok(());
        }
    }

    if request.method == "POST"
        && (request.path == "/models/plan" || request.path == "/v1/models/plan")
    {
        let payload: ModelPlanRequest = if request.body.is_empty() {
            ModelPlanRequest {
                model: None,
                model_id: None,
                context_size: None,
            }
        } else {
            match serde_json::from_slice(&request.body) {
                Ok(payload) => payload,
                Err(err) => {
                    let status = "400 Bad Request";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!("invalid JSON payload for /models/plan: {err}"),
                        },
                    )?;
                    metrics.record(
                        "models.plan",
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
            }
        };

        let model = match (payload.model.as_deref(), payload.model_id.as_deref()) {
            (None, None) => served_model_path.to_path_buf(),
            (model, model_id) => {
                match resolve_model_reference(model.map(Path::new), model_id, model_store_root) {
                    Ok(model) => model,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            "models.plan",
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
        };
        let context_size = payload.context_size.unwrap_or(4096);
        if context_size == 0 {
            let status = "400 Bad Request";
            write_json_response(
                stream,
                status,
                &ErrorResponse {
                    error: "context_size must be greater than 0".to_string(),
                },
            )?;
            metrics.record(
                "models.plan",
                status_code_value(status),
                request_started.elapsed(),
            );
            return Ok(());
        }
        let estimate = ResourcePlanner::estimate_model_requirements(&model, context_size)
            .map_err(|e| anyhow::anyhow!(e))?;
        let selector = DeviceSelector::new();
        let plan = ResourcePlanner::plan_for_estimate(&estimate, selector.devices());
        let response = build_resource_plan_report(&model, estimate, plan, selector.devices());
        let status = "200 OK";
        write_json_response(stream, status, &response)?;
        metrics.record(
            "models.plan",
            status_code_value(status),
            request_started.elapsed(),
        );
        return Ok(());
    }

    if let Some(route) = parse_model_store_api_route(&request.path) {
        let endpoint = "model_assets";
        match (request.method.as_str(), route) {
            ("GET", ModelStoreApiRoute::Collection) => match model_store
                .lock()
                .expect("model store mutex should not be poisoned")
                .list()
            {
                Ok(models) => {
                    let status = "200 OK";
                    write_json_response(stream, status, &ModelAssetListResponse { models })?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                Err(err) => {
                    let status = "500 Internal Server Error";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
            },
            ("POST", ModelStoreApiRoute::Collection) => {
                let payload: ModelAssetRegisterRequest = match serde_json::from_slice(&request.body)
                {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("invalid JSON payload for /models/assets: {err}"),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.register",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };

                match model_store
                    .lock()
                    .expect("model store mutex should not be poisoned")
                    .add_external(payload.path, payload.id, payload.name, payload.tags)
                {
                    Ok(model) => {
                        let status = "200 OK";
                        write_json_response(stream, status, &model)?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.register",
                            RuntimeEventOutcome::Success,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(model.id.clone()),
                            Some(serde_json::json!({
                                "managed": model.managed,
                                "path": model.path,
                                "source": model.source,
                                "tags": model.tags,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.register",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
            ("GET", ModelStoreApiRoute::PullJobsCollection) => {
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &ModelAssetPullJobListResponse {
                        jobs: model_pull_jobs.list_jobs(),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", ModelStoreApiRoute::PullJobsCollection) => {
                let payload: ModelAssetPullRequest = match serde_json::from_slice(&request.body) {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /models/assets/pulls: {err}"
                                ),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pulls.submit",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };

                let policy_context = build_model_pull_policy_context(&payload);
                let (policy_name, policy) = active_model_pull_policy.snapshot();
                if let Err(reason) = authorize_model_pull_request(policy.as_ref(), &policy_context)
                {
                    let status = "403 Forbidden";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "model pull denied by policy '{}': {}",
                                policy_name, reason
                            ),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::ModelAsset,
                        "model_assets.pulls.submit",
                        RuntimeEventOutcome::Denied,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        payload.id.clone().or_else(|| Some(payload.source.clone())),
                        Some(serde_json::json!({
                            "source": payload.source,
                            "policy": policy_name,
                            "reason": reason,
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let (verifier_name, verifier) = active_model_pull_verifier.snapshot();

                let pull_subject = payload.id.clone().or_else(|| Some(payload.source.clone()));
                let pull_source = payload.source.clone();
                let pull_job_request = ModelPullJobRequest {
                    source: payload.source,
                    mirrors: payload.mirrors,
                    id: payload.id,
                    name: payload.name,
                    sha256: payload.sha256,
                    resume: !payload.no_resume.unwrap_or(false),
                    tags: payload.tags,
                };
                match model_pull_jobs.submit_pull_with_governance(
                    pull_job_request,
                    Some(policy_name),
                    Some(policy),
                    Some(verifier_name),
                    Some(verifier),
                ) {
                    Ok(job) => {
                        let status = "202 Accepted";
                        write_json_response(stream, status, &job)?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pulls.submit",
                            RuntimeEventOutcome::Started,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(job.job_id.clone()),
                            Some(serde_json::json!({
                                "source": job.request.source,
                                "model_id": job.request.id,
                                "policy": job.policy_name,
                                "verifier": job.verifier_name,
                                "resume": job.request.resume,
                                "tags": job.request.tags,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        let status = "500 Internal Server Error";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pulls.submit",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            pull_subject,
                            Some(serde_json::json!({
                                "source": pull_source,
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
            ("POST", ModelStoreApiRoute::Pull) => {
                let payload: ModelAssetPullRequest = match serde_json::from_slice(&request.body) {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /models/assets/pull: {err}"
                                ),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pull",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };

                let policy_context = build_model_pull_policy_context(&payload);
                let options = ModelPullOptions {
                    mirrors: payload.mirrors.clone(),
                    expected_sha256: payload.sha256.clone(),
                    resume: !payload.no_resume.unwrap_or(false),
                };
                let (policy_name, policy) = active_model_pull_policy.snapshot();
                let (verifier_name, verifier) = active_model_pull_verifier.snapshot();
                let pull_subject = payload.id.clone().unwrap_or_else(|| payload.source.clone());
                let pull_source = payload.source.clone();
                if let Err(reason) = authorize_model_pull_request(policy.as_ref(), &policy_context)
                {
                    let status = "403 Forbidden";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "model pull denied by policy '{}': {}",
                                policy_name, reason
                            ),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::ModelAsset,
                        "model_assets.pull",
                        RuntimeEventOutcome::Denied,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(pull_subject.clone()),
                        Some(serde_json::json!({
                            "source": pull_source,
                            "policy": policy_name,
                            "reason": reason,
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let stream_pull = payload.stream.unwrap_or(false)
                    || request_query_bool(&request.path, "stream").unwrap_or(false);

                if stream_pull {
                    let status = "200 OK";
                    write_streaming_response_headers(stream, status, "application/x-ndjson")?;
                    let mut stream_write_failed = false;
                    let result = {
                        let store = model_store
                            .lock()
                            .expect("model store mutex should not be poisoned");
                        let mut emit = |progress: ModelPullProgress| {
                            if stream_write_failed {
                                return;
                            }
                            if write_ndjson_event(
                                stream,
                                &ModelAssetPullStreamEvent::Progress { progress },
                            )
                            .is_err()
                            {
                                stream_write_failed = true;
                            }
                        };
                        store
                            .pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation(
                                &payload.source,
                                payload.id.clone(),
                                payload.name.clone(),
                                payload.tags.clone(),
                                options,
                                Some(policy.as_ref()),
                                Some(verifier.as_ref()),
                                None,
                                &mut emit,
                            )
                    };
                    match result {
                        Ok(model) => {
                            let _ = write_ndjson_event(
                                stream,
                                &ModelAssetPullStreamEvent::Complete { model },
                            );
                            publish_runtime_event(
                                runtime_events,
                                RuntimeEventCategory::ModelAsset,
                                "model_assets.pull",
                                RuntimeEventOutcome::Success,
                                Some(endpoint),
                                Some(&request),
                                Some(status_code_value(status)),
                                Some(pull_subject.clone()),
                                Some(serde_json::json!({
                                    "source": pull_source,
                                    "stream": true,
                                    "policy": policy_name,
                                    "verifier": verifier_name,
                                })),
                            );
                            metrics.record(
                                endpoint,
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }
                        Err(err) => {
                            let _ = write_ndjson_event(
                                stream,
                                &ModelAssetPullStreamEvent::Error {
                                    error: err.to_string(),
                                },
                            );
                            publish_runtime_event(
                                runtime_events,
                                RuntimeEventCategory::ModelAsset,
                                "model_assets.pull",
                                RuntimeEventOutcome::Error,
                                Some(endpoint),
                                Some(&request),
                                Some(400),
                                Some(pull_subject.clone()),
                                Some(serde_json::json!({
                                    "source": pull_source,
                                    "stream": true,
                                    "policy": policy_name,
                                    "verifier": verifier_name,
                                    "error": err.to_string(),
                                })),
                            );
                            metrics.record(endpoint, 400, request_started.elapsed());
                            return Ok(());
                        }
                    }
                }

                match model_store
                    .lock()
                    .expect("model store mutex should not be poisoned")
                    .pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation(
                        &payload.source,
                        payload.id.clone(),
                        payload.name.clone(),
                        payload.tags.clone(),
                        options,
                        Some(policy.as_ref()),
                        Some(verifier.as_ref()),
                        None,
                        &mut |_| {},
                    ) {
                    Ok(model) => {
                        let status = "200 OK";
                        write_json_response(stream, status, &model)?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pull",
                            RuntimeEventOutcome::Success,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(model.id.clone()),
                            Some(serde_json::json!({
                                "source": pull_source,
                                "stream": false,
                                "policy": policy_name,
                                "verifier": verifier_name,
                                "managed": model.managed,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pull",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(pull_subject),
                            Some(serde_json::json!({
                                "source": pull_source,
                                "stream": false,
                                "policy": policy_name,
                                "verifier": verifier_name,
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
            ("GET", ModelStoreApiRoute::PullJobItem { id }) => match model_pull_jobs.get_job(&id) {
                Ok(job) => {
                    let status = "200 OK";
                    write_json_response(stream, status, &job)?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                Err(err) => {
                    let status = if matches!(err, loci::error::LociError::ModelNotFound) {
                        "404 Not Found"
                    } else {
                        "400 Bad Request"
                    };
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
            },
            ("POST", ModelStoreApiRoute::PullJobCancel { id }) => {
                match model_pull_jobs.cancel_job(&id) {
                    Ok(job) => {
                        let status = "200 OK";
                        write_json_response(stream, status, &job)?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pulls.cancel",
                            RuntimeEventOutcome::Cancelled,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(job.job_id.clone()),
                            Some(serde_json::json!({
                                "state": format!("{:?}", job.state),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        let status = if matches!(err, loci::error::LociError::ModelNotFound) {
                            "404 Not Found"
                        } else {
                            "409 Conflict"
                        };
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.pulls.cancel",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(id.clone()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
            ("GET", ModelStoreApiRoute::PullJobEvents { id }) => {
                match model_pull_jobs.subscribe(&id) {
                    Ok((snapshot, receiver)) => {
                        let status = "200 OK";
                        write_streaming_response_headers(stream, status, "application/x-ndjson")?;
                        if write_ndjson_event(
                            stream,
                            &ModelPullJobEvent::Snapshot {
                                job: snapshot.clone(),
                            },
                        )
                        .is_err()
                        {
                            metrics.record(
                                endpoint,
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }
                        if snapshot.state.is_terminal() {
                            metrics.record(
                                endpoint,
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }

                        while let Ok(event) = receiver.recv() {
                            let terminal = matches!(
                                event,
                                ModelPullJobEvent::Complete { .. }
                                    | ModelPullJobEvent::Failed { .. }
                                    | ModelPullJobEvent::Cancelled { .. }
                            );
                            if write_ndjson_event(stream, &event).is_err() {
                                break;
                            }
                            if terminal {
                                break;
                            }
                        }
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        let status = if matches!(err, loci::error::LociError::ModelNotFound) {
                            "404 Not Found"
                        } else {
                            "400 Bad Request"
                        };
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
            ("GET", ModelStoreApiRoute::Item { id }) => match model_store
                .lock()
                .expect("model store mutex should not be poisoned")
                .get(&id)
            {
                Ok(model) => {
                    let status = "200 OK";
                    write_json_response(stream, status, &model)?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                Err(err) => {
                    let status = if matches!(err, loci::error::LociError::ModelNotFound) {
                        "404 Not Found"
                    } else {
                        "400 Bad Request"
                    };
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
            },
            ("DELETE", ModelStoreApiRoute::Item { id }) => {
                let delete_file = request_query_bool(&request.path, "delete_file").unwrap_or(false);
                match model_store
                    .lock()
                    .expect("model store mutex should not be poisoned")
                    .remove(&id, delete_file)
                {
                    Ok(model) => {
                        let model_id = model.id.clone();
                        let model_managed = model.managed;
                        let model_path = model.path.clone();
                        let status = "200 OK";
                        write_json_response(
                            stream,
                            status,
                            &ModelAssetRemoveResponse {
                                model,
                                deleted_file: delete_file,
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.remove",
                            RuntimeEventOutcome::Success,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(model_id),
                            Some(serde_json::json!({
                                "deleted_file": delete_file,
                                "managed": model_managed,
                                "path": model_path,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        let status = if matches!(err, loci::error::LociError::ModelNotFound) {
                            "404 Not Found"
                        } else {
                            "400 Bad Request"
                        };
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::ModelAsset,
                            "model_assets.remove",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(id.clone()),
                            Some(serde_json::json!({
                                "delete_file": delete_file,
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
            _ => {
                let status = "405 Method Not Allowed";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: "method not allowed for model asset endpoint".to_string(),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
        }
    }

    if request.method == "GET" && request.path == "/v1/models" {
        let status = "200 OK";
        let model_id = current_runtime_model_name(engine);
        write_json_response(
            stream,
            status,
            &OpenAiModelListResponse {
                object: "list",
                data: vec![OpenAiModelDescriptor {
                    id: model_id,
                    object: "model",
                    owned_by: "loci",
                }],
            },
        )?;
        metrics.record(
            "compat.openai.models",
            status_code_value(status),
            request_started.elapsed(),
        );
        return Ok(());
    }

    if request.method == "GET" && request.path == "/api/tags" {
        let status = "200 OK";
        let model_id = current_runtime_model_name(engine);
        write_json_response(
            stream,
            status,
            &OllamaTagsResponse {
                models: vec![OllamaModelTag {
                    name: model_id.clone(),
                    model: model_id,
                    modified_at: compatibility_created_at(),
                    size: 0,
                }],
            },
        )?;
        metrics.record(
            "compat.ollama.tags",
            status_code_value(status),
            request_started.elapsed(),
        );
        return Ok(());
    }

    if request.method == "GET" && (request.path == "/metrics" || request.path == "/v1/metrics") {
        let status = "200 OK";
        let snapshot = metrics.snapshot();
        write_json_response(stream, status, &snapshot)?;
        metrics.record(
            "metrics",
            status_code_value(status),
            request_started.elapsed(),
        );
        return Ok(());
    }

    if request.method == "POST" && request.path == "/v1/chat/completions" {
        let payload: OpenAiChatCompletionsRequest = match serde_json::from_slice(&request.body) {
            Ok(payload) => payload,
            Err(err) => {
                let status = "400 Bad Request";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: format!("invalid JSON payload for /v1/chat/completions: {err}"),
                    },
                )?;
                metrics.record(
                    "compat.openai.chat",
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
        };

        let prompt = openai_chat_messages_to_prompt(&payload.messages);
        let params = GenerationParams {
            max_tokens: payload.max_tokens.unwrap_or(default_sampling.max_tokens),
            temperature: payload.temperature.unwrap_or(default_sampling.temperature),
            top_p: payload.top_p.unwrap_or(default_sampling.top_p),
            min_p: default_sampling.min_p,
            top_k: default_sampling.top_k,
            repeat_penalty: default_sampling.repetition_penalty,
        };
        let model_name = payload
            .model
            .unwrap_or_else(|| current_runtime_model_name(engine));

        if payload.stream.unwrap_or(false) {
            let stream_id = format!("chatcmpl-{}", unix_ms_now());
            let created = unix_ms_now() / 1000;
            let mut headers_written = false;
            let mut started = false;
            let mut start_stream = |stream: &mut TcpStream| -> anyhow::Result<()> {
                if !headers_written {
                    write_streaming_response_headers(stream, "200 OK", "text/event-stream")?;
                    headers_written = true;
                }
                if !started {
                    write_sse_json_event(
                        stream,
                        &openai_chat_stream_chunk(
                            &stream_id,
                            created,
                            &model_name,
                            Some("assistant"),
                            None,
                            None,
                        ),
                    )?;
                    started = true;
                }
                Ok(())
            };

            match generate_stream_with_runtime(engine, plugins, &prompt, params, |chunk| {
                start_stream(stream)?;
                write_sse_json_event(
                    stream,
                    &openai_chat_stream_chunk(
                        &stream_id,
                        created,
                        &model_name,
                        None,
                        Some(chunk.to_string()),
                        None,
                    ),
                )
            }) {
                Ok(_) => {
                    let status = "200 OK";
                    start_stream(stream)?;
                    write_sse_json_event(
                        stream,
                        &openai_chat_stream_chunk(
                            &stream_id,
                            created,
                            &model_name,
                            None,
                            None,
                            Some("stop"),
                        ),
                    )?;
                    write_sse_done(stream)?;
                    metrics.record(
                        "compat.openai.chat",
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                }
                Err(err) => {
                    let status = "500 Internal Server Error";
                    if headers_written {
                        let _ = write_sse_json_event(
                            stream,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        );
                        let _ = write_sse_done(stream);
                    } else {
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                    }
                    metrics.record(
                        "compat.openai.chat",
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                }
            }
            return Ok(());
        }

        match generate_with_runtime(engine, plugins, &prompt, params) {
            Ok(response) => {
                let usage = compatibility_usage(&prompt, &response);
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &OpenAiChatCompletionsResponse {
                        id: format!("chatcmpl-{}", unix_ms_now()),
                        object: "chat.completion",
                        created: unix_ms_now() / 1000,
                        model: model_name,
                        choices: vec![OpenAiChatChoice {
                            index: 0,
                            message: OpenAiChatMessage {
                                role: "assistant".to_string(),
                                content: response,
                            },
                            finish_reason: "stop",
                        }],
                        usage,
                    },
                )?;
                metrics.record(
                    "compat.openai.chat",
                    status_code_value(status),
                    request_started.elapsed(),
                );
            }
            Err(err) => {
                let status = "500 Internal Server Error";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: err.to_string(),
                    },
                )?;
                metrics.record(
                    "compat.openai.chat",
                    status_code_value(status),
                    request_started.elapsed(),
                );
            }
        }
        return Ok(());
    }

    if request.method == "POST" && request.path == "/v1/embeddings" {
        let payload: OpenAiEmbeddingsRequest = match serde_json::from_slice(&request.body) {
            Ok(payload) => payload,
            Err(err) => {
                let status = "400 Bad Request";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: format!("invalid JSON payload for /v1/embeddings: {err}"),
                    },
                )?;
                metrics.record(
                    "compat.openai.embeddings",
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
        };

        let inputs = normalize_openai_embedding_input(&payload.input);
        let model_name = payload
            .model
            .unwrap_or_else(|| current_runtime_model_name(engine));
        let mut data = Vec::with_capacity(inputs.len());
        let mut prompt_tokens: u32 = 0;
        let mut failed: Option<String> = None;

        for (index, input) in inputs.iter().enumerate() {
            prompt_tokens = prompt_tokens.saturating_add(estimate_token_count(input));
            let embedding = {
                let mut engine_guard = engine
                    .lock()
                    .expect("inference engine mutex should not be poisoned");
                engine_guard.generate_embeddings(input)
            };
            match embedding {
                Ok(embedding) => data.push(OpenAiEmbeddingData {
                    object: "embedding",
                    index: index as u32,
                    embedding,
                }),
                Err(err) => {
                    failed = Some(err.to_string());
                    break;
                }
            }
        }

        if let Some(error) = failed {
            let status = "500 Internal Server Error";
            write_json_response(stream, status, &ErrorResponse { error })?;
            metrics.record(
                "compat.openai.embeddings",
                status_code_value(status),
                request_started.elapsed(),
            );
            return Ok(());
        }

        let status = "200 OK";
        write_json_response(
            stream,
            status,
            &OpenAiEmbeddingsResponse {
                object: "list",
                data,
                model: model_name,
                usage: OpenAiUsage {
                    prompt_tokens,
                    completion_tokens: 0,
                    total_tokens: prompt_tokens,
                },
            },
        )?;
        metrics.record(
            "compat.openai.embeddings",
            status_code_value(status),
            request_started.elapsed(),
        );
        return Ok(());
    }

    if request.method == "POST" && request.path == "/api/generate" {
        let payload: OllamaGenerateRequest = match serde_json::from_slice(&request.body) {
            Ok(payload) => payload,
            Err(err) => {
                let status = "400 Bad Request";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: format!("invalid JSON payload for /api/generate: {err}"),
                    },
                )?;
                metrics.record(
                    "compat.ollama.generate",
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
        };

        let params = GenerationParams {
            max_tokens: payload
                .options
                .as_ref()
                .and_then(|options| options.num_predict)
                .unwrap_or(default_sampling.max_tokens),
            temperature: payload
                .options
                .as_ref()
                .and_then(|options| options.temperature)
                .unwrap_or(default_sampling.temperature),
            top_p: payload
                .options
                .as_ref()
                .and_then(|options| options.top_p)
                .unwrap_or(default_sampling.top_p),
            min_p: default_sampling.min_p,
            top_k: default_sampling.top_k,
            repeat_penalty: default_sampling.repetition_penalty,
        };
        let model_name = payload
            .model
            .unwrap_or_else(|| current_runtime_model_name(engine));
        let prompt_eval_count = estimate_token_count(&payload.prompt);

        if payload.stream.unwrap_or(false) {
            let created_at = compatibility_created_at();
            let mut headers_written = false;

            match generate_stream_with_runtime(engine, plugins, &payload.prompt, params, |chunk| {
                if !headers_written {
                    write_streaming_response_headers(stream, "200 OK", "application/x-ndjson")?;
                    headers_written = true;
                }
                write_ndjson_event(
                    stream,
                    &ollama_stream_event(
                        &model_name,
                        &created_at,
                        chunk.to_string(),
                        false,
                        None,
                        0,
                        0,
                        None,
                    ),
                )
            }) {
                Ok(response) => {
                    let status = "200 OK";
                    if !headers_written {
                        write_streaming_response_headers(stream, status, "application/x-ndjson")?;
                    }
                    write_ndjson_event(
                        stream,
                        &ollama_stream_event(
                            &model_name,
                            &created_at,
                            String::new(),
                            true,
                            Some("stop"),
                            prompt_eval_count,
                            estimate_token_count(&response),
                            None,
                        ),
                    )?;
                    metrics.record(
                        "compat.ollama.generate",
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                }
                Err(err) => {
                    let status = "500 Internal Server Error";
                    if headers_written {
                        let _ = write_ndjson_event(
                            stream,
                            &ollama_stream_event(
                                &model_name,
                                &created_at,
                                String::new(),
                                true,
                                Some("error"),
                                prompt_eval_count,
                                0,
                                Some(err.to_string()),
                            ),
                        );
                    } else {
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                    }
                    metrics.record(
                        "compat.ollama.generate",
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                }
            }
            return Ok(());
        }

        match generate_with_runtime(engine, plugins, &payload.prompt, params) {
            Ok(response) => {
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &OllamaGenerateResponse {
                        model: model_name,
                        created_at: compatibility_created_at(),
                        response: response.clone(),
                        done: true,
                        done_reason: Some("stop"),
                        prompt_eval_count,
                        eval_count: estimate_token_count(&response),
                        error: None,
                    },
                )?;
                metrics.record(
                    "compat.ollama.generate",
                    status_code_value(status),
                    request_started.elapsed(),
                );
            }
            Err(err) => {
                let status = "500 Internal Server Error";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: err.to_string(),
                    },
                )?;
                metrics.record(
                    "compat.ollama.generate",
                    status_code_value(status),
                    request_started.elapsed(),
                );
            }
        }
        return Ok(());
    }

    if let Some(route) = parse_tool_api_route(&request.path) {
        let endpoint = "tools";
        match (request.method.as_str(), route) {
            ("GET", ToolApiRoute::PluginCollection) => {
                let plugins = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard
                        .list_tool_plugins()
                        .into_iter()
                        .map(tool_plugin_descriptor_to_response)
                        .collect::<Vec<_>>()
                };
                let status = "200 OK";
                write_json_response(stream, status, &ToolPluginRegistryListResponse { plugins })?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", ToolApiRoute::PluginItem { name }) => {
                let plugin = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard
                        .list_tool_plugins()
                        .into_iter()
                        .find(|plugin| plugin.name == name)
                };
                let plugin = match plugin {
                    Some(plugin) => plugin,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("tool plugin '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let status = "200 OK";
                write_json_response(stream, status, &tool_plugin_descriptor_to_response(plugin))?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", ToolApiRoute::PluginLoad) => {
                let payload: RuntimePluginLoadRequest = match serde_json::from_slice(&request.body)
                {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /tools/plugins/load: {err}"
                                ),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };

                let descriptor = {
                    let mut guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    match guard.load_dynamic_tool_plugin(PathBuf::from(&payload.path)) {
                        Ok((name, _functions)) => guard
                            .list_tool_plugins()
                            .into_iter()
                            .find(|plugin| plugin.name == name)
                            .ok_or_else(|| {
                                anyhow::anyhow!("tool plugin '{}' missing after load", name)
                            })?,
                        Err(err) => {
                            let status = "400 Bad Request";
                            write_json_response(
                                stream,
                                status,
                                &ErrorResponse {
                                    error: err.to_string(),
                                },
                            )?;
                            publish_runtime_event(
                                runtime_events,
                                RuntimeEventCategory::Plugin,
                                "tools.plugins.load",
                                RuntimeEventOutcome::Error,
                                Some(endpoint),
                                Some(&request),
                                Some(status_code_value(status)),
                                None,
                                Some(serde_json::json!({
                                    "path": payload.path,
                                    "error": err.to_string(),
                                })),
                            );
                            metrics.record(
                                endpoint,
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }
                    }
                };

                {
                    let mut store = tool_plugin_store
                        .lock()
                        .expect("tool plugin registry mutex should not be poisoned");
                    store.add_plugin_path(PathBuf::from(&payload.path));
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting tool plugin registry: {}", e)
                    })?;
                }

                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = ToolPluginRegistryMutationResponse {
                    name: descriptor.name,
                    version: descriptor.version,
                    dynamic: descriptor.dynamic,
                    source: source.clone(),
                    functions: descriptor.function_names,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Plugin,
                    "tools.plugins.load",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "activate_requested": payload.activate.unwrap_or(false),
                        "path": payload.path,
                        "version": response.version,
                        "dynamic": response.dynamic,
                        "source": source,
                        "functions": response.functions,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", ToolApiRoute::Collection) => {
                let tools = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard
                        .function_calling_manager()
                        .list_functions()
                        .into_iter()
                        .cloned()
                        .collect::<Vec<_>>()
                };
                let status = "200 OK";
                write_json_response(stream, status, &ToolListResponse { tools })?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", ToolApiRoute::Item { name }) => {
                let tool = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard
                        .function_calling_manager()
                        .get_function(&name)
                        .cloned()
                };
                let tool = match tool {
                    Some(tool) => tool,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("tool '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let status = "200 OK";
                write_json_response(stream, status, &tool)?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", ToolApiRoute::Invoke) => {
                let payload: ToolInvokeRequest = match serde_json::from_slice(&request.body) {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("invalid JSON payload for /tools/invoke: {err}"),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Tool,
                            "tools.invoke",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let call = loci::function_calling::FunctionCall {
                    name: payload.name.clone(),
                    arguments: payload.arguments,
                };
                let argument_keys = call.arguments.keys().cloned().collect::<Vec<_>>();
                let result = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard.execute_function_call(&call)
                };
                match result {
                    Ok(value) => {
                        let status = "200 OK";
                        let tool_name = call.name.clone();
                        write_json_response(
                            stream,
                            status,
                            &ToolInvokeResponse {
                                tool: tool_name.clone(),
                                ok: true,
                                result: Some(value),
                                error: None,
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Tool,
                            "tools.invoke",
                            RuntimeEventOutcome::Success,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(tool_name),
                            Some(serde_json::json!({
                                "argument_count": argument_keys.len(),
                                "argument_keys": argument_keys,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Err(err) => {
                        let status = "400 Bad Request";
                        let tool_name = call.name.clone();
                        write_json_response(
                            stream,
                            status,
                            &ToolInvokeResponse {
                                tool: tool_name.clone(),
                                ok: false,
                                result: None,
                                error: Some(err.to_string()),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Tool,
                            "tools.invoke",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(tool_name),
                            Some(serde_json::json!({
                                "argument_count": argument_keys.len(),
                                "argument_keys": argument_keys,
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
            }
            ("POST", ToolApiRoute::PluginReload { name }) => {
                let descriptor = {
                    let mut guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    match guard.reload_dynamic_tool_plugin(&name) {
                        Ok((reloaded_name, _functions)) => guard
                            .list_tool_plugins()
                            .into_iter()
                            .find(|plugin| plugin.name == reloaded_name)
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "tool plugin '{}' missing after reload",
                                    reloaded_name
                                )
                            })?,
                        Err(err) => {
                            let status = "400 Bad Request";
                            write_json_response(
                                stream,
                                status,
                                &ErrorResponse {
                                    error: err.to_string(),
                                },
                            )?;
                            publish_runtime_event(
                                runtime_events,
                                RuntimeEventCategory::Plugin,
                                "tools.plugins.reload",
                                RuntimeEventOutcome::Error,
                                Some(endpoint),
                                Some(&request),
                                Some(status_code_value(status)),
                                Some(name.clone()),
                                Some(serde_json::json!({
                                    "error": err.to_string(),
                                })),
                            );
                            metrics.record(
                                endpoint,
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }
                    }
                };

                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = ToolPluginRegistryMutationResponse {
                    name: descriptor.name,
                    version: descriptor.version,
                    dynamic: descriptor.dynamic,
                    source: source.clone(),
                    functions: descriptor.function_names,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Plugin,
                    "tools.plugins.reload",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "version": response.version,
                        "dynamic": response.dynamic,
                        "source": source,
                        "functions": response.functions,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", ToolApiRoute::PluginUnload { name }) => {
                let descriptor = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard
                        .list_tool_plugins()
                        .into_iter()
                        .find(|plugin| plugin.name == name)
                };

                let descriptor = match descriptor {
                    Some(descriptor) => descriptor,
                    None => {
                        let status = "404 Not Found";
                        let error_message = format!("tool plugin '{}' not found", name);
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: error_message.clone(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Plugin,
                            "tools.plugins.unload",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": error_message,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };

                {
                    let mut guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    if let Err(err) = guard.unload_dynamic_tool_plugin(&name) {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Plugin,
                            "tools.plugins.unload",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }

                if let Some(source) = descriptor.source.as_ref() {
                    let mut store = tool_plugin_store
                        .lock()
                        .expect("tool plugin registry mutex should not be poisoned");
                    store.remove_plugin_path(source);
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting tool plugin registry: {}", e)
                    })?;
                }

                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = ToolPluginRegistryMutationResponse {
                    name: descriptor.name,
                    version: descriptor.version,
                    dynamic: descriptor.dynamic,
                    source: source.clone(),
                    functions: descriptor.function_names,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Plugin,
                    "tools.plugins.unload",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "version": response.version,
                        "dynamic": response.dynamic,
                        "source": source,
                        "functions": response.functions,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(route) = parse_dispatch_policy_api_route(&request.path) {
        let endpoint = "dispatch_policies";
        match (request.method.as_str(), route) {
            ("GET", PolicyApiRoute::Collection) => {
                let active_name = active_dispatch_policy.name();
                let policies = dispatch_registry
                    .descriptors()
                    .into_iter()
                    .map(|descriptor| {
                        dispatch_descriptor_to_response(descriptor, Some(active_name.as_str()))
                    })
                    .collect::<Vec<_>>();
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryListResponse {
                        active: Some(active_name),
                        policies,
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", PolicyApiRoute::Item { name }) => {
                let active_name = active_dispatch_policy.name();
                let descriptor = match dispatch_registry.describe(&name) {
                    Some(descriptor) => descriptor,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("dispatch policy '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &dispatch_descriptor_to_response(descriptor, Some(active_name.as_str())),
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Load) => {
                let payload: RuntimePluginLoadRequest = match serde_json::from_slice(&request.body)
                {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /dispatch-policies/load: {err}"
                                ),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let name = match dispatch_registry.load_dynamic_policy(PathBuf::from(&payload.path))
                {
                    Ok(name) => name,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let active = payload.activate.unwrap_or(false);
                if active {
                    let plugin = dispatch_registry.get(&name).ok_or_else(|| {
                        anyhow::anyhow!("dispatch policy '{}' missing after load", name)
                    })?;
                    active_dispatch_policy.activate_plugin(name.clone(), plugin);
                }
                {
                    let mut store = dispatch_policy_store
                        .lock()
                        .expect("dispatch policy registry mutex should not be poisoned");
                    store.add_plugin_path(PathBuf::from(&payload.path));
                    if active {
                        store.set_active(Some(name.clone()));
                    }
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting dispatch policy registry: {}", e)
                    })?;
                }
                let descriptor = dispatch_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("dispatch policy '{}' missing after load", name)
                })?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Activate { name }) => {
                let plugin = match dispatch_registry.get(&name) {
                    Some(plugin) => plugin,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("dispatch policy '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                active_dispatch_policy.activate_plugin(name.clone(), plugin);
                {
                    let mut store = dispatch_policy_store
                        .lock()
                        .expect("dispatch policy registry mutex should not be poisoned");
                    store.set_active(Some(name.clone()));
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting dispatch policy registry: {}", e)
                    })?;
                }
                let descriptor = dispatch_registry
                    .describe(&name)
                    .ok_or_else(|| anyhow::anyhow!("dispatch policy '{}' missing", name))?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: true,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Reload { name }) => {
                if active_dispatch_policy.name() == name {
                    let status = "409 Conflict";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "dispatch policy '{}' is active; switch to another policy before reload",
                                name
                            ),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                match dispatch_registry.reload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                let descriptor = dispatch_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("dispatch policy '{}' missing after reload", name)
                })?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: false,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Unload { name }) => {
                if active_dispatch_policy.name() == name {
                    let status = "409 Conflict";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "dispatch policy '{}' is active; switch to another policy before unload",
                                name
                            ),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let descriptor = dispatch_registry.describe(&name);
                match dispatch_registry.unload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                if let Some(source) = descriptor.as_ref().and_then(|item| item.source.as_ref()) {
                    let mut store = dispatch_policy_store
                        .lock()
                        .expect("dispatch policy registry mutex should not be poisoned");
                    store.remove_plugin_path(source);
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting dispatch policy registry: {}", e)
                    })?;
                }
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: false,
                        source: descriptor
                            .and_then(|item| item.source.map(|path| path.display().to_string())),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(route) = parse_execution_policy_api_route(&request.path) {
        let endpoint = "execution_policies";
        match (request.method.as_str(), route) {
            ("GET", PolicyApiRoute::Collection) => {
                let active_name = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard.execution_policy_name().to_string()
                };
                let policies = execution_policy_registry
                    .descriptors()
                    .into_iter()
                    .map(|descriptor| {
                        execution_descriptor_to_response(descriptor, Some(active_name.as_str()))
                    })
                    .collect::<Vec<_>>();
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryListResponse {
                        active: Some(active_name),
                        policies,
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", PolicyApiRoute::Item { name }) => {
                let active_name = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard.execution_policy_name().to_string()
                };
                let descriptor = match execution_policy_registry.describe(&name) {
                    Some(descriptor) => descriptor,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("execution policy '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &execution_descriptor_to_response(descriptor, Some(active_name.as_str())),
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Load) => {
                let payload: RuntimePluginLoadRequest = match serde_json::from_slice(&request.body)
                {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /execution-policies/load: {err}"
                                ),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let name = match execution_policy_registry
                    .load_dynamic_policy(PathBuf::from(&payload.path))
                {
                    Ok(name) => name,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let active = payload.activate.unwrap_or(false);
                if active {
                    let policy = execution_policy_registry.get(&name).ok_or_else(|| {
                        anyhow::anyhow!("execution policy '{}' missing after load", name)
                    })?;
                    let mut guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard.set_execution_policy_arc(policy);
                }
                {
                    let mut store = execution_policy_store
                        .lock()
                        .expect("execution policy registry mutex should not be poisoned");
                    store.add_plugin_path(PathBuf::from(&payload.path));
                    if active {
                        store.set_active(Some(name.clone()));
                    }
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting execution policy registry: {}", e)
                    })?;
                }
                let descriptor = execution_policy_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("execution policy '{}' missing after load", name)
                })?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Activate { name }) => {
                let policy = match execution_policy_registry.get(&name) {
                    Some(policy) => policy,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("execution policy '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                {
                    let mut guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard.set_execution_policy_arc(policy);
                }
                {
                    let mut store = execution_policy_store
                        .lock()
                        .expect("execution policy registry mutex should not be poisoned");
                    store.set_active(Some(name.clone()));
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting execution policy registry: {}", e)
                    })?;
                }
                let descriptor = execution_policy_registry
                    .describe(&name)
                    .ok_or_else(|| anyhow::anyhow!("execution policy '{}' missing", name))?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: true,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Reload { name }) => {
                let active_name = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard.execution_policy_name().to_string()
                };
                if active_name == name {
                    let status = "409 Conflict";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "execution policy '{}' is active; switch to another policy before reload",
                                name
                            ),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                match execution_policy_registry.reload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                let descriptor = execution_policy_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("execution policy '{}' missing after reload", name)
                })?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: false,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Unload { name }) => {
                let active_name = {
                    let guard = engine
                        .lock()
                        .expect("inference engine mutex should not be poisoned");
                    guard.execution_policy_name().to_string()
                };
                if active_name == name {
                    let status = "409 Conflict";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "execution policy '{}' is active; switch to another policy before unload",
                                name
                            ),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let descriptor = execution_policy_registry.describe(&name);
                match execution_policy_registry.unload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                if let Some(source) = descriptor.as_ref().and_then(|item| item.source.as_ref()) {
                    let mut store = execution_policy_store
                        .lock()
                        .expect("execution policy registry mutex should not be poisoned");
                    store.remove_plugin_path(source);
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting execution policy registry: {}", e)
                    })?;
                }
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: false,
                        source: descriptor
                            .and_then(|item| item.source.map(|path| path.display().to_string())),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(route) = parse_auth_policy_api_route(&request.path) {
        let endpoint = "auth_policies";
        match (request.method.as_str(), route) {
            ("GET", PolicyApiRoute::Collection) => {
                let active_name = active_management_auth_policy.name();
                let policies = management_auth_registry
                    .descriptors()
                    .into_iter()
                    .map(|descriptor| {
                        auth_descriptor_to_response(descriptor, Some(active_name.as_str()))
                    })
                    .collect::<Vec<_>>();
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryListResponse {
                        active: Some(active_name),
                        policies,
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", PolicyApiRoute::Item { name }) => {
                let active_name = active_management_auth_policy.name();
                let descriptor = match management_auth_registry.describe(&name) {
                    Some(descriptor) => descriptor,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("management auth policy '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &auth_descriptor_to_response(descriptor, Some(active_name.as_str())),
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Load) => {
                let payload: RuntimePluginLoadRequest = match serde_json::from_slice(&request.body)
                {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /auth-policies/load: {err}"
                                ),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let name = match management_auth_registry
                    .load_dynamic_policy(PathBuf::from(&payload.path))
                {
                    Ok(name) => name,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let active = payload.activate.unwrap_or(false);
                let policy = if active {
                    let policy = management_auth_registry.get(&name).ok_or_else(|| {
                        anyhow::anyhow!("management auth policy '{}' missing after load", name)
                    })?;
                    if let Err(err) = ensure_candidate_management_policy_authorizes_request(
                        &management_auth_context,
                        &name,
                        policy.as_ref(),
                    ) {
                        let _ = management_auth_registry.unload_dynamic_policy(&name);
                        let status = "409 Conflict";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                    Some(policy)
                } else {
                    None
                };
                {
                    let mut store = management_auth_store
                        .lock()
                        .expect("management auth registry mutex should not be poisoned");
                    store.add_plugin_path(PathBuf::from(&payload.path));
                    if active {
                        store.set_active(Some(name.clone()));
                    }
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting management auth registry: {}", e)
                    })?;
                }
                if let Some(policy) = policy {
                    active_management_auth_policy.activate(name.clone(), policy);
                }
                let descriptor = management_auth_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("management auth policy '{}' missing after load", name)
                })?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Activate { name }) => {
                let policy = match management_auth_registry.get(&name) {
                    Some(policy) => policy,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("management auth policy '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                if let Err(err) = ensure_candidate_management_policy_authorizes_request(
                    &management_auth_context,
                    &name,
                    policy.as_ref(),
                ) {
                    let status = "409 Conflict";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                active_management_auth_policy.activate(name.clone(), policy);
                {
                    let mut store = management_auth_store
                        .lock()
                        .expect("management auth registry mutex should not be poisoned");
                    store.set_active(Some(name.clone()));
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting management auth registry: {}", e)
                    })?;
                }
                let descriptor = management_auth_registry
                    .describe(&name)
                    .ok_or_else(|| anyhow::anyhow!("management auth policy '{}' missing", name))?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: true,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Reload { name }) => {
                if active_management_auth_policy.name() == name {
                    let status = "409 Conflict";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "management auth policy '{}' is active; switch to another policy before reload",
                                name
                            ),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                match management_auth_registry.reload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                let descriptor = management_auth_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("management auth policy '{}' missing after reload", name)
                })?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: false,
                        source: descriptor.source.map(|path| path.display().to_string()),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Unload { name }) => {
                if active_management_auth_policy.name() == name {
                    let status = "409 Conflict";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: format!(
                                "management auth policy '{}' is active; switch to another policy before unload",
                                name
                            ),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let descriptor = management_auth_registry.describe(&name);
                match management_auth_registry.unload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                if let Some(source) = descriptor.as_ref().and_then(|item| item.source.as_ref()) {
                    let mut store = management_auth_store
                        .lock()
                        .expect("management auth registry mutex should not be poisoned");
                    store.remove_plugin_path(source);
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting management auth registry: {}", e)
                    })?;
                }
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryMutationResponse {
                        name,
                        active: false,
                        source: descriptor
                            .and_then(|item| item.source.map(|path| path.display().to_string())),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(route) = parse_model_pull_policy_api_route(&request.path) {
        let endpoint = "model_pull_policies";
        match (request.method.as_str(), route) {
            ("GET", PolicyApiRoute::Collection) => {
                let active_name = active_model_pull_policy.name();
                let policies = model_pull_policy_registry
                    .descriptors()
                    .into_iter()
                    .map(|descriptor| {
                        model_pull_descriptor_to_response(descriptor, Some(active_name.as_str()))
                    })
                    .collect::<Vec<_>>();
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryListResponse {
                        active: Some(active_name),
                        policies,
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", PolicyApiRoute::Item { name }) => {
                let active_name = active_model_pull_policy.name();
                let descriptor = match model_pull_policy_registry.describe(&name) {
                    Some(descriptor) => descriptor,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("model pull policy '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &model_pull_descriptor_to_response(descriptor, Some(active_name.as_str())),
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Load) => {
                let payload: RuntimePluginLoadRequest = match serde_json::from_slice(&request.body)
                {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /model-pull-policies/load: {err}"
                                ),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_policies.load",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let name = match model_pull_policy_registry
                    .load_dynamic_policy(PathBuf::from(&payload.path))
                {
                    Ok(name) => name,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_policies.load",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "path": payload.path,
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let active = payload.activate.unwrap_or(false);
                if active {
                    let policy = model_pull_policy_registry.get(&name).ok_or_else(|| {
                        anyhow::anyhow!("model pull policy '{}' missing after load", name)
                    })?;
                    active_model_pull_policy.activate(name.clone(), policy);
                }
                {
                    let mut store = model_pull_policy_store
                        .lock()
                        .expect("model pull policy registry mutex should not be poisoned");
                    store.add_plugin_path(PathBuf::from(&payload.path));
                    if active {
                        store.set_active(Some(name.clone()));
                    }
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting model pull policy registry: {}", e)
                    })?;
                }
                let descriptor = model_pull_policy_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("model pull policy '{}' missing after load", name)
                })?;
                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_policies.load",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                        "path": payload.path,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Activate { name }) => {
                let policy = match model_pull_policy_registry.get(&name) {
                    Some(policy) => policy,
                    None => {
                        let status = "404 Not Found";
                        let error_message = format!("model pull policy '{}' not found", name);
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: error_message.clone(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_policies.activate",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": error_message,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                active_model_pull_policy.activate(name.clone(), policy);
                {
                    let mut store = model_pull_policy_store
                        .lock()
                        .expect("model pull policy registry mutex should not be poisoned");
                    store.set_active(Some(name.clone()));
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting model pull policy registry: {}", e)
                    })?;
                }
                let descriptor = model_pull_policy_registry
                    .describe(&name)
                    .ok_or_else(|| anyhow::anyhow!("model pull policy '{}' missing", name))?;
                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active: true,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_policies.activate",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Reload { name }) => {
                if active_model_pull_policy.name() == name {
                    let status = "409 Conflict";
                    let error_message = format!(
                        "model pull policy '{}' is active; switch to another policy before reload",
                        name
                    );
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: error_message.clone(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Policy,
                        "model_pull_policies.reload",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(name.clone()),
                        Some(serde_json::json!({
                            "error": error_message,
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                match model_pull_policy_registry.reload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_policies.reload",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                let descriptor = model_pull_policy_registry.describe(&name).ok_or_else(|| {
                    anyhow::anyhow!("model pull policy '{}' missing after reload", name)
                })?;
                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active: false,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_policies.reload",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Unload { name }) => {
                if active_model_pull_policy.name() == name {
                    let status = "409 Conflict";
                    let error_message = format!(
                        "model pull policy '{}' is active; switch to another policy before unload",
                        name
                    );
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: error_message.clone(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Policy,
                        "model_pull_policies.unload",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(name.clone()),
                        Some(serde_json::json!({
                            "error": error_message,
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let descriptor = model_pull_policy_registry.describe(&name);
                match model_pull_policy_registry.unload_dynamic_policy(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_policies.unload",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                if let Some(source) = descriptor.as_ref().and_then(|item| item.source.as_ref()) {
                    let mut store = model_pull_policy_store
                        .lock()
                        .expect("model pull policy registry mutex should not be poisoned");
                    store.remove_plugin_path(source);
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting model pull policy registry: {}", e)
                    })?;
                }
                let status = "200 OK";
                let source = descriptor
                    .as_ref()
                    .and_then(|item| item.source.as_ref())
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active: false,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_policies.unload",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(route) = parse_model_pull_verifier_api_route(&request.path) {
        let endpoint = "model_pull_verifiers";
        match (request.method.as_str(), route) {
            ("GET", PolicyApiRoute::Collection) => {
                let active_name = active_model_pull_verifier.name();
                let policies = model_pull_verifier_registry
                    .descriptors()
                    .into_iter()
                    .map(|descriptor| {
                        model_pull_verifier_descriptor_to_response(
                            descriptor,
                            Some(active_name.as_str()),
                        )
                    })
                    .collect::<Vec<_>>();
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &PolicyRegistryListResponse {
                        active: Some(active_name),
                        policies,
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", PolicyApiRoute::Item { name }) => {
                let active_name = active_model_pull_verifier.name();
                let descriptor = match model_pull_verifier_registry.describe(&name) {
                    Some(descriptor) => descriptor,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("model pull verifier '{}' not found", name),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &model_pull_verifier_descriptor_to_response(
                        descriptor,
                        Some(active_name.as_str()),
                    ),
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Load) => {
                let payload: RuntimePluginLoadRequest = match serde_json::from_slice(&request.body)
                {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /model-pull-verifiers/load: {err}"
                                ),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_verifiers.load",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let name = match model_pull_verifier_registry
                    .load_dynamic_verifier(PathBuf::from(&payload.path))
                {
                    Ok(name) => name,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_verifiers.load",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "path": payload.path,
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let active = payload.activate.unwrap_or(false);
                if active {
                    let verifier = model_pull_verifier_registry.get(&name).ok_or_else(|| {
                        anyhow::anyhow!("model pull verifier '{}' missing after load", name)
                    })?;
                    active_model_pull_verifier.activate(name.clone(), verifier);
                }
                {
                    let mut store = model_pull_verifier_store
                        .lock()
                        .expect("model pull verifier registry mutex should not be poisoned");
                    store.add_plugin_path(PathBuf::from(&payload.path));
                    if active {
                        store.set_active(Some(name.clone()));
                    }
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting model pull verifier registry: {}", e)
                    })?;
                }
                let descriptor = model_pull_verifier_registry
                    .describe(&name)
                    .ok_or_else(|| anyhow::anyhow!("model pull verifier '{}' missing", name))?;
                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_verifiers.load",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                        "path": payload.path,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Activate { name }) => {
                let verifier = match model_pull_verifier_registry.get(&name) {
                    Some(verifier) => verifier,
                    None => {
                        let status = "404 Not Found";
                        let error_message = format!("model pull verifier '{}' not found", name);
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: error_message.clone(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_verifiers.activate",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": error_message,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                active_model_pull_verifier.activate(name.clone(), verifier);
                {
                    let mut store = model_pull_verifier_store
                        .lock()
                        .expect("model pull verifier registry mutex should not be poisoned");
                    store.set_active(Some(name.clone()));
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting model pull verifier registry: {}", e)
                    })?;
                }
                let descriptor = model_pull_verifier_registry
                    .describe(&name)
                    .ok_or_else(|| anyhow::anyhow!("model pull verifier '{}' missing", name))?;
                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active: true,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_verifiers.activate",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Reload { name }) => {
                if active_model_pull_verifier.name() == name {
                    let status = "409 Conflict";
                    let error_message = format!(
                        "model pull verifier '{}' is active; switch to another verifier before reload",
                        name
                    );
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: error_message.clone(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Policy,
                        "model_pull_verifiers.reload",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(name.clone()),
                        Some(serde_json::json!({
                            "error": error_message,
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                match model_pull_verifier_registry.reload_dynamic_verifier(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_verifiers.reload",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                let descriptor = model_pull_verifier_registry
                    .describe(&name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("model pull verifier '{}' missing after reload", name)
                    })?;
                let status = "200 OK";
                let source = descriptor
                    .source
                    .as_ref()
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active: false,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_verifiers.reload",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", PolicyApiRoute::Unload { name }) => {
                if active_model_pull_verifier.name() == name {
                    let status = "409 Conflict";
                    let error_message = format!(
                        "model pull verifier '{}' is active; switch to another verifier before unload",
                        name
                    );
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: error_message.clone(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Policy,
                        "model_pull_verifiers.unload",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(name.clone()),
                        Some(serde_json::json!({
                            "error": error_message,
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let descriptor = model_pull_verifier_registry.describe(&name);
                match model_pull_verifier_registry.unload_dynamic_verifier(&name) {
                    Ok(()) => {}
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Policy,
                            "model_pull_verifiers.unload",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(name.clone()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                if let Some(source) = descriptor.as_ref().and_then(|item| item.source.as_ref()) {
                    let mut store = model_pull_verifier_store
                        .lock()
                        .expect("model pull verifier registry mutex should not be poisoned");
                    store.remove_plugin_path(source);
                    store.persist().map_err(|e| {
                        anyhow::anyhow!("failed persisting model pull verifier registry: {}", e)
                    })?;
                }
                let status = "200 OK";
                let source = descriptor
                    .as_ref()
                    .and_then(|item| item.source.as_ref())
                    .map(|path| path.display().to_string());
                let response = PolicyRegistryMutationResponse {
                    name,
                    active: false,
                    source: source.clone(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Policy,
                    "model_pull_verifiers.unload",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.name.clone()),
                    Some(serde_json::json!({
                        "active": response.active,
                        "source": source,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {}
        }
    }

    if let Some(route) = parse_session_api_route(&request.path) {
        let endpoint = "sessions";
        match (request.method.as_str(), route) {
            ("GET", SessionApiRoute::Collection) => {
                let active = session_manager
                    .list_sessions()
                    .into_iter()
                    .map(|info| session_info_to_summary(&info))
                    .collect::<Vec<_>>();
                let mut persisted = session_manager
                    .list_persisted_sessions()
                    .map_err(|e| anyhow::anyhow!("failed listing persisted sessions: {}", e))?
                    .into_iter()
                    .map(|id| id.as_u64())
                    .collect::<Vec<_>>();
                persisted.sort_unstable();
                let status = "200 OK";
                write_json_response(stream, status, &SessionListResponse { active, persisted })?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Collection) => {
                let payload: SessionCreateRequest = match serde_json::from_slice(&request.body) {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("invalid JSON payload for /sessions create: {err}"),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Session,
                            "sessions.create",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            None,
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let model_path = payload.model.as_ref().map(PathBuf::from);
                let model = match resolve_model_reference(
                    model_path.as_deref(),
                    payload.model_id.as_deref(),
                    model_store_root,
                ) {
                    Ok(path) => path,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Session,
                            "sessions.create",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            payload.model_id.clone().or_else(|| payload.model.clone()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let context_size = payload.context_size.unwrap_or(4096);
                let model_id = session_manager
                    .load_model(model.to_string_lossy().as_ref(), context_size)
                    .map_err(|e| anyhow::anyhow!("failed loading model for session: {}", e))?;
                let session_id = session_manager
                    .create_session(model_id)
                    .map_err(|e| anyhow::anyhow!("failed creating session: {}", e))?;
                let persisted = payload.save.unwrap_or(true);
                if persisted {
                    session_manager
                        .save_session(session_id)
                        .map_err(|e| anyhow::anyhow!("failed persisting created session: {}", e))?;
                }
                let status = "200 OK";
                let response = SessionCreateResponse {
                    session_id: session_id.as_u64(),
                    model_path: model.display().to_string(),
                    model_id: model_id.as_u64(),
                    context_size,
                    persisted,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.create",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.session_id.to_string()),
                    Some(serde_json::json!({
                        "model_path": response.model_path,
                        "model_id": response.model_id,
                        "context_size": response.context_size,
                        "persisted": response.persisted,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("GET", SessionApiRoute::Item { session_id }) => {
                if let Err(err) = ensure_session_loaded(session_manager, session_id) {
                    let status = "404 Not Found";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let handle = match session_manager.get_session(session_id) {
                    Some(handle) => handle,
                    None => {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!("session {} not found", session_id.as_u64()),
                            },
                        )?;
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let info = handle
                    .info()
                    .map_err(|e| anyhow::anyhow!("failed reading session info: {}", e))?;
                let records = handle
                    .records()
                    .map_err(|e| anyhow::anyhow!("failed reading session records: {}", e))?;
                let status = "200 OK";
                write_json_response(
                    stream,
                    status,
                    &SessionDetailResponse {
                        session: session_info_to_summary(&info),
                        records,
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Generate { session_id }) => {
                let payload: SessionGenerateRequest = match serde_json::from_slice(&request.body) {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /sessions/{{id}}/generate: {err}"
                                ),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Session,
                            "sessions.generate",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(session_id.as_u64().to_string()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                if let Err(err) = ensure_session_loaded(session_manager, session_id) {
                    let status = "404 Not Found";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Session,
                        "sessions.generate",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(session_id.as_u64().to_string()),
                        Some(serde_json::json!({
                            "error": err.to_string(),
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let handle = match session_manager.get_session(session_id) {
                    Some(handle) => handle,
                    None => {
                        let status = "404 Not Found";
                        let error_message = format!("session {} not found", session_id.as_u64());
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: error_message.clone(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Session,
                            "sessions.generate",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(session_id.as_u64().to_string()),
                            Some(serde_json::json!({
                                "error": error_message,
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                let prompt = {
                    let plugins_guard = plugins
                        .lock()
                        .expect("plugin registry mutex should not be poisoned");
                    match apply_pre_generate(&payload.prompt, plugins_guard.as_ref()) {
                        Ok(prompt) => prompt,
                        Err(err) => {
                            let status = "500 Internal Server Error";
                            write_json_response(
                                stream,
                                status,
                                &ErrorResponse {
                                    error: err.to_string(),
                                },
                            )?;
                            publish_runtime_event(
                                runtime_events,
                                RuntimeEventCategory::Session,
                                "sessions.generate",
                                RuntimeEventOutcome::Error,
                                Some(endpoint),
                                Some(&request),
                                Some(status_code_value(status)),
                                Some(session_id.as_u64().to_string()),
                                Some(serde_json::json!({
                                    "error": err.to_string(),
                                })),
                            );
                            metrics.record(
                                endpoint,
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }
                    }
                };
                let max_tokens = payload.max_tokens.unwrap_or(default_sampling.max_tokens);
                let response = handle
                    .generate(&prompt, max_tokens as usize)
                    .map_err(|e| anyhow::anyhow!("session generation failed: {}", e))?;
                let response = {
                    let plugins_guard = plugins
                        .lock()
                        .expect("plugin registry mutex should not be poisoned");
                    apply_post_generate(&response, plugins_guard.as_ref())?
                };
                let persisted = payload.save.unwrap_or(true);
                if persisted {
                    session_manager
                        .save_session(session_id)
                        .map_err(|e| anyhow::anyhow!("failed persisting session: {}", e))?;
                }
                let state = handle
                    .info()
                    .map(|info| format!("{:?}", info.state))
                    .unwrap_or_else(|_| "Unknown".to_string());
                let status = "200 OK";
                let response_payload = SessionGenerateResponse {
                    session_id: session_id.as_u64(),
                    response,
                    persisted,
                    state,
                };
                write_json_response(stream, status, &response_payload)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.generate",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response_payload.session_id.to_string()),
                    Some(serde_json::json!({
                        "persisted": response_payload.persisted,
                        "state": response_payload.state,
                        "max_tokens": max_tokens,
                        "prompt_chars": prompt.chars().count(),
                        "response_chars": response_payload.response.chars().count(),
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Suspend { session_id }) => {
                let payload: SessionSuspendRequest = match serde_json::from_slice(&request.body) {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /sessions/{{id}}/suspend: {err}"
                                ),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Session,
                            "sessions.suspend",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(session_id.as_u64().to_string()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                if let Err(err) = ensure_session_loaded(session_manager, session_id) {
                    let status = "404 Not Found";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Session,
                        "sessions.suspend",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(session_id.as_u64().to_string()),
                        Some(serde_json::json!({
                            "error": err.to_string(),
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let handle = session_manager
                    .get_session(session_id)
                    .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
                handle
                    .suspend(payload.reason.clone(), payload.data.clone())
                    .map_err(|e| anyhow::anyhow!("failed suspending session: {}", e))?;
                let persisted = payload.save.unwrap_or(true);
                if persisted {
                    session_manager
                        .save_session(session_id)
                        .map_err(|e| anyhow::anyhow!("failed persisting session: {}", e))?;
                }
                let state = handle
                    .info()
                    .map(|info| format!("{:?}", info.state))
                    .unwrap_or_else(|_| "Unknown".to_string());
                let status = "200 OK";
                let response = SessionMutationResponse {
                    session_id: session_id.as_u64(),
                    persisted,
                    state,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.suspend",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.session_id.to_string()),
                    Some(serde_json::json!({
                        "persisted": response.persisted,
                        "state": response.state,
                        "reason": payload.reason,
                        "has_data": payload.data.is_some(),
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Resume { session_id }) => {
                let payload: SessionResumeRequest = match serde_json::from_slice(&request.body) {
                    Ok(payload) => payload,
                    Err(err) => {
                        let status = "400 Bad Request";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: format!(
                                    "invalid JSON payload for /sessions/{{id}}/resume: {err}"
                                ),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Session,
                            "sessions.resume",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(session_id.as_u64().to_string()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                };
                if let Err(err) = ensure_session_loaded(session_manager, session_id) {
                    let status = "404 Not Found";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Session,
                        "sessions.resume",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(session_id.as_u64().to_string()),
                        Some(serde_json::json!({
                            "error": err.to_string(),
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let handle = session_manager
                    .get_session(session_id)
                    .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
                let external_data_chars = payload.external_data.chars().count();
                handle
                    .resume(payload.external_data)
                    .map_err(|e| anyhow::anyhow!("failed resuming session: {}", e))?;
                let persisted = payload.save.unwrap_or(true);
                if persisted {
                    session_manager
                        .save_session(session_id)
                        .map_err(|e| anyhow::anyhow!("failed persisting session: {}", e))?;
                }
                let state = handle
                    .info()
                    .map(|info| format!("{:?}", info.state))
                    .unwrap_or_else(|_| "Unknown".to_string());
                let status = "200 OK";
                let response = SessionMutationResponse {
                    session_id: session_id.as_u64(),
                    persisted,
                    state,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.resume",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.session_id.to_string()),
                    Some(serde_json::json!({
                        "persisted": response.persisted,
                        "state": response.state,
                        "external_data_chars": external_data_chars,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Save { session_id }) => {
                if let Err(err) = ensure_session_loaded(session_manager, session_id) {
                    let status = "404 Not Found";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Session,
                        "sessions.save",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(session_id.as_u64().to_string()),
                        Some(serde_json::json!({
                            "error": err.to_string(),
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                session_manager
                    .save_session(session_id)
                    .map_err(|e| anyhow::anyhow!("failed saving session: {}", e))?;
                let state = session_manager
                    .get_session(session_id)
                    .and_then(|handle| handle.info().ok().map(|x| format!("{:?}", x.state)))
                    .unwrap_or_else(|| "Unknown".to_string());
                let status = "200 OK";
                let response = SessionMutationResponse {
                    session_id: session_id.as_u64(),
                    persisted: true,
                    state,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.save",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.session_id.to_string()),
                    Some(serde_json::json!({
                        "persisted": response.persisted,
                        "state": response.state,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Restore { session_id }) => {
                if !session_manager.has_session(session_id) {
                    if let Err(err) = session_manager.restore_session(session_id) {
                        let status = "404 Not Found";
                        write_json_response(
                            stream,
                            status,
                            &ErrorResponse {
                                error: err.to_string(),
                            },
                        )?;
                        publish_runtime_event(
                            runtime_events,
                            RuntimeEventCategory::Session,
                            "sessions.restore",
                            RuntimeEventOutcome::Error,
                            Some(endpoint),
                            Some(&request),
                            Some(status_code_value(status)),
                            Some(session_id.as_u64().to_string()),
                            Some(serde_json::json!({
                                "error": err.to_string(),
                            })),
                        );
                        metrics.record(
                            endpoint,
                            status_code_value(status),
                            request_started.elapsed(),
                        );
                        return Ok(());
                    }
                }
                let state = session_manager
                    .get_session(session_id)
                    .and_then(|handle| handle.info().ok().map(|x| format!("{:?}", x.state)))
                    .unwrap_or_else(|| "Unknown".to_string());
                let status = "200 OK";
                let response = SessionMutationResponse {
                    session_id: session_id.as_u64(),
                    persisted: true,
                    state,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.restore",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.session_id.to_string()),
                    Some(serde_json::json!({
                        "persisted": response.persisted,
                        "state": response.state,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Clear { session_id }) => {
                let payload = if request.body.is_empty() {
                    SessionClearRequest { save: None }
                } else {
                    match serde_json::from_slice::<SessionClearRequest>(&request.body) {
                        Ok(payload) => payload,
                        Err(err) => {
                            let status = "400 Bad Request";
                            write_json_response(
                                stream,
                                status,
                                &ErrorResponse {
                                    error: format!(
                                        "invalid JSON payload for /sessions/{{id}}/clear: {err}"
                                    ),
                                },
                            )?;
                            publish_runtime_event(
                                runtime_events,
                                RuntimeEventCategory::Session,
                                "sessions.clear",
                                RuntimeEventOutcome::Error,
                                Some(endpoint),
                                Some(&request),
                                Some(status_code_value(status)),
                                Some(session_id.as_u64().to_string()),
                                Some(serde_json::json!({
                                    "error": err.to_string(),
                                })),
                            );
                            metrics.record(
                                endpoint,
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }
                    }
                };
                if let Err(err) = ensure_session_loaded(session_manager, session_id) {
                    let status = "404 Not Found";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Session,
                        "sessions.clear",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(session_id.as_u64().to_string()),
                        Some(serde_json::json!({
                            "error": err.to_string(),
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                let handle = session_manager
                    .get_session(session_id)
                    .ok_or_else(|| anyhow::anyhow!("session {} not found", session_id.as_u64()))?;
                handle
                    .clear_context()
                    .map_err(|e| anyhow::anyhow!("failed clearing session context: {}", e))?;
                let persisted = payload.save.unwrap_or(true);
                if persisted {
                    session_manager
                        .save_session(session_id)
                        .map_err(|e| anyhow::anyhow!("failed persisting session: {}", e))?;
                }
                let state = handle
                    .info()
                    .map(|info| format!("{:?}", info.state))
                    .unwrap_or_else(|_| "Unknown".to_string());
                let status = "200 OK";
                let response = SessionMutationResponse {
                    session_id: session_id.as_u64(),
                    persisted,
                    state,
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.clear",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.session_id.to_string()),
                    Some(serde_json::json!({
                        "persisted": response.persisted,
                        "state": response.state,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            ("POST", SessionApiRoute::Destroy { session_id })
            | ("DELETE", SessionApiRoute::Destroy { session_id }) => {
                if let Err(err) = ensure_session_loaded(session_manager, session_id) {
                    let status = "404 Not Found";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    publish_runtime_event(
                        runtime_events,
                        RuntimeEventCategory::Session,
                        "sessions.destroy",
                        RuntimeEventOutcome::Error,
                        Some(endpoint),
                        Some(&request),
                        Some(status_code_value(status)),
                        Some(session_id.as_u64().to_string()),
                        Some(serde_json::json!({
                            "error": err.to_string(),
                        })),
                    );
                    metrics.record(
                        endpoint,
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
                session_manager
                    .destroy_session(session_id)
                    .map_err(|e| anyhow::anyhow!("failed destroying session: {}", e))?;
                let status = "200 OK";
                let response = SessionMutationResponse {
                    session_id: session_id.as_u64(),
                    persisted: false,
                    state: "Destroyed".to_string(),
                };
                write_json_response(stream, status, &response)?;
                publish_runtime_event(
                    runtime_events,
                    RuntimeEventCategory::Session,
                    "sessions.destroy",
                    RuntimeEventOutcome::Success,
                    Some(endpoint),
                    Some(&request),
                    Some(status_code_value(status)),
                    Some(response.session_id.to_string()),
                    Some(serde_json::json!({
                        "persisted": response.persisted,
                        "state": response.state,
                    })),
                );
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
            _ => {
                let status = "405 Method Not Allowed";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: "method not allowed for session endpoint".to_string(),
                    },
                )?;
                metrics.record(
                    endpoint,
                    status_code_value(status),
                    request_started.elapsed(),
                );
                return Ok(());
            }
        }
    }

    if request.method == "POST" && (request.path == "/generate" || request.path == "/v1/generate") {
        let payload: GenerateRequest = match serde_json::from_slice(&request.body) {
            Ok(payload) => payload,
            Err(err) => {
                let status = "400 Bad Request";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: format!("invalid JSON payload for /generate: {err}"),
                    },
                )?;
                metrics.record(
                    "generate",
                    status_code_value(status),
                    request_started.elapsed(),
                );
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
        let prompt = {
            let plugins_guard = plugins
                .lock()
                .expect("plugin registry mutex should not be poisoned");
            match apply_pre_generate(&payload.prompt, plugins_guard.as_ref()) {
                Ok(prompt) => prompt,
                Err(err) => {
                    let status = "500 Internal Server Error";
                    write_json_response(
                        stream,
                        status,
                        &ErrorResponse {
                            error: err.to_string(),
                        },
                    )?;
                    metrics.record(
                        "generate",
                        status_code_value(status),
                        request_started.elapsed(),
                    );
                    return Ok(());
                }
            }
        };

        let generation = {
            let mut engine_guard = engine
                .lock()
                .expect("inference engine mutex should not be poisoned");
            engine_guard.generate(&prompt, params)
        };

        match generation {
            Ok(response) => {
                let response = {
                    let plugins_guard = plugins
                        .lock()
                        .expect("plugin registry mutex should not be poisoned");
                    match apply_post_generate(&response, plugins_guard.as_ref()) {
                        Ok(response) => response,
                        Err(err) => {
                            let status = "500 Internal Server Error";
                            write_json_response(
                                stream,
                                status,
                                &ErrorResponse {
                                    error: err.to_string(),
                                },
                            )?;
                            metrics.record(
                                "generate",
                                status_code_value(status),
                                request_started.elapsed(),
                            );
                            return Ok(());
                        }
                    }
                };
                let status = "200 OK";
                write_json_response(stream, status, &GenerateResponse { response })?;
                metrics.record(
                    "generate",
                    status_code_value(status),
                    request_started.elapsed(),
                );
            }
            Err(err) => {
                let status = "500 Internal Server Error";
                write_json_response(
                    stream,
                    status,
                    &ErrorResponse {
                        error: err.to_string(),
                    },
                )?;
                metrics.record(
                    "generate",
                    status_code_value(status),
                    request_started.elapsed(),
                );
            }
        }
        return Ok(());
    }

    let status = "404 Not Found";
    write_json_response(
        stream,
        status,
        &ErrorResponse {
            error: "endpoint not found".to_string(),
        },
    )?;
    metrics.record(
        "not_found",
        status_code_value(status),
        request_started.elapsed(),
    );
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
    let mut headers = HashMap::new();
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
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
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

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
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

fn write_streaming_response_headers(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
) -> anyhow::Result<()> {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(header.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn write_sse_json_event<T: Serialize>(stream: &mut TcpStream, payload: &T) -> anyhow::Result<()> {
    let body = serde_json::to_string(payload)?;
    stream.write_all(b"data: ")?;
    stream.write_all(body.as_bytes())?;
    stream.write_all(b"\n\n")?;
    stream.flush()?;
    Ok(())
}

fn write_sse_done(stream: &mut TcpStream) -> anyhow::Result<()> {
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()?;
    Ok(())
}

fn write_ndjson_event<T: Serialize>(stream: &mut TcpStream, payload: &T) -> anyhow::Result<()> {
    let body = serde_json::to_string(payload)?;
    stream.write_all(body.as_bytes())?;
    stream.write_all(b"\n")?;
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

fn generate_with_runtime(
    engine: &Arc<Mutex<InferenceEngine>>,
    plugins: &Arc<Mutex<Option<PluginRegistry>>>,
    prompt: &str,
    params: GenerationParams,
) -> anyhow::Result<String> {
    let prompt = {
        let plugins_guard = plugins
            .lock()
            .expect("plugin registry mutex should not be poisoned");
        apply_pre_generate(prompt, plugins_guard.as_ref())?
    };

    let response = {
        let mut engine_guard = engine
            .lock()
            .expect("inference engine mutex should not be poisoned");
        engine_guard.generate(&prompt, params)?
    };

    let plugins_guard = plugins
        .lock()
        .expect("plugin registry mutex should not be poisoned");
    apply_post_generate(&response, plugins_guard.as_ref())
}

fn generate_stream_with_runtime<F>(
    engine: &Arc<Mutex<InferenceEngine>>,
    plugins: &Arc<Mutex<Option<PluginRegistry>>>,
    prompt: &str,
    params: GenerationParams,
    mut on_chunk: F,
) -> anyhow::Result<String>
where
    F: FnMut(&str) -> anyhow::Result<()>,
{
    let supports_streaming = {
        let engine_guard = engine
            .lock()
            .expect("inference engine mutex should not be poisoned");
        engine_guard.supports_streaming()
    };

    let plugin_count = {
        let plugins_guard = plugins
            .lock()
            .expect("plugin registry mutex should not be poisoned");
        plugins_guard.as_ref().map_or(0, PluginRegistry::count)
    };

    if !supports_streaming || plugin_count > 0 {
        let response = generate_with_runtime(engine, plugins, prompt, params)?;
        for chunk in chunk_text_for_streaming(&response) {
            on_chunk(&chunk)?;
        }
        return Ok(response);
    }

    let plugins_guard = plugins
        .lock()
        .expect("plugin registry mutex should not be poisoned");
    let processed_prompt = apply_pre_generate(prompt, plugins_guard.as_ref())?;

    let mut streamed = String::new();
    let mut callback_error: Option<anyhow::Error> = None;
    {
        let mut engine_guard = engine
            .lock()
            .expect("inference engine mutex should not be poisoned");
        engine_guard.generate_stream(&processed_prompt, params, |token| {
            let processed_token = match apply_on_token(token, plugins_guard.as_ref()) {
                Ok(token) => token,
                Err(err) => {
                    callback_error = Some(err);
                    return false;
                }
            };

            streamed.push_str(&processed_token);
            if let Err(err) = on_chunk(&processed_token) {
                callback_error = Some(err);
                return false;
            }
            true
        })?;
    }

    if let Some(err) = callback_error {
        return Err(err);
    }

    Ok(streamed)
}

fn current_runtime_model_name(engine: &Arc<Mutex<InferenceEngine>>) -> String {
    let guard = engine
        .lock()
        .expect("inference engine mutex should not be poisoned");
    let metadata = guard.model_metadata();
    format!("loci/{}:{}", guard.backend_name(), metadata.architecture)
}

fn compatibility_usage(prompt: &str, response: &str) -> OpenAiUsage {
    let prompt_tokens = estimate_token_count(prompt);
    let completion_tokens = estimate_token_count(response);
    OpenAiUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Cursor, Read};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_file(ext: &str, content: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("loci-cli-config-{nonce}.{ext}"));
        fs::write(&path, content).expect("write temp config");
        path
    }

    fn tcp_stream_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback listener");
        let addr = listener.local_addr().expect("listener address");
        let client = TcpStream::connect(addr).expect("connect loopback client");
        let (server, _) = listener.accept().expect("accept loopback connection");
        (server, client)
    }

    fn capture_tcp_output<F>(writer: F) -> String
    where
        F: FnOnce(&mut TcpStream),
    {
        let (mut server, mut client) = tcp_stream_pair();
        writer(&mut server);
        server
            .shutdown(Shutdown::Write)
            .expect("shutdown server write");
        let mut output = String::new();
        client.read_to_string(&mut output).expect("read tcp output");
        output
    }

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
    fn parse_agent_skill_and_skill_pack_args() {
        let cli = Cli::try_parse_from([
            "loci",
            "agent",
            "--model",
            "model.gguf",
            "--tool",
            "all",
            "--skill",
            "reasoner",
            "--skill-pack",
            "skills.json",
            "--prompt",
            "hello",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Agent(cmd)) => {
                assert_eq!(cmd.skill.as_deref(), Some("reasoner"));
                assert_eq!(cmd.skill_packs.len(), 1);
                assert_eq!(
                    cmd.skill_packs[0].to_string_lossy().to_string(),
                    "skills.json".to_string()
                );
            }
            _ => panic!("expected agent command"),
        }
    }

    #[test]
    fn parse_agent_mcp_stdio_arg() {
        let cli = Cli::try_parse_from([
            "loci",
            "agent",
            "--model",
            "model.gguf",
            "--tool",
            "all",
            "--mcp-stdio",
            "fs=npx -y @modelcontextprotocol/server-filesystem C:/tmp",
            "--prompt",
            "list files",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Agent(cmd)) => {
                assert_eq!(cmd.mcp_stdio.len(), 1);
                assert!(cmd.mcp_stdio[0].starts_with("fs="));
            }
            _ => panic!("expected agent command"),
        }
    }

    #[test]
    fn parse_agent_mcp_registry_and_server_args() {
        let cli = Cli::try_parse_from([
            "loci",
            "agent",
            "--model",
            "model.gguf",
            "--tool",
            "all",
            "--mcp-registry",
            "loci_mcp.toml",
            "--mcp-server",
            "fs",
            "--prompt",
            "list files",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Agent(cmd)) => {
                assert_eq!(
                    cmd.mcp_registry
                        .as_deref()
                        .map(|p| p.to_string_lossy().to_string()),
                    Some("loci_mcp.toml".to_string())
                );
                assert_eq!(cmd.mcp_servers, vec!["fs".to_string()]);
            }
            _ => panic!("expected agent command"),
        }
    }

    #[test]
    fn parse_agent_tool_plugin_arg() {
        let cli = Cli::try_parse_from([
            "loci",
            "agent",
            "--model",
            "model.gguf",
            "--tool",
            "all",
            "--tool-plugin",
            "browser_tool_plugin.dll",
            "--prompt",
            "open example.com",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Agent(cmd)) => {
                assert_eq!(cmd.tool_plugins.len(), 1);
                assert_eq!(
                    cmd.tool_plugins[0].to_string_lossy().to_string(),
                    "browser_tool_plugin.dll".to_string()
                );
            }
            _ => panic!("expected agent command"),
        }
    }

    #[test]
    fn parse_multimodal_command_basic() {
        let cli = Cli::try_parse_from([
            "loci",
            "multimodal",
            "--model",
            "model.gguf",
            "--prompt",
            "describe this",
            "--image-input",
            "a.png",
            "--output-modality",
            "text",
            "--output-modality",
            "image",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Multimodal(cmd)) => {
                assert_eq!(cmd.image_inputs.len(), 1);
                assert_eq!(cmd.output_modalities.len(), 2);
                assert_eq!(cmd.mm_plugin_name, "descriptor");
            }
            _ => panic!("expected multimodal command"),
        }
    }

    #[test]
    fn parse_orchestrate_route_command_with_multiple_models() {
        let cli = Cli::try_parse_from([
            "loci",
            "orchestrate",
            "--model",
            "m1.gguf",
            "--model",
            "m2.gguf",
            "--prompt",
            "summarize this image",
            "--mode",
            "route",
            "--routing-strategy",
            "round-robin",
            "--image-input",
            "a.png",
            "--audio-input",
            "a.wav",
            "--output-modality",
            "text",
            "--output-modality",
            "image",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Orchestrate(cmd)) => {
                assert_eq!(cmd.models.len(), 2);
                assert_eq!(cmd.mode, OrchestrationModeArg::Route);
                assert_eq!(cmd.routing_strategy, RoutingStrategyArg::RoundRobin);
                assert_eq!(cmd.image_inputs.len(), 1);
                assert_eq!(cmd.audio_inputs.len(), 1);
                assert_eq!(cmd.output_modalities.len(), 2);
            }
            _ => panic!("expected orchestrate command"),
        }
    }

    #[test]
    fn parse_orchestrate_ensemble_judge_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "orchestrate",
            "--model",
            "m1.gguf",
            "--model",
            "m2.gguf",
            "--prompt",
            "solve this step by step",
            "--mode",
            "ensemble",
            "--ensemble-merge",
            "judge",
            "--judge-model",
            "judge.gguf",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Orchestrate(cmd)) => {
                assert_eq!(cmd.mode, OrchestrationModeArg::Ensemble);
                assert_eq!(cmd.ensemble_merge, EnsembleMergeArg::Judge);
                assert_eq!(
                    cmd.judge_model
                        .as_deref()
                        .map(|p| p.to_string_lossy().to_string()),
                    Some("judge.gguf".to_string())
                );
            }
            _ => panic!("expected orchestrate command"),
        }
    }

    #[test]
    fn parse_cli_from_json_config_with_args() {
        let path = write_temp_file(
            "json",
            r#"{"args":["orchestrate","--model","m1.gguf","--model","m2.gguf","--prompt","hello"]}"#,
        );
        let cli = parse_cli_from_config_file(&path).expect("config parse should succeed");
        let _ = fs::remove_file(path);

        match cli.command {
            Some(Commands::Orchestrate(cmd)) => {
                assert_eq!(cmd.models.len(), 2);
                assert_eq!(cmd.prompt, "hello");
            }
            _ => panic!("expected orchestrate command"),
        }
    }

    #[test]
    fn parse_cli_from_toml_config_with_commandline() {
        let path = write_temp_file(
            "toml",
            r#"commandline = "multimodal --model model.gguf --prompt \"describe\" --output-modality text""#,
        );
        let cli = parse_cli_from_config_file(&path).expect("config parse should succeed");
        let _ = fs::remove_file(path);

        match cli.command {
            Some(Commands::Multimodal(cmd)) => {
                assert_eq!(
                    cmd.model
                        .as_deref()
                        .map(|p| p.to_string_lossy().to_string()),
                    Some("model.gguf".to_string())
                );
                assert_eq!(cmd.prompt, "describe");
                assert_eq!(cmd.output_modalities, vec![ModalOutputArg::Text]);
            }
            _ => panic!("expected multimodal command"),
        }
    }

    #[test]
    fn parse_cli_from_plain_text_config() {
        let path = write_temp_file("txt", "generate --model model.gguf --prompt hello");
        let cli = parse_cli_from_config_file(&path).expect("config parse should succeed");
        let _ = fs::remove_file(path);

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert_eq!(
                    cmd.model
                        .as_deref()
                        .map(|p| p.to_string_lossy().to_string()),
                    Some("model.gguf".to_string())
                );
                assert_eq!(cmd.prompt.as_deref(), Some("hello"));
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_model_id_and_store() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model-id",
            "qwen-base",
            "--model-store",
            "custom_models",
            "--prompt",
            "hi",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert!(cmd.model.is_none());
                assert_eq!(cmd.model_id.as_deref(), Some("qwen-base"));
                assert_eq!(
                    cmd.model_store.to_string_lossy().to_string(),
                    "custom_models"
                );
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_session_create_model_id() {
        let cli = Cli::try_parse_from([
            "loci",
            "session",
            "--store-kind",
            "sqlite",
            "--store-option",
            "path=sessions/test.db",
            "create",
            "--model-id",
            "qwen-base",
            "--context-length",
            "8192",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Session(cmd)) => {
                assert_eq!(cmd.store_kind, "sqlite");
                assert_eq!(cmd.store_options, vec!["path=sessions/test.db".to_string()]);
                match cmd.command {
                    SessionAction::Create {
                        model,
                        model_id,
                        context_size,
                        no_save,
                    } => {
                        assert!(model.is_none());
                        assert_eq!(model_id.as_deref(), Some("qwen-base"));
                        assert_eq!(context_size, 8192);
                        assert!(!no_save);
                    }
                    _ => panic!("expected session create action"),
                }
            }
            _ => panic!("expected session command"),
        }
    }

    #[test]
    fn parse_serve_worker_queue_and_backpressure() {
        let cli = Cli::try_parse_from([
            "loci",
            "serve",
            "--model",
            "model.gguf",
            "--workers",
            "8",
            "--queue-size",
            "256",
            "--backpressure",
            "block",
            "--backpressure-plugin",
            "serve_dispatch_plugin.dll",
            "--backpressure-registry",
            "loci_dispatch_policies.toml",
            "--backpressure-policy-name",
            "adaptive.retry",
            "--session-store-kind",
            "sqlite",
            "--session-store-option",
            "path=sessions/server.db",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Serve(cmd)) => {
                assert_eq!(cmd.workers, 8);
                assert_eq!(cmd.queue_size, 256);
                assert_eq!(cmd.backpressure, ServeBackpressureArg::Block);
                assert_eq!(
                    cmd.backpressure_plugins
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect::<Vec<_>>(),
                    vec!["serve_dispatch_plugin.dll".to_string()]
                );
                assert_eq!(
                    cmd.backpressure_registry.to_string_lossy().to_string(),
                    "loci_dispatch_policies.toml"
                );
                assert_eq!(
                    cmd.backpressure_policy_name.as_deref(),
                    Some("adaptive.retry")
                );
                assert_eq!(cmd.session_store_kind, "sqlite");
                assert_eq!(
                    cmd.session_store_options,
                    vec!["path=sessions/server.db".to_string()]
                );
            }
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn parse_session_api_route_collection_and_item() {
        assert!(matches!(
            parse_session_api_route("/sessions"),
            Some(SessionApiRoute::Collection)
        ));
        assert!(matches!(
            parse_session_api_route("/v1/sessions"),
            Some(SessionApiRoute::Collection)
        ));
        assert!(matches!(
            parse_session_api_route("/sessions/42"),
            Some(SessionApiRoute::Item { session_id }) if session_id.as_u64() == 42
        ));
        assert!(matches!(
            parse_session_api_route("/v1/sessions/7/generate"),
            Some(SessionApiRoute::Generate { session_id }) if session_id.as_u64() == 7
        ));
    }

    #[test]
    fn parse_dispatch_policy_api_routes() {
        assert!(matches!(
            parse_dispatch_policy_api_route("/dispatch-policies"),
            Some(PolicyApiRoute::Collection)
        ));
        assert!(matches!(
            parse_dispatch_policy_api_route("/v1/dispatch-policies/load"),
            Some(PolicyApiRoute::Load)
        ));
        assert!(matches!(
            parse_dispatch_policy_api_route("/dispatch-policies/reject/activate"),
            Some(PolicyApiRoute::Activate { ref name }) if name == "reject"
        ));
    }

    #[test]
    fn parse_execution_policy_api_routes() {
        assert!(matches!(
            parse_execution_policy_api_route("/execution-policies"),
            Some(PolicyApiRoute::Collection)
        ));
        assert!(matches!(
            parse_execution_policy_api_route("/v1/execution-policies/default.execution.policy"),
            Some(PolicyApiRoute::Item { ref name }) if name == "default.execution.policy"
        ));
        assert!(matches!(
            parse_execution_policy_api_route("/execution-policies/default.execution.policy/unload"),
            Some(PolicyApiRoute::Unload { ref name }) if name == "default.execution.policy"
        ));
    }

    #[test]
    fn parse_auth_policy_api_routes() {
        assert!(matches!(
            parse_auth_policy_api_route("/auth-policies"),
            Some(PolicyApiRoute::Collection)
        ));
        assert!(matches!(
            parse_auth_policy_api_route("/v1/auth-policies/load"),
            Some(PolicyApiRoute::Load)
        ));
        assert!(matches!(
            parse_auth_policy_api_route("/auth-policies/loopback-only.management.auth/activate"),
            Some(PolicyApiRoute::Activate { ref name }) if name == "loopback-only.management.auth"
        ));
    }

    #[test]
    fn parse_model_pull_policy_api_routes() {
        assert!(matches!(
            parse_model_pull_policy_api_route("/model-pull-policies"),
            Some(PolicyApiRoute::Collection)
        ));
        assert!(matches!(
            parse_model_pull_policy_api_route("/v1/model-pull-policies/load"),
            Some(PolicyApiRoute::Load)
        ));
        assert!(matches!(
            parse_model_pull_policy_api_route("/model-pull-policies/allow-all.model.pull"),
            Some(PolicyApiRoute::Item { ref name }) if name == "allow-all.model.pull"
        ));
        assert!(matches!(
            parse_model_pull_policy_api_route(
                "/model-pull-policies/checksum-required-remote.model.pull/activate"
            ),
            Some(PolicyApiRoute::Activate { ref name })
                if name == "checksum-required-remote.model.pull"
        ));
    }

    #[test]
    fn parse_model_pull_verifier_api_routes() {
        assert!(matches!(
            parse_model_pull_verifier_api_route("/model-pull-verifiers"),
            Some(PolicyApiRoute::Collection)
        ));
        assert!(matches!(
            parse_model_pull_verifier_api_route("/v1/model-pull-verifiers/load"),
            Some(PolicyApiRoute::Load)
        ));
        assert!(matches!(
            parse_model_pull_verifier_api_route("/model-pull-verifiers/allow-all.model.verify"),
            Some(PolicyApiRoute::Item { ref name }) if name == "allow-all.model.verify"
        ));
        assert!(matches!(
            parse_model_pull_verifier_api_route(
                "/model-pull-verifiers/sidecar-sha256.model.verify/activate"
            ),
            Some(PolicyApiRoute::Activate { ref name })
                if name == "sidecar-sha256.model.verify"
        ));
    }

    #[test]
    fn parse_tool_api_routes() {
        assert!(matches!(
            parse_tool_api_route("/tools"),
            Some(ToolApiRoute::Collection)
        ));
        assert!(matches!(
            parse_tool_api_route("/v1/tools/invoke"),
            Some(ToolApiRoute::Invoke)
        ));
        assert!(matches!(
            parse_tool_api_route("/tools/browser_open_session"),
            Some(ToolApiRoute::Item { ref name }) if name == "browser_open_session"
        ));
        assert!(matches!(
            parse_tool_api_route("/tools/plugins"),
            Some(ToolApiRoute::PluginCollection)
        ));
        assert!(matches!(
            parse_tool_api_route("/v1/tools/plugins/load"),
            Some(ToolApiRoute::PluginLoad)
        ));
        assert!(matches!(
            parse_tool_api_route("/tools/plugins/browser_tool_plugin/unload"),
            Some(ToolApiRoute::PluginUnload { ref name }) if name == "browser_tool_plugin"
        ));
    }

    #[test]
    fn parse_runtime_events_api_routes() {
        assert!(matches!(
            parse_runtime_events_api_route("/events"),
            Some(RuntimeEventsApiRoute::Collection)
        ));
        assert!(matches!(
            parse_runtime_events_api_route("/v1/events/stream"),
            Some(RuntimeEventsApiRoute::Stream)
        ));
        assert!(parse_runtime_events_api_route("/events/live").is_none());
    }

    #[test]
    fn parse_dispatch_policy_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "dispatch-policy",
            "--registry",
            "dispatch.toml",
            "--plugin",
            "serve_dispatch_plugin.dll",
            "info",
            "reject",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::DispatchPolicy(cmd)) => {
                assert_eq!(cmd.registry.to_string_lossy().to_string(), "dispatch.toml");
                assert_eq!(
                    cmd.plugins
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect::<Vec<_>>(),
                    vec!["serve_dispatch_plugin.dll".to_string()]
                );
                match cmd.command {
                    DispatchPolicyAction::Info { name } => assert_eq!(name, "reject"),
                    _ => panic!("expected dispatch policy info action"),
                }
            }
            _ => panic!("expected dispatch policy command"),
        }
    }

    #[test]
    fn parse_execution_policy_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "execution-policy",
            "--registry",
            "execution.toml",
            "--plugin",
            "execution_policy_plugin.dll",
            "reload",
            "execution.policy.trace",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::ExecutionPolicy(cmd)) => {
                assert_eq!(cmd.registry.to_string_lossy().to_string(), "execution.toml");
                assert_eq!(
                    cmd.plugins
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect::<Vec<_>>(),
                    vec!["execution_policy_plugin.dll".to_string()]
                );
                match cmd.command {
                    ExecutionPolicyAction::Reload { name } => {
                        assert_eq!(name, "execution.policy.trace")
                    }
                    _ => panic!("expected execution policy reload action"),
                }
            }
            _ => panic!("expected execution policy command"),
        }
    }

    #[test]
    fn parse_auth_policy_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "auth-policy",
            "--registry",
            "management-auth.toml",
            "--plugin",
            "management_auth_plugin.dll",
            "--bearer-token",
            "secret",
            "activate",
            "bearer-token.management.auth",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::AuthPolicy(cmd)) => {
                assert_eq!(
                    cmd.registry.to_string_lossy().to_string(),
                    "management-auth.toml"
                );
                assert_eq!(cmd.bearer_token.as_deref(), Some("secret"));
                assert_eq!(
                    cmd.plugins
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect::<Vec<_>>(),
                    vec!["management_auth_plugin.dll".to_string()]
                );
                match cmd.command {
                    AuthPolicyAction::Activate { name } => {
                        assert_eq!(name, "bearer-token.management.auth")
                    }
                    _ => panic!("expected auth policy activate action"),
                }
            }
            _ => panic!("expected auth policy command"),
        }
    }

    #[test]
    fn parse_serve_management_auth_scope() {
        let cli = Cli::try_parse_from([
            "loci",
            "serve",
            "--model",
            "model.gguf",
            "--management-auth-policy-name",
            "loopback-only.management.auth",
            "--management-auth-scope",
            "custom",
            "--management-auth-prefix",
            "/tools",
            "--management-auth-prefix",
            "/browser",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Serve(cmd)) => {
                assert_eq!(
                    cmd.management_auth_policy_name.as_deref(),
                    Some("loopback-only.management.auth")
                );
                assert_eq!(
                    cmd.management_auth_scope,
                    Some(ManagementAuthScopeArg::Custom)
                );
                assert_eq!(
                    cmd.management_auth_prefixes,
                    vec!["/tools".to_string(), "/browser".to_string()]
                );
            }
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn parse_serve_tool_and_mcp_args() {
        let cli = Cli::try_parse_from([
            "loci",
            "serve",
            "--model",
            "model.gguf",
            "--tool-plugin-registry",
            "tool-plugins.toml",
            "--tool-plugin",
            "browser_tool_plugin.dll",
            "--mcp-stdio",
            "fs=npx -y @modelcontextprotocol/server-filesystem C:/tmp",
            "--mcp-registry",
            "mcp.toml",
            "--mcp-server",
            "fs",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Serve(cmd)) => {
                assert_eq!(
                    cmd.tool_plugins
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect::<Vec<_>>(),
                    vec!["browser_tool_plugin.dll".to_string()]
                );
                assert_eq!(
                    cmd.tool_plugin_registry.to_string_lossy().to_string(),
                    "tool-plugins.toml"
                );
                assert_eq!(
                    cmd.mcp_stdio,
                    vec!["fs=npx -y @modelcontextprotocol/server-filesystem C:/tmp".to_string()]
                );
                assert_eq!(
                    cmd.mcp_registry
                        .as_ref()
                        .map(|p| p.to_string_lossy().to_string()),
                    Some("mcp.toml".to_string())
                );
                assert_eq!(cmd.mcp_servers, vec!["fs".to_string()]);
            }
            _ => panic!("expected serve command"),
        }
    }

    #[test]
    fn parse_execution_policy_engine_args() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--execution-policy-registry",
            "execution.toml",
            "--execution-policy-plugin",
            "execution_policy_plugin.dll",
            "--execution-policy-name",
            "execution.policy.trace",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert_eq!(
                    cmd.engine
                        .execution_policy_registry
                        .to_string_lossy()
                        .to_string(),
                    "execution.toml"
                );
                assert_eq!(
                    cmd.engine
                        .execution_policy_plugins
                        .iter()
                        .map(|p| p.to_string_lossy().to_string())
                        .collect::<Vec<_>>(),
                    vec!["execution_policy_plugin.dll".to_string()]
                );
                assert_eq!(
                    cmd.engine.execution_policy_name.as_deref(),
                    Some("execution.policy.trace")
                );
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn management_auth_scope_control_plane_matches_builtin_routes() {
        let scope = ManagementAuthScopeConfig::from_args(ManagementAuthScopeArg::ControlPlane, &[])
            .expect("scope");
        assert!(scope.requires_auth("/sessions"));
        assert!(scope.requires_auth("/models/assets"));
        assert!(scope.requires_auth("/v1/models/assets/demo"));
        assert!(scope.requires_auth("/v1/execution-policies/default.execution.policy"));
        assert!(scope.requires_auth("/model-pull-policies/checksum-required-remote.model.pull"));
        assert!(scope.requires_auth("/model-pull-verifiers/sidecar-sha256.model.verify"));
        assert!(scope.requires_auth("/events"));
        assert!(scope.requires_auth("/v1/events/stream"));
        assert!(scope.requires_auth("/models/plan"));
        assert!(scope.requires_auth("/v1/models/plan"));
        assert!(scope.requires_auth("/auth-policies"));
        assert!(scope.requires_auth("/tools/open"));
        assert!(scope.requires_auth("/browser/open"));
        assert!(scope.requires_auth("/v1/device/keyboard/type"));
        assert!(!scope.requires_auth("/generate"));
        assert!(!scope.requires_auth("/metrics"));
        assert!(!scope.requires_auth("/openapi.yaml"));
        assert!(!scope.requires_auth("/openapi.json"));
        assert!(!scope.requires_auth("/v1/openapi.yaml"));
        assert!(!scope.requires_auth("/v1/openapi.json"));
    }

    #[test]
    fn management_auth_scope_all_matches_every_request() {
        let scope =
            ManagementAuthScopeConfig::from_args(ManagementAuthScopeArg::All, &[]).expect("scope");
        assert!(scope.requires_auth("/generate"));
        assert!(scope.requires_auth("/health"));
    }

    #[test]
    fn management_auth_scope_custom_matches_prefixes_and_v1_aliases() {
        let scope = ManagementAuthScopeConfig::from_args(
            ManagementAuthScopeArg::Custom,
            &["/tools/".to_string(), "/browser".to_string()],
        )
        .expect("scope");
        assert!(scope.requires_auth("/tools/open"));
        assert!(scope.requires_auth("/v1/tools/open"));
        assert!(scope.requires_auth("/browser"));
        assert!(!scope.requires_auth("/toolsmith"));
        assert!(!scope.requires_auth("/generate"));
    }

    #[test]
    fn management_auth_scope_custom_requires_prefixes() {
        let err = ManagementAuthScopeConfig::from_args(ManagementAuthScopeArg::Custom, &[])
            .expect_err("custom scope without prefixes should fail");
        assert!(err
            .to_string()
            .contains("--management-auth-scope=custom requires at least one"));
    }

    #[test]
    fn resolve_management_auth_scope_uses_registry_when_cli_absent() {
        let mut store = DynamicPolicyRegistry::new();
        store.set_scope(Some("custom".to_string()));
        store.set_prefixes(vec!["/browser".to_string(), "/device".to_string()]);

        let (scope, persist) = resolve_management_auth_scope(None, &[], &store).expect("scope");
        assert!(!persist);
        assert_eq!(
            scope,
            ManagementAuthScopeConfig::Custom(vec!["/browser".to_string(), "/device".to_string()])
        );
    }

    #[test]
    fn resolve_management_auth_scope_defaults_to_control_plane() {
        let store = DynamicPolicyRegistry::new();
        let (scope, persist) = resolve_management_auth_scope(None, &[], &store).expect("scope");
        assert!(!persist);
        assert_eq!(scope, ManagementAuthScopeConfig::ControlPlane);
    }

    struct AllowTestManagementPolicy;

    impl ManagementAuthPolicyPlugin for AllowTestManagementPolicy {
        fn name(&self) -> &str {
            "allow.test.management.auth"
        }

        fn authorize(&self, _context: &ManagementAuthContext) -> ManagementAuthDecision {
            ManagementAuthDecision::Allow
        }
    }

    struct DenyTestManagementPolicy;

    impl ManagementAuthPolicyPlugin for DenyTestManagementPolicy {
        fn name(&self) -> &str {
            "deny.test.management.auth"
        }

        fn authorize(&self, _context: &ManagementAuthContext) -> ManagementAuthDecision {
            ManagementAuthDecision::Deny("blocked".to_string())
        }
    }

    #[test]
    fn candidate_management_policy_self_protection_accepts_allowed_request() {
        let context = ManagementAuthContext {
            method: "POST".to_string(),
            path: "/auth-policies/demo/activate".to_string(),
            headers: HashMap::new(),
            remote_addr: Some("127.0.0.1:9000".to_string()),
        };
        ensure_candidate_management_policy_authorizes_request(
            &context,
            "allow.test.management.auth",
            &AllowTestManagementPolicy,
        )
        .expect("candidate policy should allow current request");
    }

    #[test]
    fn candidate_management_policy_self_protection_rejects_lockout() {
        let context = ManagementAuthContext {
            method: "POST".to_string(),
            path: "/auth-policies/demo/activate".to_string(),
            headers: HashMap::new(),
            remote_addr: Some("127.0.0.1:9000".to_string()),
        };
        let err = ensure_candidate_management_policy_authorizes_request(
            &context,
            "deny.test.management.auth",
            &DenyTestManagementPolicy,
        )
        .expect_err("candidate policy should be rejected");
        assert!(err.to_string().contains("current request would be denied"));
    }

    #[test]
    fn parse_mcp_connect_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "mcp",
            "connect",
            "fs=npx -y @modelcontextprotocol/server-filesystem C:/tmp",
            "--tool-prefix",
            "mcp.fs.",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Mcp(cmd)) => match cmd.command {
                McpAction::Connect {
                    spec,
                    tool_prefix,
                    probe,
                    save,
                } => {
                    assert!(spec.starts_with("fs="));
                    assert_eq!(tool_prefix.as_deref(), Some("mcp.fs."));
                    assert!(probe);
                    assert!(save);
                }
                _ => panic!("expected mcp connect action"),
            },
            _ => panic!("expected mcp command"),
        }
    }

    #[test]
    fn parse_mcp_spec_with_quotes() {
        let cfg =
            parse_mcp_stdio_spec("fs=npx -y \"@modelcontextprotocol/server-filesystem\" C:/data")
                .expect("spec parse should succeed");
        assert_eq!(cfg.server_name, "fs");
        assert_eq!(cfg.command, "npx");
        assert_eq!(cfg.args.len(), 3);
        assert_eq!(cfg.args[1], "@modelcontextprotocol/server-filesystem");
    }

    #[test]
    fn split_shell_words_keeps_windows_backslashes() {
        let words = split_shell_words("npx cmd C:\\tmp\\workspace").expect("split should succeed");
        assert_eq!(words, vec!["npx", "cmd", "C:\\tmp\\workspace"]);
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
    fn parse_model_pull_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "model",
            "--store",
            "models",
            "pull",
            "qwen.gguf",
            "--id",
            "qwen-base",
            "--tag",
            "reasoning",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Model(cmd)) => match cmd.command {
                ModelAction::Pull {
                    source,
                    mirrors,
                    id,
                    name,
                    sha256,
                    no_resume,
                    tags,
                } => {
                    assert_eq!(cmd.store.to_string_lossy().to_string(), "models");
                    assert_eq!(source, "qwen.gguf".to_string());
                    assert!(mirrors.is_empty());
                    assert_eq!(id.as_deref(), Some("qwen-base"));
                    assert!(name.is_none());
                    assert!(sha256.is_none());
                    assert!(!no_resume);
                    assert_eq!(tags, vec!["reasoning".to_string()]);
                }
                _ => panic!("expected model pull action"),
            },
            _ => panic!("expected model command"),
        }
    }

    #[test]
    fn parse_model_list_json_command() {
        let cli =
            Cli::try_parse_from(["loci", "model", "list", "--json"]).expect("parse should succeed");

        match cli.command {
            Some(Commands::Model(cmd)) => match cmd.command {
                ModelAction::List { json } => assert!(json),
                _ => panic!("expected model list action"),
            },
            _ => panic!("expected model command"),
        }
    }

    #[test]
    fn parse_model_pull_with_mirror_sha_and_no_resume() {
        let cli = Cli::try_parse_from([
            "loci",
            "model",
            "pull",
            "https://example.com/main.gguf",
            "--mirror",
            "https://mirror-a/model.gguf",
            "--mirror",
            "D:/models/fallback.gguf",
            "--sha256",
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--no-resume",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Model(cmd)) => match cmd.command {
                ModelAction::Pull {
                    source,
                    mirrors,
                    sha256,
                    no_resume,
                    ..
                } => {
                    assert_eq!(source, "https://example.com/main.gguf".to_string());
                    assert_eq!(
                        mirrors,
                        vec![
                            "https://mirror-a/model.gguf".to_string(),
                            "D:/models/fallback.gguf".to_string()
                        ]
                    );
                    assert_eq!(
                        sha256.as_deref(),
                        Some(
                            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        )
                    );
                    assert!(no_resume);
                }
                _ => panic!("expected model pull action"),
            },
            _ => panic!("expected model command"),
        }
    }

    #[test]
    fn server_metrics_snapshot_collects_counts() {
        let metrics = ServerMetrics::new();
        metrics.record("health", 200, Duration::from_millis(10));
        metrics.record("generate", 500, Duration::from_millis(30));
        metrics.record("not_found", 404, Duration::from_millis(20));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_requests, 3);
        assert_eq!(snapshot.total_client_errors, 1);
        assert_eq!(snapshot.total_server_errors, 1);
        assert!(snapshot.average_latency_ms >= 20.0);
        assert_eq!(snapshot.endpoint_hits.get("health"), Some(&1));
        assert_eq!(snapshot.endpoint_hits.get("generate"), Some(&1));
        assert_eq!(snapshot.endpoint_hits.get("not_found"), Some(&1));
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
    fn parse_tensor_split_accepts_ratio_list() {
        assert_eq!(parse_tensor_split("3,2,1").unwrap(), vec![3.0, 2.0, 1.0]);
        assert_eq!(
            parse_tensor_split("1.5, 0.5 , 0").unwrap(),
            vec![1.5, 0.5, 0.0]
        );
    }

    #[test]
    fn parse_tensor_split_rejects_invalid_values() {
        assert!(parse_tensor_split("").is_err());
        assert!(parse_tensor_split("0,0,0").is_err());
        assert!(parse_tensor_split("1,-1").is_err());
        assert!(parse_tensor_split("1,abc").is_err());
    }

    #[test]
    fn parse_positive_u32_rejects_zero() {
        assert_eq!(parse_positive_u32("8").unwrap(), 8);
        assert!(parse_positive_u32("0").is_err());
    }

    #[test]
    fn parse_generate_tiered_loading_flags() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--no-mmap",
            "--mlock",
            "--no-kv-offload",
            "--no-op-offload",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert!(!cmd.engine.resolved_mmap());
                assert!(cmd.engine.resolved_mlock());
                assert!(!cmd.engine.resolved_kv_offload());
                assert!(!cmd.engine.resolved_op_offload());
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_multi_gpu_split_flags() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--gpu-split-mode",
            "row",
            "--main-gpu",
            "1",
            "--tensor-split",
            "3,2,1",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert_eq!(cmd.engine.gpu_split_mode, GpuSplitModeArg::Row);
                assert_eq!(cmd.engine.resolved_gpu_split_mode(), GpuSplitMode::Row);
                assert_eq!(cmd.engine.main_gpu, 1);
                assert_eq!(
                    cmd.engine.resolved_tensor_split(),
                    Some(vec![3.0, 2.0, 1.0])
                );
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_multi_gpu_split_defaults() {
        let cli = Cli::try_parse_from(["loci", "generate", "--model", "model.gguf"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert_eq!(cmd.engine.gpu_split_mode, GpuSplitModeArg::Layer);
                assert_eq!(cmd.engine.resolved_gpu_split_mode(), GpuSplitMode::Layer);
                assert_eq!(cmd.engine.main_gpu, 0);
                assert!(cmd.engine.resolved_tensor_split().is_none());
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_gpu_fallback_flags() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--auto-gpu-fallback",
            "--gpu-fallback-step",
            "12",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert!(cmd.engine.auto_gpu_fallback);
                assert_eq!(cmd.engine.gpu_fallback_step, 12);
                assert_eq!(cmd.engine.resolved_gpu_fallback_step(), Some(12));
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_generate_auto_resource_plan_flag() {
        let cli = Cli::try_parse_from([
            "loci",
            "generate",
            "--model",
            "model.gguf",
            "--auto-resource-plan",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert!(cmd.engine.auto_resource_plan);
                assert!(cmd.engine.resolved_auto_resource_plan());
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_model_plan_command() {
        let cli = Cli::try_parse_from([
            "loci",
            "model",
            "plan",
            "--model",
            "model.gguf",
            "--context-length",
            "8192",
            "--json",
        ])
        .expect("parse should succeed");

        match cli.command {
            Some(Commands::Model(cmd)) => match cmd.command {
                ModelAction::Plan {
                    model,
                    model_id,
                    context_size,
                    json,
                } => {
                    assert_eq!(model, Some(PathBuf::from("model.gguf")));
                    assert_eq!(model_id, None);
                    assert_eq!(context_size, 8192);
                    assert!(json);
                }
                _ => panic!("expected model plan command"),
            },
            _ => panic!("expected model command"),
        }
    }

    #[test]
    fn parse_model_store_api_route_collection_item_and_pull() {
        assert!(matches!(
            parse_model_store_api_route("/models/assets"),
            Some(ModelStoreApiRoute::Collection)
        ));
        assert!(matches!(
            parse_model_store_api_route("/v1/models/assets"),
            Some(ModelStoreApiRoute::Collection)
        ));
        assert!(matches!(
            parse_model_store_api_route("/models/assets/pull"),
            Some(ModelStoreApiRoute::Pull)
        ));
        assert!(matches!(
            parse_model_store_api_route("/models/assets/pulls"),
            Some(ModelStoreApiRoute::PullJobsCollection)
        ));
        assert!(matches!(
            parse_model_store_api_route("/models/assets/pulls/job-1"),
            Some(ModelStoreApiRoute::PullJobItem { ref id }) if id == "job-1"
        ));
        assert!(matches!(
            parse_model_store_api_route("/models/assets/pulls/job-1/cancel"),
            Some(ModelStoreApiRoute::PullJobCancel { ref id }) if id == "job-1"
        ));
        assert!(matches!(
            parse_model_store_api_route("/v1/models/assets/pulls/job-1/events"),
            Some(ModelStoreApiRoute::PullJobEvents { ref id }) if id == "job-1"
        ));
        assert!(matches!(
            parse_model_store_api_route("/v1/models/assets/demo-id"),
            Some(ModelStoreApiRoute::Item { ref id }) if id == "demo-id"
        ));
        assert!(parse_model_store_api_route("/models/assets/demo-id/remove").is_none());
    }

    #[test]
    fn request_query_bool_parses_common_values() {
        assert_eq!(
            request_query_bool("/models/assets/id?delete_file=true", "delete_file"),
            Some(true)
        );
        assert_eq!(
            request_query_bool("/models/assets/id?delete_file=1", "delete_file"),
            Some(true)
        );
        assert_eq!(
            request_query_bool("/models/assets/id?delete_file=false", "delete_file"),
            Some(false)
        );
        assert_eq!(
            request_query_bool("/models/assets/id?other=x", "delete_file"),
            None
        );
        assert_eq!(
            request_query_bool("/models/assets/id?delete_file=maybe", "delete_file"),
            None
        );
    }

    #[test]
    fn request_query_usize_parses_limit_and_replay() {
        assert_eq!(request_query_usize("/events?limit=50", "limit"), Some(50));
        assert_eq!(
            request_query_usize("/events/stream?replay=128&follow=false", "replay"),
            Some(128)
        );
        assert_eq!(request_query_usize("/events?limit=x", "limit"), None);
        assert_eq!(request_query_usize("/events?other=1", "limit"), None);
    }

    #[test]
    fn parse_generate_tiered_loading_defaults() {
        let cli = Cli::try_parse_from(["loci", "generate", "--model", "model.gguf"])
            .expect("parse should succeed");

        match cli.command {
            Some(Commands::Generate(cmd)) => {
                assert!(cmd.engine.resolved_mmap());
                assert!(!cmd.engine.resolved_mlock());
                assert!(cmd.engine.resolved_kv_offload());
                assert!(cmd.engine.resolved_op_offload());
            }
            _ => panic!("expected generate command"),
        }
    }

    #[test]
    fn parse_http_request_from_reader_success() {
        let raw = b"POST /generate HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer abc\r\nContent-Length: 18\r\n\r\n{\"prompt\":\"hello\"}";
        let mut reader = BufReader::new(Cursor::new(raw.as_slice()));
        let request = parse_http_request_from_reader(&mut reader).expect("parse should succeed");

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/generate");
        assert_eq!(
            request.headers.get("authorization").map(|v| v.as_str()),
            Some("Bearer abc")
        );
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

    #[test]
    fn write_plain_response_over_tcp_emits_complete_http_payload() {
        let raw = capture_tcp_output(|stream| {
            write_plain_response(stream, "200 OK", "text/plain", "pong")
                .expect("write plain response");
        });

        assert!(raw.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(raw.contains("Content-Type: text/plain\r\n"));
        assert!(raw.contains("Content-Length: 4\r\n"));
        assert!(raw.ends_with("\r\n\r\npong"));
    }

    #[test]
    fn write_streaming_headers_and_sse_frames_over_tcp() {
        let raw = capture_tcp_output(|stream| {
            write_streaming_response_headers(stream, "200 OK", "text/event-stream")
                .expect("write stream headers");
            write_sse_json_event(stream, &serde_json::json!({ "delta": "hi" }))
                .expect("write sse event");
            write_sse_done(stream).expect("write sse done");
        });

        assert!(raw.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(raw.contains("Content-Type: text/event-stream\r\n"));
        assert!(raw.contains("Cache-Control: no-cache\r\n"));
        assert!(raw.contains("X-Accel-Buffering: no\r\n"));
        assert!(raw.contains("data: {\"delta\":\"hi\"}\n\n"));
        assert!(raw.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn write_ndjson_event_over_tcp_uses_single_line_json() {
        let raw = capture_tcp_output(|stream| {
            write_ndjson_event(stream, &serde_json::json!({ "event": "tick", "seq": 1 }))
                .expect("write ndjson event");
        });

        assert_eq!(raw, "{\"event\":\"tick\",\"seq\":1}\n");
    }

    #[test]
    fn openapi_spec_request_matches_base_and_v1_paths() {
        assert_eq!(
            requested_openapi_spec_format("/openapi.yaml"),
            Some(OpenApiSpecFormat::Yaml)
        );
        assert_eq!(
            requested_openapi_spec_format("/v1/openapi.yaml"),
            Some(OpenApiSpecFormat::Yaml)
        );
        assert_eq!(
            requested_openapi_spec_format("/openapi.yaml?format=yaml"),
            Some(OpenApiSpecFormat::Yaml)
        );
        assert_eq!(
            requested_openapi_spec_format("/openapi.json"),
            Some(OpenApiSpecFormat::Json)
        );
        assert_eq!(
            requested_openapi_spec_format("/v1/openapi.json"),
            Some(OpenApiSpecFormat::Json)
        );
        assert_eq!(requested_openapi_spec_format("/openapi.txt"), None);
    }

    #[test]
    fn embedded_openapi_spec_contains_core_routes() {
        assert!(OPENAPI_SPEC_YAML.contains("openapi: 3.0.3"));
        assert!(OPENAPI_SPEC_YAML.contains("/openapi.yaml:"));
        assert!(OPENAPI_SPEC_YAML.contains("/openapi.json:"));
        assert!(OPENAPI_SPEC_YAML.contains("/events:"));
        assert!(OPENAPI_SPEC_YAML.contains("/models/plan:"));
        assert!(OPENAPI_SPEC_YAML.contains("/models/assets:"));
    }

    #[test]
    fn openapi_json_spec_contains_core_routes() {
        let json = build_openapi_json_spec().expect("json conversion should succeed");
        assert!(json.contains("\"openapi\": \"3.0.3\""));
        assert!(json.contains("\"/openapi.json\""));
        assert!(json.contains("\"/events\""));
        assert!(json.contains("\"/models/assets\""));
        assert!(json.contains("\"/sessions\""));
    }
}

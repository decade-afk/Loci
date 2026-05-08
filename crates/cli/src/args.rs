use anyhow::Context;
use loci_core::EngineConfig;
use loci_protocol::{RoutingStrategy, TieredOffloadProfile};
use loci_server::RuntimeControlConfig;
use std::path::PathBuf;

/// Captures the supported command-line inputs for the standalone Loci CLI.
#[derive(Debug, Default)]
pub struct CliArgs {
    pub model_name: Option<String>,
    pub model_path: Option<PathBuf>,
    pub architecture: Option<String>,
    pub model_memory_bytes: Option<u64>,
    pub model_parameters: Option<u64>,
    pub preferred_backend: Option<String>,
    pub prompt: Option<String>,
    pub images: Vec<PathBuf>,
    pub server_bind: Option<String>,
    pub enable_routing: bool,
    pub routing_strategy: Option<RoutingStrategy>,
    pub max_loaded_models: Option<usize>,
    pub prewarm: bool,
    pub evict: bool,
    pub keep_alive_secs: Option<u64>,
    pub kv_type: Option<String>,
    pub block_size_tokens: Option<u32>,
    pub tiered_offload_enabled: Option<bool>,
    pub offload_profile: Option<TieredOffloadProfile>,
    pub large_model_mode: Option<TieredOffloadProfile>,
    pub spill_threshold_bytes: Option<u64>,
    pub clear_spill_threshold_bytes: bool,
    pub max_disk_bytes: Option<u64>,
    pub clear_max_disk_bytes: bool,
    pub prefetch_window_bytes: Option<u64>,
    pub clear_prefetch_window_bytes: bool,
    pub model_aliases: Vec<(String, String)>,
    pub structured_output: bool,
    pub tool_calling: bool,
    pub max_tokens: u32,
    pub inspect_models: bool,
    pub runtime_snapshot: bool,
    pub runtime_config: bool,
}

impl CliArgs {
    /// Parses the raw command-line iterator into a strongly typed CLI argument struct.
    pub fn parse<I>(args: I) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut parsed = Self {
            max_tokens: 128,
            ..Self::default()
        };
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-name" => {
                    parsed.model_name = Some(next_arg(&mut args, "--model-name")?);
                }
                "--model-path" => {
                    parsed.model_path = Some(PathBuf::from(next_arg(&mut args, "--model-path")?));
                }
                "--architecture" => {
                    parsed.architecture = Some(next_arg(&mut args, "--architecture")?);
                }
                "--model-memory-bytes" => {
                    parsed.model_memory_bytes =
                        Some(next_arg(&mut args, "--model-memory-bytes")?.parse()?);
                }
                "--model-parameters" => {
                    parsed.model_parameters =
                        Some(next_arg(&mut args, "--model-parameters")?.parse()?);
                }
                "--preferred-backend" => {
                    parsed.preferred_backend = Some(next_arg(&mut args, "--preferred-backend")?);
                }
                "--prompt" => {
                    parsed.prompt = Some(next_arg(&mut args, "--prompt")?);
                }
                "--image" => {
                    parsed
                        .images
                        .push(PathBuf::from(next_arg(&mut args, "--image")?));
                }
                "--server-bind" => {
                    parsed.server_bind = Some(next_arg(&mut args, "--server-bind")?);
                }
                "--enable-routing" => {
                    parsed.enable_routing = true;
                }
                "--routing-strategy" => {
                    parsed.routing_strategy = Some(parse_routing_strategy(&next_arg(
                        &mut args,
                        "--routing-strategy",
                    )?)?);
                }
                "--max-loaded-models" => {
                    parsed.max_loaded_models =
                        Some(next_arg(&mut args, "--max-loaded-models")?.parse()?);
                }
                "--prewarm" => {
                    parsed.prewarm = true;
                }
                "--evict" => {
                    parsed.evict = true;
                }
                "--keep-alive-secs" => {
                    parsed.keep_alive_secs =
                        Some(next_arg(&mut args, "--keep-alive-secs")?.parse()?);
                }
                "--type-kv" => {
                    parsed.kv_type = Some(next_arg(&mut args, "--type-kv")?);
                }
                "--block-size-tokens" => {
                    parsed.block_size_tokens =
                        Some(next_arg(&mut args, "--block-size-tokens")?.parse()?);
                }
                "--tiered-offload" => {
                    parsed.tiered_offload_enabled = Some(true);
                }
                "--no-tiered-offload" => {
                    parsed.tiered_offload_enabled = Some(false);
                }
                "--offload-profile" => {
                    parsed.offload_profile = Some(parse_offload_profile(&next_arg(
                        &mut args,
                        "--offload-profile",
                    )?)?);
                }
                "--large-model-mode" => {
                    parsed.large_model_mode = Some(parse_offload_profile(&next_arg(
                        &mut args,
                        "--large-model-mode",
                    )?)?);
                }
                "--spill-threshold-bytes" => {
                    parsed.spill_threshold_bytes =
                        Some(next_arg(&mut args, "--spill-threshold-bytes")?.parse()?);
                }
                "--clear-spill-threshold-bytes" => {
                    parsed.clear_spill_threshold_bytes = true;
                }
                "--max-disk-bytes" => {
                    parsed.max_disk_bytes = Some(next_arg(&mut args, "--max-disk-bytes")?.parse()?);
                }
                "--clear-max-disk-bytes" => {
                    parsed.clear_max_disk_bytes = true;
                }
                "--prefetch-window-bytes" => {
                    parsed.prefetch_window_bytes =
                        Some(next_arg(&mut args, "--prefetch-window-bytes")?.parse()?);
                }
                "--clear-prefetch-window-bytes" => {
                    parsed.clear_prefetch_window_bytes = true;
                }
                "--model-alias" => {
                    parsed
                        .model_aliases
                        .push(parse_model_alias(&next_arg(&mut args, "--model-alias")?)?);
                }
                "--structured-output" => {
                    parsed.structured_output = true;
                }
                "--tool-calling" => {
                    parsed.tool_calling = true;
                }
                "--max-tokens" => {
                    parsed.max_tokens = next_arg(&mut args, "--max-tokens")?.parse()?;
                }
                "--inspect-models" => {
                    parsed.inspect_models = true;
                }
                "--runtime-snapshot" => {
                    parsed.runtime_snapshot = true;
                }
                "--runtime-config" => {
                    parsed.runtime_config = true;
                }
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }

        Ok(parsed)
    }
}

/// Reads the next CLI token and raises a contextual error when the flag is missing its value.
fn next_arg(args: &mut impl Iterator<Item = String>, flag: &'static str) -> anyhow::Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

/// Parses the textual offload profile accepted by the CLI into a typed planner enum.
fn parse_offload_profile(value: &str) -> anyhow::Result<TieredOffloadProfile> {
    match value {
        "auto" => Ok(TieredOffloadProfile::Auto),
        "gpu_resident" => Ok(TieredOffloadProfile::GpuResident),
        "balanced" => Ok(TieredOffloadProfile::Balanced),
        "disk_heavy" => Ok(TieredOffloadProfile::DiskHeavy),
        other => anyhow::bail!(
            "unknown offload profile `{other}`, expected one of: auto, gpu_resident, balanced, disk_heavy"
        ),
    }
}

/// Parses the textual routing strategy accepted by the CLI into a typed routing enum.
fn parse_routing_strategy(value: &str) -> anyhow::Result<RoutingStrategy> {
    match value {
        "prompt_complexity" => Ok(RoutingStrategy::PromptComplexity),
        "latency_aware" => Ok(RoutingStrategy::LatencyAware),
        "power_aware" => Ok(RoutingStrategy::PowerAware),
        other => anyhow::bail!(
            "unknown routing strategy `{other}`, expected one of: prompt_complexity, latency_aware, power_aware"
        ),
    }
}

/// Parses `alias=target` mappings used to register model aliases from the CLI.
fn parse_model_alias(value: &str) -> anyhow::Result<(String, String)> {
    let (alias, target) = value
        .split_once('=')
        .with_context(|| "--model-alias requires the form alias=target")?;
    if alias.is_empty() || target.is_empty() {
        anyhow::bail!("--model-alias requires the form alias=target");
    }
    Ok((alias.to_string(), target.to_string()))
}

/// Converts the current engine config into the server-compatible runtime-control view used by the CLI.
pub fn runtime_control_config_from_config(config: &EngineConfig) -> RuntimeControlConfig {
    RuntimeControlConfig::new(
        config.tiered_offload.prefetch_window_bytes,
        config.routing.enabled,
        config.routing.strategy.clone(),
        config.routing.max_loaded_models,
        config.model_keep_alive_secs,
        config.tiered_offload.enabled,
        config.tiered_offload.profile,
        config.tiered_offload.spill_threshold_bytes,
        config.tiered_offload.max_disk_bytes,
        config.paged_kv.enabled,
        config.paged_kv.block_size_tokens,
        config.paged_kv.page_size_bytes,
        config.paged_kv.prefix_cache_enabled,
        config.paged_kv.type_k.clone(),
        config.paged_kv.type_v.clone(),
    )
}

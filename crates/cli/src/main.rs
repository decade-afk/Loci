use anyhow::Context;
use loci_core::{EmbeddedModelRegistration, EngineConfig, InferenceEngine};
use loci_protocol::{
    ImageInput, ModelDescriptor, RoutingConfig, RoutingStrategy, SessionRequest,
    TieredOffloadProfile,
};
use loci_server::{run_server, ServerConfig};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Default)]
struct CliArgs {
    model_name: Option<String>,
    model_path: Option<PathBuf>,
    architecture: Option<String>,
    model_memory_bytes: Option<u64>,
    model_parameters: Option<u64>,
    preferred_backend: Option<String>,
    prompt: Option<String>,
    images: Vec<PathBuf>,
    server_bind: Option<String>,
    enable_routing: bool,
    routing_strategy: Option<RoutingStrategy>,
    max_loaded_models: Option<usize>,
    prewarm: bool,
    evict: bool,
    keep_alive_secs: Option<u64>,
    kv_type: Option<String>,
    block_size_tokens: Option<u32>,
    offload_profile: Option<TieredOffloadProfile>,
    spill_threshold_bytes: Option<u64>,
    max_disk_bytes: Option<u64>,
    prefetch_window_bytes: Option<u64>,
    model_aliases: Vec<(String, String)>,
    structured_output: bool,
    tool_calling: bool,
    max_tokens: u32,
    inspect_models: bool,
    runtime_snapshot: bool,
}

impl CliArgs {
    fn parse<I>(args: I) -> anyhow::Result<Self>
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
                "--offload-profile" => {
                    parsed.offload_profile = Some(parse_offload_profile(&next_arg(
                        &mut args,
                        "--offload-profile",
                    )?)?);
                }
                "--spill-threshold-bytes" => {
                    parsed.spill_threshold_bytes =
                        Some(next_arg(&mut args, "--spill-threshold-bytes")?.parse()?);
                }
                "--max-disk-bytes" => {
                    parsed.max_disk_bytes =
                        Some(next_arg(&mut args, "--max-disk-bytes")?.parse()?);
                }
                "--prefetch-window-bytes" => {
                    parsed.prefetch_window_bytes =
                        Some(next_arg(&mut args, "--prefetch-window-bytes")?.parse()?);
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
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }

        Ok(parsed)
    }
}

fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse(env::args().skip(1))?;
    let mut config = EngineConfig::default();
    if args.enable_routing {
        config.routing = RoutingConfig {
            enabled: true,
            ..RoutingConfig::default()
        };
    }
    if let Some(strategy) = args.routing_strategy.clone() {
        config.routing.strategy = strategy;
    }
    if let Some(max_loaded_models) = args.max_loaded_models {
        config.routing.max_loaded_models = Some(max_loaded_models);
    }
    if let Some(keep_alive_secs) = args.keep_alive_secs {
        config.model_keep_alive_secs = keep_alive_secs;
    }
    if let Some(kv_type) = &args.kv_type {
        config.paged_kv.type_k = kv_type.clone();
        config.paged_kv.type_v = kv_type.clone();
    }
    if let Some(block_size_tokens) = args.block_size_tokens {
        config.paged_kv.block_size_tokens = block_size_tokens;
    }
    if let Some(offload_profile) = args.offload_profile {
        config.tiered_offload.profile = offload_profile;
    }
    if let Some(spill_threshold_bytes) = args.spill_threshold_bytes {
        config.tiered_offload.spill_threshold_bytes = Some(spill_threshold_bytes);
    }
    if let Some(max_disk_bytes) = args.max_disk_bytes {
        config.tiered_offload.max_disk_bytes = Some(max_disk_bytes);
    }
    if let Some(prefetch_window_bytes) = args.prefetch_window_bytes {
        config.tiered_offload.prefetch_window_bytes = Some(prefetch_window_bytes);
    }
    for (alias, target) in &args.model_aliases {
        config.model_aliases.insert(alias.clone(), target.clone());
    }

    let mut builder = InferenceEngine::builder().config(config);
    if let Some(backend) = &args.preferred_backend {
        builder = builder.preferred_backend(backend.clone());
    }
    if let Some(model) = build_model_descriptor(&args)? {
        builder = builder.model(model);
    }

    let mut engine = builder.build()?;

    if let Some(bind) = args.server_bind {
        return run_server(ServerConfig { bind, engine });
    }

    if args.evict {
        if let Some(model_name) = &args.model_name {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "evicted": engine.evict_model(model_name),
                    "name": model_name,
                }))?
            );
            return Ok(());
        }
        anyhow::bail!("--evict requires --model-name");
    }

    if args.prewarm {
        let prepared = engine.prepare(SessionRequest {
            prompt: "warmup".to_string(),
            max_tokens: 1,
            temperature: 0.0,
            target_model: args.model_name.clone(),
            images: collect_image_inputs(&args),
            structured_output: args.structured_output,
            tool_calling: args.tool_calling,
        })?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "prepared": prepared,
                "runtime": engine.runtime_snapshot(),
            }))?
        );
        return Ok(());
    }

    if args.inspect_models {
        if let Some(model_name) = &args.model_name {
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.inspect_model(model_name)?)?
            );
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(&engine.inspect_models())?
            );
        }
        return Ok(());
    }

    if args.runtime_snapshot {
        println!(
            "{}",
            serde_json::to_string_pretty(&engine.runtime_snapshot())?
        );
        return Ok(());
    }

    if let Some(prompt) = args.prompt.clone() {
        let response = engine.infer(SessionRequest {
            prompt,
            max_tokens: args.max_tokens,
            temperature: 0.2,
            target_model: args.model_name.clone(),
            images: collect_image_inputs(&args),
            structured_output: args.structured_output,
            tool_calling: args.tool_calling,
        })?;
        println!("{}", serde_json::to_string_pretty(&response)?);
        return Ok(());
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&engine.runtime_snapshot())?
    );
    Ok(())
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &'static str) -> anyhow::Result<String> {
    args.next()
        .with_context(|| format!("{flag} requires a value"))
}

fn build_model_descriptor(args: &CliArgs) -> anyhow::Result<Option<ModelDescriptor>> {
    let Some(path) = &args.model_path else {
        return Ok(None);
    };

    Ok(Some(loci_core::infer_model_descriptor_from_path(
        path.clone(),
        EmbeddedModelRegistration {
            name: args.model_name.clone(),
            architecture: args.architecture.clone(),
            memory_bytes: args.model_memory_bytes,
            parameter_count: args.model_parameters,
            context_length: None,
            preferred_backend: args.preferred_backend.clone(),
        },
    )?))
}

fn collect_image_inputs(args: &CliArgs) -> Vec<ImageInput> {
    args.images
        .iter()
        .cloned()
        .map(|path| ImageInput::Path { path })
        .collect()
}

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

fn parse_model_alias(value: &str) -> anyhow::Result<(String, String)> {
    let (alias, target) = value
        .split_once('=')
        .with_context(|| "--model-alias requires the form alias=target")?;
    if alias.is_empty() || target.is_empty() {
        anyhow::bail!("--model-alias requires the form alias=target");
    }
    Ok((alias.to_string(), target.to_string()))
}

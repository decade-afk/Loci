mod args;

use loci_core::{EmbeddedModelRegistration, EngineConfig, InferenceEngine};
use loci_protocol::{ImageInput, ModelDescriptor, RoutingConfig, SessionRequest};
use loci_server::{run_server_with_runtime_control, ServerConfig};
use serde_json::json;
use std::env;

use args::{runtime_control_config_from_config, CliArgs};

/// Parses CLI arguments, constructs the runtime, and dispatches the requested one-shot or service mode.
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
    if let Some(tiered_offload_enabled) = args.tiered_offload_enabled {
        config.tiered_offload.enabled = tiered_offload_enabled;
    }
    if let Some(offload_profile) = args.large_model_mode.or(args.offload_profile) {
        config.tiered_offload.profile = offload_profile;
    }
    if args.clear_spill_threshold_bytes {
        config.tiered_offload.spill_threshold_bytes = None;
    } else if let Some(spill_threshold_bytes) = args.spill_threshold_bytes {
        config.tiered_offload.spill_threshold_bytes = Some(spill_threshold_bytes);
    }
    if args.clear_max_disk_bytes {
        config.tiered_offload.max_disk_bytes = None;
    } else if let Some(max_disk_bytes) = args.max_disk_bytes {
        config.tiered_offload.max_disk_bytes = Some(max_disk_bytes);
    }
    if args.clear_prefetch_window_bytes {
        config.tiered_offload.prefetch_window_bytes = None;
    } else if let Some(prefetch_window_bytes) = args.prefetch_window_bytes {
        config.tiered_offload.prefetch_window_bytes = Some(prefetch_window_bytes);
    }
    for (alias, target) in &args.model_aliases {
        config.model_aliases.insert(alias.clone(), target.clone());
    }

    let runtime_control = runtime_control_config_from_config(&config);
    let mut builder = InferenceEngine::builder().config(config);
    if let Some(backend) = &args.preferred_backend {
        builder = builder.preferred_backend(backend.clone());
    }
    if let Some(model) = build_model_descriptor(&args)? {
        builder = builder.model(model);
    }

    let mut engine = builder.build()?;

    if let Some(bind) = args.server_bind {
        return run_server_with_runtime_control(ServerConfig { bind, engine }, runtime_control);
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

    if args.runtime_config {
        println!("{}", serde_json::to_string_pretty(&runtime_control)?);
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
        serde_json::to_string_pretty(&json!({
            "runtime": engine.runtime_snapshot(),
            "runtime_control": runtime_control,
        }))?
    );
    Ok(())
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

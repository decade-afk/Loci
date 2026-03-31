use loci_core::{InferenceEngine, PlatformTrack};
use std::env;
use std::path::PathBuf;

fn plugin_dir_from_args() -> anyhow::Result<PathBuf> {
    let mut plugin_dir = PathBuf::from("plugins");
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        if arg == "--plugin-dir" {
            let value = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--plugin-dir requires a path"))?;
            plugin_dir = PathBuf::from(value);
        }
    }

    Ok(plugin_dir)
}

fn main() -> anyhow::Result<()> {
    let plugin_dir = plugin_dir_from_args()?;
    let mut engine = InferenceEngine::builder().build()?;
    let loaded = engine.load_plugins_from_dir(&plugin_dir)?;
    let active_inference = engine
        .active_core_rewriter(loci_core::CoreComponent::Inference)
        .unwrap_or("none");
    println!(
        "loci-cli ready; plugins={}, loaded_now={}, infra_plugins={}, agent_plugins={}, active_inference={}",
        engine.plugin_count(),
        loaded,
        engine.plugins_for_track(PlatformTrack::AiInfra).len(),
        engine.plugins_for_track(PlatformTrack::AiAgent).len(),
        active_inference
    );
    Ok(())
}

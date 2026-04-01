use loci_core::{CoreComponent, InferenceEngine, PlatformTrack};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    plugin_dir: PathBuf,
    activate_legacy_text_plugins: Vec<String>,
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            plugin_dir: PathBuf::from("plugins"),
            activate_legacy_text_plugins: Vec::new(),
        }
    }
}

fn parse_args<I>(args: I) -> anyhow::Result<CliArgs>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = CliArgs::default();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--plugin-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--plugin-dir requires a path"))?;
                parsed.plugin_dir = PathBuf::from(value);
            }
            "--activate-legacy-text-plugin" => {
                let value = args.next().ok_or_else(|| {
                    anyhow::anyhow!("--activate-legacy-text-plugin requires a plugin name")
                })?;
                parsed.activate_legacy_text_plugins.push(value);
            }
            other => {
                return Err(anyhow::anyhow!("unknown argument: {other}"));
            }
        }
    }

    Ok(parsed)
}

fn comma_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "none".to_string()
    } else {
        values.join(",")
    }
}

fn main() -> anyhow::Result<()> {
    let args = parse_args(env::args().skip(1))?;
    let mut engine = InferenceEngine::builder().build()?;
    let loaded = engine.load_plugins_from_dir(&args.plugin_dir)?;

    for plugin_name in &args.activate_legacy_text_plugins {
        engine.activate_legacy_text_plugin(plugin_name)?;
    }

    let active_inference = engine
        .active_core_rewriter(CoreComponent::Inference)
        .unwrap_or("none");
    let legacy_text_candidates = engine.legacy_text_plugin_candidates();
    let active_legacy_text = engine.active_legacy_text_plugins();

    println!(
        "loci-cli ready; plugins={}, loaded_now={}, infra_plugins={}, agent_plugins={}, active_inference={}, legacy_text_candidates={}, active_legacy_text={}",
        engine.plugin_count(),
        loaded,
        engine.plugins_for_track(PlatformTrack::AiInfra).len(),
        engine.plugins_for_track(PlatformTrack::AiAgent).len(),
        active_inference,
        comma_or_none(&legacy_text_candidates),
        comma_or_none(&active_legacy_text),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_supports_plugin_dir_and_repeated_legacy_activation() {
        let parsed = parse_args([
            "--plugin-dir".to_string(),
            "custom-plugins".to_string(),
            "--activate-legacy-text-plugin".to_string(),
            "legacy-a".to_string(),
            "--activate-legacy-text-plugin".to_string(),
            "legacy-b".to_string(),
        ])
        .expect("parse args");

        assert_eq!(
            parsed,
            CliArgs {
                plugin_dir: PathBuf::from("custom-plugins"),
                activate_legacy_text_plugins: vec!["legacy-a".to_string(), "legacy-b".to_string(),],
            }
        );
    }

    #[test]
    fn parse_args_rejects_unknown_argument() {
        let err = parse_args(["--unknown".to_string()]).expect_err("should reject");
        assert!(err.to_string().contains("unknown argument"));
    }

    #[test]
    fn comma_or_none_formats_values() {
        assert_eq!(comma_or_none(&[]), "none");
        assert_eq!(
            comma_or_none(&["a".to_string(), "b".to_string()]),
            "a,b".to_string()
        );
    }
}

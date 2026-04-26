use loci_core::InferenceEngine;
use loci_server::{run_server, ServerConfig};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    plugin_dir: Option<PathBuf>,
    backend: Option<String>,
    model: Option<PathBuf>,
    server_bind: Option<String>,
}

impl CliArgs {
    fn parse<I>(args: I) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = String>,
    {
        let mut parsed = Self {
            plugin_dir: None,
            backend: None,
            model: None,
            server_bind: None,
        };
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--plugin-dir" => {
                    parsed.plugin_dir =
                        Some(PathBuf::from(args.next().ok_or_else(|| {
                            anyhow::anyhow!("--plugin-dir requires a path")
                        })?));
                }
                "--backend" => {
                    parsed.backend = Some(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("--backend requires a name"))?,
                    );
                }
                "--model" => {
                    parsed.model = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("--model requires a path"))?,
                    ));
                }
                "--server-bind" => {
                    parsed.server_bind = Some(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("--server-bind requires an address"))?,
                    );
                }
                other => return Err(anyhow::anyhow!("unknown argument: {other}")),
            }
        }

        Ok(parsed)
    }
}

fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse(env::args().skip(1))?;
    let mut engine = InferenceEngine::builder().build()?;

    if let Some(plugin_dir) = &args.plugin_dir {
        engine.load_plugins_from_dir(plugin_dir)?;
    }

    if let (Some(backend), Some(model)) = (args.backend.as_deref(), args.model.as_ref()) {
        engine.load_model(backend, model)?;
    }

    if let Some(bind) = args.server_bind {
        return run_server(ServerConfig { bind, engine });
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&engine.runtime_snapshot())?
    );
    Ok(())
}

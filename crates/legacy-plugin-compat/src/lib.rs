use anyhow::{bail, Context, Result};
use libloading::{Library, Symbol};
use loci_legacy_plugin_api::plugin::{dynamic_plugin_from_opaque, DynamicPluginOpaque, Plugin};
use std::path::Path;
use std::sync::{Arc, Mutex};

type PluginConstructor = unsafe extern "C" fn() -> DynamicPluginOpaque;

pub trait LegacyTextCompat: Send + Sync {
    fn pre_generate(&self, prompt: &str) -> Result<String>;
    fn post_generate(&self, response: &str) -> Result<String>;
    fn transform_logits(&self, logits: &mut [f32], context_tokens: &[i32]) -> Result<()>;
    fn post_sample(&self, token_id: i32) -> Result<i32>;
}

struct LoadedLegacyTextPlugin {
    plugin: Mutex<Option<Box<dyn Plugin>>>,
    _library: Option<Arc<Library>>,
    supports_pre_generate: bool,
    supports_post_generate: bool,
    supports_transform_logits: bool,
    supports_post_sample: bool,
}

impl LoadedLegacyTextPlugin {
    fn with_plugin<T>(&self, f: impl FnOnce(&dyn Plugin) -> Result<T>) -> Result<T> {
        let guard = self
            .plugin
            .lock()
            .map_err(|_| anyhow::anyhow!("legacy plugin mutex poisoned"))?;
        let plugin = guard
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("legacy plugin already released"))?;
        f(plugin)
    }
}

impl Drop for LoadedLegacyTextPlugin {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.plugin.lock() {
            if let Some(plugin) = guard.as_deref_mut() {
                let _ = plugin.cleanup();
            }
            let _ = guard.take();
        }
    }
}

impl LegacyTextCompat for LoadedLegacyTextPlugin {
    fn pre_generate(&self, prompt: &str) -> Result<String> {
        if !self.supports_pre_generate {
            return Ok(prompt.to_string());
        }

        self.with_plugin(|plugin| {
            plugin
                .pre_generate(prompt)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        })
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        if !self.supports_post_generate {
            return Ok(response.to_string());
        }

        self.with_plugin(|plugin| {
            plugin
                .post_generate(response)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        })
    }

    fn transform_logits(&self, logits: &mut [f32], context_tokens: &[i32]) -> Result<()> {
        if !self.supports_transform_logits {
            return Ok(());
        }

        let mut legacy_logits = loci_legacy_plugin_api::sampler::LogitsView::new(logits);
        self.with_plugin(|plugin| {
            plugin
                .transform_logits(&mut legacy_logits, context_tokens)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        })
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        if !self.supports_post_sample {
            return Ok(token_id);
        }

        self.with_plugin(|plugin| {
            plugin
                .post_sample(token_id)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        })
    }
}

pub fn load_legacy_text_plugin_compat(
    runtime_artifact_path: &Path,
    expected_name: &str,
    expected_version: &str,
    capabilities: &[String],
) -> Result<Option<Arc<dyn LegacyTextCompat>>> {
    let supports_pre_generate = capabilities.iter().any(|cap| cap == "pre_generate");
    let supports_post_generate = capabilities.iter().any(|cap| cap == "post_generate");
    let supports_transform_logits = capabilities.iter().any(|cap| cap == "transform_logits");
    let supports_post_sample = capabilities.iter().any(|cap| cap == "post_sample");
    if !supports_pre_generate
        && !supports_post_generate
        && !supports_transform_logits
        && !supports_post_sample
    {
        return Ok(None);
    }

    if !runtime_artifact_path.exists() {
        bail!(
            "legacy runtime artifact not found: {}",
            runtime_artifact_path.display()
        );
    }

    let library = unsafe {
        Library::new(runtime_artifact_path).with_context(|| {
            format!(
                "failed to load legacy plugin library: {}",
                runtime_artifact_path.display()
            )
        })?
    };
    let constructor: Symbol<PluginConstructor> = unsafe {
        match library.get(b"create_plugin_v1") {
            Ok(symbol) => symbol,
            Err(_) => library.get(b"create_plugin").with_context(|| {
                format!(
                    "failed to find legacy plugin constructor symbol in {}",
                    runtime_artifact_path.display()
                )
            })?,
        }
    };

    let plugin = unsafe {
        dynamic_plugin_from_opaque(constructor())
            .ok_or_else(|| anyhow::anyhow!("legacy plugin constructor returned invalid payload"))?
    };

    if plugin.name() != expected_name {
        bail!(
            "legacy plugin runtime name `{}` does not match contract name `{expected_name}`",
            plugin.name()
        );
    }
    if !expected_version.trim().is_empty() && plugin.version() != expected_version {
        bail!(
            "legacy plugin runtime version `{}` does not match contract version `{expected_version}`",
            plugin.version()
        );
    }

    let mut plugin = plugin;
    plugin
        .init()
        .map_err(|error| anyhow::anyhow!("legacy plugin init failed: {}", error))?;

    Ok(Some(Arc::new(LoadedLegacyTextPlugin {
        plugin: Mutex::new(Some(plugin)),
        _library: Some(Arc::new(library)),
        supports_pre_generate,
        supports_post_generate,
        supports_transform_logits,
        supports_post_sample,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct LegacySamplerPlugin;

    impl Plugin for LegacySamplerPlugin {
        fn name(&self) -> &str {
            "legacy-sampler"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        fn pre_generate(&self, prompt: &str) -> loci_legacy_plugin_api::error::Result<String> {
            Ok(format!("[pre]{prompt}"))
        }

        fn post_generate(&self, response: &str) -> loci_legacy_plugin_api::error::Result<String> {
            Ok(format!("{response}[post]"))
        }

        fn transform_logits(
            &self,
            logits: &mut loci_legacy_plugin_api::sampler::LogitsView<'_>,
            _context_tokens: &[i32],
        ) -> loci_legacy_plugin_api::error::Result<()> {
            logits.set_usize(1, 77.0)?;
            Ok(())
        }

        fn post_sample(&self, _token_id: i32) -> loci_legacy_plugin_api::error::Result<i32> {
            Ok(9)
        }
    }

    #[test]
    fn loaded_legacy_sampler_compat_applies_hooks() {
        let compat = LoadedLegacyTextPlugin {
            plugin: Mutex::new(Some(Box::new(LegacySamplerPlugin))),
            _library: None,
            supports_pre_generate: true,
            supports_post_generate: true,
            supports_transform_logits: true,
            supports_post_sample: true,
        };

        assert_eq!(
            compat.pre_generate("hello").expect("pre generate"),
            "[pre]hello"
        );
        let mut logits = vec![1.0, 2.0, 3.0];
        compat
            .transform_logits(&mut logits, &[])
            .expect("transform logits");
        assert_eq!(logits[1], 77.0);
        assert_eq!(compat.post_sample(1).expect("post sample"), 9);
        assert_eq!(
            compat.post_generate("world").expect("post generate"),
            "world[post]"
        );
    }
}

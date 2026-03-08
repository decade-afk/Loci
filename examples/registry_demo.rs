//! Plugin Registry comprehensive demo aligned with current API.

use loci::error::Result;
use loci::plugin::Plugin;
use loci::plugin_registry::PluginRegistry;

struct PrefixPlugin {
    prefix: String,
}

impl PrefixPlugin {
    fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }
}

impl Plugin for PrefixPlugin {
    fn name(&self) -> &str {
        "prefix"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("{}{}", self.prefix, prompt))
    }
}

struct SuffixPlugin {
    suffix: String,
}

impl SuffixPlugin {
    fn new(suffix: &str) -> Self {
        Self {
            suffix: suffix.to_string(),
        }
    }
}

impl Plugin for SuffixPlugin {
    fn name(&self) -> &str {
        "suffix"
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        Ok(format!("{}{}", response, self.suffix))
    }
}

fn main() -> Result<()> {
    let config_file = "demo_plugins.toml";

    let mut registry = PluginRegistry::with_config_path(config_file);
    registry.register_static(PrefixPlugin::new("[USER] "))?;
    registry.register_static(SuffixPlugin::new(" [END]"))?;

    println!(
        "registered={} enabled={} dynamic={}",
        registry.count(),
        registry.count_enabled(),
        registry.count_dynamic()
    );

    for (name, version, enabled, plugin_type) in registry.list() {
        println!(
            "- {} v{} enabled={} type={}",
            name, version, enabled, plugin_type
        );
    }

    let pre = registry.apply_pre_generate("hello")?;
    let post = registry.apply_post_generate("response")?;
    println!("pre='{}' post='{}'", pre, post);

    registry.disable("prefix")?;
    println!(
        "prefix disabled => {}",
        registry.apply_pre_generate("hello")?
    );

    registry.save_to_file(config_file)?;

    let mut registry2 = PluginRegistry::new();
    registry2.register_static(PrefixPlugin::new("[USER] "))?;
    registry2.register_static(SuffixPlugin::new(" [END]"))?;
    registry2.load_from_file(config_file)?;

    println!(
        "reloaded count={} enabled={}",
        registry2.count(),
        registry2.count_enabled()
    );

    std::fs::remove_file(config_file).ok();
    Ok(())
}

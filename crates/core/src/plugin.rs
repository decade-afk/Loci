use crate::error::Result as CoreResult;
use crate::sampler::LogitsView;
use anyhow::{bail, Context, Result};
use loci_plugin_api::{
    ContributionPoints, CoreComponent, CoreRewriters, PlatformTrack, PluginBootstrap,
    PluginCompatibility, PluginManifest, PluginRuntime, PluginSourceFormat,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MANIFEST_FILE_NAME: &str = "manifest.toml";
const LEGACY_PLUGIN_ABI_VERSION: u32 = 1;
const LEGACY_DYNAMIC_EXTENSIONS: &[&str] = &["dll", "so", "dylib"];
const LEGACY_WASM_EXTENSIONS: &[&str] = &["wasm"];

pub trait SamplingHook: Send + Sync {
    fn transform_logits(
        &self,
        _logits: &mut LogitsView<'_>,
        _context_tokens: &[i32],
    ) -> CoreResult<()> {
        Ok(())
    }

    fn post_sample(&self, token_id: i32) -> CoreResult<i32> {
        Ok(token_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplingLogitBias {
    pub token_id: i32,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplingHookProfile {
    #[serde(default)]
    pub logit_biases: Vec<SamplingLogitBias>,
    #[serde(default)]
    pub force_token_id: Option<i32>,
    #[serde(default = "default_forced_logit")]
    pub forced_logit: f32,
    #[serde(default)]
    pub post_sample_override: Option<i32>,
}

impl Default for SamplingHookProfile {
    fn default() -> Self {
        Self {
            logit_biases: Vec::new(),
            force_token_id: None,
            forced_logit: default_forced_logit(),
            post_sample_override: None,
        }
    }
}

fn default_forced_logit() -> f32 {
    120.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LegacyPluginContractKind {
    TextPlugin,
    ToolPlugin,
    ExecutionPolicy,
    ManagementAuthPolicy,
    ModelPullPolicy,
    ModelPullVerifier,
    ServeDispatchPolicy,
    ImageKernel,
    Backend,
}

impl LegacyPluginContractKind {
    fn as_str(self) -> &'static str {
        match self {
            LegacyPluginContractKind::TextPlugin => "text_plugin",
            LegacyPluginContractKind::ToolPlugin => "tool_plugin",
            LegacyPluginContractKind::ExecutionPolicy => "execution_policy",
            LegacyPluginContractKind::ManagementAuthPolicy => "management_auth_policy",
            LegacyPluginContractKind::ModelPullPolicy => "model_pull_policy",
            LegacyPluginContractKind::ModelPullVerifier => "model_pull_verifier",
            LegacyPluginContractKind::ServeDispatchPolicy => "serve_dispatch_policy",
            LegacyPluginContractKind::ImageKernel => "image_kernel",
            LegacyPluginContractKind::Backend => "backend",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LegacyPluginContractManifest {
    name: String,
    version: String,
    kind: LegacyPluginContractKind,
    #[serde(default = "default_legacy_plugin_abi_version")]
    abi_version: u32,
    #[serde(default)]
    min_host_version: Option<String>,
    #[serde(default)]
    max_host_version: Option<String>,
    #[serde(default)]
    capabilities: Vec<String>,
}

fn default_legacy_plugin_abi_version() -> u32 {
    LEGACY_PLUGIN_ABI_VERSION
}

#[derive(Clone, Default)]
struct RegisteredPluginRuntime {
    sampling_hook: Option<Arc<dyn SamplingHook>>,
}

#[derive(Clone, Default)]
pub struct PluginSamplingRuntime {
    hooks: Vec<RegisteredSamplingHook>,
}

#[derive(Clone)]
struct RegisteredSamplingHook {
    plugin_name: String,
    hook: Arc<dyn SamplingHook>,
}

impl PluginSamplingRuntime {
    pub fn hook_count(&self) -> usize {
        self.hooks.len()
    }

    pub fn apply_transform_logits(
        &self,
        logits: &mut LogitsView<'_>,
        context_tokens: &[i32],
    ) -> CoreResult<()> {
        for registered in &self.hooks {
            registered.hook.transform_logits(logits, context_tokens)?;
        }
        Ok(())
    }

    pub fn apply_post_sample(&self, token_id: i32) -> CoreResult<i32> {
        let mut token = token_id;
        for registered in &self.hooks {
            token = registered.hook.post_sample(token)?;
        }
        Ok(token)
    }

    pub fn plugin_names(&self) -> Vec<&str> {
        self.hooks
            .iter()
            .map(|registered| registered.plugin_name.as_str())
            .collect()
    }
}

#[derive(Clone)]
pub struct RegisteredPlugin {
    pub manifest: PluginManifest,
    manifest_path: Option<PathBuf>,
    root_dir: Option<PathBuf>,
    runtime: RegisteredPluginRuntime,
}

impl fmt::Debug for RegisteredPlugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredPlugin")
            .field("manifest", &self.manifest)
            .field("manifest_path", &self.manifest_path)
            .field("root_dir", &self.root_dir)
            .field("has_sampling_hook", &self.runtime.sampling_hook.is_some())
            .finish()
    }
}

impl RegisteredPlugin {
    pub fn new(manifest: PluginManifest) -> Self {
        Self {
            manifest,
            manifest_path: None,
            root_dir: None,
            runtime: RegisteredPluginRuntime::default(),
        }
    }

    pub fn supports_track(&self, track: PlatformTrack) -> bool {
        self.manifest.supports_track(track)
    }

    pub fn declares_core_rewriter(&self, component: CoreComponent) -> bool {
        self.manifest.declares_core_rewriter(component)
    }

    pub fn declares_inference_sampling_runtime(&self) -> bool {
        self.declares_core_rewriter(CoreComponent::Inference)
    }

    pub fn auto_activate_components(&self) -> &[CoreComponent] {
        &self.manifest.bootstrap.activate_on_load
    }

    pub fn has_sampling_hook(&self) -> bool {
        self.runtime.sampling_hook.is_some()
    }

    pub fn is_legacy_compat_bundle(&self) -> bool {
        self.manifest.is_legacy_compat_manifest()
    }

    pub fn legacy_runtime_path(&self) -> Option<&str> {
        self.manifest.compatibility.legacy_runtime_path.as_deref()
    }

    fn with_manifest_location(mut self, manifest_path: PathBuf) -> Self {
        self.root_dir = manifest_path.parent().map(Path::to_path_buf);
        self.manifest_path = Some(manifest_path);
        self
    }

    fn with_runtime(mut self, runtime: RegisteredPluginRuntime) -> Self {
        self.runtime = runtime;
        self
    }

    fn sampling_hook(&self) -> Option<Arc<dyn SamplingHook>> {
        self.runtime.sampling_hook.as_ref().map(Arc::clone)
    }
}

#[derive(Debug, Clone)]
struct ProfiledSamplingHook {
    profile: SamplingHookProfile,
}

impl ProfiledSamplingHook {
    fn new(profile: SamplingHookProfile) -> Self {
        Self { profile }
    }
}

impl SamplingHook for ProfiledSamplingHook {
    fn transform_logits(
        &self,
        logits: &mut LogitsView<'_>,
        _context_tokens: &[i32],
    ) -> CoreResult<()> {
        for bias in &self.profile.logit_biases {
            if bias.token_id < 0 {
                continue;
            }
            logits.set_usize(bias.token_id as usize, bias.logit)?;
        }

        if let Some(token_id) = self.profile.force_token_id {
            if token_id >= 0 {
                logits.set_usize(token_id as usize, self.profile.forced_logit)?;
            }
        }

        Ok(())
    }

    fn post_sample(&self, token_id: i32) -> CoreResult<i32> {
        Ok(self.profile.post_sample_override.unwrap_or(token_id))
    }
}

fn load_sampling_hook_profile(path: &Path) -> Result<SamplingHookProfile> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read sampling hook profile: {}", path.display()))?;
    toml::from_str(&content)
        .with_context(|| format!("failed to parse sampling hook profile: {}", path.display()))
}

fn build_legacy_contract_candidates(path: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
        candidates.push(parent.join(format!("{stem}.loci-plugin.json")));
        candidates.push(parent.join(format!("{stem}.loci-plugin.toml")));
        candidates.push(parent.join(format!("{stem}.plugin.json")));
        candidates.push(parent.join(format!("{stem}.plugin.toml")));
    }
    candidates
}

fn load_legacy_contract_manifest(
    runtime_artifact_path: &Path,
) -> Result<Option<LegacyPluginContractManifest>> {
    for candidate in build_legacy_contract_candidates(runtime_artifact_path) {
        if !candidate.exists() {
            continue;
        }

        let content = fs::read_to_string(&candidate).with_context(|| {
            format!(
                "failed to read legacy plugin contract: {}",
                candidate.display()
            )
        })?;
        let manifest = match candidate
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "json" => serde_json::from_str::<LegacyPluginContractManifest>(&content).with_context(
                || {
                    format!(
                        "failed to parse legacy plugin contract: {}",
                        candidate.display()
                    )
                },
            )?,
            "toml" => {
                toml::from_str::<LegacyPluginContractManifest>(&content).with_context(|| {
                    format!(
                        "failed to parse legacy plugin contract: {}",
                        candidate.display()
                    )
                })?
            }
            other => {
                bail!(
                    "unsupported legacy plugin contract format `{other}`: {}",
                    candidate.display()
                )
            }
        };

        return Ok(Some(manifest));
    }

    Ok(None)
}

fn is_legacy_runtime_artifact_extension(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    LEGACY_DYNAMIC_EXTENSIONS.contains(&extension.as_str())
        || LEGACY_WASM_EXTENSIONS.contains(&extension.as_str())
}

fn is_legacy_runtime_artifact(path: &Path) -> bool {
    path.is_file()
        && is_legacy_runtime_artifact_extension(path)
        && build_legacy_contract_candidates(path)
            .iter()
            .any(|candidate| candidate.exists())
}

fn convert_legacy_contract_manifest(
    runtime_artifact_path: &Path,
    contract: LegacyPluginContractManifest,
) -> PluginManifest {
    PluginManifest {
        name: contract.name,
        version: contract.version,
        api_version: "1.0".to_string(),
        min_host_version: contract.min_host_version,
        max_host_version: contract.max_host_version,
        target_tracks: Vec::new(),
        contributes: ContributionPoints {
            inference_hooks: contract.capabilities,
            ..Default::default()
        },
        core_rewriters: CoreRewriters::default(),
        runtime: PluginRuntime::default(),
        bootstrap: PluginBootstrap::default(),
        compatibility: PluginCompatibility {
            source_format: PluginSourceFormat::LegacyContract,
            legacy_kind: Some(contract.kind.as_str().to_string()),
            legacy_abi_version: Some(contract.abi_version),
            legacy_runtime_path: Some(runtime_artifact_path.to_string_lossy().to_string()),
        },
    }
}

fn resolve_runtime_artifact_path(manifest_path: &Path, relative_path: &str) -> PathBuf {
    manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(relative_path)
}

fn validate_runtime_artifact_within_plugin_root(
    manifest_path: &Path,
    artifact_path: &Path,
) -> Result<()> {
    let plugin_root = manifest_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("plugin manifest has no parent"))?;
    let canonical_root = plugin_root
        .canonicalize()
        .with_context(|| format!("failed to resolve plugin root: {}", plugin_root.display()))?;
    let canonical_artifact = artifact_path.canonicalize().with_context(|| {
        format!(
            "failed to resolve runtime artifact: {}",
            artifact_path.display()
        )
    })?;

    if !canonical_artifact.starts_with(&canonical_root) {
        bail!(
            "runtime artifact `{}` escapes plugin root `{}`",
            canonical_artifact.display(),
            canonical_root.display()
        );
    }

    Ok(())
}

fn load_registered_plugin_runtime(
    manifest: &PluginManifest,
    manifest_path: &Path,
) -> Result<RegisteredPluginRuntime> {
    if manifest.runtime.library_path.is_some() {
        bail!(
            "plugin `{}` declares runtime.library_path but dynamic library runtime loading is not implemented in the new architecture yet",
            manifest.name
        );
    }

    if manifest.runtime.wasm_path.is_some() {
        bail!(
            "plugin `{}` declares runtime.wasm_path but wasm runtime loading is not implemented in the new architecture yet",
            manifest.name
        );
    }

    if manifest.runtime.sampling_profile.is_some()
        && !manifest.declares_core_rewriter(CoreComponent::Inference)
    {
        bail!(
            "plugin `{}` declares a sampling profile but does not declare inference core rewriter capability",
            manifest.name
        );
    }

    let sampling_hook = manifest
        .runtime
        .sampling_profile
        .as_deref()
        .map(|profile_path| {
            let profile_path = resolve_runtime_artifact_path(manifest_path, profile_path);
            validate_runtime_artifact_within_plugin_root(manifest_path, &profile_path)?;
            let profile = load_sampling_hook_profile(&profile_path)?;
            Ok::<Arc<dyn SamplingHook>, anyhow::Error>(Arc::new(ProfiledSamplingHook::new(profile)))
        })
        .transpose()?;

    Ok(RegisteredPluginRuntime { sampling_hook })
}

fn load_legacy_plugin_bundle(runtime_artifact_path: &Path) -> Result<RegisteredPlugin> {
    let contract = load_legacy_contract_manifest(runtime_artifact_path)?.ok_or_else(|| {
        anyhow::anyhow!(
            "legacy plugin contract not found for: {}",
            runtime_artifact_path.display()
        )
    })?;
    let manifest = convert_legacy_contract_manifest(runtime_artifact_path, contract);
    Ok(RegisteredPlugin::new(manifest).with_manifest_location(runtime_artifact_path.to_path_buf()))
}

pub fn load_plugin_bundle_file(path: impl AsRef<Path>) -> Result<RegisteredPlugin> {
    let path = path.as_ref();
    let is_native_manifest = path
        .file_name()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case(MANIFEST_FILE_NAME))
        .unwrap_or(false);

    if is_native_manifest {
        return load_plugin_manifest_file(path);
    }

    if is_legacy_runtime_artifact(path) {
        return load_legacy_plugin_bundle(path);
    }

    bail!("unsupported plugin bundle entry: {}", path.display());
}

pub fn load_plugin_manifest_file(path: impl AsRef<Path>) -> Result<RegisteredPlugin> {
    let path = path.as_ref();
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read plugin manifest: {}", path.display()))?;
    let manifest: PluginManifest = toml::from_str(&content)
        .with_context(|| format!("failed to parse plugin manifest: {}", path.display()))?;
    let runtime = load_registered_plugin_runtime(&manifest, path)?;
    Ok(RegisteredPlugin::new(manifest)
        .with_manifest_location(path.to_path_buf())
        .with_runtime(runtime))
}

pub fn discover_plugin_bundle_files(plugin_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let plugin_dir = plugin_dir.as_ref();
    if !plugin_dir.exists() {
        return Ok(Vec::new());
    }

    if plugin_dir.is_file() {
        if plugin_dir
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case(MANIFEST_FILE_NAME))
            .unwrap_or(false)
            || is_legacy_runtime_artifact(plugin_dir)
        {
            return Ok(vec![plugin_dir.to_path_buf()]);
        }
        return Ok(Vec::new());
    }

    let mut bundles = Vec::new();
    let root_manifest = plugin_dir.join(MANIFEST_FILE_NAME);
    if root_manifest.exists() {
        bundles.push(root_manifest);
    }

    for entry in fs::read_dir(plugin_dir)
        .with_context(|| format!("failed to scan plugin dir: {}", plugin_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if is_legacy_runtime_artifact(&path) {
                bundles.push(path);
            }
            continue;
        }

        if !path.is_dir() {
            continue;
        }

        let manifest = path.join(MANIFEST_FILE_NAME);
        if manifest.exists() {
            bundles.push(manifest);
        }

        for child in fs::read_dir(&path)
            .with_context(|| format!("failed to scan plugin bundle dir: {}", path.display()))?
        {
            let child = child?;
            let child_path = child.path();
            if is_legacy_runtime_artifact(&child_path) {
                bundles.push(child_path);
            }
        }
    }

    bundles.sort();
    bundles.dedup();
    Ok(bundles)
}

pub fn discover_plugin_manifest_files(plugin_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let plugin_dir = plugin_dir.as_ref();
    if !plugin_dir.exists() {
        return Ok(Vec::new());
    }

    if plugin_dir.is_file() {
        if plugin_dir
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.eq_ignore_ascii_case(MANIFEST_FILE_NAME))
            .unwrap_or(false)
        {
            return Ok(vec![plugin_dir.to_path_buf()]);
        }
        return Ok(Vec::new());
    }

    let mut manifests = Vec::new();
    let root_manifest = plugin_dir.join(MANIFEST_FILE_NAME);
    if root_manifest.exists() {
        manifests.push(root_manifest);
    }

    for entry in fs::read_dir(plugin_dir)
        .with_context(|| format!("failed to scan plugin dir: {}", plugin_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let manifest = path.join(MANIFEST_FILE_NAME);
        if manifest.exists() {
            manifests.push(manifest);
        }
    }

    manifests.sort();
    manifests.dedup();
    Ok(manifests)
}

#[derive(Default)]
pub struct InMemoryPluginManager {
    plugins: Vec<RegisteredPlugin>,
    plugin_index: BTreeMap<String, usize>,
    sampling_hooks: BTreeMap<String, Arc<dyn SamplingHook>>,
}

impl crate::core::PluginManager for InMemoryPluginManager {
    fn register(&mut self, plugin: RegisteredPlugin) -> Result<()> {
        if self.plugin_index.contains_key(&plugin.manifest.name) {
            bail!("plugin already registered: {}", plugin.manifest.name);
        }

        let plugin_name = plugin.manifest.name.clone();
        let sampling_hook = plugin.sampling_hook();
        let index = self.plugins.len();
        self.plugin_index.insert(plugin_name.clone(), index);
        self.plugins.push(plugin);

        if let Some(hook) = sampling_hook {
            self.sampling_hooks.insert(plugin_name, hook);
        }

        Ok(())
    }

    fn register_sampling_hook(
        &mut self,
        plugin_name: &str,
        hook: Arc<dyn SamplingHook>,
    ) -> Result<()> {
        let plugin = self
            .get(plugin_name)
            .ok_or_else(|| anyhow::anyhow!("plugin not registered: {plugin_name}"))?;

        if !plugin.declares_inference_sampling_runtime() {
            bail!(
                "plugin `{}` does not declare inference core rewriter capability",
                plugin.manifest.name
            );
        }

        if !self.plugin_index.contains_key(plugin_name) {
            bail!("plugin not registered: {plugin_name}");
        }

        self.sampling_hooks.insert(plugin_name.to_string(), hook);
        Ok(())
    }

    fn list(&self) -> &[RegisteredPlugin] {
        &self.plugins
    }

    fn get(&self, plugin_name: &str) -> Option<&RegisteredPlugin> {
        self.plugin_index
            .get(plugin_name)
            .and_then(|index| self.plugins.get(*index))
    }

    fn plugins_for_track(&self, track: PlatformTrack) -> Vec<&RegisteredPlugin> {
        self.plugins
            .iter()
            .filter(|plugin| plugin.supports_track(track))
            .collect()
    }

    fn plugins_for_model_provider(&self, provider: &str) -> Vec<&RegisteredPlugin> {
        self.plugins
            .iter()
            .filter(|plugin| {
                plugin
                    .manifest
                    .contributes
                    .model_providers
                    .iter()
                    .any(|candidate| candidate == provider)
            })
            .collect()
    }

    fn plugins_for_core_component(&self, component: CoreComponent) -> Vec<&RegisteredPlugin> {
        self.plugins
            .iter()
            .filter(|plugin| plugin.declares_core_rewriter(component))
            .collect()
    }

    fn sampling_runtime_for_inference(
        &self,
        active_plugin_name: Option<&str>,
    ) -> PluginSamplingRuntime {
        let hooks = active_plugin_name
            .into_iter()
            .filter_map(|plugin_name| {
                self.sampling_hooks
                    .get(plugin_name)
                    .map(|hook| RegisteredSamplingHook {
                        plugin_name: plugin_name.to_string(),
                        hook: Arc::clone(hook),
                    })
            })
            .collect();

        PluginSamplingRuntime { hooks }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PluginManager;
    use crate::sampler::LogitsView;
    use loci_plugin_api::{
        ContributionPoints, CoreRewriters, PluginBootstrap, PluginCompatibility, PluginRuntime,
    };
    use std::fs;
    use std::sync::Arc;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        path.push(format!("loci-plugin-test-{name}-{nanos}"));
        path
    }

    fn plugin_manifest(name: &str) -> PluginManifest {
        PluginManifest {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            api_version: "1.0".to_string(),
            min_host_version: None,
            max_host_version: None,
            target_tracks: vec![PlatformTrack::AiInfra],
            contributes: ContributionPoints::default(),
            core_rewriters: CoreRewriters::default(),
            runtime: PluginRuntime::default(),
            bootstrap: PluginBootstrap::default(),
            compatibility: PluginCompatibility::default(),
        }
    }

    #[test]
    fn manager_indexes_plugins_by_track_provider_and_core_component() {
        let mut manager = InMemoryPluginManager::default();
        let mut manifest = plugin_manifest("infra-provider");
        manifest.contributes.model_providers = vec!["private-registry".to_string()];
        manifest.core_rewriters.inference = true;

        manager
            .register(RegisteredPlugin::new(manifest))
            .expect("register");

        assert_eq!(manager.plugins_for_track(PlatformTrack::AiInfra).len(), 1);
        assert_eq!(
            manager.plugins_for_model_provider("private-registry").len(),
            1
        );
        assert_eq!(
            manager
                .plugins_for_core_component(CoreComponent::Inference)
                .len(),
            1
        );
    }

    #[test]
    fn manager_rejects_duplicate_plugin_names() {
        let mut manager = InMemoryPluginManager::default();
        let plugin = RegisteredPlugin::new(plugin_manifest("duplicate"));

        manager.register(plugin.clone()).expect("first register");
        let err = manager.register(plugin).expect_err("should reject");

        assert!(err.to_string().contains("plugin already registered"));
    }

    #[test]
    fn discover_and_load_plugin_manifests_from_directory() {
        let dir = unique_temp_dir("discover");
        fs::create_dir_all(dir.join("plugin-a")).expect("mkdir");
        fs::create_dir_all(dir.join("plugin-b")).expect("mkdir");

        fs::write(
            dir.join("plugin-a").join(MANIFEST_FILE_NAME),
            r#"
name = "plugin-a"
version = "1.0.0"
api_version = "1.0"
"#,
        )
        .expect("write");
        fs::write(
            dir.join("plugin-b").join(MANIFEST_FILE_NAME),
            r#"
name = "plugin-b"
version = "1.0.0"
api_version = "1.0"
"#,
        )
        .expect("write");

        let manifests = discover_plugin_manifest_files(&dir).expect("discover");
        assert_eq!(manifests.len(), 2);

        let plugin = load_plugin_manifest_file(&manifests[0]).expect("load");
        assert!(!plugin.manifest.name.is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_and_load_legacy_plugin_bundle_from_directory() {
        let dir = unique_temp_dir("legacy-discover");
        fs::create_dir_all(dir.join("legacy-plugin")).expect("mkdir");
        fs::write(dir.join("legacy-plugin").join("rot13.dll"), b"binary").expect("write runtime");
        fs::write(
            dir.join("legacy-plugin").join("rot13.loci-plugin.json"),
            r#"{
  "name": "rot13_dynamic",
  "version": "1.0.0",
  "kind": "text_plugin",
  "abi_version": 1,
  "min_host_version": "0.1.0",
  "capabilities": ["pre_generate", "post_generate"]
}"#,
        )
        .expect("write contract");

        let bundles = discover_plugin_bundle_files(&dir).expect("discover bundles");
        assert_eq!(bundles.len(), 1);
        assert_eq!(
            bundles[0].file_name().and_then(|value| value.to_str()),
            Some("rot13.dll")
        );

        let plugin = load_plugin_bundle_file(&bundles[0]).expect("load legacy bundle");
        assert!(plugin.is_legacy_compat_bundle());
        assert_eq!(plugin.manifest.name, "rot13_dynamic");
        assert_eq!(
            plugin.legacy_runtime_path(),
            Some(&*bundles[0].to_string_lossy())
        );
        assert_eq!(
            plugin.manifest.compatibility.legacy_kind.as_deref(),
            Some("text_plugin")
        );
        assert_eq!(plugin.manifest.min_host_version.as_deref(), Some("0.1.0"));

        let _ = fs::remove_dir_all(&dir);
    }

    struct BiasHook;

    impl SamplingHook for BiasHook {
        fn transform_logits(
            &self,
            logits: &mut LogitsView<'_>,
            _context_tokens: &[i32],
        ) -> CoreResult<()> {
            logits.set_usize(0, -10.0)?;
            logits.set_usize(2, 99.0)?;
            Ok(())
        }

        fn post_sample(&self, _token_id: i32) -> CoreResult<i32> {
            Ok(7)
        }
    }

    #[test]
    fn manager_builds_sampling_runtime_from_registered_hooks() {
        let mut manager = InMemoryPluginManager::default();
        let mut manifest = plugin_manifest("sampler-hook");
        manifest.core_rewriters.inference = true;
        manager
            .register(RegisteredPlugin::new(manifest))
            .expect("register plugin");
        manager
            .register_sampling_hook("sampler-hook", Arc::new(BiasHook))
            .expect("register hook");

        let runtime = manager.sampling_runtime_for_inference(Some("sampler-hook"));
        assert_eq!(runtime.hook_count(), 1);
        assert_eq!(runtime.plugin_names(), vec!["sampler-hook"]);

        let mut logits = vec![1.0, 2.0, 3.0];
        runtime
            .apply_transform_logits(&mut LogitsView::new(&mut logits), &[])
            .expect("transform");
        assert_eq!(logits[0], -10.0);
        assert_eq!(logits[2], 99.0);
        assert_eq!(runtime.apply_post_sample(2).expect("post sample"), 7);
    }

    #[test]
    fn manager_rejects_sampling_hook_for_unknown_plugin() {
        let mut manager = InMemoryPluginManager::default();
        let err = manager
            .register_sampling_hook("missing", Arc::new(BiasHook))
            .expect_err("should reject");

        assert!(err.to_string().contains("plugin not registered"));
    }

    #[test]
    fn manager_rejects_sampling_hook_without_inference_declaration() {
        let mut manager = InMemoryPluginManager::default();
        manager
            .register(RegisteredPlugin::new(plugin_manifest("plain-plugin")))
            .expect("register plugin");

        let err = manager
            .register_sampling_hook("plain-plugin", Arc::new(BiasHook))
            .expect_err("should reject");

        assert!(err
            .to_string()
            .contains("does not declare inference core rewriter capability"));
    }

    #[test]
    fn manager_sampling_runtime_is_empty_without_active_inference_plugin() {
        let mut manager = InMemoryPluginManager::default();
        let mut manifest = plugin_manifest("sampler-hook");
        manifest.core_rewriters.inference = true;
        manager
            .register(RegisteredPlugin::new(manifest))
            .expect("register plugin");
        manager
            .register_sampling_hook("sampler-hook", Arc::new(BiasHook))
            .expect("register hook");

        let runtime = manager.sampling_runtime_for_inference(None);
        assert_eq!(runtime.hook_count(), 0);
    }

    #[test]
    fn load_plugin_manifest_file_wires_sampling_hook_from_sidecar_profile() {
        let dir = unique_temp_dir("sampling-sidecar");
        fs::create_dir_all(dir.join("sampler-plugin")).expect("mkdir");
        fs::write(
            dir.join("sampler-plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "sampler-plugin"
version = "1.0.0"
api_version = "1.0"
target_tracks = ["ai_infra"]

[core_rewriters]
inference = true

[runtime]
sampling_profile = "sampling-hook.toml"
"#,
        )
        .expect("write manifest");
        fs::write(
            dir.join("sampler-plugin").join("sampling-hook.toml"),
            r#"
force_token_id = 1
forced_logit = 55.0
post_sample_override = 7

[[logit_biases]]
token_id = 2
logit = 99.0
"#,
        )
        .expect("write profile");

        let plugin = load_plugin_manifest_file(dir.join("sampler-plugin").join(MANIFEST_FILE_NAME))
            .expect("load plugin bundle");
        assert!(plugin.has_sampling_hook());

        let mut manager = InMemoryPluginManager::default();
        manager.register(plugin).expect("register plugin");

        let runtime = manager.sampling_runtime_for_inference(Some("sampler-plugin"));
        assert_eq!(runtime.hook_count(), 1);

        let mut logits = vec![1.0, 2.0, 3.0];
        runtime
            .apply_transform_logits(&mut LogitsView::new(&mut logits), &[])
            .expect("transform");
        assert_eq!(logits[1], 55.0);
        assert_eq!(logits[2], 99.0);
        assert_eq!(runtime.apply_post_sample(3).expect("post sample"), 7);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_plugin_manifest_file_rejects_sampling_profile_without_inference_rewriter() {
        let dir = unique_temp_dir("bad-sampling-sidecar");
        fs::create_dir_all(dir.join("plain-plugin")).expect("mkdir");
        fs::write(
            dir.join("plain-plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "plain-plugin"
version = "1.0.0"
api_version = "1.0"

[runtime]
sampling_profile = "sampling-hook.toml"
"#,
        )
        .expect("write manifest");
        fs::write(
            dir.join("plain-plugin").join("sampling-hook.toml"),
            "post_sample_override = 4\n",
        )
        .expect("write profile");

        let err = load_plugin_manifest_file(dir.join("plain-plugin").join(MANIFEST_FILE_NAME))
            .expect_err("load should fail");

        assert!(err
            .to_string()
            .contains("does not declare inference core rewriter capability"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_plugin_manifest_file_rejects_sampling_profile_outside_plugin_root() {
        let dir = unique_temp_dir("escaped-sampling-sidecar");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(dir.join("outside.toml"), "post_sample_override = 4\n")
            .expect("write outside profile");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "escaped-plugin"
version = "1.0.0"
api_version = "1.0"

[core_rewriters]
inference = true

[runtime]
sampling_profile = "../outside.toml"
"#,
        )
        .expect("write manifest");

        let err = load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME))
            .expect_err("load should fail");

        assert!(err.to_string().contains("escapes plugin root"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_plugin_manifest_file_rejects_unimplemented_runtime_kinds() {
        let dir = unique_temp_dir("unsupported-runtime");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "dynamic-plugin"
version = "1.0.0"
api_version = "1.0"

[runtime]
library_path = "plugin.dll"
"#,
        )
        .expect("write manifest");

        let err = load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME))
            .expect_err("load should fail");

        assert!(err
            .to_string()
            .contains("dynamic library runtime loading is not implemented"));

        let _ = fs::remove_dir_all(&dir);
    }
}

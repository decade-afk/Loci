use crate::error::Result as CoreResult;
use crate::sampler::LogitsView;
use anyhow::{bail, Context, Result};
use loci_legacy_plugin_compat::LegacyTextCompat;
use loci_plugin_api::{
    ContributionPoints, CoreComponent, CoreRewriters, LegacyRuntimeBridge, PlatformTrack,
    PluginBootstrap, PluginCompatibility, PluginManifest, PluginRuntime, PluginSourceFormat,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MANIFEST_FILE_NAME: &str = "manifest.toml";
const HOST_PLUGIN_API_VERSION: &str = "1.0";
const LEGACY_PLUGIN_ABI_VERSION_CURRENT: u32 = 2;
const LEGACY_PLUGIN_ABI_VERSION_SUPPORTED: &[u32] = &[1, 2];
const LEGACY_DYNAMIC_EXTENSIONS: &[&str] = &["dll", "so", "dylib"];
const LEGACY_WASM_EXTENSIONS: &[&str] = &["wasm"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LegacyCapability {
    PreGenerate,
    PostGenerate,
    OnToken,
    TransformLogits,
    PostSample,
}

impl LegacyCapability {
    fn from_str(raw: &str) -> Option<Self> {
        match raw {
            "pre_generate" => Some(Self::PreGenerate),
            "post_generate" => Some(Self::PostGenerate),
            "on_token" => Some(Self::OnToken),
            "transform_logits" => Some(Self::TransformLogits),
            "post_sample" => Some(Self::PostSample),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::PreGenerate => "pre_generate",
            Self::PostGenerate => "post_generate",
            Self::OnToken => "on_token",
            Self::TransformLogits => "transform_logits",
            Self::PostSample => "post_sample",
        }
    }

    fn is_sampling(self) -> bool {
        matches!(self, Self::TransformLogits | Self::PostSample)
    }

    fn supports_text_compat_bridge(self) -> bool {
        matches!(
            self,
            Self::PreGenerate | Self::PostGenerate | Self::TransformLogits | Self::PostSample
        )
    }
}

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
    LEGACY_PLUGIN_ABI_VERSION_CURRENT
}

#[derive(Clone)]
struct LegacySamplingCompatHook {
    compat: Arc<dyn LegacyTextCompat>,
}

impl LegacySamplingCompatHook {
    fn new(compat: Arc<dyn LegacyTextCompat>) -> Self {
        Self { compat }
    }
}

impl SamplingHook for LegacySamplingCompatHook {
    fn transform_logits(
        &self,
        logits: &mut LogitsView<'_>,
        context_tokens: &[i32],
    ) -> CoreResult<()> {
        self.compat
            .transform_logits(logits.as_mut_slice(), context_tokens)
            .map_err(crate::error::LociError::from)
    }

    fn post_sample(&self, token_id: i32) -> CoreResult<i32> {
        self.compat
            .post_sample(token_id)
            .map_err(crate::error::LociError::from)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegisteredHostRuntimeKind {
    DynamicLibrary,
    WasmModule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RegisteredHostRuntime {
    kind: RegisteredHostRuntimeKind,
    declared_path: String,
    resolved_path: PathBuf,
}

impl RegisteredHostRuntime {
    pub(crate) fn kind(&self) -> RegisteredHostRuntimeKind {
        self.kind
    }

    pub(crate) fn declared_path(&self) -> &str {
        &self.declared_path
    }

    pub(crate) fn resolved_path(&self) -> &Path {
        &self.resolved_path
    }
}

#[derive(Clone, Default)]
struct RegisteredPluginRuntime {
    host_runtimes: Vec<RegisteredHostRuntime>,
    sampling_hook: Option<Arc<dyn SamplingHook>>,
    legacy_text_compat: Option<Arc<dyn LegacyTextCompat>>,
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

    pub fn has_legacy_text_compat_runtime(&self) -> bool {
        self.runtime.legacy_text_compat.is_some()
    }

    pub fn supports_legacy_pre_generate(&self) -> bool {
        self.declares_legacy_capability(LegacyCapability::PreGenerate)
    }

    pub fn supports_legacy_post_generate(&self) -> bool {
        self.declares_legacy_capability(LegacyCapability::PostGenerate)
    }

    pub fn supports_legacy_sampling(&self) -> bool {
        self.manifest
            .compatibility
            .legacy_capabilities
            .iter()
            .filter_map(|candidate| LegacyCapability::from_str(candidate))
            .any(LegacyCapability::is_sampling)
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

    pub(crate) fn legacy_text_compat_runtime(&self) -> Option<Arc<dyn LegacyTextCompat>> {
        self.runtime.legacy_text_compat.as_ref().map(Arc::clone)
    }

    pub(crate) fn declares_legacy_capability(&self, capability: LegacyCapability) -> bool {
        self.manifest
            .compatibility
            .legacy_capabilities
            .iter()
            .filter_map(|candidate| LegacyCapability::from_str(candidate))
            .any(|candidate| candidate == capability)
    }

    pub(crate) fn registered_host_runtimes(&self) -> &[RegisteredHostRuntime] {
        &self.runtime.host_runtimes
    }
}

pub(crate) fn legacy_sampling_hook_from_compat(
    compat: Arc<dyn LegacyTextCompat>,
) -> Arc<dyn SamplingHook> {
    Arc::new(LegacySamplingCompatHook::new(compat))
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

fn parse_version_tuple(raw: &str) -> Option<(u64, u64, u64)> {
    let core = raw.split(['-', '+']).next()?.trim();
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    let patch = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor, patch))
}

fn validate_host_version_range(
    plugin_name: &str,
    min_host_version: Option<&str>,
    max_host_version: Option<&str>,
) -> Result<()> {
    let host_version = parse_version_tuple(env!("CARGO_PKG_VERSION")).ok_or_else(|| {
        anyhow::anyhow!(
            "host package version is not valid semver: {}",
            env!("CARGO_PKG_VERSION")
        )
    })?;

    if let Some(min_version) = min_host_version {
        let min_version_tuple = parse_version_tuple(min_version).ok_or_else(|| {
            anyhow::anyhow!("plugin `{plugin_name}` has invalid min_host_version `{min_version}`")
        })?;
        if host_version < min_version_tuple {
            bail!(
                "plugin `{plugin_name}` requires host >= `{min_version}`, current host is `{}`",
                env!("CARGO_PKG_VERSION")
            );
        }
    }

    if let Some(max_version) = max_host_version {
        let max_version_tuple = parse_version_tuple(max_version).ok_or_else(|| {
            anyhow::anyhow!("plugin `{plugin_name}` has invalid max_host_version `{max_version}`")
        })?;
        if host_version > max_version_tuple {
            bail!(
                "plugin `{plugin_name}` supports host <= `{max_version}`, current host is `{}`",
                env!("CARGO_PKG_VERSION")
            );
        }
    }

    Ok(())
}

fn validate_plugin_manifest(manifest: &PluginManifest) -> Result<()> {
    if manifest.name.trim().is_empty() {
        bail!("plugin manifest name cannot be empty");
    }
    if manifest.version.trim().is_empty() {
        bail!("plugin `{}` has empty version", manifest.name);
    }
    if manifest.api_version != HOST_PLUGIN_API_VERSION {
        bail!(
            "plugin `{}` declares api_version `{}`, host supports `{HOST_PLUGIN_API_VERSION}`",
            manifest.name,
            manifest.api_version
        );
    }
    if let Some(contract) = manifest.runtime.host_contract.as_ref() {
        if contract.protocol.trim().is_empty() {
            bail!(
                "plugin `{}` declares an empty runtime.host_contract.protocol",
                manifest.name
            );
        }
        if contract.entrypoint.trim().is_empty() {
            bail!(
                "plugin `{}` declares an empty runtime.host_contract.entrypoint",
                manifest.name
            );
        }
        if manifest.runtime.library_path.is_none() && manifest.runtime.wasm_path.is_none() {
            bail!(
                "plugin `{}` declares runtime.host_contract without a host runtime artifact",
                manifest.name
            );
        }
    }

    validate_host_version_range(
        &manifest.name,
        manifest.min_host_version.as_deref(),
        manifest.max_host_version.as_deref(),
    )
}

fn validate_legacy_contract_manifest(contract: &LegacyPluginContractManifest) -> Result<()> {
    if contract.name.trim().is_empty() {
        bail!("legacy plugin contract name cannot be empty");
    }
    if contract.version.trim().is_empty() {
        bail!("legacy plugin `{}` has empty version", contract.name);
    }
    if !LEGACY_PLUGIN_ABI_VERSION_SUPPORTED.contains(&contract.abi_version) {
        bail!(
            "legacy plugin `{}` requires ABI v{}, host supports {:?}",
            contract.name,
            contract.abi_version,
            LEGACY_PLUGIN_ABI_VERSION_SUPPORTED
        );
    }

    validate_host_version_range(
        &contract.name,
        contract.min_host_version.as_deref(),
        contract.max_host_version.as_deref(),
    )
}

fn legacy_kind_target_tracks(kind: LegacyPluginContractKind) -> Vec<PlatformTrack> {
    match kind {
        LegacyPluginContractKind::ToolPlugin => vec![PlatformTrack::AiAgent],
        LegacyPluginContractKind::TextPlugin
        | LegacyPluginContractKind::ExecutionPolicy
        | LegacyPluginContractKind::ManagementAuthPolicy
        | LegacyPluginContractKind::ModelPullPolicy
        | LegacyPluginContractKind::ModelPullVerifier
        | LegacyPluginContractKind::ServeDispatchPolicy
        | LegacyPluginContractKind::ImageKernel
        | LegacyPluginContractKind::Backend => vec![PlatformTrack::AiInfra],
    }
}

fn legacy_sampling_capabilities(contract: &LegacyPluginContractManifest) -> Vec<String> {
    if contract.kind != LegacyPluginContractKind::TextPlugin {
        return Vec::new();
    }

    contract
        .capabilities
        .iter()
        .filter_map(|capability| LegacyCapability::from_str(capability))
        .filter(|capability| capability.is_sampling())
        .map(|capability| capability.as_str().to_string())
        .collect()
}

fn legacy_text_compat_capabilities(contract: &LegacyPluginContractManifest) -> Vec<String> {
    if contract.kind != LegacyPluginContractKind::TextPlugin {
        return Vec::new();
    }

    contract
        .capabilities
        .iter()
        .filter_map(|capability| LegacyCapability::from_str(capability))
        .filter(|capability| capability.supports_text_compat_bridge())
        .map(|capability| capability.as_str().to_string())
        .collect()
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

fn is_legacy_dynamic_library(path: &Path) -> bool {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    LEGACY_DYNAMIC_EXTENSIONS.contains(&extension.as_str())
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
    contract: &LegacyPluginContractManifest,
) -> PluginManifest {
    let compat_capabilities = legacy_text_compat_capabilities(contract);
    let sampling_capabilities = legacy_sampling_capabilities(contract);
    let enables_text_compat_bridge =
        is_legacy_dynamic_library(runtime_artifact_path) && !compat_capabilities.is_empty();
    let enables_sampling_bridge = enables_text_compat_bridge && !sampling_capabilities.is_empty();

    PluginManifest {
        name: contract.name.clone(),
        version: contract.version.clone(),
        api_version: HOST_PLUGIN_API_VERSION.to_string(),
        min_host_version: contract.min_host_version.clone(),
        max_host_version: contract.max_host_version.clone(),
        target_tracks: legacy_kind_target_tracks(contract.kind),
        contributes: ContributionPoints {
            inference_hooks: sampling_capabilities,
            ..Default::default()
        },
        core_rewriters: CoreRewriters {
            inference: enables_sampling_bridge,
            ..Default::default()
        },
        runtime: PluginRuntime::default(),
        bootstrap: PluginBootstrap::default(),
        compatibility: PluginCompatibility {
            source_format: PluginSourceFormat::LegacyContract,
            legacy_kind: Some(contract.kind.as_str().to_string()),
            legacy_abi_version: Some(contract.abi_version),
            legacy_runtime_path: Some(runtime_artifact_path.to_string_lossy().to_string()),
            legacy_capabilities: contract.capabilities.clone(),
            runtime_bridge: if enables_text_compat_bridge {
                LegacyRuntimeBridge::LegacyTextPluginV1
            } else {
                LegacyRuntimeBridge::None
            },
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
) -> Result<PathBuf> {
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

    if !canonical_artifact.is_file() {
        bail!(
            "runtime artifact `{}` is not a file",
            canonical_artifact.display()
        );
    }

    if !canonical_artifact.starts_with(&canonical_root) {
        bail!(
            "runtime artifact `{}` escapes plugin root `{}`",
            canonical_artifact.display(),
            canonical_root.display()
        );
    }

    Ok(canonical_artifact)
}

fn register_host_runtime_artifact(
    manifest_path: &Path,
    relative_path: &str,
    kind: RegisteredHostRuntimeKind,
) -> Result<RegisteredHostRuntime> {
    let artifact_path = resolve_runtime_artifact_path(manifest_path, relative_path);
    let resolved_path =
        validate_runtime_artifact_within_plugin_root(manifest_path, &artifact_path)?;

    Ok(RegisteredHostRuntime {
        kind,
        declared_path: relative_path.to_string(),
        resolved_path,
    })
}

fn load_registered_plugin_runtime(
    manifest: &PluginManifest,
    manifest_path: &Path,
) -> Result<RegisteredPluginRuntime> {
    validate_plugin_manifest(manifest)?;

    if manifest.runtime.library_path.is_some() && manifest.runtime.wasm_path.is_some() {
        bail!(
            "plugin `{}` declares both runtime.library_path and runtime.wasm_path; declare exactly one executable host runtime artifact",
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
            let profile_path =
                validate_runtime_artifact_within_plugin_root(manifest_path, &profile_path)?;
            let profile = load_sampling_hook_profile(&profile_path)?;
            Ok::<Arc<dyn SamplingHook>, anyhow::Error>(Arc::new(ProfiledSamplingHook::new(profile)))
        })
        .transpose()?;

    let mut host_runtimes = Vec::new();
    if let Some(library_path) = manifest.runtime.library_path.as_deref() {
        host_runtimes.push(register_host_runtime_artifact(
            manifest_path,
            library_path,
            RegisteredHostRuntimeKind::DynamicLibrary,
        )?);
    }
    if let Some(wasm_path) = manifest.runtime.wasm_path.as_deref() {
        host_runtimes.push(register_host_runtime_artifact(
            manifest_path,
            wasm_path,
            RegisteredHostRuntimeKind::WasmModule,
        )?);
    }

    Ok(RegisteredPluginRuntime {
        host_runtimes,
        sampling_hook,
        legacy_text_compat: None,
    })
}

fn load_legacy_plugin_runtime(
    _runtime_artifact_path: &Path,
    _manifest: &PluginManifest,
) -> Result<RegisteredPluginRuntime> {
    // Legacy bundles register governance metadata only. Runtime materialization
    // happens during explicit activation in the engine.
    Ok(RegisteredPluginRuntime::default())
}

fn load_legacy_plugin_bundle(runtime_artifact_path: &Path) -> Result<RegisteredPlugin> {
    let contract = load_legacy_contract_manifest(runtime_artifact_path)?.ok_or_else(|| {
        anyhow::anyhow!(
            "legacy plugin contract not found for: {}",
            runtime_artifact_path.display()
        )
    })?;
    validate_legacy_contract_manifest(&contract)?;
    let manifest = convert_legacy_contract_manifest(runtime_artifact_path, &contract);
    let runtime = load_legacy_plugin_runtime(runtime_artifact_path, &manifest)?;
    Ok(RegisteredPlugin::new(manifest)
        .with_manifest_location(runtime_artifact_path.to_path_buf())
        .with_runtime(runtime))
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
    validate_plugin_manifest(&manifest)?;
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
    let root_has_manifest = root_manifest.exists();
    if root_has_manifest {
        bundles.push(root_manifest);
    }

    for entry in fs::read_dir(plugin_dir)
        .with_context(|| format!("failed to scan plugin dir: {}", plugin_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if !root_has_manifest && is_legacy_runtime_artifact(&path) {
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
            continue;
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
struct TestLegacyTextCompat {
    plugin_name: String,
}

#[cfg(test)]
impl LegacyTextCompat for TestLegacyTextCompat {
    fn pre_generate(&self, prompt: &str) -> Result<String> {
        Ok(format!("[{}:pre]{prompt}", self.plugin_name))
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        Ok(format!("{response}[{}:post]", self.plugin_name))
    }

    fn transform_logits(&self, _logits: &mut [f32], _context_tokens: &[i32]) -> Result<()> {
        Ok(())
    }

    fn post_sample(&self, token_id: i32) -> Result<i32> {
        Ok(token_id)
    }
}

#[cfg(test)]
pub(crate) fn registered_legacy_text_plugin_for_tests(
    plugin_name: &str,
    capabilities: &[&str],
) -> RegisteredPlugin {
    RegisteredPlugin {
        manifest: PluginManifest {
            name: plugin_name.to_string(),
            version: "1.0.0".to_string(),
            api_version: HOST_PLUGIN_API_VERSION.to_string(),
            min_host_version: None,
            max_host_version: None,
            target_tracks: vec![PlatformTrack::AiInfra],
            contributes: ContributionPoints {
                inference_hooks: capabilities
                    .iter()
                    .filter_map(|capability| LegacyCapability::from_str(capability))
                    .filter(|capability| capability.is_sampling())
                    .map(|capability| capability.as_str().to_string())
                    .collect(),
                ..Default::default()
            },
            core_rewriters: CoreRewriters {
                inference: capabilities
                    .iter()
                    .filter_map(|capability| LegacyCapability::from_str(capability))
                    .any(LegacyCapability::is_sampling),
                ..Default::default()
            },
            runtime: PluginRuntime::default(),
            bootstrap: PluginBootstrap::default(),
            compatibility: PluginCompatibility {
                source_format: PluginSourceFormat::LegacyContract,
                legacy_kind: Some("text_plugin".to_string()),
                legacy_abi_version: Some(LEGACY_PLUGIN_ABI_VERSION_CURRENT),
                legacy_runtime_path: Some(format!("plugins/{plugin_name}.dll")),
                legacy_capabilities: capabilities
                    .iter()
                    .map(|capability| (*capability).to_string())
                    .collect(),
                runtime_bridge: LegacyRuntimeBridge::LegacyTextPluginV1,
            },
        },
        manifest_path: None,
        root_dir: None,
        runtime: RegisteredPluginRuntime {
            host_runtimes: Vec::new(),
            sampling_hook: None,
            legacy_text_compat: Some(Arc::new(TestLegacyTextCompat {
                plugin_name: plugin_name.to_string(),
            })),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PluginManager;
    use crate::sampler::LogitsView;
    use loci_plugin_api::{
        ContributionPoints, CoreRewriters, LegacyRuntimeBridge, PluginBootstrap,
        PluginCompatibility, PluginRuntime,
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
        assert_eq!(plugin.manifest.target_tracks, vec![PlatformTrack::AiInfra]);
        assert_eq!(
            plugin.manifest.compatibility.legacy_capabilities,
            vec!["pre_generate".to_string(), "post_generate".to_string()]
        );
        assert_eq!(
            plugin.manifest.compatibility.runtime_bridge,
            LegacyRuntimeBridge::LegacyTextPluginV1
        );
        assert!(!plugin.declares_inference_sampling_runtime());
        assert!(!plugin.has_legacy_text_compat_runtime());
        assert!(plugin.supports_legacy_pre_generate());
        assert!(plugin.supports_legacy_post_generate());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_legacy_sampling_bundle_keeps_runtime_unmaterialized() {
        let dir = unique_temp_dir("legacy-sampling-discover");
        fs::create_dir_all(dir.join("legacy-sampler")).expect("mkdir");
        fs::write(dir.join("legacy-sampler").join("sampler.dll"), b"binary")
            .expect("write runtime");
        fs::write(
            dir.join("legacy-sampler").join("sampler.loci-plugin.json"),
            r#"{
  "name": "legacy_sampler",
  "version": "1.0.0",
  "kind": "text_plugin",
  "abi_version": 1,
  "capabilities": ["transform_logits", "post_sample"]
}"#,
        )
        .expect("write contract");

        let plugin = load_plugin_bundle_file(dir.join("legacy-sampler").join("sampler.dll"))
            .expect("load legacy sampling bundle");
        assert!(plugin.is_legacy_compat_bundle());
        assert!(plugin.declares_inference_sampling_runtime());
        assert!(plugin.supports_legacy_sampling());
        assert!(!plugin.has_sampling_hook());
        assert!(!plugin.has_legacy_text_compat_runtime());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn convert_legacy_sampling_contract_enables_controlled_bridge_metadata() {
        let runtime_artifact = PathBuf::from("plugins/legacy-sampler.dll");
        let manifest = convert_legacy_contract_manifest(
            &runtime_artifact,
            &LegacyPluginContractManifest {
                name: "legacy-sampler".to_string(),
                version: "1.0.0".to_string(),
                kind: LegacyPluginContractKind::TextPlugin,
                abi_version: 1,
                min_host_version: None,
                max_host_version: None,
                capabilities: vec![
                    "transform_logits".to_string(),
                    "post_sample".to_string(),
                    "post_generate".to_string(),
                ],
            },
        );

        assert_eq!(manifest.target_tracks, vec![PlatformTrack::AiInfra]);
        assert!(manifest.core_rewriters.inference);
        assert_eq!(
            manifest.contributes.inference_hooks,
            vec!["transform_logits".to_string(), "post_sample".to_string()]
        );
        assert_eq!(
            manifest.compatibility.legacy_capabilities,
            vec![
                "transform_logits".to_string(),
                "post_sample".to_string(),
                "post_generate".to_string(),
            ]
        );
        assert_eq!(
            manifest.compatibility.runtime_bridge,
            LegacyRuntimeBridge::LegacyTextPluginV1
        );
        let plugin = RegisteredPlugin::new(manifest);
        assert!(!plugin.supports_legacy_pre_generate());
        assert!(plugin.supports_legacy_post_generate());
    }

    #[test]
    fn discover_plugin_bundle_files_prefers_manifest_over_legacy_runtime_in_same_dir() {
        let dir = unique_temp_dir("mixed-bundles");
        fs::create_dir_all(dir.join("mixed-plugin")).expect("mkdir");
        fs::write(
            dir.join("mixed-plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "mixed-plugin"
version = "1.0.0"
api_version = "1.0"
"#,
        )
        .expect("write manifest");
        fs::write(dir.join("mixed-plugin").join("mixed.dll"), b"binary").expect("write runtime");
        fs::write(
            dir.join("mixed-plugin").join("mixed.loci-plugin.json"),
            r#"{
  "name": "mixed-plugin",
  "version": "1.0.0",
  "kind": "text_plugin",
  "abi_version": 1,
  "capabilities": ["transform_logits"]
}"#,
        )
        .expect("write contract");

        let bundles = discover_plugin_bundle_files(&dir).expect("discover bundles");
        assert_eq!(
            bundles,
            vec![dir.join("mixed-plugin").join(MANIFEST_FILE_NAME)]
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn workspace_example_plugins_include_ui_shell_bundle() {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir
            .parent()
            .and_then(Path::parent)
            .expect("workspace root");
        let plugin_root = workspace_root.join("plugins");
        let bundles = discover_plugin_bundle_files(&plugin_root).expect("discover bundles");
        let manifest_path = plugin_root
            .join("example-ui-shell")
            .join(MANIFEST_FILE_NAME);

        assert!(
            bundles.iter().any(|path| path == &manifest_path),
            "expected example-ui-shell bundle to be discoverable from workspace plugins dir"
        );

        let plugin = load_plugin_manifest_file(&manifest_path).expect("load example ui bundle");
        assert_eq!(plugin.manifest.name, "example-ui-shell");
        assert!(plugin.manifest.core_rewriters.ui_host);
        assert_eq!(
            plugin
                .manifest
                .runtime
                .host_contract
                .as_ref()
                .expect("host contract")
                .protocol,
            "loci.host-runtime.v1"
        );
        assert_eq!(plugin.runtime.host_runtimes.len(), 1);
        assert_eq!(
            plugin.runtime.host_runtimes[0].kind,
            RegisteredHostRuntimeKind::DynamicLibrary
        );
        assert_eq!(
            plugin.runtime.host_runtimes[0].declared_path,
            "runtime/plugin.dll"
        );
        assert_eq!(
            plugin.manifest.contributes.ui_contributes.panels,
            vec![
                "workspace-overview".to_string(),
                "model-catalog".to_string()
            ]
        );
        assert_eq!(
            plugin.manifest.contributes.ui_contributes.windows,
            vec!["operations-console".to_string()]
        );
        assert_eq!(
            plugin.manifest.contributes.ui_contributes.widgets,
            vec!["runtime-status".to_string()]
        );
    }

    #[test]
    fn load_plugin_manifest_file_rejects_host_contract_without_host_runtime() {
        let dir = unique_temp_dir("host-contract-without-runtime");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "host-contract-only"
version = "1.0.0"
api_version = "1.0"

[runtime.host_contract]
protocol = "loci.host-runtime.v1"
entrypoint = "bootstrap"
"#,
        )
        .expect("write manifest");

        let err = load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME))
            .expect_err("load should fail");
        assert!(err
            .to_string()
            .contains("declares runtime.host_contract without a host runtime artifact"));

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
    fn load_plugin_manifest_file_registers_dynamic_library_runtime_metadata() {
        let dir = unique_temp_dir("dynamic-runtime");
        fs::create_dir_all(dir.join("plugin").join("runtime")).expect("mkdir");
        fs::write(
            dir.join("plugin").join("runtime").join("plugin.dll"),
            b"binary",
        )
        .expect("write runtime");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "dynamic-plugin"
version = "1.0.0"
api_version = "1.0"

[runtime]
library_path = "runtime/plugin.dll"
"#,
        )
        .expect("write manifest");

        let plugin =
            load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME)).expect("load");

        assert_eq!(plugin.runtime.host_runtimes.len(), 1);
        assert_eq!(
            plugin.runtime.host_runtimes[0].kind,
            RegisteredHostRuntimeKind::DynamicLibrary
        );
        assert_eq!(
            plugin.runtime.host_runtimes[0].declared_path,
            "runtime/plugin.dll"
        );
        assert!(plugin.runtime.host_runtimes[0]
            .resolved_path
            .ends_with(Path::new("runtime").join("plugin.dll")));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_plugin_manifest_file_registers_wasm_runtime_metadata() {
        let dir = unique_temp_dir("wasm-runtime");
        fs::create_dir_all(dir.join("plugin").join("runtime")).expect("mkdir");
        fs::write(
            dir.join("plugin").join("runtime").join("plugin.wasm"),
            b"\0asm",
        )
        .expect("write runtime");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "wasm-plugin"
version = "1.0.0"
api_version = "1.0"

[runtime]
wasm_path = "runtime/plugin.wasm"
"#,
        )
        .expect("write manifest");

        let plugin =
            load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME)).expect("load");

        assert_eq!(plugin.runtime.host_runtimes.len(), 1);
        assert_eq!(
            plugin.runtime.host_runtimes[0].kind,
            RegisteredHostRuntimeKind::WasmModule
        );
        assert_eq!(
            plugin.runtime.host_runtimes[0].declared_path,
            "runtime/plugin.wasm"
        );
        assert!(plugin.runtime.host_runtimes[0]
            .resolved_path
            .ends_with(Path::new("runtime").join("plugin.wasm")));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_plugin_manifest_file_rejects_host_runtime_outside_plugin_root() {
        let dir = unique_temp_dir("escaped-runtime-sidecar");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(dir.join("outside.dll"), b"binary").expect("write outside runtime");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "escaped-runtime-plugin"
version = "1.0.0"
api_version = "1.0"

[runtime]
library_path = "../outside.dll"
"#,
        )
        .expect("write manifest");

        let err = load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME))
            .expect_err("load should fail");

        assert!(err.to_string().contains("escapes plugin root"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_plugin_manifest_file_rejects_multiple_host_runtime_artifacts() {
        let dir = unique_temp_dir("multiple-host-runtimes");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "hybrid-runtime-plugin"
version = "1.0.0"
api_version = "1.0"

[runtime]
library_path = "runtime/plugin.dll"
wasm_path = "runtime/plugin.wasm"
"#,
        )
        .expect("write manifest");

        let err = load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME))
            .expect_err("load should fail");

        assert!(err
            .to_string()
            .contains("declare exactly one executable host runtime artifact"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_plugin_manifest_file_rejects_incompatible_api_version() {
        let dir = unique_temp_dir("bad-api-version");
        fs::create_dir_all(dir.join("plugin")).expect("mkdir");
        fs::write(
            dir.join("plugin").join(MANIFEST_FILE_NAME),
            r#"
name = "bad-api-plugin"
version = "1.0.0"
api_version = "2.0"
"#,
        )
        .expect("write manifest");

        let err = load_plugin_manifest_file(dir.join("plugin").join(MANIFEST_FILE_NAME))
            .expect_err("load should fail");
        assert!(err.to_string().contains("declares api_version"));

        let _ = fs::remove_dir_all(&dir);
    }
}

//! Persistent model asset store for built-in model lifecycle management.
//!
//! This module keeps model asset metadata in a local JSON index and supports:
//! - Registering external model files (`add_external`)
//! - Importing/copying model files into managed store (`pull_from_source`)
//! - Listing/querying/removing model records

use crate::error::{LociError, Result};
use crate::model_pull_policy::{
    authorize_model_pull_request, ModelPullPolicyContext, ModelPullPolicyPlugin,
};
use crate::model_pull_verifier::{
    verify_model_pull, ModelPullVerificationContext, ModelPullVerifierPlugin,
};
use crate::timeout_controller::CancellationHandle;
use reqwest::blocking::Client;
use reqwest::header::RANGE;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::fs::OpenOptions;
use std::hash::Hasher;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use xxhash_rust::xxh64::Xxh64;

const STORE_VERSION: u32 = 1;
const INDEX_FILE: &str = "loci_models.json";
const BLOB_DIR: &str = "blobs";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModel {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub source: String,
    pub size_bytes: u64,
    pub checksum_xxh64: String,
    #[serde(default)]
    pub checksum_sha256: Option<String>,
    pub created_at_unix_ms: u64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub managed: bool,
}

#[derive(Debug, Clone)]
pub struct ModelPullOptions {
    pub mirrors: Vec<String>,
    pub expected_sha256: Option<String>,
    pub resume: bool,
}

impl Default for ModelPullOptions {
    fn default() -> Self {
        Self {
            mirrors: Vec::new(),
            expected_sha256: None,
            resume: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPullPhase {
    Resolving,
    Fetching,
    Verifying,
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPullProgress {
    pub phase: ModelPullPhase,
    pub source: Option<String>,
    pub model_id: Option<String>,
    pub destination: Option<PathBuf>,
    pub bytes_transferred: u64,
    pub total_bytes: Option<u64>,
    pub resumed_bytes: u64,
    pub done: bool,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoreFile {
    #[serde(default = "default_store_version")]
    version: u32,
    #[serde(default)]
    models: Vec<StoredModel>,
}

fn default_store_version() -> u32 {
    STORE_VERSION
}

impl Default for StoreFile {
    fn default() -> Self {
        Self {
            version: STORE_VERSION,
            models: Vec::new(),
        }
    }
}

pub struct ModelStore {
    root: PathBuf,
    index_path: PathBuf,
}

impl ModelStore {
    pub fn new<P: AsRef<Path>>(root: P) -> Self {
        let root = root.as_ref().to_path_buf();
        let index_path = root.join(INDEX_FILE);
        Self { root, index_path }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn add_external<P: AsRef<Path>>(
        &self,
        model_path: P,
        id: Option<String>,
        name: Option<String>,
        tags: Vec<String>,
    ) -> Result<StoredModel> {
        self.ensure_layout()?;
        let model_path = model_path.as_ref();
        if !model_path.exists() {
            return Err(LociError::ConfigError(format!(
                "model file does not exist: {}",
                model_path.display()
            )));
        }
        let normalized_path = normalize_path(model_path)?;
        let mut store = self.load_store_file()?;
        let model_id = ensure_unique_id(
            sanitize_id(
                id.as_deref()
                    .unwrap_or_else(|| file_stem_fallback(model_path)),
            ),
            &store.models,
        );
        let model_name = name.unwrap_or_else(|| file_stem_fallback(model_path).to_string());

        if store.models.iter().any(|m| m.path == normalized_path) {
            return Err(LociError::ConfigError(format!(
                "model path already registered: {}",
                normalized_path.display()
            )));
        }

        let size_bytes = file_size(&normalized_path)?;
        let checksum_xxh64 = compute_file_xxh64(&normalized_path)?;
        let checksum_sha256 = Some(compute_file_sha256(&normalized_path)?);
        let model = StoredModel {
            id: model_id,
            name: model_name,
            path: normalized_path.clone(),
            source: normalized_path.to_string_lossy().to_string(),
            size_bytes,
            checksum_xxh64,
            checksum_sha256,
            created_at_unix_ms: unix_ms_now(),
            tags,
            managed: false,
        };
        store.models.push(model.clone());
        self.save_store_file(&store)?;
        Ok(model)
    }

    pub fn pull_from_source(
        &self,
        source: &str,
        id: Option<String>,
        name: Option<String>,
        tags: Vec<String>,
    ) -> Result<StoredModel> {
        self.pull_from_source_with_options(source, id, name, tags, ModelPullOptions::default())
    }

    pub fn pull_from_source_with_options(
        &self,
        source: &str,
        id: Option<String>,
        name: Option<String>,
        tags: Vec<String>,
        options: ModelPullOptions,
    ) -> Result<StoredModel> {
        self.pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation(
            source,
            id,
            name,
            tags,
            options,
            None,
            None,
            None,
            &mut |_| {},
        )
    }

    pub fn pull_from_source_with_options_and_progress<F>(
        &self,
        source: &str,
        id: Option<String>,
        name: Option<String>,
        tags: Vec<String>,
        options: ModelPullOptions,
        on_progress: &mut F,
    ) -> Result<StoredModel>
    where
        F: FnMut(ModelPullProgress),
    {
        self.pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation(
            source,
            id,
            name,
            tags,
            options,
            None,
            None,
            None,
            on_progress,
        )
    }

    pub fn pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation<F>(
        &self,
        source: &str,
        id: Option<String>,
        name: Option<String>,
        tags: Vec<String>,
        options: ModelPullOptions,
        policy: Option<&dyn ModelPullPolicyPlugin>,
        verifier: Option<&dyn ModelPullVerifierPlugin>,
        cancellation: Option<&CancellationHandle>,
        on_progress: &mut F,
    ) -> Result<StoredModel>
    where
        F: FnMut(ModelPullProgress),
    {
        authorize_model_pull_policy(
            policy,
            source,
            &options,
            id.as_deref(),
            name.as_deref(),
            &tags,
        )?;
        self.ensure_layout()?;
        let mut store = self.load_store_file()?;
        let default_name = source_default_name(source, is_url_source(source));
        let base_name = name.unwrap_or(default_name);
        let model_id = ensure_unique_id(
            sanitize_id(id.as_deref().unwrap_or_else(|| base_name.as_str())),
            &store.models,
        );
        let extension = infer_source_extension(source, is_url_source(source));
        let dest_rel = PathBuf::from(BLOB_DIR).join(format!("{model_id}.{extension}"));
        let dest_abs = self.root.join(&dest_rel);
        let candidate_sources = build_candidate_sources(source, &options.mirrors);
        let mut selected_source: Option<String> = None;
        let mut errors = Vec::new();

        for candidate in &candidate_sources {
            check_pull_cancellation(cancellation, Some(&dest_abs))?;
            emit_model_pull_progress(
                on_progress,
                ModelPullProgress {
                    phase: ModelPullPhase::Resolving,
                    source: Some(candidate.clone()),
                    model_id: Some(model_id.clone()),
                    destination: Some(dest_abs.clone()),
                    bytes_transferred: 0,
                    total_bytes: None,
                    resumed_bytes: 0,
                    done: false,
                    message: format!("trying source '{}'", candidate),
                    error: None,
                },
            );

            if let Err(e) = fetch_source_to_path_with_progress(
                candidate,
                &dest_abs,
                options.resume,
                Some(model_id.as_str()),
                cancellation,
                on_progress,
            ) {
                if is_cancelled_error(&e) {
                    return Err(e);
                }
                emit_model_pull_progress(
                    on_progress,
                    ModelPullProgress {
                        phase: ModelPullPhase::Fetching,
                        source: Some(candidate.clone()),
                        model_id: Some(model_id.clone()),
                        destination: Some(dest_abs.clone()),
                        bytes_transferred: 0,
                        total_bytes: None,
                        resumed_bytes: 0,
                        done: false,
                        message: format!("source '{}' failed", candidate),
                        error: Some(e.to_string()),
                    },
                );
                errors.push(format!("{}: {}", candidate, e));
                if dest_abs.exists() {
                    let _ = fs::remove_file(&dest_abs);
                }
                continue;
            }
            selected_source = Some(candidate.clone());
            break;
        }

        let selected_source = match selected_source {
            Some(source) => source,
            None => {
                return Err(LociError::NetworkError(format!(
                    "all model pull sources failed: {}",
                    errors.join(" | ")
                )));
            }
        };

        let normalized_dest = normalize_path(&dest_abs)?;
        check_pull_cancellation(cancellation, Some(&normalized_dest))?;
        let size_bytes = file_size(&normalized_dest)?;
        emit_model_pull_progress(
            on_progress,
            ModelPullProgress {
                phase: ModelPullPhase::Verifying,
                source: Some(selected_source.clone()),
                model_id: Some(model_id.clone()),
                destination: Some(normalized_dest.clone()),
                bytes_transferred: size_bytes,
                total_bytes: Some(size_bytes),
                resumed_bytes: 0,
                done: false,
                message: "verifying checksums".to_string(),
                error: None,
            },
        );
        check_pull_cancellation(cancellation, Some(&normalized_dest))?;
        let checksum_xxh64 = compute_file_xxh64(&normalized_dest)?;
        let checksum_sha256 = compute_file_sha256(&normalized_dest)?;
        if let Some(expected_sha256) = options.expected_sha256.as_deref() {
            let normalized_expected = normalize_sha256(expected_sha256)?;
            if checksum_sha256 != normalized_expected {
                let _ = fs::remove_file(&normalized_dest);
                return Err(LociError::ConfigError(format!(
                    "sha256 mismatch for model '{}': expected {}, got {}",
                    model_id, normalized_expected, checksum_sha256
                )));
            }
        }
        verify_model_pull_verifier(
            verifier,
            source,
            &selected_source,
            &options,
            id.as_deref(),
            Some(base_name.as_str()),
            &tags,
            &normalized_dest,
            size_bytes,
            &checksum_sha256,
        )?;

        let model = StoredModel {
            id: model_id,
            name: base_name,
            path: normalized_dest.clone(),
            source: selected_source.clone(),
            size_bytes,
            checksum_xxh64,
            checksum_sha256: Some(checksum_sha256),
            created_at_unix_ms: unix_ms_now(),
            tags,
            managed: true,
        };
        check_pull_cancellation(cancellation, Some(&model.path))?;
        store.models.push(model.clone());
        self.save_store_file(&store)?;
        emit_model_pull_progress(
            on_progress,
            ModelPullProgress {
                phase: ModelPullPhase::Completed,
                source: Some(selected_source),
                model_id: Some(model.id.clone()),
                destination: Some(model.path.clone()),
                bytes_transferred: model.size_bytes,
                total_bytes: Some(model.size_bytes),
                resumed_bytes: 0,
                done: true,
                message: "model import completed".to_string(),
                error: None,
            },
        );
        Ok(model)
    }

    pub fn pull_from_source_with_options_and_progress_and_cancellation<F>(
        &self,
        source: &str,
        id: Option<String>,
        name: Option<String>,
        tags: Vec<String>,
        options: ModelPullOptions,
        cancellation: Option<&CancellationHandle>,
        on_progress: &mut F,
    ) -> Result<StoredModel>
    where
        F: FnMut(ModelPullProgress),
    {
        self.pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation(
            source,
            id,
            name,
            tags,
            options,
            None,
            None,
            cancellation,
            on_progress,
        )
    }

    pub fn pull_from_source_with_options_and_progress_and_policy_and_cancellation<F>(
        &self,
        source: &str,
        id: Option<String>,
        name: Option<String>,
        tags: Vec<String>,
        options: ModelPullOptions,
        policy: Option<&dyn ModelPullPolicyPlugin>,
        cancellation: Option<&CancellationHandle>,
        on_progress: &mut F,
    ) -> Result<StoredModel>
    where
        F: FnMut(ModelPullProgress),
    {
        self.pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation(
            source,
            id,
            name,
            tags,
            options,
            policy,
            None,
            cancellation,
            on_progress,
        )
    }

    pub fn list(&self) -> Result<Vec<StoredModel>> {
        let mut models = self.load_store_file()?.models;
        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }

    pub fn get(&self, id: &str) -> Result<StoredModel> {
        self.load_store_file()?
            .models
            .into_iter()
            .find(|m| m.id == id)
            .ok_or_else(|| LociError::ModelNotFound)
    }

    pub fn remove(&self, id: &str, delete_file: bool) -> Result<StoredModel> {
        let mut store = self.load_store_file()?;
        let index = store
            .models
            .iter()
            .position(|m| m.id == id)
            .ok_or(LociError::ModelNotFound)?;
        let removed = store.models.remove(index);
        self.save_store_file(&store)?;

        if delete_file && removed.path.exists() {
            fs::remove_file(&removed.path).map_err(|e| {
                LociError::IoError(std::io::Error::new(
                    e.kind(),
                    format!(
                        "failed to remove model file '{}': {}",
                        removed.path.display(),
                        e
                    ),
                ))
            })?;
        }
        Ok(removed)
    }

    fn ensure_layout(&self) -> Result<()> {
        fs::create_dir_all(self.root.join(BLOB_DIR))?;
        Ok(())
    }

    fn load_store_file(&self) -> Result<StoreFile> {
        if !self.index_path.exists() {
            return Ok(StoreFile::default());
        }
        let raw = fs::read_to_string(&self.index_path).map_err(|e| {
            LociError::ConfigError(format!(
                "failed reading model store '{}': {}",
                self.index_path.display(),
                e
            ))
        })?;
        let file: StoreFile = serde_json::from_str(&raw).map_err(|e| {
            LociError::ConfigError(format!(
                "failed parsing model store '{}': {}",
                self.index_path.display(),
                e
            ))
        })?;
        Ok(file)
    }

    fn save_store_file(&self, file: &StoreFile) -> Result<()> {
        fs::create_dir_all(&self.root)?;
        let raw = serde_json::to_string_pretty(file)?;
        fs::write(&self.index_path, raw).map_err(|e| {
            LociError::ConfigError(format!(
                "failed writing model store '{}': {}",
                self.index_path.display(),
                e
            ))
        })?;
        Ok(())
    }
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    Ok(fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?.len())
}

fn compute_file_xxh64(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut buffer = [0u8; 64 * 1024];
    let mut hasher = Xxh64::new(0);

    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.write(&buffer[..bytes]);
    }
    Ok(format!("{:016x}", hasher.finish()))
}

fn compute_file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut buffer = [0u8; 64 * 1024];
    let mut hasher = Sha256::new();

    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn source_default_name(source: &str, is_url: bool) -> String {
    let candidate = if is_url {
        let cleaned = strip_url_query_and_fragment(source);
        cleaned
            .rsplit('/')
            .next()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("model")
            .to_string()
    } else {
        file_stem_fallback(Path::new(source)).to_string()
    };
    let stem = Path::new(&candidate)
        .file_stem()
        .and_then(|x| x.to_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("model");
    stem.to_string()
}

fn infer_source_extension(source: &str, is_url: bool) -> String {
    if is_url {
        let cleaned = strip_url_query_and_fragment(source);
        let filename = cleaned.rsplit('/').next().unwrap_or("model.gguf");
        return Path::new(filename)
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or("gguf")
            .to_string();
    }
    Path::new(source)
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("gguf")
        .to_string()
}

fn strip_url_query_and_fragment(url: &str) -> &str {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    &url[..end]
}

fn is_url_source(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://")
}

fn build_candidate_sources(primary: &str, mirrors: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for mirror in mirrors {
        let trimmed = mirror.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !out.iter().any(|x| x == trimmed) {
            out.push(trimmed.to_string());
        }
    }
    if !out.iter().any(|x| x == primary) {
        out.push(primary.to_string());
    }
    out
}

fn emit_model_pull_progress<F>(on_progress: &mut F, progress: ModelPullProgress)
where
    F: FnMut(ModelPullProgress),
{
    on_progress(progress);
}

fn authorize_model_pull_policy(
    policy: Option<&dyn ModelPullPolicyPlugin>,
    source: &str,
    options: &ModelPullOptions,
    requested_id: Option<&str>,
    requested_name: Option<&str>,
    tags: &[String],
) -> Result<()> {
    let Some(policy) = policy else {
        return Ok(());
    };
    let context = ModelPullPolicyContext::new(
        source,
        options.mirrors.clone(),
        requested_id.map(str::to_string),
        requested_name.map(str::to_string),
        options.expected_sha256.clone(),
        options.resume,
        tags.to_vec(),
    );
    authorize_model_pull_request(policy, &context).map_err(|reason| {
        LociError::ConfigError(format!(
            "model pull denied by policy '{}': {}",
            policy.name(),
            reason
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_model_pull_verifier(
    verifier: Option<&dyn ModelPullVerifierPlugin>,
    requested_source: &str,
    selected_source: &str,
    options: &ModelPullOptions,
    requested_id: Option<&str>,
    requested_name: Option<&str>,
    tags: &[String],
    path: &Path,
    size_bytes: u64,
    checksum_sha256: &str,
) -> Result<()> {
    let Some(verifier) = verifier else {
        return Ok(());
    };
    let context = ModelPullVerificationContext::new(
        requested_source,
        selected_source,
        options.mirrors.clone(),
        requested_id.map(str::to_string),
        requested_name.map(str::to_string),
        options.expected_sha256.clone(),
        tags.to_vec(),
        path.to_path_buf(),
        size_bytes,
        checksum_sha256.to_string(),
    );
    verify_model_pull(verifier, &context).map_err(|reason| {
        let _ = fs::remove_file(path);
        LociError::ConfigError(format!(
            "model pull verifier '{}' rejected asset: {}",
            verifier.name(),
            reason
        ))
    })
}

fn is_cancelled_error(err: &LociError) -> bool {
    matches!(err, LociError::Timeout(message) if message.to_ascii_lowercase().contains("cancel"))
}

fn check_pull_cancellation(
    cancellation: Option<&CancellationHandle>,
    cleanup_path: Option<&Path>,
) -> Result<()> {
    if cancellation
        .map(CancellationHandle::is_cancelled)
        .unwrap_or(false)
    {
        if let Some(path) = cleanup_path {
            let _ = fs::remove_file(path);
        }
        return Err(LociError::Timeout("Operation cancelled".to_string()));
    }
    Ok(())
}

fn fetch_source_to_path_with_progress<F>(
    source: &str,
    destination: &Path,
    resume: bool,
    model_id: Option<&str>,
    cancellation: Option<&CancellationHandle>,
    on_progress: &mut F,
) -> Result<()>
where
    F: FnMut(ModelPullProgress),
{
    if is_url_source(source) {
        download_url_to_file_with_progress(
            source,
            destination,
            resume,
            model_id,
            cancellation,
            on_progress,
        )
    } else {
        let source_path = PathBuf::from(source);
        if !source_path.exists() {
            return Err(LociError::ConfigError(format!(
                "source file does not exist: {}",
                source_path.display()
            )));
        }
        copy_file_with_progress(
            &source_path,
            destination,
            model_id,
            cancellation,
            on_progress,
        )?;
        Ok(())
    }
}

fn normalize_sha256(raw: &str) -> Result<String> {
    let mut normalized = raw.trim().to_ascii_lowercase();
    if let Some(rest) = normalized.strip_prefix("sha256:") {
        normalized = rest.to_string();
    }
    if normalized.len() != 64 || !normalized.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(LociError::ConfigError(format!(
            "invalid sha256 '{}': expected 64 hex characters",
            raw
        )));
    }
    Ok(normalized)
}

#[cfg(test)]
fn download_url_to_file(url: &str, destination: &Path, resume: bool) -> Result<()> {
    download_url_to_file_with_progress(url, destination, resume, None, None, &mut |_| {})
}

fn download_url_to_file_with_progress<F>(
    url: &str,
    destination: &Path,
    resume: bool,
    model_id: Option<&str>,
    cancellation: Option<&CancellationHandle>,
    on_progress: &mut F,
) -> Result<()>
where
    F: FnMut(ModelPullProgress),
{
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| LociError::NetworkError(format!("failed creating HTTP client: {e}")))?;

    let existing_bytes = if resume {
        fs::metadata(destination).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let mut request = client.get(url);
    if existing_bytes > 0 {
        request = request.header(RANGE, format!("bytes={existing_bytes}-"));
    }

    let mut response = request
        .send()
        .map_err(|e| LociError::NetworkError(format!("failed downloading '{url}': {e}")))?;

    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE && existing_bytes > 0 {
        emit_model_pull_progress(
            on_progress,
            ModelPullProgress {
                phase: ModelPullPhase::Fetching,
                source: Some(url.to_string()),
                model_id: model_id.map(|id| id.to_string()),
                destination: Some(destination.to_path_buf()),
                bytes_transferred: existing_bytes,
                total_bytes: Some(existing_bytes),
                resumed_bytes: existing_bytes,
                done: false,
                message: "range request already satisfied".to_string(),
                error: None,
            },
        );
        return Ok(());
    }
    check_pull_cancellation(cancellation, Some(destination))?;
    if existing_bytes > 0
        && response.status() != StatusCode::PARTIAL_CONTENT
        && response.status() == StatusCode::OK
    {
        let _ = fs::remove_file(destination);
    }
    if !response.status().is_success() {
        return Err(LociError::NetworkError(format!(
            "download failed for '{url}': HTTP {}",
            response.status()
        )));
    }

    let append_mode = existing_bytes > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    let mut output = if append_mode {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(destination)
            .map_err(|e| {
                LociError::IoError(std::io::Error::new(
                    e.kind(),
                    format!(
                        "failed opening destination file for resume '{}': {}",
                        destination.display(),
                        e
                    ),
                ))
            })?
    } else {
        fs::File::create(destination).map_err(|e| {
            LociError::IoError(std::io::Error::new(
                e.kind(),
                format!(
                    "failed creating destination file '{}': {}",
                    destination.display(),
                    e
                ),
            ))
        })?
    };

    let total_bytes = if append_mode {
        response
            .content_length()
            .map(|remaining| remaining.saturating_add(existing_bytes))
    } else {
        response.content_length()
    };
    let mut transferred = existing_bytes;
    emit_model_pull_progress(
        on_progress,
        ModelPullProgress {
            phase: ModelPullPhase::Fetching,
            source: Some(url.to_string()),
            model_id: model_id.map(|id| id.to_string()),
            destination: Some(destination.to_path_buf()),
            bytes_transferred: transferred,
            total_bytes,
            resumed_bytes: existing_bytes,
            done: false,
            message: if append_mode {
                "resuming download".to_string()
            } else {
                "starting download".to_string()
            },
            error: None,
        },
    );

    let mut buffer = [0u8; 64 * 1024];
    loop {
        check_pull_cancellation(cancellation, Some(destination))?;
        let bytes = response.read(&mut buffer).map_err(|e| {
            LociError::IoError(std::io::Error::new(
                e.kind(),
                format!("failed reading response body from '{url}': {}", e),
            ))
        })?;
        if bytes == 0 {
            break;
        }
        output.write_all(&buffer[..bytes]).map_err(|e| {
            LociError::IoError(std::io::Error::new(
                e.kind(),
                format!(
                    "failed writing downloaded bytes to '{}': {}",
                    destination.display(),
                    e
                ),
            ))
        })?;
        transferred = transferred.saturating_add(bytes as u64);
        emit_model_pull_progress(
            on_progress,
            ModelPullProgress {
                phase: ModelPullPhase::Fetching,
                source: Some(url.to_string()),
                model_id: model_id.map(|id| id.to_string()),
                destination: Some(destination.to_path_buf()),
                bytes_transferred: transferred,
                total_bytes,
                resumed_bytes: existing_bytes,
                done: false,
                message: "downloading model bytes".to_string(),
                error: None,
            },
        );
    }
    output.flush()?;
    Ok(())
}

fn copy_file_with_progress<F>(
    source: &Path,
    destination: &Path,
    model_id: Option<&str>,
    cancellation: Option<&CancellationHandle>,
    on_progress: &mut F,
) -> Result<()>
where
    F: FnMut(ModelPullProgress),
{
    let total_bytes = fs::metadata(source).map(|metadata| metadata.len()).ok();
    let mut input = fs::File::open(source).map_err(|e| {
        LociError::IoError(std::io::Error::new(
            e.kind(),
            format!("failed opening source model '{}': {}", source.display(), e),
        ))
    })?;
    let mut output = fs::File::create(destination).map_err(|e| {
        LociError::IoError(std::io::Error::new(
            e.kind(),
            format!(
                "failed creating destination file '{}': {}",
                destination.display(),
                e
            ),
        ))
    })?;

    let mut transferred = 0u64;
    emit_model_pull_progress(
        on_progress,
        ModelPullProgress {
            phase: ModelPullPhase::Fetching,
            source: Some(source.display().to_string()),
            model_id: model_id.map(|id| id.to_string()),
            destination: Some(destination.to_path_buf()),
            bytes_transferred: 0,
            total_bytes,
            resumed_bytes: 0,
            done: false,
            message: "starting local model import".to_string(),
            error: None,
        },
    );

    let mut buffer = [0u8; 64 * 1024];
    loop {
        check_pull_cancellation(cancellation, Some(destination))?;
        let bytes = input.read(&mut buffer).map_err(|e| {
            LociError::IoError(std::io::Error::new(
                e.kind(),
                format!("failed reading source model '{}': {}", source.display(), e),
            ))
        })?;
        if bytes == 0 {
            break;
        }
        output.write_all(&buffer[..bytes]).map_err(|e| {
            LociError::IoError(std::io::Error::new(
                e.kind(),
                format!(
                    "failed writing imported model to '{}': {}",
                    destination.display(),
                    e
                ),
            ))
        })?;
        transferred = transferred.saturating_add(bytes as u64);
        emit_model_pull_progress(
            on_progress,
            ModelPullProgress {
                phase: ModelPullPhase::Fetching,
                source: Some(source.display().to_string()),
                model_id: model_id.map(|id| id.to_string()),
                destination: Some(destination.to_path_buf()),
                bytes_transferred: transferred,
                total_bytes,
                resumed_bytes: 0,
                done: false,
                message: "importing local model bytes".to_string(),
                error: None,
            },
        );
    }
    output.flush()?;
    Ok(())
}

fn sanitize_id(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch.to_ascii_lowercase());
        } else if ch.is_ascii_whitespace() {
            out.push('-');
        }
    }
    let cleaned = out.trim_matches(['-', '.']);
    if cleaned.is_empty() {
        "model".to_string()
    } else {
        cleaned.to_string()
    }
}

fn ensure_unique_id(base: String, existing: &[StoredModel]) -> String {
    if !existing.iter().any(|m| m.id == base) {
        return base;
    }
    let mut i = 2u32;
    loop {
        let candidate = format!("{base}-{i}");
        if !existing.iter().any(|m| m.id == candidate) {
            return candidate;
        }
        i += 1;
    }
}

fn file_stem_fallback(path: &Path) -> &str {
    path.file_stem()
        .and_then(|x| x.to_str())
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("model")
}

fn unix_ms_now() -> u64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_pull_policy::LocalOnlyModelPullPolicy;
    use crate::timeout_controller::TimeoutContext;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-model-store-{nonce}"))
    }

    #[test]
    fn add_external_and_get() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let model_file = root.join("tiny.gguf");
        fs::write(&model_file, b"tiny model bytes").unwrap();

        let store = ModelStore::new(&root);
        let added = store
            .add_external(&model_file, None, None, vec!["base".to_string()])
            .unwrap();
        assert!(!added.id.is_empty());
        assert!(!added.managed);
        assert_eq!(added.tags, vec!["base".to_string()]);

        let got = store.get(&added.id).unwrap();
        assert_eq!(got.id, added.id);
        assert_eq!(got.size_bytes, added.size_bytes);
        assert!(got.checksum_sha256.is_some());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pull_local_file_into_store() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.gguf");
        fs::write(&source, b"source model bytes").unwrap();

        let store_root = root.join("store");
        let store = ModelStore::new(&store_root);
        let pulled = store
            .pull_from_source(&source.to_string_lossy(), None, None, vec![])
            .unwrap();
        assert!(pulled.managed);
        assert!(pulled.path.exists());
        assert!(pulled.checksum_sha256.is_some());
        assert!(pulled
            .path
            .to_string_lossy()
            .contains(&format!("{}blobs", std::path::MAIN_SEPARATOR)));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn remove_with_delete_file() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("remove.gguf");
        fs::write(&source, b"remove model bytes").unwrap();

        let store = ModelStore::new(root.join("store"));
        let pulled = store
            .pull_from_source(
                &source.to_string_lossy(),
                Some("deleteme".to_string()),
                None,
                vec![],
            )
            .unwrap();
        assert!(pulled.path.exists());

        let removed = store.remove("deleteme", true).unwrap();
        assert_eq!(removed.id, "deleteme");
        assert!(!removed.path.exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pull_http_file_into_store() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let body = b"remote model bytes";
        let url = spawn_one_shot_http(body);

        let store = ModelStore::new(root.join("store"));
        let pulled = store
            .pull_from_source(&url, Some("remote-model".to_string()), None, vec![])
            .unwrap();

        assert!(pulled.managed);
        assert_eq!(pulled.id, "remote-model");
        assert_eq!(pulled.source, url);
        assert_eq!(fs::read(&pulled.path).unwrap(), body);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pull_reports_progress_for_local_file() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("progress.gguf");
        fs::write(&source, b"progress bytes for local import").unwrap();

        let store = ModelStore::new(root.join("store"));
        let mut events = Vec::new();
        let pulled = store
            .pull_from_source_with_options_and_progress(
                &source.to_string_lossy(),
                Some("progress-local".to_string()),
                None,
                vec![],
                ModelPullOptions::default(),
                &mut |event| events.push(event),
            )
            .unwrap();

        assert_eq!(pulled.id, "progress-local");
        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|event| event.phase == ModelPullPhase::Fetching));
        assert!(events
            .iter()
            .any(|event| event.phase == ModelPullPhase::Verifying));
        assert_eq!(events.last().map(|event| event.done), Some(true));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pull_can_be_cancelled_before_copy_starts() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("cancel.gguf");
        fs::write(&source, b"cancel bytes for local import").unwrap();

        let store = ModelStore::new(root.join("store"));
        let cancellation = TimeoutContext::disabled().cancellation_handle();
        cancellation.cancel();

        let result = store.pull_from_source_with_options_and_progress_and_cancellation(
            &source.to_string_lossy(),
            Some("cancel-local".to_string()),
            None,
            vec![],
            ModelPullOptions::default(),
            Some(&cancellation),
            &mut |_| {},
        );

        assert!(
            matches!(result, Err(LociError::Timeout(message)) if message.contains("cancelled"))
        );
        assert!(store.list().unwrap().is_empty());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pull_policy_can_deny_remote_source_before_fetch() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let store = ModelStore::new(root.join("store"));
        let policy = LocalOnlyModelPullPolicy;

        let result = store.pull_from_source_with_options_and_progress_and_policy_and_cancellation(
            "https://example.com/model.gguf",
            Some("remote-blocked".to_string()),
            None,
            vec!["managed".to_string()],
            ModelPullOptions::default(),
            Some(&policy),
            None,
            &mut |_| {},
        );

        assert!(
            matches!(result, Err(LociError::ConfigError(message)) if message.contains("model pull denied by policy"))
        );
        assert!(store.list().unwrap_or_default().is_empty());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pull_uses_mirror_priority() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();

        let primary = root.join("primary.gguf");
        fs::write(&primary, b"primary").unwrap();
        let mirror_good = root.join("mirror.gguf");
        fs::write(&mirror_good, b"mirror-bytes").unwrap();
        let mirror_missing = root.join("missing.gguf");

        let store = ModelStore::new(root.join("store"));
        let pulled = store
            .pull_from_source_with_options(
                &primary.to_string_lossy(),
                Some("mirror-first".to_string()),
                None,
                vec![],
                ModelPullOptions {
                    mirrors: vec![
                        mirror_missing.to_string_lossy().to_string(),
                        mirror_good.to_string_lossy().to_string(),
                    ],
                    expected_sha256: None,
                    resume: true,
                },
            )
            .unwrap();

        assert_eq!(pulled.id, "mirror-first");
        assert_eq!(pulled.source, mirror_good.to_string_lossy().to_string());
        assert_eq!(fs::read(&pulled.path).unwrap(), b"mirror-bytes");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pull_with_sha256_mismatch_fails() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let source = root.join("sha.gguf");
        fs::write(&source, b"checksum-data").unwrap();
        let store = ModelStore::new(root.join("store"));

        let result = store.pull_from_source_with_options(
            &source.to_string_lossy(),
            Some("sha-test".to_string()),
            None,
            vec![],
            ModelPullOptions {
                mirrors: Vec::new(),
                expected_sha256: Some(
                    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string(),
                ),
                resume: true,
            },
        );
        assert!(result.is_err());
        assert!(store.list().unwrap().is_empty());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn download_url_resume_with_range_request() {
        let root = temp_dir();
        fs::create_dir_all(&root).unwrap();
        let body = b"resume-enabled-body";
        let url = spawn_range_http(body);
        let destination = root.join("resume.gguf");

        fs::write(&destination, &body[..6]).unwrap();
        download_url_to_file(&url, &destination, true).unwrap();

        assert_eq!(fs::read(&destination).unwrap(), body);
        let _ = fs::remove_dir_all(root);
    }

    fn spawn_one_shot_http(body: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request_buf = [0u8; 1024];
                let _ = stream.read(&mut request_buf);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.write_all(body);
                let _ = stream.flush();
            }
        });

        format!("http://{addr}/model.gguf")
    }

    fn spawn_range_http(body: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request_buf = [0u8; 2048];
                let bytes = stream.read(&mut request_buf).unwrap_or(0);
                let request = String::from_utf8_lossy(&request_buf[..bytes]);
                let mut range_start = 0usize;
                for line in request.lines() {
                    if let Some(value) = line.strip_prefix("Range: bytes=") {
                        if let Some((start, _)) = value.split_once('-') {
                            range_start = start.parse::<usize>().unwrap_or(0);
                        }
                    }
                }

                if range_start > 0 && range_start < body.len() {
                    let slice = &body[range_start..];
                    let response = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                        slice.len(),
                        range_start,
                        body.len() - 1,
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.write_all(slice);
                    let _ = stream.flush();
                } else {
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.write_all(body);
                    let _ = stream.flush();
                }
            }
        });

        format!("http://{addr}/model.gguf")
    }
}

//! Policy builder for tiered weight and KV offload decisions.

use loci_protocol::{
    AcceleratorKind, HardwareTopology, ModelAssetInventory, ModelDescriptor, ModelShardRole,
    ThermalState, TieredOffloadConfig, TieredOffloadPlan, TieredOffloadPolicy,
    TieredOffloadProfile, TieredPlacementPercentages,
};
use memmap2::{Mmap, MmapOptions};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};

const PAGE_SIZE_BYTES: u64 = 4096;

/// Carries backend-agnostic host signals that improve spill planning quality.
#[derive(Debug, Clone, PartialEq)]
pub struct HostTieringHints {
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub free_disk_bytes: Option<u64>,
    pub disk_read_mbps: f64,
    pub disk_write_mbps: f64,
}

/// Represents one runtime-managed spill segment backed by the shared spill file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpillSegment {
    pub tensor: SpillTensorKind,
    pub offset_bytes: u64,
    pub length_bytes: u64,
}

/// Identifies which tensor class owns a segment within the spill file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpillTensorKind {
    Weights,
    KvCache,
    Activations,
}

/// Reports the runtime state of one spill session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TieredSessionSnapshot {
    pub session_key: String,
    pub model_name: String,
    pub spill_path: PathBuf,
    pub mapped_bytes: u64,
    pub prefetched_bytes: u64,
    pub scheduled_prefetch_requests: usize,
    pub completed_prefetch_requests: usize,
    pub segments: Vec<SpillSegment>,
}

/// Reports aggregated spill runtime state across all active sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TieredRuntimeSnapshot {
    pub root_dir: PathBuf,
    pub active_sessions: Vec<TieredSessionSnapshot>,
    pub total_spill_bytes: u64,
    pub total_prefetched_bytes: u64,
}

/// Represents failures raised by the disk-backed spill runtime.
#[derive(Debug, thiserror::Error)]
pub enum TieredOffloadError {
    #[error("spill runtime session `{session_key}` was not found")]
    SessionNotFound { session_key: String },
    #[error("spill runtime state is poisoned")]
    Poisoned,
    #[error("spill runtime io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Converts runtime configuration and topology constraints into spill policies.
pub struct TieredOffloadManager {
    config: TieredOffloadConfig,
}

impl TieredOffloadManager {
    /// Creates a manager from the supplied tiered-offload configuration.
    pub fn new(config: TieredOffloadConfig) -> Self {
        Self { config }
    }

    /// Produces a spill plan when the model does not fit comfortably in active memory.
    pub fn plan(
        &self,
        model: &ModelDescriptor,
        topology: &HardwareTopology,
        host: Option<&HostTieringHints>,
    ) -> Option<TieredOffloadPlan> {
        let model_bytes = model.memory_bytes?;
        let topology_active_memory_bytes = topology
            .devices
            .iter()
            .filter(|device| device.kind != loci_protocol::AcceleratorKind::Disk)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>()
            .saturating_mul(3)
            / 4;
        let host_active_memory_bytes = host
            .map(|host| host.available_memory_bytes.saturating_mul(3) / 4)
            .unwrap_or(u64::MAX);
        let active_memory_bytes = topology_active_memory_bytes.min(host_active_memory_bytes);
        let gpu_memory_bytes = topology
            .devices
            .iter()
            .filter(|device| device.kind == AcceleratorKind::Gpu)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>();
        let cpu_memory_bytes = topology
            .devices
            .iter()
            .filter(|device| device.kind == AcceleratorKind::Cpu)
            .filter_map(|device| device.memory_bytes)
            .sum::<u64>();
        let disk_device = topology
            .devices
            .iter()
            .find(|device| device.kind == loci_protocol::AcceleratorKind::Disk)?;

        let threshold = self
            .config
            .spill_threshold_bytes
            .unwrap_or(active_memory_bytes);
        if model_bytes <= threshold || model_bytes <= active_memory_bytes {
            return None;
        }

        let spill_bytes = model_bytes.saturating_sub(active_memory_bytes);
        let spill_budget_bytes = spill_bytes.min(capped_disk_budget(
            self.config.max_disk_bytes,
            host.and_then(|host| host.free_disk_bytes),
        )?);
        let prefetch_window_bytes =
            self.dynamic_prefetch_window(spill_budget_bytes, model_bytes, topology, host);
        let profile = self.select_profile(
            model_bytes,
            spill_budget_bytes,
            gpu_memory_bytes,
            cpu_memory_bytes,
            topology,
        );
        let policy = self.build_policy(
            profile,
            model_bytes,
            spill_budget_bytes,
            gpu_memory_bytes,
            cpu_memory_bytes,
        );

        Some(TieredOffloadPlan {
            spill_bytes: spill_budget_bytes,
            prefetch_window_bytes,
            target_device: disk_device.id.clone(),
            profile,
            policy,
        })
    }

    /// Widens or narrows the prefetch window as spill pressure changes.
    fn dynamic_prefetch_window(
        &self,
        spill_bytes: u64,
        model_bytes: u64,
        topology: &HardwareTopology,
        host: Option<&HostTieringHints>,
    ) -> u64 {
        let configured = self.config.prefetch_window_bytes.unwrap_or(0);
        if configured == 0 || model_bytes == 0 {
            return configured;
        }

        // Heavier spill pressure benefits from a wider prefetch window so the
        // planner can hide more storage latency behind compute.
        let spill_ratio_percent = spill_bytes.saturating_mul(100) / model_bytes;
        let scaled = if spill_ratio_percent >= 50 {
            configured.saturating_mul(2)
        } else if spill_ratio_percent >= 25 {
            configured.saturating_mul(3) / 2
        } else {
            configured
        };
        let io_scaled = match host {
            Some(host) if host.disk_read_mbps > 0.0 && host.disk_read_mbps < 1500.0 => {
                scaled.saturating_mul(5) / 4
            }
            Some(host) if host.disk_read_mbps >= 3000.0 => scaled.saturating_mul(3) / 4,
            _ => scaled,
        };

        if has_power_pressure(topology) {
            io_scaled.saturating_mul(3) / 4
        } else {
            io_scaled
        }
    }

    /// Chooses the effective offload profile when `auto` mode is enabled.
    fn select_profile(
        &self,
        model_bytes: u64,
        spill_bytes: u64,
        gpu_memory_bytes: u64,
        cpu_memory_bytes: u64,
        topology: &HardwareTopology,
    ) -> TieredOffloadProfile {
        if self.config.profile != TieredOffloadProfile::Auto {
            return self.config.profile;
        }

        if model_bytes == 0 {
            return TieredOffloadProfile::Balanced;
        }

        let spill_ratio_percent = spill_bytes.saturating_mul(100) / model_bytes;
        if has_power_pressure(topology) && spill_ratio_percent >= 20 {
            TieredOffloadProfile::DiskHeavy
        } else if gpu_memory_bytes == 0 || spill_ratio_percent >= 45 {
            TieredOffloadProfile::DiskHeavy
        } else if (cpu_memory_bytes == 0 || spill_ratio_percent <= 15) && !battery_limited(topology)
        {
            TieredOffloadProfile::GpuResident
        } else {
            TieredOffloadProfile::Balanced
        }
    }

    /// Converts a profile template into a capacity-aware placement policy.
    fn build_policy(
        &self,
        profile: TieredOffloadProfile,
        model_bytes: u64,
        spill_bytes: u64,
        gpu_memory_bytes: u64,
        cpu_memory_bytes: u64,
    ) -> TieredOffloadPolicy {
        let template = policy_template(profile);
        let weights = fit_percentages(
            model_bytes,
            template.weights,
            gpu_memory_bytes,
            cpu_memory_bytes,
        );
        let kv_cache = fit_percentages(
            model_bytes / 6,
            template.kv_cache,
            gpu_memory_bytes.saturating_mul(3) / 4,
            cpu_memory_bytes / 2,
        );
        let activations = fit_percentages(
            model_bytes / 12,
            template.activations,
            gpu_memory_bytes / 2,
            cpu_memory_bytes / 2,
        );

        TieredOffloadPolicy {
            weights,
            kv_cache,
            activations,
            cpu_cache_compute: template.cpu_cache_compute
                || (kv_cache.disk_percent > 0 && kv_cache.cpu_percent >= kv_cache.gpu_percent),
            compress_weights: template.compress_weights
                || weights.disk_percent > 0
                || weights.cpu_percent >= 50,
            compress_kv_cache: template.compress_kv_cache
                || spill_bytes > 0
                || kv_cache.cpu_percent >= 50,
        }
    }
}

/// Owns mmap-backed spill artifacts and background prefetch workers.
pub struct TieredOffloadRuntime {
    root_dir: PathBuf,
    sessions: Mutex<HashMap<String, SpillSession>>,
}

struct SpillSession {
    session_key: String,
    model_name: String,
    spill_path: PathBuf,
    mapped_bytes: u64,
    segments: Vec<SpillSegment>,
    stats: Arc<Mutex<PrefetchStats>>,
    sender: Option<Sender<Vec<PrefetchRange>>>,
    worker: Option<JoinHandle<()>>,
}

#[derive(Debug, Default)]
struct PrefetchStats {
    prefetched_bytes: u64,
    scheduled_prefetch_requests: usize,
    completed_prefetch_requests: usize,
}

#[derive(Debug, Clone, Copy)]
struct PrefetchRange {
    offset_bytes: u64,
    length_bytes: u64,
}

impl TieredOffloadRuntime {
    /// Creates a spill runtime rooted at the supplied directory.
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the default root used for spill artifacts on the current host.
    pub fn default_root_dir() -> PathBuf {
        std::env::temp_dir().join("loci-tiered-offload")
    }

    /// Creates or reuses a spill session and schedules the initial warmup prefetch.
    pub fn prepare_session(
        &self,
        session_key: &str,
        model: &ModelDescriptor,
        assets: &ModelAssetInventory,
        plan: &TieredOffloadPlan,
    ) -> Result<TieredSessionSnapshot, TieredOffloadError> {
        if let Some(snapshot) = self.session_snapshot(session_key)? {
            return Ok(snapshot);
        }

        fs::create_dir_all(&self.root_dir)?;
        let spill_path = self.root_dir.join(format!(
            "{}-{}.spill",
            sanitize_name(&model.name),
            sanitize_name(session_key)
        ));
        let mapped_bytes = plan.spill_bytes.max(PAGE_SIZE_BYTES);
        materialize_spill_file(&spill_path, assets, mapped_bytes)?;
        let mmap = open_mmap(&spill_path)?;
        let segments = allocate_spill_segments(plan, mapped_bytes);
        let stats = Arc::new(Mutex::new(PrefetchStats::default()));
        let (sender, worker) = spawn_prefetch_worker(Arc::new(mmap), Arc::clone(&stats));

        let session = SpillSession {
            session_key: session_key.to_string(),
            model_name: model.name.clone(),
            spill_path,
            mapped_bytes,
            segments,
            stats,
            sender: Some(sender),
            worker: Some(worker),
        };

        let warmup_ranges = session.initial_prefetch_ranges(plan.prefetch_window_bytes);
        let snapshot = session.snapshot()?;

        self.sessions()?.insert(session_key.to_string(), session);

        if !warmup_ranges.is_empty() {
            self.schedule_prefetch(session_key, warmup_ranges)?;
        }

        Ok(snapshot)
    }

    /// Schedules an explicit prefetch for the supplied session ranges.
    pub fn schedule_prefetch(
        &self,
        session_key: &str,
        ranges: Vec<(u64, u64)>,
    ) -> Result<(), TieredOffloadError> {
        let mut sessions = self.sessions()?;
        let session =
            sessions
                .get_mut(session_key)
                .ok_or_else(|| TieredOffloadError::SessionNotFound {
                    session_key: session_key.to_string(),
                })?;
        session.schedule_prefetch(ranges)
    }

    /// Returns the runtime snapshot for a single session if it exists.
    pub fn session_snapshot(
        &self,
        session_key: &str,
    ) -> Result<Option<TieredSessionSnapshot>, TieredOffloadError> {
        let sessions = self.sessions()?;
        sessions
            .get(session_key)
            .map(SpillSession::snapshot)
            .transpose()
    }

    /// Returns an aggregated snapshot across all active spill sessions.
    pub fn snapshot(&self) -> Result<TieredRuntimeSnapshot, TieredOffloadError> {
        let sessions = self.sessions()?;
        let mut active_sessions = Vec::with_capacity(sessions.len());
        let mut total_spill_bytes = 0u64;
        let mut total_prefetched_bytes = 0u64;

        for session in sessions.values() {
            let snapshot = session.snapshot()?;
            total_spill_bytes = total_spill_bytes.saturating_add(snapshot.mapped_bytes);
            total_prefetched_bytes =
                total_prefetched_bytes.saturating_add(snapshot.prefetched_bytes);
            active_sessions.push(snapshot);
        }

        Ok(TieredRuntimeSnapshot {
            root_dir: self.root_dir.clone(),
            active_sessions,
            total_spill_bytes,
            total_prefetched_bytes,
        })
    }

    /// Removes a spill session and drops its worker, mmap, and file artifact.
    pub fn evict_session(&self, session_key: &str) -> Result<bool, TieredOffloadError> {
        Ok(self.sessions()?.remove(session_key).is_some())
    }

    fn sessions(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<String, SpillSession>>, TieredOffloadError> {
        self.sessions
            .lock()
            .map_err(|_| TieredOffloadError::Poisoned)
    }
}

impl Default for TieredOffloadRuntime {
    fn default() -> Self {
        Self::new(Self::default_root_dir())
    }
}

impl SpillSession {
    /// Builds a user-facing snapshot of the spill artifact and worker state.
    fn snapshot(&self) -> Result<TieredSessionSnapshot, TieredOffloadError> {
        let stats = self
            .stats
            .lock()
            .map_err(|_| TieredOffloadError::Poisoned)?;
        Ok(TieredSessionSnapshot {
            session_key: self.session_key.clone(),
            model_name: self.model_name.clone(),
            spill_path: self.spill_path.clone(),
            mapped_bytes: self.mapped_bytes,
            prefetched_bytes: stats.prefetched_bytes,
            scheduled_prefetch_requests: stats.scheduled_prefetch_requests,
            completed_prefetch_requests: stats.completed_prefetch_requests,
            segments: self.segments.clone(),
        })
    }

    /// Converts byte ranges into prefetch commands for the worker thread.
    fn schedule_prefetch(&mut self, ranges: Vec<(u64, u64)>) -> Result<(), TieredOffloadError> {
        let normalized: Vec<PrefetchRange> = ranges
            .into_iter()
            .filter(|(_, length_bytes)| *length_bytes > 0)
            .map(|(offset_bytes, length_bytes)| PrefetchRange {
                offset_bytes: offset_bytes.min(self.mapped_bytes),
                length_bytes: length_bytes.min(self.mapped_bytes.saturating_sub(offset_bytes)),
            })
            .filter(|range| range.length_bytes > 0)
            .collect();

        if normalized.is_empty() {
            return Ok(());
        }

        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| TieredOffloadError::Poisoned)?;
            stats.scheduled_prefetch_requests += normalized.len();
        }

        if let Some(sender) = &self.sender {
            sender.send(normalized).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "spill prefetch worker is unavailable",
                )
            })?;
        }

        Ok(())
    }

    /// Builds the initial warmup ranges capped by the requested prefetch window.
    fn initial_prefetch_ranges(&self, prefetch_window_bytes: u64) -> Vec<(u64, u64)> {
        if prefetch_window_bytes == 0 {
            return Vec::new();
        }

        let mut remaining = prefetch_window_bytes;
        let mut ranges = Vec::new();
        for segment in &self.segments {
            if remaining == 0 {
                break;
            }
            let length_bytes = segment.length_bytes.min(remaining);
            if length_bytes > 0 {
                ranges.push((segment.offset_bytes, length_bytes));
                remaining = remaining.saturating_sub(length_bytes);
            }
        }
        ranges
    }
}

impl Drop for SpillSession {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let _ = fs::remove_file(&self.spill_path);
    }
}

fn materialize_spill_file(
    spill_path: &Path,
    assets: &ModelAssetInventory,
    mapped_bytes: u64,
) -> Result<(), TieredOffloadError> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .truncate(true)
        .open(spill_path)?;
    file.set_len(mapped_bytes)?;

    let source_paths = source_model_blobs(assets);
    if !source_paths.is_empty() {
        copy_source_prefixes(&source_paths, &mut file, mapped_bytes)?;
    }

    file.flush()?;
    Ok(())
}

fn source_model_blobs(assets: &ModelAssetInventory) -> Vec<PathBuf> {
    let mut candidates = assets
        .shards
        .iter()
        .filter(|shard| shard.mmap_candidate)
        .filter(|shard| matches!(shard.role, ModelShardRole::Weights | ModelShardRole::Graph))
        .map(|shard| assets.root.join(&shard.path))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();

    if candidates.is_empty() && assets.root.is_file() {
        candidates.push(assets.root.clone());
    }

    if candidates.is_empty() && assets.root.is_dir() {
        for fallback in ["model.bin", "openvino_model.bin", "weights.bin"] {
            let candidate = assets.root.join(fallback);
            if candidate.is_file() {
                candidates.push(candidate);
                break;
            }
        }
    }

    candidates
}

fn copy_source_prefixes(
    source_paths: &[PathBuf],
    spill_file: &mut File,
    mapped_bytes: u64,
) -> Result<(), TieredOffloadError> {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut remaining = mapped_bytes;

    spill_file.seek(SeekFrom::Start(0))?;
    for source_path in source_paths {
        if remaining == 0 {
            break;
        }

        let mut source = File::open(source_path)?;
        let mut source_remaining = remaining.min(source.metadata()?.len());
        while source_remaining > 0 && remaining > 0 {
            let next = source_remaining.min(buffer.len() as u64) as usize;
            let read = source.read(&mut buffer[..next])?;
            if read == 0 {
                break;
            }
            spill_file.write_all(&buffer[..read])?;
            let read_u64 = read as u64;
            source_remaining = source_remaining.saturating_sub(read_u64);
            remaining = remaining.saturating_sub(read_u64);
        }
    }

    Ok(())
}

fn open_mmap(path: &Path) -> Result<Mmap, TieredOffloadError> {
    let file = File::open(path)?;
    // The spill file length is set before mapping and never mutated afterward.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(mmap)
}

fn spawn_prefetch_worker(
    mmap: Arc<Mmap>,
    stats: Arc<Mutex<PrefetchStats>>,
) -> (Sender<Vec<PrefetchRange>>, JoinHandle<()>) {
    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || prefetch_worker_loop(mmap, stats, receiver));
    (sender, worker)
}

fn prefetch_worker_loop(
    mmap: Arc<Mmap>,
    stats: Arc<Mutex<PrefetchStats>>,
    receiver: Receiver<Vec<PrefetchRange>>,
) {
    while let Ok(batch) = receiver.recv() {
        for range in batch {
            let prefetched = prefetch_range(&mmap, range);
            if let Ok(mut stats) = stats.lock() {
                stats.prefetched_bytes = stats.prefetched_bytes.saturating_add(prefetched);
                stats.completed_prefetch_requests += 1;
            } else {
                return;
            }
        }
    }
}

fn prefetch_range(mmap: &Mmap, range: PrefetchRange) -> u64 {
    let start = range.offset_bytes.min(mmap.len() as u64) as usize;
    let end = range
        .offset_bytes
        .saturating_add(range.length_bytes)
        .min(mmap.len() as u64) as usize;
    if start >= end {
        return 0;
    }

    // Touch one byte per page to force the operating system to fault in the
    // mapped region without copying the full buffer into a separate allocation.
    let mut accumulator = 0u8;
    let mut cursor = start;
    while cursor < end {
        accumulator ^= mmap[cursor];
        cursor = cursor.saturating_add(PAGE_SIZE_BYTES as usize);
    }
    if end > start {
        accumulator ^= mmap[end - 1];
    }

    std::hint::black_box(accumulator);
    (end - start) as u64
}

fn allocate_spill_segments(plan: &TieredOffloadPlan, mapped_bytes: u64) -> Vec<SpillSegment> {
    let disk_scores = [
        (
            SpillTensorKind::Weights,
            plan.policy.weights.disk_percent as u64,
        ),
        (
            SpillTensorKind::KvCache,
            plan.policy.kv_cache.disk_percent as u64,
        ),
        (
            SpillTensorKind::Activations,
            plan.policy.activations.disk_percent as u64,
        ),
    ];
    let total_score = disk_scores
        .iter()
        .map(|(_, score)| *score)
        .sum::<u64>()
        .max(1);

    let mut segments = Vec::new();
    let mut offset_bytes = 0u64;
    for (index, (tensor, score)) in disk_scores.iter().enumerate() {
        let remaining_bytes = mapped_bytes.saturating_sub(offset_bytes);
        if remaining_bytes == 0 {
            break;
        }

        let length_bytes = if index == disk_scores.len() - 1 {
            remaining_bytes
        } else {
            mapped_bytes
                .saturating_mul(*score)
                .saturating_div(total_score)
                .min(remaining_bytes)
        };

        segments.push(SpillSegment {
            tensor: *tensor,
            offset_bytes,
            length_bytes,
        });
        offset_bytes = offset_bytes.saturating_add(length_bytes);
    }

    segments
}

fn sanitize_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn capped_disk_budget(configured_max: Option<u64>, free_disk_bytes: Option<u64>) -> Option<u64> {
    let configured = configured_max.unwrap_or(u64::MAX);
    let host_budget = free_disk_bytes
        .map(|bytes| bytes.saturating_mul(85) / 100)
        .unwrap_or(u64::MAX);
    let budget = configured.min(host_budget);
    (budget > 0).then_some(budget)
}

/// Returns whether thermal, battery, or power-budget pressure is present.
fn has_power_pressure(topology: &HardwareTopology) -> bool {
    matches!(
        topology.power.thermal_state,
        ThermalState::Warm | ThermalState::Hot | ThermalState::Critical
    ) || battery_limited(topology)
        || topology
            .power
            .power_budget_watts
            .map(|budget| budget <= 18)
            .unwrap_or(false)
}

/// Returns whether the device is operating in a low-battery regime.
fn battery_limited(topology: &HardwareTopology) -> bool {
    topology
        .power
        .battery_percent
        .map(|value| value < 25)
        .unwrap_or(false)
}

/// Static placement template used before capacity fitting is applied.
#[derive(Clone, Copy)]
struct PolicyTemplate {
    weights: TieredPlacementPercentages,
    kv_cache: TieredPlacementPercentages,
    activations: TieredPlacementPercentages,
    cpu_cache_compute: bool,
    compress_weights: bool,
    compress_kv_cache: bool,
}

/// Returns the baseline placement template associated with a profile.
fn policy_template(profile: TieredOffloadProfile) -> PolicyTemplate {
    match profile {
        TieredOffloadProfile::Auto | TieredOffloadProfile::Balanced => PolicyTemplate {
            weights: TieredPlacementPercentages {
                gpu_percent: 40,
                cpu_percent: 40,
                disk_percent: 20,
            },
            kv_cache: TieredPlacementPercentages {
                gpu_percent: 30,
                cpu_percent: 50,
                disk_percent: 20,
            },
            activations: TieredPlacementPercentages {
                gpu_percent: 50,
                cpu_percent: 50,
                disk_percent: 0,
            },
            cpu_cache_compute: false,
            compress_weights: false,
            compress_kv_cache: false,
        },
        TieredOffloadProfile::GpuResident => PolicyTemplate {
            weights: TieredPlacementPercentages {
                gpu_percent: 80,
                cpu_percent: 20,
                disk_percent: 0,
            },
            kv_cache: TieredPlacementPercentages {
                gpu_percent: 90,
                cpu_percent: 10,
                disk_percent: 0,
            },
            activations: TieredPlacementPercentages {
                gpu_percent: 100,
                cpu_percent: 0,
                disk_percent: 0,
            },
            cpu_cache_compute: false,
            compress_weights: false,
            compress_kv_cache: false,
        },
        TieredOffloadProfile::DiskHeavy => PolicyTemplate {
            weights: TieredPlacementPercentages {
                gpu_percent: 15,
                cpu_percent: 35,
                disk_percent: 50,
            },
            kv_cache: TieredPlacementPercentages {
                gpu_percent: 10,
                cpu_percent: 40,
                disk_percent: 50,
            },
            activations: TieredPlacementPercentages {
                gpu_percent: 20,
                cpu_percent: 60,
                disk_percent: 20,
            },
            cpu_cache_compute: true,
            compress_weights: true,
            compress_kv_cache: true,
        },
    }
}

/// Fits a preferred placement split to the available GPU and CPU capacities.
fn fit_percentages(
    demand_bytes: u64,
    preferred: TieredPlacementPercentages,
    gpu_capacity_bytes: u64,
    cpu_capacity_bytes: u64,
) -> TieredPlacementPercentages {
    if demand_bytes == 0 {
        return TieredPlacementPercentages {
            gpu_percent: 100,
            cpu_percent: 0,
            disk_percent: 0,
        };
    }

    let gpu_target = demand_bytes.saturating_mul(preferred.gpu_percent as u64) / 100;
    let cpu_target = demand_bytes.saturating_mul(preferred.cpu_percent as u64) / 100;

    let gpu_bytes = gpu_target.min(gpu_capacity_bytes);
    let remaining_after_gpu = demand_bytes.saturating_sub(gpu_bytes);
    let cpu_bytes = cpu_target
        .saturating_add(gpu_target.saturating_sub(gpu_bytes))
        .min(cpu_capacity_bytes)
        .min(remaining_after_gpu);
    let gpu_percent = ((gpu_bytes * 100) / demand_bytes) as u8;
    let cpu_percent = ((cpu_bytes * 100) / demand_bytes) as u8;
    let disk_percent = 100u8.saturating_sub(gpu_percent.saturating_add(cpu_percent));

    TieredPlacementPercentages {
        gpu_percent,
        cpu_percent,
        disk_percent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loci_protocol::{
        AcceleratorKind, DeviceDescriptor, ModelAssetInventory, ModelAssetLayout, ModelFormat,
        ModelShardDescriptor, ModelShardRole, PowerState, ThermalState,
    };
    use std::{
        path::PathBuf,
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    fn topology() -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    compute_units: Some(8),
                    power_watts: Some(15.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "disk".to_string(),
                    kind: AcceleratorKind::Disk,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(256 * 1024 * 1024 * 1024),
                    compute_units: None,
                    power_watts: None,
                },
            ],
            power: PowerState {
                battery_powered: false,
                battery_percent: None,
                thermal_state: ThermalState::Nominal,
                power_budget_watts: Some(25),
            },
        }
    }

    fn topology_with_gpu() -> HardwareTopology {
        HardwareTopology {
            devices: vec![
                DeviceDescriptor {
                    id: "gpu:0".to_string(),
                    name: "gpu".to_string(),
                    kind: AcceleratorKind::Gpu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(6 * 1024 * 1024 * 1024),
                    compute_units: Some(32),
                    power_watts: Some(35.0),
                },
                DeviceDescriptor {
                    id: "cpu:0".to_string(),
                    name: "cpu".to_string(),
                    kind: AcceleratorKind::Cpu,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    compute_units: Some(8),
                    power_watts: Some(15.0),
                },
                DeviceDescriptor {
                    id: "disk:0".to_string(),
                    name: "disk".to_string(),
                    kind: AcceleratorKind::Disk,
                    platform: Some(std::env::consts::OS.to_string()),
                    memory_bytes: Some(256 * 1024 * 1024 * 1024),
                    compute_units: None,
                    power_watts: None,
                },
            ],
            power: PowerState {
                battery_powered: false,
                battery_percent: None,
                thermal_state: ThermalState::Nominal,
                power_budget_watts: Some(45),
            },
        }
    }

    fn hot_topology_with_gpu() -> HardwareTopology {
        let mut topology = topology_with_gpu();
        topology.power.battery_powered = true;
        topology.power.battery_percent = Some(12);
        topology.power.thermal_state = ThermalState::Hot;
        topology.power.power_budget_watts = Some(15);
        topology
    }

    fn model(memory_bytes: u64) -> ModelDescriptor {
        ModelDescriptor {
            name: "demo".to_string(),
            path: PathBuf::from("D:/models/demo.gguf"),
            architecture: "llama".to_string(),
            memory_bytes: Some(memory_bytes),
            parameter_count: None,
            context_length: Some(8192),
            preferred_backend: None,
        }
    }

    fn host_hints(available_memory_bytes: u64, free_disk_bytes: u64) -> HostTieringHints {
        HostTieringHints {
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes,
            free_disk_bytes: Some(free_disk_bytes),
            disk_read_mbps: 1200.0,
            disk_write_mbps: 1000.0,
        }
    }

    fn inventory_for_model(model: &ModelDescriptor) -> ModelAssetInventory {
        ModelAssetInventory {
            root: model.path.clone(),
            layout: ModelAssetLayout::GgufFile,
            total_bytes: model.memory_bytes.unwrap_or_default(),
            shards: vec![ModelShardDescriptor {
                name: model.path.to_string_lossy().to_string(),
                path: model.path.clone(),
                bytes: model.memory_bytes.unwrap_or_default(),
                format: ModelFormat::Gguf,
                role: ModelShardRole::Weights,
                mmap_candidate: true,
            }],
        }
    }

    #[test]
    fn plan_uses_real_disk_target_and_scaled_prefetch_window() {
        let manager = TieredOffloadManager::new(TieredOffloadConfig::default());
        let plan = manager
            .plan(&model(12 * 1024 * 1024 * 1024), &topology(), None)
            .expect("plan");

        assert_eq!(plan.target_device, "disk:0");
        assert_eq!(plan.profile, TieredOffloadProfile::DiskHeavy);
        assert!(plan.prefetch_window_bytes >= 256 * 1024 * 1024);
        assert_eq!(
            plan.policy.weights.gpu_percent
                + plan.policy.weights.cpu_percent
                + plan.policy.weights.disk_percent,
            100
        );
        assert!(plan.policy.compress_kv_cache);
    }

    #[test]
    fn explicit_gpu_resident_profile_biases_policy_towards_gpu() {
        let mut config = TieredOffloadConfig::default();
        config.profile = TieredOffloadProfile::GpuResident;
        let manager = TieredOffloadManager::new(config);
        let plan = manager
            .plan(&model(12 * 1024 * 1024 * 1024), &topology_with_gpu(), None)
            .expect("plan");

        assert_eq!(plan.profile, TieredOffloadProfile::GpuResident);
        assert!(plan.policy.weights.gpu_percent >= plan.policy.weights.cpu_percent);
        assert!(!plan.policy.cpu_cache_compute);
    }

    #[test]
    fn auto_profile_switches_to_disk_heavy_under_power_pressure() {
        let manager = TieredOffloadManager::new(TieredOffloadConfig::default());
        let plan = manager
            .plan(
                &model(14 * 1024 * 1024 * 1024),
                &hot_topology_with_gpu(),
                None,
            )
            .expect("plan");

        assert_eq!(plan.profile, TieredOffloadProfile::DiskHeavy);
        assert!(plan.prefetch_window_bytes < 256 * 1024 * 1024 * 2);
        assert!(plan.policy.weights.disk_percent >= plan.policy.weights.gpu_percent);
    }

    #[test]
    fn host_hints_cap_spill_bytes_to_available_disk_budget() {
        let manager = TieredOffloadManager::new(TieredOffloadConfig::default());
        let host = host_hints(2 * 1024 * 1024 * 1024, 3 * 1024 * 1024 * 1024);
        let plan = manager
            .plan(
                &model(20 * 1024 * 1024 * 1024),
                &topology_with_gpu(),
                Some(&host),
            )
            .expect("plan");

        assert!(plan.spill_bytes <= (3 * 1024 * 1024 * 1024_u64).saturating_mul(85) / 100);
    }

    fn unique_root() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        std::env::temp_dir().join(format!("loci-tiered-offload-test-{stamp}"))
    }

    #[test]
    fn runtime_prepares_spill_artifact_and_snapshot() {
        let runtime = TieredOffloadRuntime::new(unique_root());
        let plan = TieredOffloadPlan {
            spill_bytes: 2 * 1024 * 1024,
            prefetch_window_bytes: 256 * 1024,
            target_device: "disk:0".to_string(),
            profile: TieredOffloadProfile::Balanced,
            policy: TieredOffloadPolicy {
                weights: TieredPlacementPercentages {
                    gpu_percent: 20,
                    cpu_percent: 40,
                    disk_percent: 40,
                },
                kv_cache: TieredPlacementPercentages {
                    gpu_percent: 10,
                    cpu_percent: 50,
                    disk_percent: 40,
                },
                activations: TieredPlacementPercentages {
                    gpu_percent: 50,
                    cpu_percent: 25,
                    disk_percent: 25,
                },
                cpu_cache_compute: true,
                compress_weights: true,
                compress_kv_cache: true,
            },
        };

        let snapshot = runtime
            .prepare_session(
                "demo-session",
                &model(12 * 1024 * 1024 * 1024),
                &inventory_for_model(&model(12 * 1024 * 1024 * 1024)),
                &plan,
            )
            .expect("session");

        assert_eq!(snapshot.session_key, "demo-session");
        assert!(snapshot.spill_path.exists());
        assert_eq!(snapshot.mapped_bytes, plan.spill_bytes);
        assert_eq!(snapshot.segments.len(), 3);
    }

    #[test]
    fn runtime_prefetch_worker_updates_snapshot() {
        let runtime = TieredOffloadRuntime::new(unique_root());
        let plan = TieredOffloadPlan {
            spill_bytes: 1024 * 1024,
            prefetch_window_bytes: 128 * 1024,
            target_device: "disk:0".to_string(),
            profile: TieredOffloadProfile::DiskHeavy,
            policy: TieredOffloadPolicy {
                weights: TieredPlacementPercentages {
                    gpu_percent: 10,
                    cpu_percent: 20,
                    disk_percent: 70,
                },
                kv_cache: TieredPlacementPercentages {
                    gpu_percent: 0,
                    cpu_percent: 20,
                    disk_percent: 80,
                },
                activations: TieredPlacementPercentages {
                    gpu_percent: 20,
                    cpu_percent: 30,
                    disk_percent: 50,
                },
                cpu_cache_compute: true,
                compress_weights: true,
                compress_kv_cache: true,
            },
        };

        runtime
            .prepare_session(
                "warmup",
                &model(16 * 1024 * 1024 * 1024),
                &inventory_for_model(&model(16 * 1024 * 1024 * 1024)),
                &plan,
            )
            .expect("session");
        runtime
            .schedule_prefetch("warmup", vec![(0, 64 * 1024)])
            .expect("prefetch");

        for _ in 0..20 {
            let snapshot = runtime
                .session_snapshot("warmup")
                .expect("snapshot")
                .expect("session");
            if snapshot.completed_prefetch_requests > 0 {
                assert!(snapshot.prefetched_bytes > 0);
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }

        panic!("prefetch worker did not complete in time");
    }

    #[test]
    fn evict_session_removes_spill_artifact() {
        let runtime = TieredOffloadRuntime::new(unique_root());
        let plan = TieredOffloadPlan {
            spill_bytes: 1024 * 1024,
            prefetch_window_bytes: 64 * 1024,
            target_device: "disk:0".to_string(),
            profile: TieredOffloadProfile::Balanced,
            policy: TieredOffloadPolicy {
                weights: TieredPlacementPercentages {
                    gpu_percent: 20,
                    cpu_percent: 40,
                    disk_percent: 40,
                },
                kv_cache: TieredPlacementPercentages {
                    gpu_percent: 20,
                    cpu_percent: 40,
                    disk_percent: 40,
                },
                activations: TieredPlacementPercentages {
                    gpu_percent: 50,
                    cpu_percent: 30,
                    disk_percent: 20,
                },
                cpu_cache_compute: false,
                compress_weights: false,
                compress_kv_cache: false,
            },
        };

        let snapshot = runtime
            .prepare_session(
                "evict",
                &model(12 * 1024 * 1024 * 1024),
                &inventory_for_model(&model(12 * 1024 * 1024 * 1024)),
                &plan,
            )
            .expect("session");
        assert!(snapshot.spill_path.exists());

        assert!(runtime.evict_session("evict").expect("evicted"));
        assert!(!snapshot.spill_path.exists());
    }
}

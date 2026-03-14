//! Background model asset pull jobs for host-side orchestration.

use crate::error::{LociError, Result};
use crate::model_pull_policy::ModelPullPolicyPlugin;
use crate::model_pull_verifier::ModelPullVerifierPlugin;
use crate::model_store::{ModelPullOptions, ModelPullProgress, ModelStore, StoredModel};
use crate::timeout_controller::{CancellationHandle, TimeoutContext};
use crossbeam::channel::{unbounded, Receiver, Sender};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPullJobRequest {
    pub source: String,
    #[serde(default)]
    pub mirrors: Vec<String>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub sha256: Option<String>,
    #[serde(default = "default_resume")]
    pub resume: bool,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_resume() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPullJobState {
    Queued,
    Running,
    CancelRequested,
    Completed,
    Failed,
    Cancelled,
}

impl ModelPullJobState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPullJobSnapshot {
    pub job_id: String,
    pub state: ModelPullJobState,
    pub created_at_unix_ms: u64,
    pub started_at_unix_ms: Option<u64>,
    pub finished_at_unix_ms: Option<u64>,
    #[serde(default)]
    pub policy_name: Option<String>,
    #[serde(default)]
    pub verifier_name: Option<String>,
    pub request: ModelPullJobRequest,
    #[serde(default)]
    pub progress: Option<ModelPullProgress>,
    #[serde(default)]
    pub model: Option<StoredModel>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelPullJobEvent {
    Snapshot {
        job: ModelPullJobSnapshot,
    },
    Progress {
        job_id: String,
        progress: ModelPullProgress,
    },
    Complete {
        job: ModelPullJobSnapshot,
    },
    Failed {
        job: ModelPullJobSnapshot,
    },
    CancelRequested {
        job: ModelPullJobSnapshot,
    },
    Cancelled {
        job: ModelPullJobSnapshot,
    },
}

struct ModelPullJobEntry {
    snapshot: Mutex<ModelPullJobSnapshot>,
    subscribers: Mutex<Vec<Sender<ModelPullJobEvent>>>,
    cancellation: CancellationHandle,
    policy: Option<Arc<dyn ModelPullPolicyPlugin>>,
    verifier: Option<Arc<dyn ModelPullVerifierPlugin>>,
}

struct ModelPullJobManagerInner {
    store: Arc<std::sync::Mutex<ModelStore>>,
    jobs: RwLock<HashMap<String, Arc<ModelPullJobEntry>>>,
    next_job_id: AtomicU64,
}

#[derive(Clone)]
pub struct ModelPullJobManager {
    inner: Arc<ModelPullJobManagerInner>,
}

impl ModelPullJobManager {
    pub fn new(store: Arc<std::sync::Mutex<ModelStore>>) -> Self {
        Self {
            inner: Arc::new(ModelPullJobManagerInner {
                store,
                jobs: RwLock::new(HashMap::new()),
                next_job_id: AtomicU64::new(1),
            }),
        }
    }

    pub fn submit_pull(&self, request: ModelPullJobRequest) -> Result<ModelPullJobSnapshot> {
        self.submit_pull_with_governance(request, None, None, None, None)
    }

    pub fn submit_pull_with_policy(
        &self,
        request: ModelPullJobRequest,
        policy_name: Option<String>,
        policy: Option<Arc<dyn ModelPullPolicyPlugin>>,
    ) -> Result<ModelPullJobSnapshot> {
        self.submit_pull_with_governance(request, policy_name, policy, None, None)
    }

    pub fn submit_pull_with_governance(
        &self,
        request: ModelPullJobRequest,
        policy_name: Option<String>,
        policy: Option<Arc<dyn ModelPullPolicyPlugin>>,
        verifier_name: Option<String>,
        verifier: Option<Arc<dyn ModelPullVerifierPlugin>>,
    ) -> Result<ModelPullJobSnapshot> {
        let sequence = self.inner.next_job_id.fetch_add(1, Ordering::Relaxed);
        let job_id = format!("pull-{}-{}", unix_ms_now(), sequence);
        let timeout_context = TimeoutContext::disabled();
        let cancellation = timeout_context.cancellation_handle();
        let snapshot = ModelPullJobSnapshot {
            job_id: job_id.clone(),
            state: ModelPullJobState::Queued,
            created_at_unix_ms: unix_ms_now(),
            started_at_unix_ms: None,
            finished_at_unix_ms: None,
            policy_name,
            verifier_name,
            request: request.clone(),
            progress: None,
            model: None,
            error: None,
        };
        let entry = Arc::new(ModelPullJobEntry {
            snapshot: Mutex::new(snapshot.clone()),
            subscribers: Mutex::new(Vec::new()),
            cancellation,
            policy,
            verifier,
        });
        self.inner.jobs.write().insert(job_id, Arc::clone(&entry));

        let manager = self.clone();
        thread::spawn(move || manager.run_job(entry));

        Ok(snapshot)
    }

    pub fn list_jobs(&self) -> Vec<ModelPullJobSnapshot> {
        let jobs = self.inner.jobs.read();
        let mut snapshots = jobs
            .values()
            .map(|entry| entry.snapshot.lock().clone())
            .collect::<Vec<_>>();
        snapshots.sort_by(|a, b| {
            a.created_at_unix_ms
                .cmp(&b.created_at_unix_ms)
                .then_with(|| a.job_id.cmp(&b.job_id))
        });
        snapshots
    }

    pub fn get_job(&self, job_id: &str) -> Result<ModelPullJobSnapshot> {
        Ok(self.job_entry(job_id)?.snapshot.lock().clone())
    }

    pub fn cancel_job(&self, job_id: &str) -> Result<ModelPullJobSnapshot> {
        let entry = self.job_entry(job_id)?;
        let snapshot = {
            let mut snapshot = entry.snapshot.lock();
            match snapshot.state {
                ModelPullJobState::Queued | ModelPullJobState::Running => {
                    entry.cancellation.cancel();
                    snapshot.state = ModelPullJobState::CancelRequested;
                    snapshot.error = Some("cancellation requested".to_string());
                }
                ModelPullJobState::CancelRequested => {}
                terminal => {
                    return Err(LociError::InvalidArgument(format!(
                        "model pull job '{}' is already {}",
                        job_id,
                        format_job_state(terminal)
                    )));
                }
            }
            snapshot.clone()
        };
        self.publish(
            &entry,
            ModelPullJobEvent::CancelRequested {
                job: snapshot.clone(),
            },
        );
        Ok(snapshot)
    }

    pub fn subscribe(
        &self,
        job_id: &str,
    ) -> Result<(ModelPullJobSnapshot, Receiver<ModelPullJobEvent>)> {
        let entry = self.job_entry(job_id)?;
        let (tx, rx) = unbounded();
        entry.subscribers.lock().push(tx);
        let snapshot = entry.snapshot.lock().clone();
        Ok((snapshot, rx))
    }

    fn run_job(&self, entry: Arc<ModelPullJobEntry>) {
        let request = entry.snapshot.lock().request.clone();

        let started = {
            let mut snapshot = entry.snapshot.lock();
            snapshot.started_at_unix_ms = Some(unix_ms_now());
            if entry.cancellation.is_cancelled() {
                snapshot.state = ModelPullJobState::Cancelled;
                snapshot.finished_at_unix_ms = Some(unix_ms_now());
                snapshot.error = Some("operation cancelled".to_string());
            } else {
                snapshot.state = ModelPullJobState::Running;
                snapshot.error = None;
            }
            snapshot.clone()
        };

        if started.state == ModelPullJobState::Cancelled {
            self.publish(&entry, ModelPullJobEvent::Cancelled { job: started });
            return;
        }
        self.publish(&entry, ModelPullJobEvent::Snapshot { job: started });

        let options = ModelPullOptions {
            mirrors: request.mirrors.clone(),
            expected_sha256: request.sha256.clone(),
            resume: request.resume,
        };
        let result = {
            let store = self
                .inner
                .store
                .lock()
                .expect("model store mutex should not be poisoned");
            let mut emit = |progress: ModelPullProgress| {
                let job_id = {
                    let mut snapshot = entry.snapshot.lock();
                    snapshot.progress = Some(progress.clone());
                    if matches!(snapshot.state, ModelPullJobState::Queued) {
                        snapshot.state = ModelPullJobState::Running;
                    }
                    snapshot.job_id.clone()
                };
                self.publish(&entry, ModelPullJobEvent::Progress { job_id, progress });
            };

            store.pull_from_source_with_options_and_progress_and_policy_and_verifier_and_cancellation(
                &request.source,
                request.id.clone(),
                request.name.clone(),
                request.tags.clone(),
                options,
                entry.policy.as_deref(),
                entry.verifier.as_deref(),
                Some(&entry.cancellation),
                &mut emit,
            )
        };

        match result {
            Ok(model) => {
                let snapshot = {
                    let mut snapshot = entry.snapshot.lock();
                    snapshot.state = ModelPullJobState::Completed;
                    snapshot.finished_at_unix_ms = Some(unix_ms_now());
                    snapshot.model = Some(model);
                    snapshot.error = None;
                    snapshot.clone()
                };
                self.publish(&entry, ModelPullJobEvent::Complete { job: snapshot });
            }
            Err(err) => {
                let cancelled = entry.cancellation.is_cancelled() || is_cancelled_error(&err);
                let snapshot = {
                    let mut snapshot = entry.snapshot.lock();
                    snapshot.state = if cancelled {
                        ModelPullJobState::Cancelled
                    } else {
                        ModelPullJobState::Failed
                    };
                    snapshot.finished_at_unix_ms = Some(unix_ms_now());
                    snapshot.error = Some(err.to_string());
                    snapshot.clone()
                };
                if cancelled {
                    self.publish(&entry, ModelPullJobEvent::Cancelled { job: snapshot });
                } else {
                    self.publish(&entry, ModelPullJobEvent::Failed { job: snapshot });
                }
            }
        }
    }

    fn job_entry(&self, job_id: &str) -> Result<Arc<ModelPullJobEntry>> {
        self.inner
            .jobs
            .read()
            .get(job_id)
            .cloned()
            .ok_or(LociError::ModelNotFound)
    }

    fn publish(&self, entry: &Arc<ModelPullJobEntry>, event: ModelPullJobEvent) {
        let mut subscribers = entry.subscribers.lock();
        subscribers.retain(|sender| sender.send(event.clone()).is_ok());
    }
}

fn is_cancelled_error(err: &LociError) -> bool {
    matches!(err, LociError::Timeout(message) if message.to_ascii_lowercase().contains("cancel"))
}

fn format_job_state(state: ModelPullJobState) -> &'static str {
    match state {
        ModelPullJobState::Queued => "queued",
        ModelPullJobState::Running => "running",
        ModelPullJobState::CancelRequested => "cancel_requested",
        ModelPullJobState::Completed => "completed",
        ModelPullJobState::Failed => "failed",
        ModelPullJobState::Cancelled => "cancelled",
    }
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
    use std::fs;
    use std::path::PathBuf;
    use std::time::Duration;

    fn temp_dir() -> PathBuf {
        let root = std::env::temp_dir().join(format!("loci-model-pull-jobs-{}", unix_ms_now()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn background_pull_job_completes_for_local_file() {
        let root = temp_dir();
        let source = root.join("source.gguf");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"background pull test bytes").unwrap();

        let store = Arc::new(std::sync::Mutex::new(ModelStore::new(root.join("store"))));
        let manager = ModelPullJobManager::new(store);
        let job = manager
            .submit_pull(ModelPullJobRequest {
                source: source.to_string_lossy().to_string(),
                mirrors: Vec::new(),
                id: Some("background-test".to_string()),
                name: None,
                sha256: None,
                resume: true,
                tags: vec!["managed".to_string()],
            })
            .unwrap();

        let mut snapshot = manager.get_job(&job.job_id).unwrap();
        for _ in 0..100 {
            if snapshot.state.is_terminal() {
                break;
            }
            thread::sleep(Duration::from_millis(20));
            snapshot = manager.get_job(&job.job_id).unwrap();
        }

        assert_eq!(snapshot.state, ModelPullJobState::Completed);
        assert!(snapshot.model.is_some());
        assert!(snapshot.progress.is_some());

        let _ = fs::remove_dir_all(root);
    }
}

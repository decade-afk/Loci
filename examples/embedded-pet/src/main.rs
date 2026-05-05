use anyhow::Context;
use loci_core::{EmbeddedModelRegistration, EngineConfig, InferenceEngine, SessionRequest, TieredOffloadProfile};

const SPILL_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DISK_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const PREFETCH_WINDOW_BYTES: u64 = 128 * 1024 * 1024;
const KV_BLOCK_SIZE_TOKENS: u32 = 32;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .context("usage: embedded-local <model-path>")?;

    let mut config = EngineConfig::default();
    config.tiered_offload.profile = TieredOffloadProfile::DiskHeavy;
    config.tiered_offload.spill_threshold_bytes = Some(SPILL_THRESHOLD_BYTES);
    config.tiered_offload.max_disk_bytes = Some(MAX_DISK_BYTES);
    config.tiered_offload.prefetch_window_bytes = Some(PREFETCH_WINDOW_BYTES);
    config.paged_kv.block_size_tokens = KV_BLOCK_SIZE_TOKENS;
    config.paged_kv.type_k = "q8_0".to_string();
    config.paged_kv.type_v = "q4_0".to_string();

    let mut engine = InferenceEngine::builder()
        .config(config)
        .local_model(
            model_path,
            EmbeddedModelRegistration {
                name: Some("embedded-demo".to_string()),
                ..EmbeddedModelRegistration::default()
            },
        )?
        .build()?;

    let response = engine.infer(SessionRequest {
        prompt: "Reply in one short friendly sentence for a desktop pet.".to_string(),
        max_tokens: 48,
        temperature: 0.7,
        target_model: Some("embedded-demo".to_string()),
        images: Vec::new(),
        structured_output: false,
        tool_calling: false,
    })?;

    println!("backend: {}", response.backend);
    println!("model: {}", response.model);
    println!("text: {}", response.text);
    let snapshot = engine.runtime_snapshot();
    println!(
        "planner: profile=disk_heavy spill_threshold_bytes={} max_disk_bytes={} prefetch_window_bytes={} kv_block_size_tokens={} kv_types=q8_0/q4_0",
        SPILL_THRESHOLD_BYTES,
        MAX_DISK_BYTES,
        PREFETCH_WINDOW_BYTES,
        KV_BLOCK_SIZE_TOKENS
    );
    println!(
        "runtime config: profile={:?} spill_threshold_bytes={:?} max_disk_bytes={:?} kv_block_size_tokens={} kv_type_k={} kv_type_v={}",
        snapshot.config.tiered_offload_profile,
        snapshot.config.spill_threshold_bytes,
        snapshot.config.max_disk_bytes,
        snapshot.config.kv_block_size_tokens,
        snapshot.config.kv_type_k,
        snapshot.config.kv_type_v
    );
    if let Some(tiered) = snapshot.tiered_offload_runtime {
        println!("spill root: {}", tiered.root_dir);
        println!("total spill bytes: {}", tiered.total_spill_bytes);
        println!("total prefetched bytes: {}", tiered.total_prefetched_bytes);
        for session in tiered.sessions {
            println!(
                "spill session: {} mapped={} prefetched={} weights={} kv={} activations={}",
                session.session_key,
                session.mapped_bytes,
                session.prefetched_bytes,
                session.weights_bytes,
                session.kv_cache_bytes,
                session.activations_bytes
            );
        }
    }
    Ok(())
}

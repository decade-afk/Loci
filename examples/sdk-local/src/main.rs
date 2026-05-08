use anyhow::Context;
use loci_sdk::{
    LocalModelRegistrationRequest, Loci, ModelPreparationRequest, TextSessionConfig,
    TieredOffloadProfile,
};

const SPILL_THRESHOLD_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DISK_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const PREFETCH_WINDOW_BYTES: u64 = 128 * 1024 * 1024;
const KV_BLOCK_SIZE_TOKENS: u32 = 32;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .context("usage: sdk-local <model-path>")?;

    let mut loci = Loci::builder()
        .tiered_offload_profile(TieredOffloadProfile::DiskHeavy)
        .spill_threshold_bytes(SPILL_THRESHOLD_BYTES)
        .max_disk_bytes(MAX_DISK_BYTES)
        .prefetch_window_bytes(PREFETCH_WINDOW_BYTES)
        .kv_block_size_tokens(KV_BLOCK_SIZE_TOKENS)
        .kv_types("q8_0", "q4_0")
        .build()?;
    let registered =
        loci.register_model(LocalModelRegistrationRequest::new(model_path).name("sdk-demo"))?;
    let inspection = loci.inspect_model("sdk-demo")?;
    let prepared = loci.prepare_model(ModelPreparationRequest::new().model("sdk-demo"))?;
    let mut session = loci.open_text_session(
        TextSessionConfig::new()
            .model("sdk-demo")
            .system_prompt("you are a local assistant")
            .max_tokens(48)
            .temperature(0.7),
    )?;
    let response = loci.generate_in_text_session_with_callback(
        &mut session,
        "Reply in one short sentence as a local assistant.",
        |chunk| {
            if !chunk.delta.is_empty() {
                println!("chunk: {}", chunk.delta);
            }
        },
    )?;

    println!("registered: {} ({})", registered.name, registered.format);
    println!(
        "inspection: ready={} recommended_backend={:?}",
        inspection.ready_for_inference, inspection.recommended_backend
    );
    println!(
        "planner: profile=disk_heavy spill_threshold_bytes={} max_disk_bytes={} prefetch_window_bytes={} kv_block_size_tokens={} kv_types=q8_0/q4_0",
        SPILL_THRESHOLD_BYTES,
        MAX_DISK_BYTES,
        PREFETCH_WINDOW_BYTES,
        KV_BLOCK_SIZE_TOKENS
    );
    println!("prepared session: {}", prepared.session_key);
    println!("session prepared: {}", session.prepared().session_key);
    println!("prepared residency: {:?}", prepared.residency);
    println!(
        "prepared estimated_memory_bytes: {:?}",
        prepared.estimated_memory_bytes
    );
    println!("backend: {}", response.backend);
    println!("model: {}", response.model);
    println!("text: {}", response.text);
    let runtime_config = loci.runtime_config();
    println!(
        "runtime config: profile={:?} spill_threshold_bytes={:?} max_disk_bytes={:?} kv_block_size_tokens={} kv_type_k={} kv_type_v={}",
        runtime_config.tiered_offload_profile,
        runtime_config.spill_threshold_bytes,
        runtime_config.max_disk_bytes,
        runtime_config.kv_block_size_tokens,
        runtime_config.kv_type_k,
        runtime_config.kv_type_v
    );
    if let Some(tiered) = loci.tiered_offload_runtime() {
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

use anyhow::Context;
use loci_sdk::{
    LocalModelRegistrationRequest, Loci, ModelPreparationRequest, TextGenerationRequest,
};
use loci_sdk::loci_core::TieredOffloadProfile;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .context("usage: sdk-local <model-path>")?;

    let mut loci = Loci::builder()
        .tiered_offload_profile(TieredOffloadProfile::DiskHeavy)
        .spill_threshold_bytes(512 * 1024 * 1024)
        .max_disk_bytes(64 * 1024 * 1024 * 1024)
        .prefetch_window_bytes(128 * 1024 * 1024)
        .kv_block_size_tokens(32)
        .kv_types("q8_0", "q4_0")
        .build()?;
    let registered =
        loci.register_model(LocalModelRegistrationRequest::new(model_path).name("sdk-demo"))?;
    let inspection = loci.inspect_model("sdk-demo")?;
    let prepared = loci.prepare_model(ModelPreparationRequest::new().model("sdk-demo"))?;
    let response = loci.generate_text(
        TextGenerationRequest::new("Reply in one short sentence as a local assistant.")
            .model("sdk-demo")
            .max_tokens(48)
            .temperature(0.7),
    )?;

    println!("registered: {} ({})", registered.name, registered.format);
    println!(
        "inspection: ready={} recommended_backend={:?}",
        inspection.ready_for_inference, inspection.recommended_backend
    );
    println!("prepared session: {}", prepared.session_key);
    println!("prepared residency: {:?}", prepared.residency);
    println!(
        "prepared estimated_memory_bytes: {:?}",
        prepared.estimated_memory_bytes
    );
    println!("backend: {}", response.backend);
    println!("model: {}", response.model);
    println!("text: {}", response.text);
    let snapshot = loci.runtime_snapshot();
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

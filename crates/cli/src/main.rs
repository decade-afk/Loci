use loci_core::InferenceEngine;

fn main() -> anyhow::Result<()> {
    let engine = InferenceEngine::builder().build()?;
    println!("loci-cli ready; plugins={}", engine.plugin_count());
    Ok(())
}

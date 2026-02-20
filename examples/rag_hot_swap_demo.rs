use loci::prelude::*;

fn main() -> Result<()> {
    let mut plugin_a = InMemoryRagPlugin::new("product_docs");
    plugin_a.set_top_k(2)?;
    plugin_a.ingest_documents(vec![
        RagDocument::new("p1", "Loci supports plugin-based architecture."),
        RagDocument::new("p2", "Backends can be selected at runtime."),
    ])?;

    let mut plugin_b = InMemoryRagPlugin::new("rust_docs");
    plugin_b.set_top_k(2)?;
    plugin_b.ingest_documents(vec![
        RagDocument::new("r1", "Rust ownership eliminates data races."),
        RagDocument::new("r2", "Borrow checker enforces aliasing rules."),
    ])?;

    // In real application, engine comes from loaded model:
    // let mut engine = InferenceEngine::builder().model_path(\"...\").build()?;
    // Here we just demonstrate API usage signatures.
    let _ = (plugin_a, plugin_b);

    println!("RAG hot-swap API example:");
    println!("  engine.register_rag_plugin(plugin_a)?;");
    println!("  engine.register_rag_plugin(plugin_b)?;");
    println!("  engine.activate_rag_plugin(\"product_docs\")?;");
    println!("  engine.activate_rag_plugin(\"rust_docs\")?; // hot-swap");
    println!("  engine.deactivate_rag_plugin();");
    Ok(())
}

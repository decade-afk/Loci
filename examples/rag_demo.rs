use loci::prelude::*;

fn main() -> Result<()> {
    let mut rag = RagEngine::new(HashEmbeddingProvider::default()).with_chunking(ChunkingConfig {
        chunk_size: 32,
        chunk_overlap: 8,
    })?;

    rag.ingest_documents(vec![
        RagDocument::new(
            "rust-book",
            "Rust ownership guarantees memory safety without garbage collection.",
        ),
        RagDocument::new(
            "loci-guide",
            "Loci supports local inference with plugin-based architecture and backend abstraction.",
        ),
    ])?;

    let prompt = rag.augment_prompt(
        "How does Rust keep memory safe?",
        2,
        Some("Answer concisely based on context."),
    );

    println!("Indexed chunks: {}", rag.indexed_chunks());
    println!("\nAugmented prompt:\n{}", prompt);
    Ok(())
}

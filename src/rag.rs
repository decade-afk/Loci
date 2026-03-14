use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagDocument {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

impl RagDocument {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagChunk {
    pub document_id: String,
    pub chunk_id: usize,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkingConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 240,
            chunk_overlap: 40,
        }
    }
}

pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
}

#[derive(Debug, Clone)]
pub struct HashEmbeddingProvider {
    dims: usize,
}

impl HashEmbeddingProvider {
    pub fn new(dims: usize) -> Result<Self> {
        if dims == 0 {
            return Err(LociError::InvalidArgument(
                "Embedding dimension must be greater than 0".to_string(),
            ));
        }

        Ok(Self { dims })
    }
}

impl Default for HashEmbeddingProvider {
    fn default() -> Self {
        Self { dims: 384 }
    }
}

impl EmbeddingProvider for HashEmbeddingProvider {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dims];

        for token in text
            .split_whitespace()
            .map(|token| token.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|token| !token.is_empty())
        {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            token.to_ascii_lowercase().hash(&mut hasher);
            let idx = (hasher.finish() as usize) % self.dims;
            vector[idx] += 1.0;
        }

        normalize_l2(&mut vector);
        vector
    }
}

#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    pub chunk: RagChunk,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct InMemoryVectorStore {
    entries: Vec<(RagChunk, Vec<f32>)>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn insert(&mut self, chunk: RagChunk, embedding: Vec<f32>) {
        self.entries.push((chunk, embedding));
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<RetrievedChunk> {
        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }

        let mut scored = self
            .entries
            .iter()
            .map(|(chunk, embedding)| RetrievedChunk {
                chunk: chunk.clone(),
                score: cosine_similarity(query_embedding, embedding),
            })
            .collect::<Vec<_>>();

        scored.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored.truncate(top_k.min(scored.len()));
        scored
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RagEngine<E>
where
    E: EmbeddingProvider,
{
    embedding_provider: E,
    store: InMemoryVectorStore,
    chunking: ChunkingConfig,
}

impl<E> RagEngine<E>
where
    E: EmbeddingProvider,
{
    pub fn new(embedding_provider: E) -> Self {
        Self {
            embedding_provider,
            store: InMemoryVectorStore::new(),
            chunking: ChunkingConfig::default(),
        }
    }

    pub fn with_chunking(mut self, chunking: ChunkingConfig) -> Result<Self> {
        if chunking.chunk_size == 0 {
            return Err(LociError::InvalidArgument(
                "chunk_size must be greater than 0".to_string(),
            ));
        }

        if chunking.chunk_overlap >= chunking.chunk_size {
            return Err(LociError::InvalidArgument(
                "chunk_overlap must be smaller than chunk_size".to_string(),
            ));
        }

        self.chunking = chunking;
        Ok(self)
    }

    pub fn ingest_document(&mut self, document: RagDocument) -> Result<usize> {
        if document.text.trim().is_empty() {
            return Err(LociError::InvalidArgument(
                "Document text cannot be empty".to_string(),
            ));
        }

        let chunks = split_into_chunks(&document, self.chunking);
        for chunk in &chunks {
            let embedding = self.embedding_provider.embed(&chunk.text);
            self.store.insert(chunk.clone(), embedding);
        }

        Ok(chunks.len())
    }

    pub fn ingest_documents<I>(&mut self, documents: I) -> Result<usize>
    where
        I: IntoIterator<Item = RagDocument>,
    {
        let mut ingested_chunks = 0;
        for document in documents {
            ingested_chunks += self.ingest_document(document)?;
        }

        Ok(ingested_chunks)
    }

    pub fn retrieve(&self, query: &str, top_k: usize) -> Vec<RetrievedChunk> {
        let query_embedding = self.embedding_provider.embed(query);
        self.store.search(&query_embedding, top_k)
    }

    pub fn augment_prompt(&self, query: &str, top_k: usize, instruction: Option<&str>) -> String {
        let context = self.retrieve(query, top_k);
        let mut output = String::new();

        if let Some(instruction) = instruction {
            output.push_str(instruction);
            output.push_str("\n\n");
        }

        if !context.is_empty() {
            output.push_str("Context:\n");
            for (index, item) in context.iter().enumerate() {
                output.push_str(&format!("[{}] {}\n", index + 1, item.chunk.text));
            }
            output.push('\n');
        }

        output.push_str("Question:\n");
        output.push_str(query);
        output
    }

    pub fn indexed_chunks(&self) -> usize {
        self.store.len()
    }
}

/// Hot-swappable RAG plugin interface.
pub trait RagPlugin: Send + Sync {
    fn name(&self) -> &str;
    fn augment_prompt(&self, prompt: &str) -> Result<String>;
    fn indexed_chunks(&self) -> usize;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Built-in in-memory RAG plugin.
pub struct InMemoryRagPlugin {
    name: String,
    top_k: usize,
    instruction: Option<String>,
    engine: RagEngine<HashEmbeddingProvider>,
}

impl InMemoryRagPlugin {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            top_k: 3,
            instruction: None,
            engine: RagEngine::new(HashEmbeddingProvider::default()),
        }
    }

    pub fn with_chunking(mut self, chunking: ChunkingConfig) -> Result<Self> {
        self.engine = self.engine.with_chunking(chunking)?;
        Ok(self)
    }

    pub fn set_top_k(&mut self, top_k: usize) -> Result<()> {
        if top_k == 0 {
            return Err(LociError::InvalidArgument(
                "top_k must be greater than 0".to_string(),
            ));
        }
        self.top_k = top_k;
        Ok(())
    }

    pub fn set_instruction(&mut self, instruction: Option<String>) {
        self.instruction = instruction;
    }

    pub fn ingest_document(&mut self, document: RagDocument) -> Result<usize> {
        self.engine.ingest_document(document)
    }

    pub fn ingest_documents<I>(&mut self, documents: I) -> Result<usize>
    where
        I: IntoIterator<Item = RagDocument>,
    {
        self.engine.ingest_documents(documents)
    }
}

impl RagPlugin for InMemoryRagPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn augment_prompt(&self, prompt: &str) -> Result<String> {
        if self.engine.indexed_chunks() == 0 {
            return Ok(prompt.to_string());
        }

        Ok(self
            .engine
            .augment_prompt(prompt, self.top_k, self.instruction.as_deref()))
    }

    fn indexed_chunks(&self) -> usize {
        self.engine.indexed_chunks()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn split_into_chunks(document: &RagDocument, config: ChunkingConfig) -> Vec<RagChunk> {
    let words = document
        .text
        .split_whitespace()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    if words.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let step = config.chunk_size - config.chunk_overlap;
    let mut start = 0;

    while start < words.len() {
        let end = (start + config.chunk_size).min(words.len());
        let text = words[start..end].join(" ");
        chunks.push(RagChunk {
            document_id: document.id.clone(),
            chunk_id: chunks.len(),
            text,
            metadata: document.metadata.clone(),
        });

        if end == words.len() {
            break;
        }

        start += step;
    }

    chunks
}

fn normalize_l2(values: &mut [f32]) {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> f32 {
    let dims = lhs.len().min(rhs.len());
    if dims == 0 {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut lhs_norm = 0.0;
    let mut rhs_norm = 0.0;

    for index in 0..dims {
        dot += lhs[index] * rhs[index];
        lhs_norm += lhs[index] * lhs[index];
        rhs_norm += rhs[index] * rhs[index];
    }

    if lhs_norm == 0.0 || rhs_norm == 0.0 {
        0.0
    } else {
        dot / (lhs_norm.sqrt() * rhs_norm.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_with_overlap() {
        let document = RagDocument::new(
            "doc-1",
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
        );
        let chunks = split_into_chunks(
            &document,
            ChunkingConfig {
                chunk_size: 4,
                chunk_overlap: 1,
            },
        );

        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].text, "alpha beta gamma delta");
        assert_eq!(chunks[1].text, "delta epsilon zeta eta");
    }

    #[test]
    fn test_retrieval_returns_relevant_chunk() {
        let mut engine = RagEngine::new(HashEmbeddingProvider::default())
            .with_chunking(ChunkingConfig {
                chunk_size: 8,
                chunk_overlap: 0,
            })
            .unwrap();

        engine
            .ingest_documents(vec![
                RagDocument::new("doc-1", "Rust ownership and borrowing rules"),
                RagDocument::new("doc-2", "Neural network batch normalization"),
            ])
            .unwrap();

        let retrieved = engine.retrieve("How does Rust borrowing work?", 1);
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].chunk.document_id, "doc-1");
    }

    #[test]
    fn test_prompt_augmentation_contains_context() {
        let mut engine = RagEngine::new(HashEmbeddingProvider::default());
        engine
            .ingest_document(RagDocument::new(
                "doc-1",
                "Paris is the capital city of France.",
            ))
            .unwrap();

        let prompt = engine.augment_prompt(
            "What is the capital of France?",
            1,
            Some("Answer using the context."),
        );

        assert!(prompt.contains("Context:"));
        assert!(prompt.contains("capital city of France"));
        assert!(prompt.contains("Question:"));
    }

    #[test]
    fn test_in_memory_rag_plugin_augmentation() {
        let mut plugin = InMemoryRagPlugin::new("knowledge");
        plugin
            .ingest_document(RagDocument::new(
                "doc-1",
                "Rust prevents data races through ownership and borrowing rules.",
            ))
            .unwrap();

        let prompt = plugin
            .augment_prompt("How does Rust avoid data races?")
            .unwrap();
        assert!(prompt.contains("Context:"));
        assert!(prompt.contains("ownership"));
    }
}

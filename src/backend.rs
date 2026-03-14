//! Backend abstraction layer for inference engines
//!
//! This module provides trait-based abstractions for different inference backends
//! (llama.cpp, Candle, ONNX, etc.), enabling pluggable architecture while
//! maintaining zero-overhead for static dispatch.
//!
//! ## Architecture
//!
//! - `InferenceBackend`: Factory trait for creating models
//! - `Model`: Core trait for loaded models (text + multimodal inference)
//! - `BackendCapabilities`: Declarative metadata about backend features
//!
//! ## Design Principles
//!
//! 1. **Single-track Native (Phase 1)**: Hardcoded backend selection via config
//! 2. **Future Dynamic Loading (Phase 2)**: libloading for .so/.dll plugins
//! 3. **Future WASM Track (Phase 3)**: wasmtime for sandboxed lightweight plugins
//!
//! Current implementation focuses on Phase 1 with trait design ready for Phase 2/3.

use crate::backends::{CandleBackend, DynamicBackend, LlamaCppBackend};
use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Image data for multimodal inference
#[derive(Debug, Clone)]
pub struct Image {
    /// Image data (raw bytes, base64, or file path)
    pub data: ImageData,
    /// Optional image format hint
    pub format: Option<ImageFormat>,
}

#[derive(Debug, Clone)]
pub enum ImageData {
    /// Raw image bytes
    Bytes(Vec<u8>),
    /// Base64 encoded image
    Base64(String),
    /// Path to image file
    Path(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Jpeg,
    Gif,
    Webp,
}

/// Strategy for splitting model tensors across multiple GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GpuSplitMode {
    /// Keep the model on a single selected GPU.
    None,
    /// Split layers (and KV) across multiple GPUs.
    #[default]
    Layer,
    /// Split rows/tensors across multiple GPUs when backend supports tensor parallelism.
    Row,
}

/// Parameters for backend initialization
#[derive(Debug, Clone)]
pub struct BackendParams {
    /// Number of GPU layers to offload (-1 = all)
    pub n_gpu_layers: i32,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Use memory-mapped model loading when available
    pub use_mmap: bool,
    /// Lock model pages into RAM when available
    pub use_mlock: bool,
    /// Offload K/Q/V ops and KV cache to device
    pub kv_offload: bool,
    /// Offload host tensor ops to device
    pub op_offload: bool,
    /// Multi-GPU split strategy
    pub split_mode: GpuSplitMode,
    /// Primary GPU index used for single-GPU placement
    pub main_gpu: u32,
    /// Relative split weights for each GPU
    pub tensor_split: Option<Vec<f32>>,
    /// Additional backend-specific options
    pub options: Vec<(String, String)>,
}

impl Default for BackendParams {
    fn default() -> Self {
        Self {
            n_gpu_layers: -1,
            use_gpu: true,
            use_mmap: true,
            use_mlock: false,
            kv_offload: true,
            op_offload: true,
            split_mode: GpuSplitMode::Layer,
            main_gpu: 0,
            tensor_split: None,
            options: Vec::new(),
        }
    }
}

/// Parameters for inference
#[derive(Debug, Clone)]
pub struct InferenceParams {
    /// Context size (number of tokens)
    pub n_ctx: u32,
    /// Batch size for prompt processing
    pub n_batch: u32,
    /// Number of threads (None = auto-detect)
    pub n_threads: Option<u32>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Min-p sampling
    pub min_p: f32,
    /// Top-k sampling
    pub top_k: u32,
    /// Repetition penalty
    pub repeat_penalty: f32,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            n_batch: 512,
            n_threads: None,
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 40,
            repeat_penalty: 1.1,
        }
    }
}

/// Backend capabilities (declarative metadata)
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend name (e.g., "llama.cpp", "candle", "onnx")
    pub name: String,
    /// Backend version
    pub version: String,
    /// Supports text-only inference
    pub supports_text: bool,
    /// Supports multimodal inference (text + images)
    pub supports_multimodal: bool,
    /// Supports embeddings generation
    pub supports_embeddings: bool,
    /// Supports streaming output
    pub supports_streaming: bool,
    /// GPU acceleration available
    pub has_gpu_support: bool,
    /// Supported model formats (e.g., ["gguf", "safetensors"])
    pub supported_formats: Vec<String>,
}

/// Core trait for inference backends (factory pattern)
///
/// Implementations create `Model` instances. This trait will be used for:
/// - Phase 1: Hardcoded backend selection (llama.cpp)
/// - Phase 2: Dynamic loading via libloading
/// - Phase 3: WASM backends via wasmtime
pub trait InferenceBackend: Send + Sync {
    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Load a model from file
    fn load_model(
        &self,
        model_path: &Path,
        backend_params: BackendParams,
    ) -> Result<Box<dyn Model>>;

    /// Initialize the backend (called once on startup)
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    /// Cleanup the backend (called on shutdown)
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Core trait for loaded models
///
/// This trait abstracts over different model implementations, supporting:
/// - Text-only inference (always required)
/// - Multimodal inference (optional, check `supports_multimodal`)
/// - Embeddings (optional, check `supports_embeddings`)
/// - Streaming (optional, via `ModelExt` trait)
pub trait Model: Send + Sync {
    /// Get model metadata
    fn metadata(&self) -> ModelMetadata;

    /// Check if multimodal inference is supported
    fn supports_multimodal(&self) -> bool {
        false
    }

    /// Check if embeddings are supported
    fn supports_embeddings(&self) -> bool {
        false
    }

    /// Check if streaming is supported
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Perform text-only inference
    fn infer_text(&mut self, prompt: &str, params: &InferenceParams) -> Result<String>;

    /// Perform multimodal inference (text + images)
    ///
    /// # Errors
    /// Returns error if backend doesn't support multimodal (check `supports_multimodal` first)
    fn infer_multimodal(
        &mut self,
        _text: &str,
        _images: &[Image],
        _params: &InferenceParams,
    ) -> Result<String> {
        Err(crate::error::LociError::UnsupportedOperation(
            "Multimodal inference not supported by this backend".to_string(),
        ))
    }

    /// Generate embeddings for text
    ///
    /// # Errors
    /// Returns error if backend doesn't support embeddings (check `supports_embeddings` first)
    fn generate_embeddings(&mut self, _text: &str) -> Result<Vec<f32>> {
        Err(crate::error::LociError::UnsupportedOperation(
            "Embeddings not supported by this backend".to_string(),
        ))
    }

    /// Perform streaming inference with callback (object-safe variant)
    ///
    /// Implement this for backends that support streaming when working through
    /// trait objects (`Box<dyn Model>`).
    fn infer_stream(
        &mut self,
        _prompt: &str,
        _params: &InferenceParams,
        _callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        Err(crate::error::LociError::UnsupportedOperation(
            "Streaming inference not supported by this backend".to_string(),
        ))
    }

    /// Perform streaming multimodal inference (object-safe variant)
    fn infer_multimodal_stream(
        &mut self,
        _text: &str,
        _images: &[Image],
        _params: &InferenceParams,
        _callback: &mut dyn FnMut(&str) -> bool,
    ) -> Result<()> {
        Err(crate::error::LociError::UnsupportedOperation(
            "Multimodal streaming inference not supported by this backend".to_string(),
        ))
    }
}

/// Extension trait for streaming inference (separate to maintain dyn compatibility)
///
/// This trait is separated from `Model` because generic methods prevent trait objects.
/// Concrete types implement both `Model` and `ModelExt`.
pub trait ModelExt: Model {
    /// Perform streaming inference with callback
    ///
    /// The callback receives each generated token. Return `false` to stop generation early.
    fn infer_stream<F>(
        &mut self,
        prompt: &str,
        params: &InferenceParams,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool;

    /// Perform streaming multimodal inference
    fn infer_multimodal_stream<F>(
        &mut self,
        text: &str,
        images: &[Image],
        params: &InferenceParams,
        callback: F,
    ) -> Result<()>
    where
        F: FnMut(&str) -> bool;
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model architecture name (e.g., "llama", "qwen", "phi")
    pub architecture: String,
    /// Vocabulary size
    pub n_vocab: u32,
    /// Training context size
    pub n_ctx_train: u32,
    /// Embedding dimension
    pub n_embd: u32,
    /// Number of layers
    pub n_layer: u32,
    /// Model parameter count (approximate)
    pub param_count: Option<u64>,
}

/// Backend registry for managing available backends
///
/// Phase 1: Hardcoded backends (llama.cpp only)
/// Phase 2: Dynamic registration via libloading
/// Phase 3: WASM backends via wasmtime
pub struct BackendRegistry {
    backends: std::collections::HashMap<String, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            backends: std::collections::HashMap::new(),
        }
    }

    /// Register a backend
    pub fn register(&mut self, name: String, backend: Box<dyn InferenceBackend>) {
        self.backends.insert(name, backend);
    }

    /// Register built-in backends
    pub fn register_builtin_backends(&mut self) {
        self.register("llama.cpp".to_string(), Box::new(LlamaCppBackend::new()));
        self.register("candle".to_string(), Box::new(CandleBackend::new()));
    }

    /// Create a registry pre-populated with built-in backends
    pub fn with_builtin_backends() -> Self {
        let mut registry = Self::new();
        registry.register_builtin_backends();
        registry
    }

    /// Load and register a dynamic backend
    pub fn load_dynamic_backend<P: AsRef<Path>>(&mut self, name: String, path: P) -> Result<()> {
        if self.backends.contains_key(&name) {
            return Err(LociError::InvalidArgument(format!(
                "Backend already registered: {name}"
            )));
        }

        let backend = DynamicBackend::load(path)?;
        self.register(name, Box::new(backend));
        Ok(())
    }

    /// Unregister a backend by name
    pub fn unregister(&mut self, name: &str) -> Option<Box<dyn InferenceBackend>> {
        self.backends.remove(name)
    }

    /// Check if backend exists
    pub fn contains(&self, name: &str) -> bool {
        self.backends.contains_key(name)
    }

    /// Get a backend by name
    pub fn get(&self, name: &str) -> Option<&dyn InferenceBackend> {
        self.backends.get(name).map(|backend| backend.as_ref())
    }

    /// Get mutable backend by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Box<dyn InferenceBackend>> {
        self.backends.get_mut(name)
    }

    /// Load model from a named backend
    pub fn load_model(
        &mut self,
        backend_name: &str,
        model_path: &Path,
        params: BackendParams,
    ) -> Result<Box<dyn Model>> {
        let backend = self.backends.get_mut(backend_name).ok_or_else(|| {
            LociError::BackendNotAvailable(format!("Backend not found: {backend_name}"))
        })?;

        backend.init()?;
        backend.load_model(model_path, params)
    }

    /// List backend names
    pub fn names(&self) -> Vec<&str> {
        self.backends.keys().map(String::as_str).collect()
    }

    /// List all registered backends
    pub fn list(&self) -> Vec<BackendCapabilities> {
        self.backends.values().map(|b| b.capabilities()).collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_with_builtins() {
        let registry = BackendRegistry::with_builtin_backends();
        assert!(registry.contains("llama.cpp"));
        assert!(registry.contains("candle"));
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = BackendRegistry::new();
        registry.register("candle".to_string(), Box::new(CandleBackend::new()));
        assert!(registry.contains("candle"));
        registry.unregister("candle");
        assert!(!registry.contains("candle"));
    }
}

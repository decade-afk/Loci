//! Deep Programmable Multimodal Plugin System
//!
//! This module provides a plugin-based architecture for multimodal processing:
//!
//! ## Plugin Types
//!
//! 1. **Vision Encoder Plugins**
//!    - Custom image encoders
//!    - Preprocessing pipelines
//!    - Feature extractors
//!
//! 2. **Audio Encoder Plugins**
//!    - Custom audio processors
//!    - Speech recognition
//!    - Audio classification
//!
//! 3. **Fusion Strategy Plugins**
//!    - Custom fusion algorithms
//!    - Attention mechanisms
//!    - Cross-modal transformations
//!
//! 4. **Preprocessing Plugins**
//!    - Image augmentation
//!    - Audio enhancement
//!    - Data normalization
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │  MultimodalPluginRegistry                              │
//! │  - HashMap<PluginId, Box<dyn MultimodalPlugin>>        │
//! │  - Plugin composition                                   │
//! └────────────────────────────────────────────────────────┘
//!          ↓
//! ┌─────────────────┬─────────────────┬────────────────────┐
//! │  VisionPlugin   │  AudioPlugin    │  FusionPlugin      │
//! │  (trait)        │  (trait)        │  (trait)           │
//! └─────────────────┴─────────────────┴────────────────────┘
//!          ↓
//! ┌────────────────────────────────────────────────────────┐
//! │  Plugin Implementations                                │
//! │  - Native (built-in)                                    │
//! │  - Dynamic (libloading)                                 │
//! │  - WASM (wasmtime)                                      │
//! └────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use loci::multimodal_plugin::*;
//!
//! // Register built-in plugins
//! let mut registry = MultimodalPluginRegistry::new();
//! registry.register_builtin_vision_plugins();
//!
//! // Load custom WASM plugin
//! registry.load_wasm_plugin("custom_encoder.wasm")?;
//!
//! // Use plugin
//! let encoder = registry.get_vision_encoder("clip-vit-large")?;
//! let embeddings = encoder.encode(image)?;
//! ```

use crate::error::Result;
use crate::multimodal::{Audio, Image};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Plugin ID for multimodal processors
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModalPluginId(pub String);

impl ModalPluginId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

/// Vision encoder plugin trait
///
/// All vision encoders must implement this trait to be usable by the system.
pub trait VisionEncoderPlugin: Send + Sync {
    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Encode image to embeddings
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    ///
    /// # Returns
    ///
    /// Embedding vector (shape: [hidden_dim])
    fn encode(&self, image: &Image) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get supported image sizes
    fn supported_sizes(&self) -> Vec<(usize, usize)>;

    /// Preprocess image (resize, normalize, etc.)
    fn preprocess(&self, image: &Image) -> Result<Image>;

    /// Get plugin metadata as JSON
    fn metadata(&self) -> String {
        serde_json::json!({
            "name": self.name(),
            "version": self.version(),
            "type": "vision_encoder",
            "embedding_dim": self.embedding_dim(),
        })
        .to_string()
    }
}

/// Audio encoder plugin trait
pub trait AudioEncoderPlugin: Send + Sync {
    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Encode audio to embeddings
    fn encode(&self, audio: &Audio) -> Result<Vec<f32>>;

    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Get supported sample rates
    fn supported_sample_rates(&self) -> Vec<usize>;

    /// Preprocess audio
    fn preprocess(&self, audio: &Audio) -> Result<Audio>;

    /// Get plugin metadata
    fn metadata(&self) -> String {
        serde_json::json!({
            "name": self.name(),
            "version": self.version(),
            "type": "audio_encoder",
            "embedding_dim": self.embedding_dim(),
        })
        .to_string()
    }
}

/// Fusion strategy plugin trait
pub trait FusionStrategyPlugin: Send + Sync {
    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Fuse multimodal embeddings
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Map of modality name to embedding vectors
    ///
    /// # Returns
    ///
    /// Fused embedding vector
    fn fuse(&self, embeddings: &HashMap<String, Vec<f32>>) -> Result<Vec<f32>>;

    /// Get metadata
    fn metadata(&self) -> String {
        serde_json::json!({
            "name": self.name(),
            "version": self.version(),
            "type": "fusion_strategy",
        })
        .to_string()
    }
}

/// Preprocessing plugin trait
pub trait PreprocessingPlugin: Send + Sync {
    /// Get plugin name
    fn name(&self) -> &str;

    /// Get plugin version
    fn version(&self) -> &str;

    /// Preprocess image
    fn preprocess_image(&self, image: &Image) -> Result<Image>;

    /// Preprocess audio
    fn preprocess_audio(&self, audio: &Audio) -> Result<Audio>;

    /// Get metadata
    fn metadata(&self) -> String {
        serde_json::json!({
            "name": self.name(),
            "version": self.version(),
            "type": "preprocessing",
        })
        .to_string()
    }
}

/// Multimodal plugin registry
pub struct MultimodalPluginRegistry {
    /// Vision encoder plugins
    vision_encoders: HashMap<ModalPluginId, Box<dyn VisionEncoderPlugin>>,

    /// Audio encoder plugins
    audio_encoders: HashMap<ModalPluginId, Box<dyn AudioEncoderPlugin>>,

    /// Fusion strategy plugins
    fusion_strategies: HashMap<ModalPluginId, Box<dyn FusionStrategyPlugin>>,

    /// Preprocessing plugins
    preprocessors: HashMap<ModalPluginId, Box<dyn PreprocessingPlugin>>,
}

impl MultimodalPluginRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            vision_encoders: HashMap::new(),
            audio_encoders: HashMap::new(),
            fusion_strategies: HashMap::new(),
            preprocessors: HashMap::new(),
        }
    }

    /// Register a vision encoder plugin
    pub fn register_vision_encoder(
        &mut self,
        id: ModalPluginId,
        plugin: Box<dyn VisionEncoderPlugin>,
    ) {
        self.vision_encoders.insert(id, plugin);
    }

    /// Register an audio encoder plugin
    pub fn register_audio_encoder(
        &mut self,
        id: ModalPluginId,
        plugin: Box<dyn AudioEncoderPlugin>,
    ) {
        self.audio_encoders.insert(id, plugin);
    }

    /// Register a fusion strategy plugin
    pub fn register_fusion_strategy(
        &mut self,
        id: ModalPluginId,
        plugin: Box<dyn FusionStrategyPlugin>,
    ) {
        self.fusion_strategies.insert(id, plugin);
    }

    /// Register a preprocessing plugin
    pub fn register_preprocessor(
        &mut self,
        id: ModalPluginId,
        plugin: Box<dyn PreprocessingPlugin>,
    ) {
        self.preprocessors.insert(id, plugin);
    }

    /// Get vision encoder by ID
    pub fn get_vision_encoder(&self, id: &ModalPluginId) -> Option<&dyn VisionEncoderPlugin> {
        self.vision_encoders.get(id).map(|b| b.as_ref())
    }

    /// Get audio encoder by ID
    pub fn get_audio_encoder(&self, id: &ModalPluginId) -> Option<&dyn AudioEncoderPlugin> {
        self.audio_encoders.get(id).map(|b| b.as_ref())
    }

    /// Get fusion strategy by ID
    pub fn get_fusion_strategy(&self, id: &ModalPluginId) -> Option<&dyn FusionStrategyPlugin> {
        self.fusion_strategies.get(id).map(|b| b.as_ref())
    }

    /// Get preprocessor by ID
    pub fn get_preprocessor(&self, id: &ModalPluginId) -> Option<&dyn PreprocessingPlugin> {
        self.preprocessors.get(id).map(|b| b.as_ref())
    }

    /// List all registered vision encoders
    pub fn list_vision_encoders(&self) -> Vec<ModalPluginId> {
        self.vision_encoders.keys().cloned().collect()
    }

    /// List all registered audio encoders
    pub fn list_audio_encoders(&self) -> Vec<ModalPluginId> {
        self.audio_encoders.keys().cloned().collect()
    }

    /// List all fusion strategies
    pub fn list_fusion_strategies(&self) -> Vec<ModalPluginId> {
        self.fusion_strategies.keys().cloned().collect()
    }

    /// Load WASM plugin
    ///
    /// # Arguments
    ///
    /// * `path` - Path to WASM module
    ///
    /// # Note
    ///
    /// This would integrate with the existing WASM plugin system
    pub fn load_wasm_plugin<P: AsRef<std::path::Path>>(&mut self, _path: P) -> Result<()> {
        // Placeholder: Integration with crate::wasm_plugin
        Ok(())
    }

    /// Register built-in vision encoder plugins
    pub fn register_builtin_vision_plugins(&mut self) {
        // CLIP ViT-B/16
        self.register_vision_encoder(
            ModalPluginId::new("clip-vit-b-16"),
            Box::new(CLIPVisionEncoder {
                name: "CLIP ViT-B/16".to_string(),
                version: "1.0.0".to_string(),
                embedding_dim: 768,
                image_size: 224,
                patch_size: 16,
            }),
        );

        // CLIP ViT-L/14
        self.register_vision_encoder(
            ModalPluginId::new("clip-vit-l-14"),
            Box::new(CLIPVisionEncoder {
                name: "CLIP ViT-L/14".to_string(),
                version: "1.0.0".to_string(),
                embedding_dim: 1024,
                image_size: 224,
                patch_size: 14,
            }),
        );

        // SigLIP
        self.register_vision_encoder(
            ModalPluginId::new("siglip-so400m"),
            Box::new(SigLIPVisionEncoder {
                name: "SigLIP SO400M".to_string(),
                version: "1.0.0".to_string(),
                embedding_dim: 1152,
            }),
        );
    }

    /// Register built-in audio encoder plugins
    pub fn register_builtin_audio_plugins(&mut self) {
        // Whisper Tiny
        self.register_audio_encoder(
            ModalPluginId::new("whisper-tiny"),
            Box::new(WhisperAudioEncoder {
                name: "Whisper Tiny".to_string(),
                version: "1.0.0".to_string(),
                embedding_dim: 384,
                sample_rate: 16000,
            }),
        );

        // Whisper Base
        self.register_audio_encoder(
            ModalPluginId::new("whisper-base"),
            Box::new(WhisperAudioEncoder {
                name: "Whisper Base".to_string(),
                version: "1.0.0".to_string(),
                embedding_dim: 512,
                sample_rate: 16000,
            }),
        );
    }

    /// Register built-in fusion strategies
    pub fn register_builtin_fusion_plugins(&mut self) {
        // Concatenation fusion
        self.register_fusion_strategy(
            ModalPluginId::new("concatenate"),
            Box::new(ConcatenateFusion {
                name: "Concatenation Fusion".to_string(),
            }),
        );

        // Attention fusion
        self.register_fusion_strategy(
            ModalPluginId::new("attention"),
            Box::new(AttentionFusion {
                name: "Attention-based Fusion".to_string(),
                num_heads: 8,
            }),
        );

        // Q-Former fusion
        self.register_fusion_strategy(
            ModalPluginId::new("qformer"),
            Box::new(QFormerFusion {
                name: "Q-Former Fusion".to_string(),
                num_queries: 32,
            }),
        );
    }
}

impl Default for MultimodalPluginRegistry {
    fn default() -> Self {
        let mut registry = Self::new();

        // Register all built-in plugins
        registry.register_builtin_vision_plugins();
        registry.register_builtin_audio_plugins();
        registry.register_builtin_fusion_plugins();

        registry
    }
}

// ===== Built-in Vision Encoder Plugins =====

/// CLIP Vision Encoder
struct CLIPVisionEncoder {
    name: String,
    version: String,
    embedding_dim: usize,
    image_size: usize,
    #[allow(dead_code)]
    patch_size: usize,
}

impl VisionEncoderPlugin for CLIPVisionEncoder {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn encode(&self, _image: &Image) -> Result<Vec<f32>> {
        // Placeholder: Real CLIP encoding
        Ok(vec![0.0; self.embedding_dim])
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn supported_sizes(&self) -> Vec<(usize, usize)> {
        vec![(self.image_size, self.image_size)]
    }

    fn preprocess(&self, image: &Image) -> Result<Image> {
        let mut processed = image.clone();

        // Resize to CLIP size
        if processed.width != self.image_size || processed.height != self.image_size {
            processed.resize(self.image_size, self.image_size)?;
        }

        Ok(processed)
    }
}

/// SigLIP Vision Encoder
struct SigLIPVisionEncoder {
    name: String,
    version: String,
    embedding_dim: usize,
}

impl VisionEncoderPlugin for SigLIPVisionEncoder {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn encode(&self, _image: &Image) -> Result<Vec<f32>> {
        Ok(vec![0.0; self.embedding_dim])
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn supported_sizes(&self) -> Vec<(usize, usize)> {
        vec![(384, 384)]
    }

    fn preprocess(&self, image: &Image) -> Result<Image> {
        let mut processed = image.clone();
        processed.resize(384, 384)?;
        Ok(processed)
    }
}

// ===== Built-in Audio Encoder Plugins =====

/// Whisper Audio Encoder
struct WhisperAudioEncoder {
    name: String,
    version: String,
    embedding_dim: usize,
    sample_rate: usize,
}

impl AudioEncoderPlugin for WhisperAudioEncoder {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn encode(&self, _audio: &Audio) -> Result<Vec<f32>> {
        Ok(vec![0.0; self.embedding_dim])
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn supported_sample_rates(&self) -> Vec<usize> {
        vec![self.sample_rate]
    }

    fn preprocess(&self, audio: &Audio) -> Result<Audio> {
        let mut processed = audio.clone();

        // Resample if needed
        if processed.sample_rate != self.sample_rate {
            processed.resample(self.sample_rate)?;
        }

        Ok(processed)
    }
}

// ===== Built-in Fusion Strategy Plugins =====

/// Concatenation fusion
struct ConcatenateFusion {
    name: String,
}

impl FusionStrategyPlugin for ConcatenateFusion {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn fuse(&self, embeddings: &HashMap<String, Vec<f32>>) -> Result<Vec<f32>> {
        let mut result = Vec::new();

        // Concatenate all embeddings
        for (_modality, emb) in embeddings.iter() {
            result.extend_from_slice(emb);
        }

        Ok(result)
    }
}

/// Attention-based fusion
struct AttentionFusion {
    name: String,
    #[allow(dead_code)]
    num_heads: usize,
}

impl FusionStrategyPlugin for AttentionFusion {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn fuse(&self, embeddings: &HashMap<String, Vec<f32>>) -> Result<Vec<f32>> {
        // Placeholder: Real cross-attention fusion
        let mut result = Vec::new();
        for emb in embeddings.values() {
            result.extend_from_slice(emb);
        }
        Ok(result)
    }
}

/// Q-Former fusion (BLIP-2 style)
struct QFormerFusion {
    name: String,
    num_queries: usize,
}

impl FusionStrategyPlugin for QFormerFusion {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn fuse(&self, embeddings: &HashMap<String, Vec<f32>>) -> Result<Vec<f32>> {
        // Placeholder: Q-Former with learnable queries
        // Real implementation would use attention over queries
        let query_dim = embeddings.values().next().map(|v| v.len()).unwrap_or(768);
        Ok(vec![0.0; self.num_queries * query_dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::ImageFormat;

    #[test]
    fn test_plugin_registry_creation() {
        let registry = MultimodalPluginRegistry::new();
        assert_eq!(registry.list_vision_encoders().len(), 0);
    }

    #[test]
    fn test_builtin_plugins() {
        let registry = MultimodalPluginRegistry::default();

        assert!(registry.list_vision_encoders().len() > 0);
        assert!(registry.list_audio_encoders().len() > 0);
        assert!(registry.list_fusion_strategies().len() > 0);
    }

    #[test]
    fn test_clip_encoder() {
        let registry = MultimodalPluginRegistry::default();

        let clip_id = ModalPluginId::new("clip-vit-b-16");
        let encoder = registry.get_vision_encoder(&clip_id);

        assert!(encoder.is_some());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.name(), "CLIP ViT-B/16");
        assert_eq!(encoder.embedding_dim(), 768);
    }

    #[test]
    fn test_vision_encoding() {
        let registry = MultimodalPluginRegistry::default();

        let clip_id = ModalPluginId::new("clip-vit-b-16");
        let encoder = registry.get_vision_encoder(&clip_id).unwrap();

        let image = Image::new(vec![0; 224 * 224 * 3], 224, 224, ImageFormat::RGB);
        let embeddings = encoder.encode(&image);

        assert!(embeddings.is_ok());
        assert_eq!(embeddings.unwrap().len(), 768);
    }

    #[test]
    fn test_fusion_strategies() {
        let registry = MultimodalPluginRegistry::default();

        let fusion_id = ModalPluginId::new("concatenate");
        let fusion = registry.get_fusion_strategy(&fusion_id).unwrap();

        let mut embeddings = HashMap::new();
        embeddings.insert("vision".to_string(), vec![1.0; 768]);
        embeddings.insert("text".to_string(), vec![2.0; 512]);

        let result = fusion.fuse(&embeddings);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 768 + 512);
    }
}

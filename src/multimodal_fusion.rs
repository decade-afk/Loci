//! Multimodal Token Fusion
//!
//! This module implements advanced token-level fusion for multimodal inputs.

use crate::error::{LociError, Result};
use crate::vision_clip::ImageEmbedding;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Token type in fused sequence
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    Text(u32),
    Vision(usize),
    Audio(usize),
    Special(SpecialToken),
}

/// Special tokens
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpecialToken {
    BOS,
    EOS,
    SEP,
    PAD,
    VisionStart,
    VisionEnd,
}

/// Fused token sequence
#[derive(Clone)]
pub struct FusedTokenSequence {
    tokens: Vec<TokenType>,
    vision_embeddings: Option<Arc<ImageEmbedding>>,
    text_embeddings: Option<Arc<[f32]>>,
    embedding_dim: usize,
}

impl FusedTokenSequence {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            tokens: Vec::new(),
            vision_embeddings: None,
            text_embeddings: None,
            embedding_dim,
        }
    }

    pub fn push_text(&mut self, token_id: u32) {
        self.tokens.push(TokenType::Text(token_id));
    }

    pub fn push_vision(&mut self, patch_idx: usize) {
        self.tokens.push(TokenType::Vision(patch_idx));
    }

    pub fn push_special(&mut self, special: SpecialToken) {
        self.tokens.push(TokenType::Special(special));
    }

    pub fn set_vision_embeddings(&mut self, embeddings: ImageEmbedding) {
        self.vision_embeddings = Some(Arc::new(embeddings));
    }

    pub fn token_at(&self, idx: usize) -> Option<&TokenType> {
        self.tokens.get(idx)
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn tokens(&self) -> &[TokenType] {
        &self.tokens
    }

    pub fn to_embeddings(&self) -> Vec<f32> {
        let mut embeddings = Vec::with_capacity(self.tokens.len() * self.embedding_dim);
        for _token in &self.tokens {
            embeddings.extend_from_slice(&vec![0.0; self.embedding_dim]);
        }
        embeddings
    }
}

/// Fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub strategy: FusionStrategyType,
    pub vision_position: VisionPosition,
    pub use_cls_token: bool,
    pub max_vision_tokens: usize,
    pub embedding_dim: usize,
    pub add_vision_markers: bool,
}

/// Fusion strategy type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusionStrategyType {
    DirectInject,
    LinearProjection,
    CrossAttention,
    QFormer,
    Perceiver,
}

/// Vision token position in sequence
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VisionPosition {
    Prefix,
    Suffix,
    Interleaved(usize),
    Replace(usize, usize),
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategyType::DirectInject,
            vision_position: VisionPosition::Prefix,
            use_cls_token: true,
            max_vision_tokens: 256,
            embedding_dim: 1024,
            add_vision_markers: true,
        }
    }
}

/// Multimodal fusion manager
pub struct MultimodalFusion {
    config: FusionConfig,
    projection_weights: Option<Vec<f32>>,
}

impl MultimodalFusion {
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            projection_weights: None,
        }
    }

    pub fn fuse_with_vision(
        &self,
        text_tokens: Vec<u32>,
        image_embedding: ImageEmbedding,
    ) -> Result<FusedTokenSequence> {
        let mut fused = FusedTokenSequence::new(self.config.embedding_dim);

        fused.push_special(SpecialToken::BOS);

        match self.config.vision_position {
            VisionPosition::Prefix => {
                self.inject_vision_prefix(&mut fused, &image_embedding)?;
                self.inject_text(&mut fused, &text_tokens)?;
            }
            VisionPosition::Suffix => {
                self.inject_text(&mut fused, &text_tokens)?;
                self.inject_vision_suffix(&mut fused, &image_embedding)?;
            }
            VisionPosition::Interleaved(pos) => {
                self.inject_interleaved(&mut fused, &text_tokens, &image_embedding, pos)?;
            }
            VisionPosition::Replace(start, end) => {
                self.inject_replace(&mut fused, &text_tokens, &image_embedding, start, end)?;
            }
        }

        fused.set_vision_embeddings(image_embedding);
        Ok(fused)
    }

    fn inject_vision_prefix(
        &self,
        fused: &mut FusedTokenSequence,
        image_embedding: &ImageEmbedding,
    ) -> Result<()> {
        if self.config.add_vision_markers {
            fused.push_special(SpecialToken::VisionStart);
        }

        if self.config.use_cls_token {
            fused.push_vision(0);
        }

        let num_patches = std::cmp::min(
            image_embedding.seq_len() - 1,
            self.config.max_vision_tokens,
        );

        for i in 0..num_patches {
            fused.push_vision(i + 1);
        }

        if self.config.add_vision_markers {
            fused.push_special(SpecialToken::VisionEnd);
        }

        Ok(())
    }

    fn inject_vision_suffix(
        &self,
        fused: &mut FusedTokenSequence,
        image_embedding: &ImageEmbedding,
    ) -> Result<()> {
        if self.config.add_vision_markers {
            fused.push_special(SpecialToken::VisionStart);
        }

        if self.config.use_cls_token {
            fused.push_vision(0);
        }

        let num_patches = std::cmp::min(
            image_embedding.seq_len() - 1,
            self.config.max_vision_tokens,
        );

        for i in 0..num_patches {
            fused.push_vision(i + 1);
        }

        if self.config.add_vision_markers {
            fused.push_special(SpecialToken::VisionEnd);
        }

        Ok(())
    }

    fn inject_interleaved(
        &self,
        fused: &mut FusedTokenSequence,
        text_tokens: &[u32],
        image_embedding: &ImageEmbedding,
        position: usize,
    ) -> Result<()> {
        for &token_id in text_tokens.iter().take(position) {
            fused.push_text(token_id);
        }

        self.inject_vision_prefix(fused, image_embedding)?;

        for &token_id in text_tokens.iter().skip(position) {
            fused.push_text(token_id);
        }

        Ok(())
    }

    fn inject_replace(
        &self,
        fused: &mut FusedTokenSequence,
        text_tokens: &[u32],
        image_embedding: &ImageEmbedding,
        start: usize,
        end: usize,
    ) -> Result<()> {
        for &token_id in text_tokens.iter().take(start) {
            fused.push_text(token_id);
        }

        self.inject_vision_prefix(fused, image_embedding)?;

        for &token_id in text_tokens.iter().skip(end) {
            fused.push_text(token_id);
        }

        Ok(())
    }

    fn inject_text(&self, fused: &mut FusedTokenSequence, text_tokens: &[u32]) -> Result<()> {
        for &token_id in text_tokens {
            fused.push_text(token_id);
        }
        Ok(())
    }

    pub fn get_stats(&self, fused: &FusedTokenSequence) -> FusionStats {
        let mut text_count = 0;
        let mut vision_count = 0;
        let mut special_count = 0;

        for token in fused.tokens() {
            match token {
                TokenType::Text(_) => text_count += 1,
                TokenType::Vision(_) => vision_count += 1,
                TokenType::Special(_) => special_count += 1,
                _ => {}
            }
        }

        FusionStats {
            total_tokens: fused.len(),
            text_tokens: text_count,
            vision_tokens: vision_count,
            special_tokens: special_count,
            embedding_dim: self.config.embedding_dim,
        }
    }
}

/// Fusion statistics
#[derive(Debug, Clone)]
pub struct FusionStats {
    pub total_tokens: usize,
    pub text_tokens: usize,
    pub vision_tokens: usize,
    pub special_tokens: usize,
    pub embedding_dim: usize,
}

impl FusionStats {
    pub fn vision_ratio(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.vision_tokens as f32 / self.total_tokens as f32
        }
    }

    pub fn text_ratio(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.text_tokens as f32 / self.total_tokens as f32
        }
    }
}
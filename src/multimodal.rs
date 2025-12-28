/**
 * Loci Phase 4 Week 3-4: 多模态支持（Vision）
 *
 * 核心特性：
 * 1. Vision Encoder 接口（CLIP/SigLIP）
 * 2. CLIP ViT-L/14@336 实现
 * 3. 图像预处理 pipeline
 * 4. 多模态 KV Cache 扩展
 * 5. 图像 + 文本混合推理
 *
 * 架构：
 * - 图像 → Resize → Normalize → Patch Extraction → ViT Encoder → Embedding
 * - Embedding → Projector → 注入 KV Cache → 文本生成
 */

use anyhow::{Result, bail};
use std::path::Path;

// ==================== 图像数据类型 ====================

/// RGB 图像缓冲区
#[derive(Debug, Clone)]
pub struct ImageBuffer {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,  // RGB 格式，每像素 3 字节
}

impl ImageBuffer {
    /// 从文件加载图像
    pub fn from_file(path: &Path) -> Result<Self> {
        // TODO: 使用 image crate 加载图像
        // let img = image::open(path)?;
        // let rgb = img.to_rgb8();

        bail!("Image loading requires 'image' crate dependency");
    }

    /// 创建空白图像
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0u8; (width * height * 3) as usize],
        }
    }

    /// 获取像素值（RGB）
    pub fn get_pixel(&self, x: u32, y: u32) -> (u8, u8, u8) {
        let idx = ((y * self.width + x) * 3) as usize;
        (self.data[idx], self.data[idx + 1], self.data[idx + 2])
    }

    /// 设置像素值（RGB）
    pub fn set_pixel(&mut self, x: u32, y: u32, rgb: (u8, u8, u8)) {
        let idx = ((y * self.width + x) * 3) as usize;
        self.data[idx] = rgb.0;
        self.data[idx + 1] = rgb.1;
        self.data[idx + 2] = rgb.2;
    }
}

// ==================== Tensor 类型 ====================

/// 简单的 Tensor 类型（用于图像处理）
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; size],
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ==================== Vision Encoder 接口 ====================

/// Vision Encoder 接口（统一抽象）
pub trait VisionEncoder: Send + Sync {
    /// 编码图像为 embedding
    fn encode_image(&self, image: &ImageBuffer) -> Result<Vec<f32>>;

    /// Embedding 维度
    fn embedding_dim(&self) -> usize;

    /// 支持的图像尺寸
    fn supported_sizes(&self) -> Vec<(u32, u32)>;

    /// 模型名称
    fn model_name(&self) -> &str;
}

// ==================== CLIP ViT-L/14 实现 ====================

/// CLIP ViT-L/14@336 Vision Encoder
pub struct CLIPVisionEncoder {
    /// 图像尺寸（CLIP ViT-L/14@336 使用 336×336）
    image_size: u32,

    /// Patch 大小（ViT-L/14 使用 14×14）
    patch_size: u32,

    /// Embedding 维度（ViT-L 为 1024）
    embedding_dim: usize,

    /// ImageNet 归一化参数（mean）
    normalize_mean: [f32; 3],

    /// ImageNet 归一化参数（std）
    normalize_std: [f32; 3],

    // TODO: 实际的 ViT 模型权重
    // model_weights: GGUFModel,
}

impl CLIPVisionEncoder {
    /// 创建新的 CLIP encoder
    pub fn new() -> Self {
        Self {
            image_size: 336,
            patch_size: 14,
            embedding_dim: 1024,
            // ImageNet 归一化参数
            normalize_mean: [0.48145466, 0.4578275, 0.40821073],
            normalize_std: [0.26862954, 0.26130258, 0.27577711],
        }
    }

    /// 预处理图像
    ///
    /// 步骤：
    /// 1. Resize 到 336×336
    /// 2. 归一化（ImageNet mean/std）
    /// 3. 转换为 Tensor（C×H×W 格式）
    pub fn preprocess_image(&self, image: &ImageBuffer) -> Result<Tensor> {
        // 1. Resize（简化实现：使用最近邻插值）
        let resized = self.resize_image(image, self.image_size, self.image_size)?;

        // 2. 归一化并转换为 Tensor
        let mut tensor_data = Vec::with_capacity((3 * self.image_size * self.image_size) as usize);

        for c in 0..3 {
            for y in 0..self.image_size {
                for x in 0..self.image_size {
                    let (r, g, b) = resized.get_pixel(x, y);
                    let pixel = match c {
                        0 => r,
                        1 => g,
                        2 => b,
                        _ => unreachable!(),
                    };

                    // 归一化: (pixel/255 - mean) / std
                    let normalized = ((pixel as f32 / 255.0) - self.normalize_mean[c]) / self.normalize_std[c];
                    tensor_data.push(normalized);
                }
            }
        }

        Ok(Tensor::from_vec(
            tensor_data,
            &[3, self.image_size as usize, self.image_size as usize],
        ))
    }

    /// Resize 图像（最近邻插值）
    fn resize_image(&self, image: &ImageBuffer, new_width: u32, new_height: u32) -> Result<ImageBuffer> {
        let mut resized = ImageBuffer::new(new_width, new_height);

        let x_ratio = image.width as f32 / new_width as f32;
        let y_ratio = image.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * x_ratio) as u32;
                let src_y = (y as f32 * y_ratio) as u32;

                let pixel = image.get_pixel(src_x, src_y);
                resized.set_pixel(x, y, pixel);
            }
        }

        Ok(resized)
    }

    /// 提取 patches（将图像分割为 14×14 的 patches）
    pub fn extract_patches(&self, tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
        assert_eq!(tensor.shape[0], 3);  // RGB
        assert_eq!(tensor.shape[1] as u32, self.image_size);
        assert_eq!(tensor.shape[2] as u32, self.image_size);

        let num_patches_per_side = self.image_size / self.patch_size;
        let patch_dim = (3 * self.patch_size * self.patch_size) as usize;

        let mut patches = Vec::new();

        for patch_y in 0..num_patches_per_side {
            for patch_x in 0..num_patches_per_side {
                let mut patch = Vec::with_capacity(patch_dim);

                for c in 0..3 {
                    for py in 0..self.patch_size {
                        for px in 0..self.patch_size {
                            let y = patch_y * self.patch_size + py;
                            let x = patch_x * self.patch_size + px;

                            let idx = (c * self.image_size * self.image_size
                                + y * self.image_size
                                + x) as usize;

                            patch.push(tensor.data[idx]);
                        }
                    }
                }

                patches.push(patch);
            }
        }

        Ok(patches)
    }

    /// ViT Forward Pass（简化实现）
    ///
    /// 真实实现需要：
    /// - Patch Embedding
    /// - Position Embedding
    /// - Transformer Encoder (12 layers for ViT-L)
    /// - Layer Normalization
    /// - CLS Token Pooling
    pub fn forward(&self, patches: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        // TODO: 实现完整的 ViT forward pass
        // 当前返回 mock embedding

        println!("[CLIP] Processing {} patches", patches.len());
        println!("[CLIP] Patch dimension: {}", patches[0].len());

        // Mock: 返回随机 embedding
        Ok(vec![0.1; self.embedding_dim])
    }

    /// Pooling（从 patch embeddings 中提取全局特征）
    ///
    /// CLIP 使用 CLS token 的输出
    pub fn pool_embeddings(&self, embeddings: Vec<f32>) -> Vec<f32> {
        // 简化：直接返回（实际应提取 CLS token）
        embeddings
    }
}

impl Default for CLIPVisionEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionEncoder for CLIPVisionEncoder {
    fn encode_image(&self, image: &ImageBuffer) -> Result<Vec<f32>> {
        // 1. 预处理
        let tensor = self.preprocess_image(image)?;
        println!("[CLIP] Preprocessed tensor shape: {:?}", tensor.shape);

        // 2. 提取 patches
        let patches = self.extract_patches(&tensor)?;
        println!("[CLIP] Extracted {} patches", patches.len());

        // 3. ViT Forward Pass
        let embeddings = self.forward(patches)?;

        // 4. Pooling
        let pooled = self.pool_embeddings(embeddings);

        Ok(pooled)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn supported_sizes(&self) -> Vec<(u32, u32)> {
        vec![(336, 336)]
    }

    fn model_name(&self) -> &str {
        "CLIP ViT-L/14@336"
    }
}

// ==================== Token 类型扩展 ====================

/// Token 类型（扩展支持图像 token）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// 文本 token
    Text,

    /// 图像 token（关联图像 ID）
    Image { image_id: usize },
}

/// 带类型的 Token
#[derive(Debug, Clone)]
pub struct TypedToken {
    pub id: u32,
    pub token_type: TokenType,
}

// ==================== 多模态 KV Cache ====================

/// 多模态 KV Cache
///
/// 扩展标准 KV Cache 以支持图像 token
pub struct MultimodalKVCache {
    /// 文本 token 的 KV pairs
    pub text_cache: Vec<(Vec<f32>, Vec<f32>)>,  // (Key, Value)

    /// 图像 token 的 KV pairs
    pub image_cache: Vec<(Vec<f32>, Vec<f32>)>,

    /// Token 类型序列（记录每个位置的 token 类型）
    pub token_types: Vec<TokenType>,
}

impl MultimodalKVCache {
    pub fn new() -> Self {
        Self {
            text_cache: Vec::new(),
            image_cache: Vec::new(),
            token_types: Vec::new(),
        }
    }

    /// 添加文本 token 的 KV
    pub fn append_text(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.text_cache.push((key, value));
        self.token_types.push(TokenType::Text);
    }

    /// 添加图像 token 的 KV
    pub fn append_image(&mut self, key: Vec<f32>, value: Vec<f32>, image_id: usize) {
        self.image_cache.push((key, value));
        self.token_types.push(TokenType::Image { image_id });
    }

    /// 获取指定位置的 KV（根据 token 类型）
    pub fn get(&self, pos: usize) -> Option<(&Vec<f32>, &Vec<f32>)> {
        if pos >= self.token_types.len() {
            return None;
        }

        match self.token_types[pos] {
            TokenType::Text => {
                let text_idx = self.token_types[..=pos]
                    .iter()
                    .filter(|t| **t == TokenType::Text)
                    .count() - 1;
                self.text_cache.get(text_idx).map(|(k, v)| (k, v))
            }
            TokenType::Image { .. } => {
                let image_idx = self.token_types[..=pos]
                    .iter()
                    .filter(|t| matches!(t, TokenType::Image { .. }))
                    .count() - 1;
                self.image_cache.get(image_idx).map(|(k, v)| (k, v))
            }
        }
    }

    /// 获取总 token 数
    pub fn len(&self) -> usize {
        self.token_types.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.token_types.is_empty()
    }
}

impl Default for MultimodalKVCache {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_buffer_creation() {
        let img = ImageBuffer::new(100, 100);
        assert_eq!(img.width, 100);
        assert_eq!(img.height, 100);
        assert_eq!(img.data.len(), 100 * 100 * 3);
    }

    #[test]
    fn test_clip_encoder_creation() {
        let encoder = CLIPVisionEncoder::new();
        assert_eq!(encoder.embedding_dim(), 1024);
        assert_eq!(encoder.model_name(), "CLIP ViT-L/14@336");
    }

    #[test]
    fn test_image_preprocessing() {
        let encoder = CLIPVisionEncoder::new();
        let img = ImageBuffer::new(224, 224);

        let tensor = encoder.preprocess_image(&img).unwrap();
        assert_eq!(tensor.shape, vec![3, 336, 336]);
    }

    #[test]
    fn test_patch_extraction() {
        let encoder = CLIPVisionEncoder::new();
        let tensor = Tensor::new(vec![3, 336, 336]);

        let patches = encoder.extract_patches(&tensor).unwrap();

        let expected_patches = (336 / 14) * (336 / 14);
        assert_eq!(patches.len(), expected_patches);
        assert_eq!(patches[0].len(), 3 * 14 * 14);
    }

    #[test]
    fn test_multimodal_kv_cache() {
        let mut cache = MultimodalKVCache::new();

        cache.append_text(vec![1.0], vec![2.0]);
        cache.append_image(vec![3.0], vec![4.0], 0);
        cache.append_text(vec![5.0], vec![6.0]);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.token_types[0], TokenType::Text);
        assert_eq!(cache.token_types[1], TokenType::Image { image_id: 0 });
        assert_eq!(cache.token_types[2], TokenType::Text);
    }
}

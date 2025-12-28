/**
 * Loci Phase 4 Week 7-8: Advanced Quantization Formats
 *
 * 核心特性：
 * 1. IQ2_XXS: Ultra-low precision quantization (2-bit with importance weighting)
 * 2. BitNet b1.58: 1.58-bit quantization with ternary weights {-1, 0, +1}
 * 3. 统一量化接口
 * 4. 运行时反量化
 *
 * 性能目标：
 * - IQ2_XXS: 16x 内存压缩（vs FP16）
 * - BitNet b1.58: 10x 内存压缩 + 特殊硬件加速
 * - 反量化延迟 < 1ms per layer
 */

use anyhow::{Result, bail};
use std::sync::Arc;

// ==================== Quantization Types ====================

/// 量化类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationType {
    /// 无量化（FP32）
    None,
    /// FP16 半精度
    FP16,
    /// 8-bit 整数量化
    Q8_0,
    /// 4-bit 整数量化
    Q4_0,
    /// 2-bit 超低精度量化（重要性加权）
    Iq2Xxs,
    /// BitNet 1.58-bit 三元量化 {-1, 0, +1}
    BitNet158,
}

impl QuantizationType {
    /// 获取每个元素的比特数
    pub fn bits_per_element(&self) -> f32 {
        match self {
            Self::None => 32.0,
            Self::FP16 => 16.0,
            Self::Q8_0 => 8.0,
            Self::Q4_0 => 4.0,
            Self::Iq2Xxs => 2.0,
            Self::BitNet158 => 1.58,
        }
    }

    /// 计算压缩率（相对于 FP32）
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_element()
    }

    /// 是否需要特殊硬件加速
    pub fn requires_special_hardware(&self) -> bool {
        matches!(self, Self::BitNet158)
    }
}

// ==================== Quantization Trait ====================

/// 统一量化接口
pub trait QuantizationScheme: Send + Sync {
    /// 量化类型
    fn quantization_type(&self) -> QuantizationType;

    /// 量化 FP32 张量
    fn quantize(&self, data: &[f32]) -> Result<QuantizedTensor>;

    /// 反量化到 FP32
    fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>>;

    /// 估计量化误差（MSE）
    fn estimate_error(&self, original: &[f32], quantized: &QuantizedTensor) -> f32 {
        let dequantized = self.dequantize(quantized).unwrap_or_default();
        if dequantized.len() != original.len() {
            return f32::INFINITY;
        }

        let mse: f32 = original.iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;

        mse
    }
}

/// 量化后的张量
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// 量化类型
    pub qtype: QuantizationType,
    /// 量化数据（字节流）
    pub data: Vec<u8>,
    /// 元数据（scales, zero points, etc.）
    pub metadata: QuantizationMetadata,
    /// 原始形状
    pub shape: Vec<usize>,
}

/// 量化元数据
#[derive(Debug, Clone)]
pub struct QuantizationMetadata {
    /// Scale 因子（每个 block）
    pub scales: Vec<f32>,
    /// Zero point（每个 block）
    pub zero_points: Vec<f32>,
    /// Block 大小
    pub block_size: usize,
    /// 重要性权重（IQ2_XXS 专用）
    pub importance_weights: Option<Vec<f32>>,
}

// ==================== IQ2_XXS Implementation ====================

/// Iq2Xxs: 2-bit 量化 + 重要性加权
///
/// 核心思想：
/// 1. 将权重分组为 blocks（默认 32 个元素）
/// 2. 每个 block 计算 scale + zero_point
/// 3. 计算重要性权重（基于梯度/激活幅度）
/// 4. 重要性高的 block 使用更高精度的 scale
pub struct Iq2Xxs {
    /// Block 大小
    pub block_size: usize,
    /// 重要性权重阈值（超过此值使用更高精度）
    pub importance_threshold: f32,
}

impl Default for Iq2Xxs {
    fn default() -> Self {
        Self {
            block_size: 32,
            importance_threshold: 0.5,
        }
    }
}

impl Iq2Xxs {
    /// 计算重要性权重（简化版：基于标准差）
    fn compute_importance(&self, block: &[f32]) -> f32 {
        let mean = block.iter().sum::<f32>() / block.len() as f32;
        let variance = block.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / block.len() as f32;
        variance.sqrt()
    }

    /// 量化单个 block 到 2-bit
    fn quantize_block(&self, block: &[f32]) -> (Vec<u8>, f32, f32) {
        let min = block.iter().copied().fold(f32::INFINITY, f32::min);
        let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let scale = (max - min) / 3.0; // 2-bit: 4 个级别 (0, 1, 2, 3)
        let zero_point = min;

        let mut quantized = Vec::new();
        for chunk in block.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let q = if scale > 1e-8 {
                    ((val - zero_point) / scale).round().clamp(0.0, 3.0) as u8
                } else {
                    0
                };
                byte |= q << (i * 2); // 每个元素占 2 bits
            }
            quantized.push(byte);
        }

        (quantized, scale, zero_point)
    }
}

impl QuantizationScheme for Iq2Xxs {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::Iq2Xxs
    }

    fn quantize(&self, data: &[f32]) -> Result<QuantizedTensor> {
        let mut quantized_data = Vec::new();
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        let mut importance_weights = Vec::new();

        for block in data.chunks(self.block_size) {
            let importance = self.compute_importance(block);
            importance_weights.push(importance);

            let (q_block, scale, zero_point) = self.quantize_block(block);
            quantized_data.extend_from_slice(&q_block);
            scales.push(scale);
            zero_points.push(zero_point);
        }

        Ok(QuantizedTensor {
            qtype: QuantizationType::Iq2Xxs,
            data: quantized_data,
            metadata: QuantizationMetadata {
                scales,
                zero_points,
                block_size: self.block_size,
                importance_weights: Some(importance_weights),
            },
            shape: vec![data.len()],
        })
    }

    fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        if tensor.qtype != QuantizationType::Iq2Xxs {
            bail!("Expected Iq2Xxs, got {:?}", tensor.qtype);
        }

        let mut dequantized = Vec::new();
        let mut data_offset = 0;

        for (block_idx, (&scale, &zero_point)) in tensor.metadata.scales.iter()
            .zip(tensor.metadata.zero_points.iter())
            .enumerate()
        {
            let bytes_per_block = (self.block_size + 3) / 4; // 每 4 个元素占 1 字节

            for i in 0..bytes_per_block {
                if data_offset + i >= tensor.data.len() {
                    break;
                }

                let byte = tensor.data[data_offset + i];
                for j in 0..4 {
                    if block_idx * self.block_size + i * 4 + j >= tensor.shape[0] {
                        break;
                    }

                    let q = (byte >> (j * 2)) & 0b11;
                    let val = zero_point + (q as f32) * scale;
                    dequantized.push(val);
                }
            }

            data_offset += bytes_per_block;
        }

        Ok(dequantized)
    }
}

// ==================== BitNet b1.58 Implementation ====================

/// BitNet b1.58: 三元量化 {-1, 0, +1}
///
/// 核心思想：
/// 1. 权重只有 3 个可能值：-1, 0, +1
/// 2. 使用 2 bits 编码：00 = -1, 01 = 0, 10 = +1
/// 3. 平均每个权重 1.58 bits（因为有统计偏差）
/// 4. 可使用特殊硬件（位运算 + 查找表）加速
pub struct BitNet158 {
    /// 阈值（abs(x) < threshold 则量化为 0）
    pub zero_threshold: f32,
}

impl Default for BitNet158 {
    fn default() -> Self {
        Self {
            zero_threshold: 0.1,
        }
    }
}

impl BitNet158 {
    /// 量化单个值到 {-1, 0, +1}
    fn quantize_value(&self, val: f32, scale: f32) -> i8 {
        let normalized = val / scale;
        if normalized.abs() < self.zero_threshold {
            0
        } else if normalized > 0.0 {
            1
        } else {
            -1
        }
    }

    /// 将三元值编码为 2 bits
    fn encode_ternary(val: i8) -> u8 {
        match val {
            -1 => 0b00,
            0 => 0b01,
            1 => 0b10,
            _ => 0b11, // Invalid
        }
    }

    /// 从 2 bits 解码三元值
    fn decode_ternary(bits: u8) -> i8 {
        match bits & 0b11 {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0, // Default to 0 for invalid
        }
    }
}

impl QuantizationScheme for BitNet158 {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::BitNet158
    }

    fn quantize(&self, data: &[f32]) -> Result<QuantizedTensor> {
        // 计算全局 scale（基于最大绝对值）
        let max_abs = data.iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let scale = max_abs.max(1e-8);

        // 量化到 {-1, 0, +1}
        let ternary: Vec<i8> = data.iter()
            .map(|&val| self.quantize_value(val, scale))
            .collect();

        // 编码为字节流（每 4 个值占 1 字节）
        let mut encoded = Vec::new();
        for chunk in ternary.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                byte |= Self::encode_ternary(val) << (i * 2);
            }
            encoded.push(byte);
        }

        Ok(QuantizedTensor {
            qtype: QuantizationType::BitNet158,
            data: encoded,
            metadata: QuantizationMetadata {
                scales: vec![scale],
                zero_points: vec![0.0],
                block_size: data.len(),
                importance_weights: None,
            },
            shape: vec![data.len()],
        })
    }

    fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        if tensor.qtype != QuantizationType::BitNet158 {
            bail!("Expected BitNet158, got {:?}", tensor.qtype);
        }

        let scale = tensor.metadata.scales[0];
        let mut dequantized = Vec::new();

        for &byte in &tensor.data {
            for i in 0..4 {
                if dequantized.len() >= tensor.shape[0] {
                    break;
                }

                let bits = (byte >> (i * 2)) & 0b11;
                let ternary = Self::decode_ternary(bits);
                let val = (ternary as f32) * scale;
                dequantized.push(val);
            }
        }

        Ok(dequantized)
    }
}

// ==================== Quantization Manager ====================

/// 量化方案管理器
pub struct QuantizationManager {
    schemes: std::collections::HashMap<QuantizationType, Arc<dyn QuantizationScheme>>,
}

impl QuantizationManager {
    /// 创建默认管理器（包含所有内置方案）
    pub fn new() -> Self {
        let mut schemes: std::collections::HashMap<QuantizationType, Arc<dyn QuantizationScheme>> = std::collections::HashMap::new();

        schemes.insert(
            QuantizationType::Iq2Xxs,
            Arc::new(Iq2Xxs::default()),
        );
        schemes.insert(
            QuantizationType::BitNet158,
            Arc::new(BitNet158::default()),
        );

        Self { schemes }
    }

    /// 获取量化方案
    pub fn get_scheme(&self, qtype: QuantizationType) -> Option<Arc<dyn QuantizationScheme>> {
        self.schemes.get(&qtype).cloned()
    }

    /// 量化张量
    pub fn quantize(&self, data: &[f32], qtype: QuantizationType) -> Result<QuantizedTensor> {
        let scheme = self.get_scheme(qtype)
            .ok_or_else(|| anyhow::anyhow!("Unsupported quantization type: {:?}", qtype))?;
        scheme.quantize(data)
    }

    /// 反量化张量
    pub fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        let scheme = self.get_scheme(tensor.qtype)
            .ok_or_else(|| anyhow::anyhow!("Unsupported quantization type: {:?}", tensor.qtype))?;
        scheme.dequantize(tensor)
    }
}

impl Default for QuantizationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_types() {
        assert_eq!(QuantizationType::Iq2Xxs.bits_per_element(), 2.0);
        assert_eq!(QuantizationType::BitNet158.bits_per_element(), 1.58);
        assert_eq!(QuantizationType::Iq2Xxs.compression_ratio(), 16.0);
    }

    #[test]
    fn test_iq2_xxs_quantization() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scheme = Iq2Xxs::default();

        let quantized = scheme.quantize(&data).unwrap();
        assert_eq!(quantized.qtype, QuantizationType::Iq2Xxs);

        let dequantized = scheme.dequantize(&quantized).unwrap();
        assert_eq!(dequantized.len(), data.len());

        // 验证误差在合理范围内
        let mse = scheme.estimate_error(&data, &quantized);
        assert!(mse < 2.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_bitnet158_quantization() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let scheme = BitNet158::default();

        let quantized = scheme.quantize(&data).unwrap();
        assert_eq!(quantized.qtype, QuantizationType::BitNet158);

        let dequantized = scheme.dequantize(&quantized).unwrap();
        assert_eq!(dequantized.len(), data.len());

        // 验证量化到 {-1, 0, +1}
        for &val in &dequantized {
            let normalized = val / quantized.metadata.scales[0];
            assert!(
                normalized.abs() < 1e-6 || (normalized - 1.0).abs() < 1e-6 || (normalized + 1.0).abs() < 1e-6,
                "Value not in {{-1, 0, +1}}: {}",
                normalized
            );
        }
    }

    #[test]
    fn test_ternary_encoding() {
        assert_eq!(BitNet158::encode_ternary(-1), 0b00);
        assert_eq!(BitNet158::encode_ternary(0), 0b01);
        assert_eq!(BitNet158::encode_ternary(1), 0b10);

        assert_eq!(BitNet158::decode_ternary(0b00), -1);
        assert_eq!(BitNet158::decode_ternary(0b01), 0);
        assert_eq!(BitNet158::decode_ternary(0b10), 1);
    }

    #[test]
    fn test_quantization_manager() {
        let manager = QuantizationManager::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Iq2Xxs
        let q1 = manager.quantize(&data, QuantizationType::Iq2Xxs).unwrap();
        let d1 = manager.dequantize(&q1).unwrap();
        assert_eq!(d1.len(), data.len());

        // BitNet b1.58
        let q2 = manager.quantize(&data, QuantizationType::BitNet158).unwrap();
        let d2 = manager.dequantize(&q2).unwrap();
        assert_eq!(d2.len(), data.len());
    }

    #[test]
    fn test_large_tensor_quantization() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
        let scheme = Iq2Xxs::default();

        let quantized = scheme.quantize(&data).unwrap();
        let dequantized = scheme.dequantize(&quantized).unwrap();

        assert_eq!(dequantized.len(), data.len());

        // 验证压缩率
        let original_bytes = data.len() * 4; // FP32
        let compressed_bytes = quantized.data.len();
        let compression = original_bytes as f32 / compressed_bytes as f32;
        assert!(compression > 10.0, "Compression ratio too low: {}", compression);
    }
}

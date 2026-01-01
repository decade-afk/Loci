//! Quantization Module
//!
//! This module provides core functionality for the Loci project.
//!


use anyhow::{Result, bail};
use std::sync::Arc;




#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// QuantizationType enumeration
pub enum QuantizationType {
    
    None,
    
    FP16,
    
    Q8_0,
    
    Q4_0,
    
    Iq2Xxs,
    
    BitNet158,
}

// Implementation for QuantizationType
impl QuantizationType {
    
    /// bits_per_element function
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

    
    /// compression_ratio function
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_element()
    }

    
    /// requires_special_hardware function
    pub fn requires_special_hardware(&self) -> bool {
        matches!(self, Self::BitNet158)
    }
}




pub trait QuantizationScheme: Send + Sync {
    
    fn quantization_type(&self) -> QuantizationType;

    
    fn quantize(&self, data: &[f32]) -> Result<QuantizedTensor>;

    
    fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>>;

    
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


#[derive(Debug, Clone)]
    /// QuantizedTensor structure
pub struct QuantizedTensor {
    
    pub qtype: QuantizationType,
    
    pub data: Vec<u8>,
    
    pub metadata: QuantizationMetadata,
    
    pub shape: Vec<usize>,
}


#[derive(Debug, Clone)]
    /// QuantizationMetadata structure
pub struct QuantizationMetadata {
    
    pub scales: Vec<f32>,
    
    pub zero_points: Vec<f32>,
    
    pub block_size: usize,
    
    pub importance_weights: Option<Vec<f32>>,
}










    /// Iq2Xxs structure
pub struct Iq2Xxs {
    
    pub block_size: usize,
    
    pub importance_threshold: f32,
}

// Implementation for Default
impl Default for Iq2Xxs {
    fn default() -> Self {
        Self {
            block_size: 32,
            importance_threshold: 0.5,
        }
    }
}

// Implementation for Iq2Xxs
impl Iq2Xxs {
    
    fn compute_importance(&self, block: &[f32]) -> f32 {
        let mean = block.iter().sum::<f32>() / block.len() as f32;
        let variance = block.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / block.len() as f32;
        variance.sqrt()
    }

    
    fn quantize_block(&self, block: &[f32]) -> (Vec<u8>, f32, f32) {
        let min = block.iter().copied().fold(f32::INFINITY, f32::min);
        let max = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let scale = (max - min) / 3.0; 
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
                byte |= q << (i * 2); 
            }
            quantized.push(byte);
        }

        (quantized, scale, zero_point)
    }
}

// Implementation for QuantizationScheme
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
            let bytes_per_block = (self.block_size + 3) / 4; 

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










    /// BitNet158 structure
pub struct BitNet158 {
    
    pub zero_threshold: f32,
}

// Implementation for Default
impl Default for BitNet158 {
    fn default() -> Self {
        Self {
            zero_threshold: 0.1,
        }
    }
}

// Implementation for BitNet158
impl BitNet158 {
    
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

    
    fn encode_ternary(val: i8) -> u8 {
        match val {
            -1 => 0b00,
            0 => 0b01,
            1 => 0b10,
            _ => 0b11, 
        }
    }

    
    fn decode_ternary(bits: u8) -> i8 {
        match bits & 0b11 {
            0b00 => -1,
            0b01 => 0,
            0b10 => 1,
            _ => 0, 
        }
    }
}

// Implementation for QuantizationScheme
impl QuantizationScheme for BitNet158 {
    fn quantization_type(&self) -> QuantizationType {
        QuantizationType::BitNet158
    }

    fn quantize(&self, data: &[f32]) -> Result<QuantizedTensor> {
        
        let max_abs = data.iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let scale = max_abs.max(1e-8);

        
        let ternary: Vec<i8> = data.iter()
            .map(|&val| self.quantize_value(val, scale))
            .collect();

        
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




    /// QuantizationManager structure
pub struct QuantizationManager {
    schemes: std::collections::HashMap<QuantizationType, Arc<dyn QuantizationScheme>>,
}

// Implementation for QuantizationManager
impl QuantizationManager {
    
    /// new function
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

    
    /// get_scheme function
    pub fn get_scheme(&self, qtype: QuantizationType) -> Option<Arc<dyn QuantizationScheme>> {
        self.schemes.get(&qtype).cloned()
    }

    
    /// quantize function
    pub fn quantize(&self, data: &[f32], qtype: QuantizationType) -> Result<QuantizedTensor> {
        let scheme = self.get_scheme(qtype)
            .ok_or_else(|| anyhow::anyhow!("Unsupported quantization type: {:?}", qtype))?;
        scheme.quantize(data)
    }

    
    /// dequantize function
    pub fn dequantize(&self, tensor: &QuantizedTensor) -> Result<Vec<f32>> {
        let scheme = self.get_scheme(tensor.qtype)
            .ok_or_else(|| anyhow::anyhow!("Unsupported quantization type: {:?}", tensor.qtype))?;
        scheme.dequantize(tensor)
    }
}

// Implementation for Default
impl Default for QuantizationManager {
    fn default() -> Self {
        Self::new()
    }
}



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

        
        let q1 = manager.quantize(&data, QuantizationType::Iq2Xxs).unwrap();
        let d1 = manager.dequantize(&q1).unwrap();
        assert_eq!(d1.len(), data.len());

        
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

        
        let original_bytes = data.len() * 4; 
        let compressed_bytes = quantized.data.len();
        let compression = original_bytes as f32 / compressed_bytes as f32;
        assert!(compression > 10.0, "Compression ratio too low: {}", compression);
    }
}

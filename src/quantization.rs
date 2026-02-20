use crate::error::{LociError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationScheme {
    Int8Symmetric,
    Int4Symmetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizedData {
    Int8(Vec<i8>),
    Int4Packed(Vec<u8>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor {
    pub scheme: QuantizationScheme,
    pub shape: Vec<usize>,
    pub scale: f32,
    pub data: QuantizedData,
}

impl QuantizedTensor {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn quantized_size_bytes(&self) -> usize {
        match &self.data {
            QuantizedData::Int8(values) => values.len(),
            QuantizedData::Int4Packed(values) => values.len(),
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        match &self.data {
            QuantizedData::Int8(values) => values
                .iter()
                .map(|value| *value as f32 * self.scale)
                .collect(),
            QuantizedData::Int4Packed(values) => {
                let mut output = Vec::with_capacity(self.element_count());
                for byte in values {
                    let low = (byte & 0x0F) as i8;
                    let high = ((byte >> 4) & 0x0F) as i8;
                    output.push(int4_to_signed(low) as f32 * self.scale);
                    if output.len() < self.element_count() {
                        output.push(int4_to_signed(high) as f32 * self.scale);
                    }
                }
                output
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationReport {
    pub scheme: QuantizationScheme,
    pub original_bytes: usize,
    pub quantized_bytes: usize,
    pub compression_ratio: f32,
    pub mse: f32,
    pub max_abs_error: f32,
}

pub struct QuantizationTool;

impl QuantizationTool {
    pub fn quantize(values: &[f32], shape: Vec<usize>, scheme: QuantizationScheme) -> Result<QuantizedTensor> {
        let expected = shape.iter().product::<usize>();
        if expected != values.len() {
            return Err(LociError::InvalidArgument(format!(
                "Shape element count ({expected}) does not match values ({})",
                values.len()
            )));
        }

        if values.is_empty() {
            return Err(LociError::InvalidArgument(
                "Cannot quantize empty tensor".to_string(),
            ));
        }

        let max_abs = values
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()))
            .max(f32::EPSILON);

        match scheme {
            QuantizationScheme::Int8Symmetric => {
                let scale = max_abs / 127.0;
                let data = values
                    .iter()
                    .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
                    .collect();

                Ok(QuantizedTensor {
                    scheme,
                    shape,
                    scale,
                    data: QuantizedData::Int8(data),
                })
            }
            QuantizationScheme::Int4Symmetric => {
                let scale = max_abs / 7.0;
                let mut packed = Vec::with_capacity(values.len().div_ceil(2));
                let mut index = 0;
                while index < values.len() {
                    let low = quantize_to_int4(values[index], scale);
                    let high = if index + 1 < values.len() {
                        quantize_to_int4(values[index + 1], scale)
                    } else {
                        0
                    };
                    packed.push((low & 0x0F) | ((high & 0x0F) << 4));
                    index += 2;
                }

                Ok(QuantizedTensor {
                    scheme,
                    shape,
                    scale,
                    data: QuantizedData::Int4Packed(packed),
                })
            }
        }
    }

    pub fn quantize_with_report(
        values: &[f32],
        shape: Vec<usize>,
        scheme: QuantizationScheme,
    ) -> Result<(QuantizedTensor, QuantizationReport)> {
        let quantized = Self::quantize(values, shape, scheme)?;
        let reconstructed = quantized.dequantize();
        let mut squared_error = 0.0_f32;
        let mut max_abs_error = 0.0_f32;
        for (actual, approx) in values.iter().zip(reconstructed.iter()) {
            let error = actual - approx;
            squared_error += error * error;
            max_abs_error = max_abs_error.max(error.abs());
        }

        let original_bytes = values.len() * std::mem::size_of::<f32>();
        let quantized_bytes = quantized.quantized_size_bytes();
        let report = QuantizationReport {
            scheme,
            original_bytes,
            quantized_bytes,
            compression_ratio: original_bytes as f32 / quantized_bytes.max(1) as f32,
            mse: squared_error / values.len() as f32,
            max_abs_error,
        };

        Ok((quantized, report))
    }

    pub fn quantize_f32_file<PIn, POut>(
        input_path: PIn,
        output_path: POut,
        shape: Vec<usize>,
        scheme: QuantizationScheme,
    ) -> Result<QuantizationReport>
    where
        PIn: AsRef<Path>,
        POut: AsRef<Path>,
    {
        let bytes = fs::read(input_path)?;
        if bytes.len() % 4 != 0 {
            return Err(LociError::ModelFormatError(
                "Input file length is not aligned to f32".to_string(),
            ));
        }

        let mut values = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let (quantized, report) = Self::quantize_with_report(&values, shape, scheme)?;
        let serialized = serde_json::to_vec(&quantized)
            .map_err(|e| LociError::SerializationError(e.to_string()))?;
        fs::write(output_path, serialized)?;
        Ok(report)
    }

    pub fn load_quantized_tensor<P: AsRef<Path>>(path: P) -> Result<QuantizedTensor> {
        let data = fs::read(path)?;
        let tensor: QuantizedTensor = serde_json::from_slice(&data)
            .map_err(|e| LociError::SerializationError(e.to_string()))?;
        Ok(tensor)
    }
}

fn quantize_to_int4(value: f32, scale: f32) -> u8 {
    let quantized = (value / scale).round().clamp(-7.0, 7.0) as i8;
    signed_to_int4(quantized)
}

fn signed_to_int4(value: i8) -> u8 {
    if value < 0 {
        (16 + value) as u8
    } else {
        value as u8
    }
}

fn int4_to_signed(value: i8) -> i8 {
    if value >= 8 {
        value - 16
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(lhs: &[f32], rhs: &[f32], tolerance: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            assert!((a - b).abs() <= tolerance, "{a} != {b}");
        }
    }

    #[test]
    fn test_int8_roundtrip() {
        let values = vec![0.25, -1.4, 0.0, 3.2, -2.0, 1.0, 0.8, -0.3];
        let (tensor, report) = QuantizationTool::quantize_with_report(
            &values,
            vec![2, 4],
            QuantizationScheme::Int8Symmetric,
        )
        .unwrap();

        let reconstructed = tensor.dequantize();
        assert_eq!(report.scheme, QuantizationScheme::Int8Symmetric);
        assert!(report.compression_ratio > 1.5);
        assert_close(&values, &reconstructed, 0.05);
    }

    #[test]
    fn test_int4_roundtrip() {
        let values = vec![0.1, -0.2, 0.7, -1.0, 1.5, -1.8, 0.3, 0.0, 0.9];
        let (tensor, report) = QuantizationTool::quantize_with_report(
            &values,
            vec![9],
            QuantizationScheme::Int4Symmetric,
        )
        .unwrap();

        let reconstructed = tensor.dequantize();
        assert_eq!(report.scheme, QuantizationScheme::Int4Symmetric);
        assert!(report.compression_ratio > 3.0);
        assert_close(&values, &reconstructed, 0.4);
    }

    #[test]
    fn test_shape_validation() {
        let values = vec![1.0, 2.0, 3.0];
        let error = QuantizationTool::quantize(
            &values,
            vec![2, 2],
            QuantizationScheme::Int8Symmetric,
        )
        .unwrap_err();
        assert!(matches!(error, LociError::InvalidArgument(_)));
    }
}

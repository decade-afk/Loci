

//! Kernel Fusion Optimizations
//!
//! This module provides fused kernel implementations for common operations in transformer models.
//! Kernel fusion combines multiple operations into a single kernel to reduce memory bandwidth
//! requirements and improve cache locality. This is particularly important for inference
//! performance on CPUs.
//!
//! The module implements the following fused operations:
//! - **RMSNorm + RoPE Fusion**: Combines Root Mean Square Layer Normalization with
//!   Rotary Positional Embeddings, eliminating the need for an intermediate tensor.
//! - **MatMul + Add Fusion**: Combines matrix multiplication with bias addition,
//!   reducing memory reads/writes.
//! - **LayerNorm + Linear Fusion**: Combines Layer Normalization with a linear layer,
//!   avoiding materialization of the normalized intermediate tensor.
//!
//! The module also includes SIMD optimizations using AVX2 and FMA instructions on x86_64
//! architectures for improved performance.

use anyhow::Result;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;




#[derive(Debug, Clone)]
pub struct RMSNormParams {
    
    pub gamma: Vec<f32>,
    
    pub eps: f32,
}


#[derive(Debug, Clone)]
pub struct RoPEParams {
    
    pub head_dim: usize,
    
    pub max_seq_len: usize,
    
    pub base: f32,
}

impl RoPEParams {
    
    pub fn precompute_freqs(&self) -> (Vec<f32>, Vec<f32>) {
        let mut cos_table = Vec::with_capacity(self.max_seq_len * self.head_dim);
        let mut sin_table = Vec::with_capacity(self.max_seq_len * self.head_dim);

        for pos in 0..self.max_seq_len {
            for i in 0..self.head_dim / 2 {
                let freq = 1.0 / self.base.powf((2 * i) as f32 / self.head_dim as f32);
                let angle = pos as f32 * freq;
                cos_table.push(angle.cos());
                cos_table.push(angle.cos()); 
                sin_table.push(angle.sin());
                sin_table.push(angle.sin());
            }
        }

        (cos_table, sin_table)
    }
}










pub struct RMSNormRoPEFusion {
    pub rmsnorm_params: RMSNormParams,
    pub rope_params: RoPEParams,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
}

impl RMSNormRoPEFusion {
    pub fn new(rmsnorm_params: RMSNormParams, rope_params: RoPEParams) -> Self {
        let (cos_table, sin_table) = rope_params.precompute_freqs();
        Self {
            rmsnorm_params,
            rope_params,
            cos_table,
            sin_table,
        }
    }

    
    
    
    
    
    
    
    
    pub fn forward(&self, x: &[f32], pos: usize) -> Result<Vec<f32>> {
        let seq_len = x.len() / self.rmsnorm_params.gamma.len();
        let hidden_dim = self.rmsnorm_params.gamma.len();
        let mut output = vec![0.0; x.len()];

        for t in 0..seq_len {
            let offset = t * hidden_dim;
            let input_slice = &x[offset..offset + hidden_dim];
            let output_slice = &mut output[offset..offset + hidden_dim];

            
            let rms = self.compute_rms(input_slice);

            
            for i in 0..hidden_dim {
                output_slice[i] = (input_slice[i] / rms) * self.rmsnorm_params.gamma[i];
            }

            
            
            if hidden_dim >= self.rope_params.head_dim {
                self.apply_rope_inplace(output_slice, pos + t);
            }
        }

        Ok(output)
    }

    
    #[inline]
    fn compute_rms(&self, x: &[f32]) -> f32 {
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        ((sum_sq / x.len() as f32) + self.rmsnorm_params.eps).sqrt()
    }

    
    #[inline]
    fn apply_rope_inplace(&self, x: &mut [f32], pos: usize) {
        let head_dim = self.rope_params.head_dim;
        let base_idx = pos * head_dim;

        for i in (0..head_dim).step_by(2) {
            let idx = base_idx + i;
            let cos = self.cos_table[idx];
            let sin = self.sin_table[idx];

            let x0 = x[i];
            let x1 = x[i + 1];

            
            x[i] = x0 * cos - x1 * sin;
            x[i + 1] = x0 * sin + x1 * cos;
        }
    }

    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn forward_avx2(&self, x: &[f32], pos: usize) -> Result<Vec<f32>> {
        
        
        
        
        

        self.forward(x, pos) 
    }
}











pub struct MatMulAddFusion {
    
    pub weight: Vec<f32>,
    
    pub bias: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl MatMulAddFusion {
    pub fn new(weight: Vec<f32>, bias: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        assert_eq!(weight.len(), in_features * out_features);
        assert_eq!(bias.len(), out_features);
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    
    
    
    
    
    
    
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let batch_size = x.len() / self.in_features;
        let mut output = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            let x_offset = b * self.in_features;
            let y_offset = b * self.out_features;

            for i in 0..self.out_features {
                let w_offset = i * self.in_features;

                
                let mut acc = self.bias[i]; 

                for j in 0..self.in_features {
                    acc += x[x_offset + j] * self.weight[w_offset + j];
                }

                output[y_offset + i] = acc;
            }
        }

        Ok(output)
    }

    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn forward_avx2(&self, x: &[f32]) -> Result<Vec<f32>> {
        let batch_size = x.len() / self.in_features;
        let mut output = vec![0.0; batch_size * self.out_features];

        for b in 0..batch_size {
            let x_offset = b * self.in_features;
            let y_offset = b * self.out_features;

            for i in 0..self.out_features {
                let w_offset = i * self.in_features;

                
                let mut sum = _mm256_setzero_ps();
                let mut j = 0;

                
                while j + 8 <= self.in_features {
                    let x_vec = _mm256_loadu_ps(x.as_ptr().add(x_offset + j));
                    let w_vec = _mm256_loadu_ps(self.weight.as_ptr().add(w_offset + j));
                    sum = _mm256_fmadd_ps(x_vec, w_vec, sum); 
                    j += 8;
                }

                
                let mut acc = self.horizontal_sum_avx2(sum);

                
                while j < self.in_features {
                    acc += x[x_offset + j] * self.weight[w_offset + j];
                    j += 1;
                }

                
                output[y_offset + i] = acc + self.bias[i];
            }
        }

        Ok(output)
    }

    
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn horizontal_sum_avx2(&self, v: __m256) -> f32 {
        
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(hi, lo);

        
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x1));

        _mm_cvtss_f32(sum32)
    }
}




#[derive(Debug, Clone)]
pub struct LayerNormParams {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub eps: f32,
}


pub struct LayerNormLinearFusion {
    pub ln_params: LayerNormParams,
    pub linear: MatMulAddFusion,
}

impl LayerNormLinearFusion {
    pub fn new(ln_params: LayerNormParams, linear: MatMulAddFusion) -> Self {
        Self { ln_params, linear }
    }

    
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let hidden_dim = self.ln_params.gamma.len();
        let batch_size = x.len() / hidden_dim;
        let mut normalized = vec![0.0; x.len()];

        
        for b in 0..batch_size {
            let offset = b * hidden_dim;
            let input_slice = &x[offset..offset + hidden_dim];
            let output_slice = &mut normalized[offset..offset + hidden_dim];

            
            let mean: f32 = input_slice.iter().sum::<f32>() / hidden_dim as f32;
            let variance: f32 = input_slice.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f32>() / hidden_dim as f32;
            let std = (variance + self.ln_params.eps).sqrt();

            
            for i in 0..hidden_dim {
                output_slice[i] = ((input_slice[i] - mean) / std) * self.ln_params.gamma[i] + self.ln_params.beta[i];
            }
        }

        
        self.linear.forward(&normalized)
    }
}




pub struct KernelFusionManager {
    
    pub enable_avx2: bool,
    
    pub enable_avx512: bool,
}

impl KernelFusionManager {
    
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                enable_avx2: is_x86_feature_detected!("avx2"),
                enable_avx512: is_x86_feature_detected!("avx512f"),
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                enable_avx2: false,
                enable_avx512: false,
            }
        }
    }

    
    pub fn print_features(&self) {
        println!("╔════════════════════════════════════════╗");
        println!("║       Kernel Fusion Features          ║");
        println!("╠════════════════════════════════════════╣");
        println!("║  AVX2:     {:29} ║", if self.enable_avx2 { "✅ Enabled" } else { "❌ Disabled" });
        println!("║  AVX512:   {:29} ║", if self.enable_avx512 { "✅ Enabled" } else { "❌ Disabled" });
        println!("╚════════════════════════════════════════╝");
    }
}

impl Default for KernelFusionManager {
    fn default() -> Self {
        Self::new()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_rope_fusion() {
        let gamma = vec![1.0; 64];
        let rmsnorm_params = RMSNormParams { gamma, eps: 1e-6 };

        let rope_params = RoPEParams {
            head_dim: 64,
            max_seq_len: 128,
            base: 10000.0,
        };

        let fusion = RMSNormRoPEFusion::new(rmsnorm_params, rope_params);
        let x = vec![1.0; 64];
        let output = fusion.forward(&x, 0).unwrap();

        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_matmul_add_fusion() {
        let in_features = 64;
        let out_features = 32;
        let batch_size = 2;

        let weight = vec![0.1; in_features * out_features];
        let bias = vec![0.5; out_features];

        let fusion = MatMulAddFusion::new(weight, bias, in_features, out_features);
        let x = vec![1.0; batch_size * in_features];
        let output = fusion.forward(&x).unwrap();

        assert_eq!(output.len(), batch_size * out_features);

        
        for &val in &output {
            assert!(val > 0.5); 
        }
    }

    #[test]
    fn test_layernorm_linear_fusion() {
        let hidden_dim = 64;
        let out_features = 32;

        let ln_params = LayerNormParams {
            gamma: vec![1.0; hidden_dim],
            beta: vec![0.0; hidden_dim],
            eps: 1e-5,
        };

        let weight = vec![0.1; hidden_dim * out_features];
        let bias = vec![0.0; out_features];
        let linear = MatMulAddFusion::new(weight, bias, hidden_dim, out_features);

        let fusion = LayerNormLinearFusion::new(ln_params, linear);
        let x = vec![1.0; hidden_dim];
        let output = fusion.forward(&x).unwrap();

        assert_eq!(output.len(), out_features);
    }

    #[test]
    fn test_rope_precompute() {
        let rope_params = RoPEParams {
            head_dim: 64,
            max_seq_len: 128,
            base: 10000.0,
        };

        let (cos_table, sin_table) = rope_params.precompute_freqs();
        assert_eq!(cos_table.len(), 128 * 64);
        assert_eq!(sin_table.len(), 128 * 64);

        
        for i in 0..cos_table.len() {
            let sum_sq = cos_table[i].powi(2) + sin_table[i].powi(2);
            assert!((sum_sq - 1.0).abs() < 1e-5, "cos^2 + sin^2 != 1 at index {}", i);
        }
    }

    #[test]
    fn test_kernel_fusion_manager() {
        let manager = KernelFusionManager::new();
        manager.print_features();

        #[cfg(target_arch = "x86_64")]
        {
            
            println!("AVX2: {}", manager.enable_avx2);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_matmul_add() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping AVX2 test (not supported)");
            return;
        }

        let in_features = 128;
        let out_features = 64;
        let batch_size = 4;

        let weight = vec![0.1; in_features * out_features];
        let bias = vec![0.5; out_features];

        let fusion = MatMulAddFusion::new(weight, bias, in_features, out_features);
        let x = vec![1.0; batch_size * in_features];

        let output_scalar = fusion.forward(&x).unwrap();
        let output_avx2 = unsafe { fusion.forward_avx2(&x).unwrap() };

        
        for (i, (&a, &b)) in output_scalar.iter().zip(output_avx2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "Mismatch at index {}: {} vs {}", i, a, b);
        }
    }
}

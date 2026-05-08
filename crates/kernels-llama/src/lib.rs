//! Curated llama.cpp-inspired kernels and migration boundaries.
//!
//! This crate now contains both provenance metadata and one concrete CPU-side
//! kernel implementation (`RMSNorm`) so the porting path is no longer purely
//! declarative.

use loci_protocol::{
    AcceleratorKind, ChipOperatorClass, KernelDescriptor, KernelImplementationKind, KernelMaturity,
    KernelOrigin, ModelFormat,
};

/// Executes a simple CPU RMSNorm pass over one contiguous vector.
///
/// The implementation is intentionally portable and safe Rust first. It is not
/// yet vectorized, but it establishes a numerically testable migration target
/// for a future SIMD path.
pub fn rms_norm_f32(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>, String> {
    if input.len() != weight.len() {
        return Err(format!(
            "rms_norm length mismatch: input={} weight={}",
            input.len(),
            weight.len()
        ));
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0 / (mean_square + epsilon).sqrt();

    Ok(input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * inv_rms * w)
        .collect())
}

/// Applies a baseline RoPE transform to one contiguous head vector.
///
/// The input length must be even. Each consecutive pair forms one complex lane.
pub fn rope_f32(
    input: &[f32],
    position: usize,
    base_theta: f32,
    rotary_dim: usize,
) -> Result<Vec<f32>, String> {
    if !input.len().is_multiple_of(2) {
        return Err(format!(
            "rope expects an even hidden size, got {}",
            input.len()
        ));
    }
    if rotary_dim > input.len() {
        return Err(format!(
            "rope rotary_dim {} exceeds input size {}",
            rotary_dim,
            input.len()
        ));
    }
    if !rotary_dim.is_multiple_of(2) {
        return Err(format!("rope rotary_dim must be even, got {rotary_dim}"));
    }
    if !(base_theta.is_finite() && base_theta > 0.0) {
        return Err(format!(
            "rope base_theta must be positive and finite, got {base_theta}"
        ));
    }

    let mut output = input.to_vec();
    for pair_offset in (0..rotary_dim).step_by(2) {
        let lane = pair_offset / 2;
        let exponent = (2.0 * lane as f32) / rotary_dim as f32;
        let inv_freq = 1.0 / base_theta.powf(exponent);
        let angle = position as f32 * inv_freq;
        let cos = angle.cos();
        let sin = angle.sin();

        let x0 = input[pair_offset];
        let x1 = input[pair_offset + 1];
        output[pair_offset] = x0 * cos - x1 * sin;
        output[pair_offset + 1] = x0 * sin + x1 * cos;
    }

    Ok(output)
}

/// Projects one hidden state against a row-major output matrix and returns one
/// score per candidate row.
///
/// This is the portable baseline for the logits/projection hotspot that sits
/// underneath greedy token selection in the current Candle path.
pub fn projection_scores_f32(
    hidden_state: &[f32],
    row_major_weights: &[f32],
    row_width: usize,
    candidate_count: usize,
) -> Result<Vec<f32>, String> {
    if row_width == 0 {
        return Err("projection row_width must be greater than zero".to_string());
    }
    if hidden_state.is_empty() {
        return Err("projection hidden_state must not be empty".to_string());
    }

    let row_count = row_major_weights.len() / row_width;
    if row_count == 0 {
        return Err("projection weights do not contain any complete rows".to_string());
    }
    if candidate_count == 0 {
        return Ok(Vec::new());
    }

    let hidden_width = hidden_state.len().min(row_width);
    Ok((0..candidate_count.min(row_count))
        .map(|row_index| {
            let start = row_index * row_width;
            let row = &row_major_weights[start..start + row_width];
            row.iter()
                .zip(hidden_state.iter())
                .take(hidden_width)
                .map(|(weight, hidden)| weight * hidden)
                .sum::<f32>()
        })
        .collect())
}

/// Returns the curated MVP kernel catalog for llama.cpp-inspired ports.
pub fn curated_kernel_descriptors() -> Vec<KernelDescriptor> {
    vec![
        KernelDescriptor {
            backend: "candle".to_string(),
            kernel_name: "llama_rmsnorm_port".to_string(),
            operator_class: ChipOperatorClass::RmsNorm,
            implementation: KernelImplementationKind::Rust,
            maturity: KernelMaturity::Integrated,
            origin: KernelOrigin {
                project: "llama.cpp".to_string(),
                component: "rms_norm".to_string(),
                license: Some("MIT".to_string()),
                notes: vec![
                    "portable safe Rust baseline landed in loci-kernels-llama".to_string(),
                    "future SIMD path should preserve this behavior".to_string(),
                ],
            },
            supported_targets: vec![AcceleratorKind::Cpu, AcceleratorKind::Gpu],
            supported_formats: vec![ModelFormat::Gguf],
            supported_architectures: vec![
                "llama".to_string(),
                "mistral".to_string(),
                "qwen".to_string(),
            ],
            dispatch_keys: vec!["norm".to_string(), "decoder".to_string()],
            notes: vec!["MVP hotspot candidate".to_string()],
        },
        KernelDescriptor {
            backend: "candle".to_string(),
            kernel_name: "llama_rope_port".to_string(),
            operator_class: ChipOperatorClass::Attention,
            implementation: KernelImplementationKind::Rust,
            maturity: KernelMaturity::Integrated,
            origin: KernelOrigin {
                project: "llama.cpp".to_string(),
                component: "rope".to_string(),
                license: Some("MIT".to_string()),
                notes: vec![
                    "portable safe Rust baseline landed in loci-kernels-llama".to_string(),
                    "future fused or SIMD path should preserve lane ordering".to_string(),
                ],
            },
            supported_targets: vec![AcceleratorKind::Cpu, AcceleratorKind::Gpu],
            supported_formats: vec![ModelFormat::Gguf],
            supported_architectures: vec![
                "llama".to_string(),
                "mistral".to_string(),
                "qwen".to_string(),
            ],
            dispatch_keys: vec!["rope".to_string(), "attention".to_string()],
            notes: vec!["MVP rotary baseline".to_string()],
        },
        KernelDescriptor {
            backend: "candle".to_string(),
            kernel_name: "llama_projection_score_port".to_string(),
            operator_class: ChipOperatorClass::Matmul,
            implementation: KernelImplementationKind::Rust,
            maturity: KernelMaturity::Integrated,
            origin: KernelOrigin {
                project: "llama.cpp".to_string(),
                component: "output projection".to_string(),
                license: Some("MIT".to_string()),
                notes: vec![
                    "portable safe Rust baseline landed in loci-kernels-llama".to_string(),
                    "future quantized path should preserve row-major score ordering".to_string(),
                ],
            },
            supported_targets: vec![AcceleratorKind::Cpu, AcceleratorKind::Gpu],
            supported_formats: vec![ModelFormat::Gguf],
            supported_architectures: vec![
                "llama".to_string(),
                "mistral".to_string(),
                "qwen".to_string(),
            ],
            dispatch_keys: vec!["projection".to_string(), "decode".to_string()],
            notes: vec!["MVP logits scoring baseline".to_string()],
        },
        KernelDescriptor {
            backend: "candle".to_string(),
            kernel_name: "llama_quantized_matmul_port".to_string(),
            operator_class: ChipOperatorClass::Matmul,
            implementation: KernelImplementationKind::Rust,
            maturity: KernelMaturity::Planned,
            origin: KernelOrigin {
                project: "llama.cpp".to_string(),
                component: "quantized matmul".to_string(),
                license: Some("MIT".to_string()),
                notes: vec!["high-value GGUF hotspot".to_string()],
            },
            supported_targets: vec![AcceleratorKind::Cpu, AcceleratorKind::Gpu],
            supported_formats: vec![ModelFormat::Gguf],
            supported_architectures: vec![
                "llama".to_string(),
                "mistral".to_string(),
                "qwen".to_string(),
            ],
            dispatch_keys: vec!["quantized".to_string(), "matmul".to_string()],
            notes: vec!["portable SIMD candidate".to_string()],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_curated_porting_catalog() {
        let kernels = curated_kernel_descriptors();
        assert!(kernels
            .iter()
            .any(|kernel| kernel.kernel_name == "llama_rmsnorm_port"));
        assert!(kernels
            .iter()
            .any(|kernel| kernel.kernel_name == "llama_projection_score_port"));
        assert!(kernels
            .iter()
            .all(|kernel| kernel.origin.project == "llama.cpp"));
        assert!(kernels
            .iter()
            .all(|kernel| kernel.supported_formats.contains(&ModelFormat::Gguf)));
        assert!(kernels
            .iter()
            .any(|kernel| kernel.kernel_name == "llama_rmsnorm_port"
                && kernel.maturity == KernelMaturity::Integrated));
        assert!(kernels
            .iter()
            .any(|kernel| kernel.kernel_name == "llama_rope_port"
                && kernel.maturity == KernelMaturity::Integrated));
    }

    #[test]
    fn rms_norm_matches_expected_reference_values() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0_f32, 1.0, 1.0, 1.0];
        let output = rms_norm_f32(&input, &weight, 1e-5).expect("output");

        let mean_square = (1.0_f32 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv_rms = 1.0 / (mean_square + 1e-5).sqrt();
        let expected: Vec<f32> = input.iter().map(|x| x * inv_rms).collect();

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_rejects_length_mismatch() {
        let error = rms_norm_f32(&[1.0, 2.0], &[1.0], 1e-5).expect_err("error");
        assert!(error.contains("length mismatch"));
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let input = [1.0_f32, -2.0, 3.0, -4.0];
        let output = rope_f32(&input, 0, 10_000.0, 4).expect("output");
        assert_eq!(output, input);
    }

    #[test]
    fn rope_rotates_first_lane_at_position_one() {
        let input = [1.0_f32, 0.0, 0.0, 1.0];
        let output = rope_f32(&input, 1, 10_000.0, 4).expect("output");

        assert!((output[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((output[1] - 1.0_f32.sin()).abs() < 1e-6);

        let lane1_angle = 1.0 / 100.0_f32;
        assert!((output[2] - (-lane1_angle.sin())).abs() < 1e-6);
        assert!((output[3] - lane1_angle.cos()).abs() < 1e-6);
    }

    #[test]
    fn rope_preserves_tail_outside_rotary_window() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let output = rope_f32(&input, 3, 10_000.0, 4).expect("output");
        assert_eq!(&output[4..], &[5.0, 6.0]);
    }

    #[test]
    fn rope_rejects_invalid_shapes() {
        let error = rope_f32(&[1.0, 2.0, 3.0], 1, 10_000.0, 2).expect_err("error");
        assert!(error.contains("even hidden size"));

        let error = rope_f32(&[1.0, 2.0, 3.0, 4.0], 1, 10_000.0, 3).expect_err("error");
        assert!(error.contains("rotary_dim must be even"));
    }

    #[test]
    fn projection_scores_match_reference_rows() {
        let hidden = [1.0_f32, 2.0, 3.0];
        let weights = [
            1.0_f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.5, 0.5, 0.5,
        ];

        let scores = projection_scores_f32(&hidden, &weights, 3, 3).expect("scores");
        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[1] - 2.0).abs() < 1e-6);
        assert!((scores[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn projection_scores_reject_invalid_shapes() {
        let error = projection_scores_f32(&[1.0], &[1.0], 0, 1).expect_err("error");
        assert!(error.contains("row_width"));

        let error = projection_scores_f32(&[], &[1.0], 1, 1).expect_err("error");
        assert!(error.contains("hidden_state"));
    }
}

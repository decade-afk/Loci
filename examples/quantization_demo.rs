use loci::prelude::*;

fn main() -> Result<()> {
    let weights = vec![0.12, -0.98, 1.56, 0.0, -0.44, 0.77, 2.01, -1.3];

    let (_int8_tensor, int8_report) = QuantizationTool::quantize_with_report(
        &weights,
        vec![2, 4],
        QuantizationScheme::Int8Symmetric,
    )?;

    let (_int4_tensor, int4_report) = QuantizationTool::quantize_with_report(
        &weights,
        vec![2, 4],
        QuantizationScheme::Int4Symmetric,
    )?;

    println!(
        "INT8: {} -> {} bytes, ratio {:.2}x, mse {:.6}",
        int8_report.original_bytes,
        int8_report.quantized_bytes,
        int8_report.compression_ratio,
        int8_report.mse
    );

    println!(
        "INT4: {} -> {} bytes, ratio {:.2}x, mse {:.6}",
        int4_report.original_bytes,
        int4_report.quantized_bytes,
        int4_report.compression_ratio,
        int4_report.mse
    );

    Ok(())
}

//! Batch inference demonstration
//!
//! This example shows how to process multiple prompts efficiently using batch inference

use loci::batch_inference::{
    BatchConfig, BatchInferenceBuilder, BatchInferenceProcessor, PromptBatch,
};
use loci::prelude::*;

fn main() -> Result<()> {
    println!("=== Batch Inference Demo ===\n");

    // Mock inference function for demonstration
    let mock_inference = |prompt: &str, _params: &InferenceParams| -> Result<String> {
        // Simulate processing time
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok(format!("Response to: {}", prompt))
    };

    // Example 1: Sequential batch processing
    println!("1. Sequential Batch Processing:");
    let config = BatchConfig::default();
    let processor = BatchInferenceProcessor::new(mock_inference, config);

    let prompts = vec![
        "What is machine learning?".to_string(),
        "Explain neural networks.".to_string(),
        "What is deep learning?".to_string(),
    ];

    let batch = PromptBatch::new(prompts, InferenceParams::default());
    let result = processor.process_batch(batch)?;

    println!("Total time: {}ms", result.total_time_ms);
    println!(
        "Average time per prompt: {:.2}ms",
        result.avg_time_per_prompt_ms
    );
    println!("Success rate: {:.1}%", result.success_rate() * 100.0);
    println!(
        "Successful: {}, Failed: {}\n",
        result.successful_count(),
        result.failed_count()
    );

    for (i, response) in result.responses.iter().enumerate() {
        match response {
            Ok(text) => println!("  [{}] {}", i + 1, text),
            Err(e) => println!("  [{}] Error: {}", i + 1, e),
        }
    }

    // Example 2: Parallel batch processing
    println!("\n2. Parallel Batch Processing (max 2 concurrent):");
    let parallel_config = BatchConfig::parallel(2);
    let parallel_processor = BatchInferenceProcessor::new(mock_inference, parallel_config);

    let large_batch = PromptBatch::new(
        vec![
            "Prompt 1".to_string(),
            "Prompt 2".to_string(),
            "Prompt 3".to_string(),
            "Prompt 4".to_string(),
        ],
        InferenceParams::default(),
    );

    let result = parallel_processor.process_batch(large_batch)?;
    println!(
        "Total time: {}ms (should be faster than sequential)",
        result.total_time_ms
    );
    println!("Processed {} prompts\n", result.responses.len());

    // Example 3: Batch with progress callback
    println!("3. Batch with Progress Tracking:");
    let progress_processor = BatchInferenceProcessor::new(mock_inference, BatchConfig::default());

    let batch = PromptBatch::new(
        vec![
            "Question 1".to_string(),
            "Question 2".to_string(),
            "Question 3".to_string(),
        ],
        InferenceParams::default(),
    );

    let result = progress_processor.process_with_progress(batch, |current, total| {
        println!(
            "  Progress: {}/{} ({:.1}%)",
            current,
            total,
            (current as f64 / total as f64) * 100.0
        );
    })?;

    println!("Completed in {}ms\n", result.total_time_ms);

    // Example 4: Using builder pattern
    println!("4. Using Builder Pattern:");
    let processor = BatchInferenceBuilder::new()
        .max_concurrent(3)
        .timeout(5000)
        .continue_on_error(true)
        .build(mock_inference);

    let batch = PromptBatch::new(vec!["Test prompt".to_string()], InferenceParams::default());

    let result = processor.process_batch(batch)?;
    println!(
        "Processed with custom config: {} prompts",
        result.responses.len()
    );

    Ok(())
}

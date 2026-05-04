use anyhow::Context;
use loci_core::{EmbeddedModelRegistration, InferenceEngine, SessionRequest};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .context("usage: embedded-local <model-path>")?;

    let mut engine = InferenceEngine::builder()
        .local_model(
            model_path,
            EmbeddedModelRegistration {
                name: Some("embedded-demo".to_string()),
                ..EmbeddedModelRegistration::default()
            },
        )?
        .build()?;

    let response = engine.infer(SessionRequest {
        prompt: "Reply in one short friendly sentence for a desktop pet.".to_string(),
        max_tokens: 48,
        temperature: 0.7,
        target_model: Some("embedded-demo".to_string()),
        images: Vec::new(),
        structured_output: false,
        tool_calling: false,
    })?;

    println!("backend: {}", response.backend);
    println!("model: {}", response.model);
    println!("text: {}", response.text);
    Ok(())
}

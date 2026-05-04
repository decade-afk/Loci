use anyhow::Context;
use loci_sdk::{
    LocalModelRegistrationRequest, Loci, ModelPreparationRequest, TextGenerationRequest,
};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .context("usage: sdk-local <model-path>")?;

    let mut loci = Loci::builder().build()?;
    let registered =
        loci.register_model(LocalModelRegistrationRequest::new(model_path).name("sdk-demo"))?;
    let inspection = loci.inspect_model("sdk-demo")?;
    let prepared = loci.prepare_model(ModelPreparationRequest::new().model("sdk-demo"))?;
    let response = loci.generate_text(
        TextGenerationRequest::new("Reply in one short sentence as a local assistant.")
            .model("sdk-demo")
            .max_tokens(48)
            .temperature(0.7),
    )?;

    println!("registered: {} ({})", registered.name, registered.format);
    println!(
        "inspection: ready={} recommended_backend={:?}",
        inspection.ready_for_inference, inspection.recommended_backend
    );
    println!("prepared session: {}", prepared.session_key);
    println!("backend: {}", response.backend);
    println!("model: {}", response.model);
    println!("text: {}", response.text);
    Ok(())
}

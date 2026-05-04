use anyhow::Context;
use loci_sdk::{LocalModelRegistrationRequest, Loci, LociServiceConfig};

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .context("usage: sdk-service <model-path> [bind]")?;
    let bind = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "127.0.0.1:18081".to_string());

    let mut loci = Loci::builder().build()?;
    loci.register_model(LocalModelRegistrationRequest::new(model_path).name("service-demo"))?;
    loci.run_service(LociServiceConfig::with_bind(bind))
}

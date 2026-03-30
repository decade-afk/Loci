use loci_core::InferenceEngine;

pub fn build_engine() -> loci_core::Result<InferenceEngine> {
    InferenceEngine::builder().build()
}

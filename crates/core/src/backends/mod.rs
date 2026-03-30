pub mod mock;
#[cfg(feature = "llama")]
pub mod llamacpp;

pub use mock::{MockBackend, MockModel};
#[cfg(feature = "llama")]
pub use llamacpp::LlamaCppBackend;

use crate::backend::BackendRegistry;

pub fn register_builtin_backends(registry: &mut BackendRegistry) {
    registry.register("mock".to_string(), Box::new(MockBackend::new()));
    #[cfg(feature = "llama")]
    registry.register("llama.cpp".to_string(), Box::new(LlamaCppBackend::new()));
}

pub fn default_backend_name() -> &'static str {
    #[cfg(feature = "llama")]
    {
        "llama.cpp"
    }
    #[cfg(not(feature = "llama"))]
    {
        "mock"
    }
}

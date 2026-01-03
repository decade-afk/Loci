//! Backend implementations
//!
//! This module contains concrete implementations of the `InferenceBackend` trait.
//!
//! ## Available Backends
//!
//! - `llamacpp`: Native llama.cpp backend (Phase 1 - production ready)
//!
//! ## Planned Backends
//!
//! - `candle`: Pure Rust backend using HuggingFace Candle (Phase 2)
//! - `onnx`: ONNX Runtime backend (Phase 2)
//! - Dynamic loading via libloading (Phase 2)
//! - WASM backends via wasmtime (Phase 3)

pub mod llamacpp;

pub use llamacpp::{LlamaCppBackend, LlamaCppModel};

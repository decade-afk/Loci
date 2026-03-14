//! Backend implementations
//!
//! This module contains concrete implementations of the `InferenceBackend` trait.
//!
//! ## Available Backends
//!
//! - `llamacpp`: Native llama.cpp backend (Phase 1 - production ready)
//! - `candle`: Pure Rust backend
//! - `dynamic`: Runtime loaded backend via shared library
//!
//! ## Planned Backends
//!
//! - `onnx`: ONNX Runtime backend
//! - WASM backends via wasmtime

pub mod candle;
pub mod dynamic;
pub mod llamacpp;

pub use candle::{CandleBackend, CandleModel};
pub use dynamic::DynamicBackend;
pub use llamacpp::{LlamaCppBackend, LlamaCppModel};

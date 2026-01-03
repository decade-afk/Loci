//! Temperature Booster WASM Plugin Example
//!
//! This plugin demonstrates dynamic temperature adjustment based on context length.
//! It increases temperature (more randomness) as the generated text gets longer,
//! helping to prevent repetitive outputs.

#![no_std]

use loci_wasm_plugin_sdk::*;

// Define plugin metadata
plugin_metadata! {
    name: "temperature_booster",
    version: "0.1.0"
}

// Configuration
const BASE_TEMPERATURE: f32 = 1.0;
const TEMPERATURE_INCREASE_RATE: f32 = 0.01; // Per 10 tokens
const MAX_TEMPERATURE: f32 = 1.5;

/// Transform logits hook - apply dynamic temperature based on context length
#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: i32, n_vocab: i32, context_len: i32) -> i32 {
    unsafe {
        // Get mutable access to logits
        let logits = get_mut_slice(logits_ptr, n_vocab);

        // Calculate dynamic temperature based on context length
        let temperature_boost = (context_len as f32 / 10.0) * TEMPERATURE_INCREASE_RATE;
        let temperature = (BASE_TEMPERATURE + temperature_boost).min(MAX_TEMPERATURE);

        // Apply temperature scaling
        apply_temperature(logits, temperature);
    }

    0 // Success
}

/// Pre-generate hook - could be used to log generation start
#[no_mangle]
pub extern "C" fn pre_generate(prompt_ptr: i32) -> i32 {
    // In a real plugin, you might want to process the prompt
    // For now, just return the same pointer
    prompt_ptr
}

//! Token Banner WASM Plugin Example
//!
//! This plugin demonstrates how to build a WASM plugin for Loci that bans
//! specific tokens from being sampled.

#![no_std]

use loci_wasm_plugin_sdk::*;

// Define plugin metadata
plugin_metadata! {
    name: "token_banner",
    version: "0.1.0"
}

// List of banned tokens (example: common profanity tokens)
// In a real plugin, this would be configurable
static BANNED_TOKENS: &[i32] = &[
    100,  // Example token ID
    200,  // Example token ID
    300,  // Example token ID
];

/// Transform logits hook - ban specific tokens
#[no_mangle]
pub extern "C" fn transform_logits(logits_ptr: i32, n_vocab: i32, _context_len: i32) -> i32 {
    unsafe {
        // Get mutable access to logits
        let logits = get_mut_slice(logits_ptr, n_vocab);

        // Ban tokens by setting their logits to negative infinity
        for &token_id in BANNED_TOKENS {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] = f32::NEG_INFINITY;
            }
        }
    }

    0 // Success
}

/// Post-sample hook - verify token is not banned (safety check)
#[no_mangle]
pub extern "C" fn post_sample(token_id: i32) -> i32 {
    // If somehow a banned token was sampled, replace with space token (ID: 1)
    for &banned in BANNED_TOKENS {
        if token_id == banned {
            return 1; // Space token
        }
    }

    token_id
}

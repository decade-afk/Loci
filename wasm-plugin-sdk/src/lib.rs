//! Loci WASM Plugin SDK
//!
//! This SDK provides helper functions and macros for building WASM plugins
//! for the Loci LLM inference framework.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use loci_wasm_plugin_sdk::*;
//!
//! // Define plugin metadata
//! plugin_metadata! {
//!     name: "my_plugin",
//!     version: "0.1.0"
//! }
//!
//! // Implement plugin hooks
//! #[plugin_hook]
//! pub fn transform_logits(logits_ptr: i32, n_vocab: i32, context_len: i32) -> i32 {
//!     // Your logits transformation code here
//!     0 // Success
//! }
//!
//! #[plugin_hook]
//! pub fn post_sample(token_id: i32) -> i32 {
//!     // Your post-sample processing here
//!     token_id
//! }
//! ```

#![no_std]

use core::panic::PanicInfo;
use core::slice;

/// Plugin metadata - name and version
#[macro_export]
macro_rules! plugin_metadata {
    (name: $name:expr, version: $version:expr) => {
        static PLUGIN_NAME: &str = $name;
        static PLUGIN_VERSION: &str = $version;

        #[no_mangle]
        pub extern "C" fn plugin_name() -> i32 {
            PLUGIN_NAME.as_ptr() as i32
        }

        #[no_mangle]
        pub extern "C" fn plugin_name_len() -> i32 {
            PLUGIN_NAME.len() as i32
        }

        #[no_mangle]
        pub extern "C" fn plugin_version() -> i32 {
            PLUGIN_VERSION.as_ptr() as i32
        }

        #[no_mangle]
        pub extern "C" fn plugin_version_len() -> i32 {
            PLUGIN_VERSION.len() as i32
        }
    };
}

/// Attribute macro for plugin hooks (syntax sugar)
/// Usage: #[plugin_hook]
#[macro_export]
macro_rules! plugin_hook {
    () => {
        #[no_mangle]
    };
}

/// Allocator for WASM linear memory
/// This function is called by the host to allocate memory for passing data to WASM
#[no_mangle]
pub extern "C" fn alloc(size: i32) -> i32 {
    let layout = core::alloc::Layout::from_size_align(size as usize, 4).unwrap();
    unsafe { core::alloc::alloc(layout) as i32 }
}

/// Deallocator for WASM linear memory
#[no_mangle]
pub extern "C" fn dealloc(ptr: i32, size: i32) {
    let layout = core::alloc::Layout::from_size_align(size as usize, 4).unwrap();
    unsafe { core::alloc::dealloc(ptr as *mut u8, layout) }
}

/// Helper: Read string from WASM memory
pub unsafe fn read_string(ptr: i32, len: i32) -> &'static str {
    let slice = slice::from_raw_parts(ptr as *const u8, len as usize);
    core::str::from_utf8_unchecked(slice)
}

/// Helper: Write string to WASM memory (allocates)
pub fn write_string(s: &str) -> (i32, i32) {
    let len = s.len() as i32;
    let ptr = alloc(len);
    unsafe {
        let dest = slice::from_raw_parts_mut(ptr as *mut u8, len as usize);
        dest.copy_from_slice(s.as_bytes());
    }
    (ptr, len)
}

/// Helper: Get mutable slice from WASM memory
pub unsafe fn get_mut_slice(ptr: i32, len: i32) -> &'static mut [f32] {
    slice::from_raw_parts_mut(ptr as *mut f32, len as usize)
}

/// Helper: Get immutable slice from WASM memory
pub unsafe fn get_slice(ptr: i32, len: i32) -> &'static [f32] {
    slice::from_raw_parts(ptr as *const f32, len as usize)
}

/// Helper: Get i32 slice from WASM memory
pub unsafe fn get_i32_slice(ptr: i32, len: i32) -> &'static [i32] {
    slice::from_raw_parts(ptr as *const i32, len as usize)
}

/// Panic handler for no_std environment
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    // In WASM, we can't really panic, so we just trap
    loop {}
}

/// Global allocator for WASM
#[global_allocator]
static ALLOCATOR: WasmAllocator = WasmAllocator;

struct WasmAllocator;

unsafe impl core::alloc::GlobalAlloc for WasmAllocator {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        // Simple bump allocator (for demonstration)
        // In production, use wee_alloc or similar
        static mut HEAP: [u8; 65536] = [0; 65536]; // 64KB heap
        static mut OFFSET: usize = 0;

        let size = layout.size();
        let align = layout.align();

        let aligned_offset = (OFFSET + align - 1) & !(align - 1);
        if aligned_offset + size > HEAP.len() {
            return core::ptr::null_mut();
        }

        let ptr = HEAP.as_mut_ptr().add(aligned_offset);
        OFFSET = aligned_offset + size;
        ptr
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: core::alloc::Layout) {
        // Bump allocator doesn't deallocate
    }
}

/// Example: Token banning plugin helper
pub struct TokenBanner {
    banned_tokens: &'static [i32],
}

impl TokenBanner {
    pub const fn new(banned_tokens: &'static [i32]) -> Self {
        Self { banned_tokens }
    }

    pub fn apply(&self, logits: &mut [f32]) {
        for &token in self.banned_tokens {
            if (token as usize) < logits.len() {
                logits[token as usize] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Example: Temperature scaling helper
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature != 1.0 && temperature > 0.0 {
        for logit in logits.iter_mut() {
            *logit /= temperature;
        }
    }
}

/// Example: Softmax helper (for probability normalization)
pub fn softmax(logits: &mut [f32]) {
    // Find max for numerical stability
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp and sum
    let mut sum = 0.0f32;
    for logit in logits.iter_mut() {
        *logit = (*logit - max).exp();
        sum += *logit;
    }

    // Normalize
    for logit in logits.iter_mut() {
        *logit /= sum;
    }
}

/// Example: Find top-k tokens
pub fn find_top_k(logits: &[f32], k: usize) -> [i32; 10] {
    let mut top_k = [0i32; 10];
    let k = k.min(10).min(logits.len());

    for i in 0..k {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (idx, &logit) in logits.iter().enumerate() {
            if logit > max_val {
                let mut is_already_selected = false;
                for j in 0..i {
                    if top_k[j] == idx as i32 {
                        is_already_selected = true;
                        break;
                    }
                }

                if !is_already_selected {
                    max_val = logit;
                    max_idx = idx;
                }
            }
        }

        top_k[i] = max_idx as i32;
    }

    top_k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_temperature() {
        let mut logits = [1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        assert_eq!(logits, [0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_softmax() {
        let mut logits = [1.0, 2.0, 3.0];
        softmax(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

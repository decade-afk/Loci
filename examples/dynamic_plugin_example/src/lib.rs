//! Example dynamic plugin for Loci
//!
//! This plugin can be compiled as a shared library and loaded at runtime.
//! Build with: cargo build --release
//! The output will be in target/release/example_dynamic_plugin.dll (Windows)

use loci::plugin::Plugin;
use loci::error::Result;

/// A dynamic plugin that converts text to ROT13
pub struct Rot13Plugin {
    name: String,
}

impl Rot13Plugin {
    pub fn new() -> Self {
        Self {
            name: "rot13_dynamic".to_string(),
        }
    }

    fn rot13(&self, text: &str) -> String {
        text.chars()
            .map(|c| {
                if c.is_ascii_alphabetic() {
                    let base = if c.is_ascii_lowercase() { b'a' } else { b'A' };
                    let offset = (c as u8 - base + 13) % 26;
                    (base + offset) as char
                } else {
                    c
                }
            })
            .collect()
    }
}

impl Plugin for Rot13Plugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        "1.0.0"
    }

    fn init(&mut self) -> Result<()> {
        println!("[Rot13Plugin] Dynamic plugin initialized!");
        Ok(())
    }

    fn post_generate(&self, response: &str) -> Result<String> {
        Ok(self.rot13(response))
    }

    fn cleanup(&mut self) -> Result<()> {
        println!("[Rot13Plugin] Cleanup");
        Ok(())
    }
}

/// Required export for dynamic loading
/// This function is called by the PluginRegistry to create an instance
#[no_mangle]
pub extern "C" fn create_plugin() -> *mut dyn Plugin {
    let plugin = Rot13Plugin::new();
    Box::into_raw(Box::new(plugin))
}

/// Optional: Destroy function for cleanup
#[no_mangle]
pub extern "C" fn destroy_plugin(ptr: *mut dyn Plugin) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

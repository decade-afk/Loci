#![allow(clippy::missing_safety_doc)]

// The native bridge currently exposes its ABI from the C/C++ side.
// This Rust crate exists to participate in the workspace build graph and produce a cdylib.

#[no_mangle]
pub extern "C" fn loci_backend_openvino_bridge_anchor() -> u32 {
    1
}

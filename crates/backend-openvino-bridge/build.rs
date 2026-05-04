fn main() {
    println!("cargo:rerun-if-changed=native/bridge.cpp");
    println!("cargo:rerun-if-changed=native/materialize_text.cpp");
    println!("cargo:rerun-if-changed=native/bridge.h");
    println!("cargo:rerun-if-changed=include/loci_openvino_bridge.h");

    cc::Build::new()
        .cpp(true)
        .include("include")
        .include("native")
        .file("native/bridge.cpp")
        .file("native/materialize_text.cpp")
        .warnings(false)
        .compile("loci_openvino_bridge_native");
}

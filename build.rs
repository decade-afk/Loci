use std::env;
use std::path::PathBuf;

/// Path to the llama.cpp dependency
const LLAMA_CPP_PATH: &str = "deps/llama.cpp";

fn main() {
    // Tell Cargo to rerun this build script when the llama.cpp dependency changes
    println!("cargo:rerun-if-changed={}", LLAMA_CPP_PATH);

    // Get the output directory where generated files should be placed
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Configure CMake to build llama.cpp with specific options
    let mut config = cmake::Config::new(LLAMA_CPP_PATH);
    config
        // Disable shared library building to use static linking
        .define("BUILD_SHARED_LIBS", "OFF")
        // Disable building tests to reduce compilation time
        .define("LLAMA_BUILD_TESTS", "OFF")
        // Disable building examples to reduce compilation time
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        // Disable building server to reduce compilation time
        .define("LLAMA_BUILD_SERVER", "OFF")
        // Disable building tools to reduce compilation time
        .define("LLAMA_BUILD_TOOLS", "OFF")
        // Disable curl dependency to simplify build process
        .define("LLAMA_CURL", "OFF")
        // Disable native optimizations to improve portability
        .define("GGML_NATIVE", "OFF")
        // Enable OpenMP for CPU parallelization
        .define("GGML_OPENMP", "ON");

    // Configure GPU backends based on feature flags
    #[cfg(feature = "cuda")]
    {
        println!("cargo:warning=Building with CUDA support");
        config.define("GGML_CUDA", "ON");
        // Note: CUDA libraries will be linked automatically by llama.cpp CMake
    }

    #[cfg(feature = "metal")]
    {
        println!("cargo:warning=Building with Metal support");
        config.define("GGML_METAL", "ON");
    }

    #[cfg(feature = "vulkan")]
    {
        println!("cargo:warning=Building with Vulkan support");
        config.define("GGML_VULKAN", "ON");
    }

    #[cfg(feature = "rocm")]
    {
        println!("cargo:warning=Building with ROCm support");
        config.define("GGML_HIPBLAS", "ON");  // ROCm uses HIP BLAS
    }

    #[cfg(feature = "opencl")]
    {
        println!("cargo:warning=Building with OpenCL support");
        config.define("GGML_CLBLAST", "ON");
    }

    // Fix for "file too big" error on Windows
    // MinGW uses -Wa,-mbig-obj, MSVC uses /bigobj
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("windows-gnu") {
        // MinGW toolchain
        config.cxxflag("-Wa,-mbig-obj");
        config.cflag("-Wa,-mbig-obj");
    } else if target.contains("windows-msvc") {
        // MSVC toolchain
        config.cxxflag("/bigobj");
        config.cflag("/bigobj");
    }

    let dst = config.build();

    // Specify library search paths for linking
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // Link llama.cpp libraries based on the target platform
    link_libraries(&target);

    // Link system libraries based on the target platform
    link_system_libraries(&target);

    // Generate Rust bindings for llama.cpp header
    let bindings = bindgen::Builder::default()
        .header("deps/llama.cpp/include/llama.h")
        .clang_arg(format!("-I{}/include", dst.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

/// Links the required llama.cpp libraries based on the target platform
fn link_libraries(target: &str) {
    if target.contains("windows-gnu") {
        // MinGW uses .a files with lib prefix
        println!("cargo:rustc-link-lib=static:+verbatim=libllama.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml-cpu.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml-base.a");
    } else if target.contains("windows-msvc") {
        // MSVC uses .lib files without lib prefix
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
    } else {
        // Unix-like systems (Linux, macOS, etc.)
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
    }
}

/// Links the required system libraries based on the target platform
fn link_system_libraries(target: &str) {
    if target.contains("windows-gnu") {
        // MinGW specific libraries
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-lib=dylib=winpthread");
        println!("cargo:rustc-link-lib=dylib=advapi32");
    } else if target.contains("windows-msvc") {
        // MSVC specific libraries
        println!("cargo:rustc-link-lib=dylib=advapi32");
        // Note: MSVC uses its own C++ runtime and doesn't need explicit stdc++/gomp/winpthread
    } else {
        // Unix-like systems (Linux, macOS, etc.)
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
}
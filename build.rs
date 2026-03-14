use std::env;
use std::path::PathBuf;

/// Path to the llama.cpp dependency
const LLAMA_CPP_PATH: &str = "deps/llama.cpp";

fn main() {
    // Tell Cargo to rerun this build script when the llama.cpp dependency changes
    println!("cargo:rerun-if-changed={}", LLAMA_CPP_PATH);
    println!("cargo:rerun-if-changed=src/ffi_shim.c");
    println!("cargo:rerun-if-env-changed=LOCI_CPU_OPT");
    println!("cargo:rerun-if-env-changed=LOCI_CMAKE_BUILD_JOBS");

    // Get the output directory where generated files should be placed
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let target = env::var("TARGET").unwrap_or_default();

    if let Some(cmake_jobs) = resolve_cmake_build_jobs(&target) {
        env::set_var("NUM_JOBS", cmake_jobs);
    }

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
        // Avoid noisy warnings about optional build-cache tools on machines
        // where ccache is not installed.
        .define("GGML_CCACHE", "OFF")
        // Disable native optimizations to improve portability
        .define("GGML_NATIVE", "OFF")
        // Disable OpenMP for better Windows MinGW runtime stability.
        .define("GGML_OPENMP", "OFF");

    // Windows MinGW optimization tiers (stability first):
    // - safe : disable SIMD extensions
    // - sse42: enable only SSE4.2
    // - avx  : enable AVX (+SSE4.2), keep AVX2/FMA/F16C/BMI2 off
    // - avx2 : enable AVX2 stack (highest perf, may be less stable on some setups)
    if target.contains("windows-gnu") {
        let opt = env::var("LOCI_CPU_OPT").unwrap_or_else(|_| "sse42".to_string());
        match opt.as_str() {
            "safe" => {
                config
                    .define("GGML_SSE42", "OFF")
                    .define("GGML_AVX", "OFF")
                    .define("GGML_AVX2", "OFF")
                    .define("GGML_FMA", "OFF")
                    .define("GGML_F16C", "OFF")
                    .define("GGML_BMI2", "OFF");
            }
            "avx" => {
                config
                    .define("GGML_SSE42", "ON")
                    .define("GGML_AVX", "ON")
                    .define("GGML_AVX2", "OFF")
                    .define("GGML_FMA", "OFF")
                    .define("GGML_F16C", "OFF")
                    .define("GGML_BMI2", "OFF");
            }
            "avx2" => {
                config
                    .define("GGML_SSE42", "ON")
                    .define("GGML_AVX", "ON")
                    .define("GGML_AVX2", "ON")
                    .define("GGML_FMA", "ON")
                    .define("GGML_F16C", "ON")
                    .define("GGML_BMI2", "ON");
            }
            _ => {
                // Default to a balanced, safer tier.
                config
                    .define("GGML_SSE42", "ON")
                    .define("GGML_AVX", "OFF")
                    .define("GGML_AVX2", "OFF")
                    .define("GGML_FMA", "OFF")
                    .define("GGML_F16C", "OFF")
                    .define("GGML_BMI2", "OFF");
            }
        }
    }

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
        config.define("GGML_HIPBLAS", "ON"); // ROCm uses HIP BLAS
    }

    #[cfg(feature = "opencl")]
    {
        println!("cargo:warning=Building with OpenCL support");
        config.define("GGML_CLBLAST", "ON");
    }

    // Fix for "file too big" error on Windows
    // MinGW uses -Wa,-mbig-obj, MSVC uses /bigobj
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

    // Build a small C shim to avoid by-value FFI calls for llama_batch on Windows.
    cc::Build::new()
        .file("src/ffi_shim.c")
        .include("deps/llama.cpp/include")
        .include("deps/llama.cpp/ggml/include")
        .compile("loci_ffi_shim");

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

fn resolve_cmake_build_jobs(target: &str) -> Option<String> {
    match env::var("LOCI_CMAKE_BUILD_JOBS") {
        Ok(raw) if !raw.trim().is_empty() => Some(raw),
        _ if target.contains("windows-msvc") => Some("1".to_string()),
        _ => None,
    }
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
        println!("cargo:rustc-link-lib=dylib=advapi32");
    } else if target.contains("windows-msvc") {
        // MSVC specific libraries
        println!("cargo:rustc-link-lib=dylib=advapi32");
        // Note: MSVC uses its own C++ runtime and doesn't need explicit stdc++/gomp/winpthread
    } else {
        // Unix-like systems (Linux, macOS, etc.)
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
    }
}

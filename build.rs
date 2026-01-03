use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=deps/llama.cpp");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 使用 CMake 编译 llama.cpp
    let mut config = cmake::Config::new("deps/llama.cpp");
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "ON");

    // GPU backend configuration based on feature flags
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

    // 链接库
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // 根据目标平台链接 llama.cpp 库
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

    // 链接系统库
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

    // 生成绑定
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

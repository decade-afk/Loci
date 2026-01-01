// Loci Phase 3: Mobile Build Script
// Supports Android NDK and iOS cross-compilation

use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=IPHONEOS_DEPLOYMENT_TARGET");

    // Detect target platform and configure
    if target.contains("android") {
        setup_android_toolchain(&target, &out_dir);
    } else if target.contains("ios") {
        setup_ios_toolchain(&target, &out_dir);
    } else {
        setup_desktop_toolchain(&target);
    }

    // Configure llama.cpp compilation options (common to all platforms)
    configure_llama_cpp(&target);

    // Generate C header file (loci.h)
    generate_c_header();
}

/// Android NDK toolchain configuration
fn setup_android_toolchain(target: &str, _out_dir: &PathBuf) {
    println!("cargo:warning=Configuring Android NDK toolchain for target: {}", target);

    // Check NDK environment variables (supports multiple common variable names)
    let ndk_root = env::var("ANDROID_NDK_ROOT")
        .or_else(|_| env::var("NDK_HOME"))
        .or_else(|_| env::var("ANDROID_NDK_HOME"))
        .expect("ANDROID_NDK_ROOT, NDK_HOME, or ANDROID_NDK_HOME must be set for Android builds");

    println!("cargo:warning=Using Android NDK at: {}", ndk_root);

    // Select compiler based on target architecture (supports 4 mainstream ABIs)
    let (arch, abi, triple) = if target.contains("aarch64") {
        ("arm64-v8a", "aarch64-linux-android", "aarch64-linux-android")
    } else if target.contains("armv7") {
        ("armeabi-v7a", "armv7a-linux-androideabi", "armv7a-linux-androideabi")
    } else if target.contains("i686") {
        ("x86", "i686-linux-android", "i686-linux-android")
    } else if target.contains("x86_64") {
        ("x86_64", "x86_64-linux-android", "x86_64-linux-android")
    } else {
        panic!("Unsupported Android target: {}", target);
    };

    println!("cargo:warning=Android ABI: {} (arch={}, triple={})", abi, arch, triple);

    // Set API Level (minimum support for Android 7.0 Nougat)
    let api_level = env::var("ANDROID_API_LEVEL").unwrap_or_else(|_| "24".to_string());
    println!("cargo:warning=Android API Level: {}", api_level);

    // Detect NDK toolchain path (supports multiple operating systems)
    let host_tag = if cfg!(target_os = "windows") {
        "windows-x86_64"
    } else if cfg!(target_os = "macos") {
        "darwin-x86_64"
    } else if cfg!(target_os = "linux") {
        "linux-x86_64"
    } else {
        "linux-x86_64"
    };

    let toolchain_dir = PathBuf::from(&ndk_root)
        .join("toolchains")
        .join("llvm")
        .join("prebuilt")
        .join(host_tag);

    if !toolchain_dir.exists() {
        panic!("NDK toolchain not found at: {:?}", toolchain_dir);
    }

    println!("cargo:warning=Using NDK toolchain: {}", toolchain_dir.display());

    // Set linker search path
    let lib_dir = toolchain_dir
        .join("sysroot")
        .join("usr")
        .join("lib")
        .join(triple)
        .join(&api_level);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link Android system libraries
    println!("cargo:rustc-link-lib=log");       // Android logcat
    println!("cargo:rustc-link-lib=android");   // Android native API
    println!("cargo:rustc-link-lib=c++_shared"); // LLVM libc++

    // Optional: Link Vulkan (for GPU acceleration)
    if arch == "arm64-v8a" || arch == "x86_64" {
        println!("cargo:rustc-link-lib=vulkan");
    }

    // Configure C/C++ compiler environment variables (for building C dependencies)
    let cc_path = toolchain_dir.join("bin").join(format!("{}{}-clang", triple, api_level));
    let cxx_path = toolchain_dir.join("bin").join(format!("{}{}-clang++", triple, api_level));

    // Set environment variables (for cc crate)
    println!("cargo:rustc-env=CC={}", cc_path.display());
    println!("cargo:rustc-env=CXX={}", cxx_path.display());

    // Configure JNI header file path
    let jni_include = toolchain_dir.join("sysroot").join("usr").join("include");
    println!("cargo:include={}", jni_include.display());
}

/// iOS toolchain configuration
fn setup_ios_toolchain(target: &str, _out_dir: &PathBuf) {
    println!("cargo:warning=Configuring iOS toolchain for target: {}", target);

    // Determine iOS SDK type (device vs simulator)
    let (sdk, arch, is_simulator) = if target.contains("aarch64-apple-ios-sim") {
        ("iphonesimulator", "arm64", true)  // Apple Silicon Mac simulator
    } else if target.contains("aarch64") && !target.contains("sim") {
        ("iphoneos", "arm64", false)        // Real iOS device (iPhone/iPad)
    } else if target.contains("x86_64") {
        ("iphonesimulator", "x86_64", true) // Intel Mac simulator
    } else if target.contains("arm64e") {
        ("iphoneos", "arm64e", false)       // iOS 14+ new architecture (reserved for future)
    } else {
        panic!("Unsupported iOS target: {}", target);
    };

    println!("cargo:warning=iOS SDK: {} (arch={}, simulator={})", sdk, arch, is_simulator);

    // Set minimum iOS version (iOS 14+ required, supports Metal + Neural Engine)
    let min_version = env::var("IPHONEOS_DEPLOYMENT_TARGET")
        .unwrap_or_else(|_| "14.0".to_string());

    println!("cargo:rustc-env=IPHONEOS_DEPLOYMENT_TARGET={}", min_version);
    println!("cargo:warning=Minimum iOS version: {}", min_version);

    // Link iOS system frameworks (Metal required for GPU acceleration)
    println!("cargo:rustc-link-lib=framework=Foundation");   // Base framework
    println!("cargo:rustc-link-lib=framework=Metal");        // GPU acceleration (required)
    println!("cargo:rustc-link-lib=framework=MetalKit");     // Metal helper tools
    println!("cargo:rustc-link-lib=framework=Accelerate");   // SIMD optimization (BLAS/LAPACK)

    // Optional: Link CoreML (Phase 4 may use for model acceleration)
    if env::var("ENABLE_COREML").is_ok() {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:warning=CoreML framework enabled");
    }

    // Configure bitcode (iOS App Store requirement, device only)
    if !is_simulator && sdk == "iphoneos" {
        println!("cargo:rustc-link-arg=-fembed-bitcode");
        println!("cargo:warning=Bitcode enabled for App Store submission");
    }

    // Configure architecture-specific optimization flags
    if arch == "arm64" || arch == "arm64e" {
        // Enable ARM NEON SIMD instructions
        println!("cargo:rustc-cfg=feature=\"neon\"");
        println!("cargo:warning=ARM NEON SIMD enabled");
    }

    // Set Xcode SDK path (for finding system header files)
    if let Ok(sdk_path) = env::var("SDKROOT") {
        println!("cargo:warning=Using SDK at: {}", sdk_path);
    } else {
        // Auto-detect Xcode SDK (using xcrun command)
        let sdk_output = std::process::Command::new("xcrun")
            .args(&["--sdk", sdk, "--show-sdk-path"])
            .output();

        if let Ok(output) = sdk_output {
            if output.status.success() {
                let sdk_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:warning=Auto-detected SDK: {}", sdk_path);
                println!("cargo:rustc-env=SDKROOT={}", sdk_path);
            }
        }
    }

    // Universal Binary hint (requires cargo-lipo)
    println!("cargo:warning=For universal binary (device + simulator), use: cargo lipo");
}

/// Desktop platform toolchain configuration (Windows/Linux/macOS)
fn setup_desktop_toolchain(target: &str) {
    println!("cargo:warning=Configuring desktop toolchain for target: {}", target);

    // macOS specific configuration
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    // Windows specific configuration
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=dylib=kernel32");
        println!("cargo:rustc-link-lib=dylib=user32");
    }

    // Linux specific configuration
    if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}

/// llama.cpp compilation configuration (cross-platform common)
fn configure_llama_cpp(target: &str) {
    println!("cargo:warning=Configuring llama.cpp for target: {}", target);

    // llama.cpp submodule path
    let llama_dir = PathBuf::from("deps/llama.cpp");

    // Check if llama.cpp exists
    if !llama_dir.exists() {
        println!("cargo:warning=llama.cpp submodule not found at deps/llama.cpp");
        println!("cargo:warning=Please run: git submodule update --init --recursive");
        // Don't panic, allow build to continue (using placeholder implementation)
        return;
    }

    // Use cmake to build llama.cpp
    let mut cmake_config = cmake::Config::new(&llama_dir);

    // Enable corresponding acceleration features based on target platform
    if target.contains("android") || target.contains("ios") {
        // Mobile platforms prioritize NEON (ARM SIMD)
        if target.contains("aarch64") || target.contains("arm") {
            cmake_config.define("LLAMA_ARM_NEON", "ON");
            println!("cargo:rustc-cfg=feature=\"neon\"");
        }
    } else {
        // Desktop platforms select SIMD based on CPU architecture
        if target.contains("x86_64") {
            cmake_config.define("LLAMA_AVX2", "ON");
            println!("cargo:rustc-cfg=feature=\"avx2\"");
        }
    }

    // GPU acceleration configuration
    if target.contains("apple") {
        cmake_config.define("LLAMA_METAL", "ON");
        println!("cargo:rustc-cfg=feature=\"metal\"");
    } else if target.contains("linux") && env::var("CUDA_AVAILABLE").is_ok() {
        cmake_config.define("LLAMA_CUDA", "ON");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }

    // Build static library
    cmake_config.define("BUILD_SHARED_LIBS", "OFF");
    cmake_config.define("LLAMA_BUILD_TESTS", "OFF");
    cmake_config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    cmake_config.define("LLAMA_CURL", "OFF");

    // Execute build
    let dst = cmake_config.build();

    // Add multiple possible library search paths (llama.cpp directory structure may vary by version)
    let possible_lib_dirs = vec![
        dst.join("lib"),
        dst.join("lib64"),
        dst.join("build/ggml/src"),
        dst.join("build/src"),
        dst.join("ggml/src"),
        dst.join("src"),
        dst.join("bin"),  // Windows may place DLL in bin directory
    ];

    for lib_dir in &possible_lib_dirs {
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:warning=✅ Found library directory: {}", lib_dir.display());
        } else {
            println!("cargo:warning=⚠️ Library directory not found: {}", lib_dir.display());
        }
    }

    // Link ggml and llama static libraries (note order: llama depends on ggml, so llama comes first)
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static:+verbatim=ggml.a");
    println!("cargo:rustc-link-lib=static:+verbatim=ggml-base.a");
    println!("cargo:rustc-link-lib=static:+verbatim=ggml-cpu.a");

    // Link C++ standard library
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");  // OpenMP for Linux
    } else if target.contains("windows") {
        // Windows MinGW uses dynamically linked C++ runtime
        if target.contains("gnu") {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=gomp");  // OpenMP for MinGW
        }
    }

    // Optimization level hint
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".to_string());
    if opt_level == "3" || opt_level == "z" || opt_level == "s" {
        println!("cargo:warning=Building with optimization level: {}", opt_level);
    }
}

/// Generate C header file (loci.h)
///
/// Uses cbindgen to generate C/C++ compatible header file from Rust FFI code
fn generate_c_header() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&crate_dir).join("loci.h");

    println!("cargo:warning=Generating C header file at: {}", header_path.display());
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=src/mobile_ffi.rs");

    // Use cbindgen to generate header file
    match cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("LOCI_H")
        .with_documentation(true)
        .with_parse_deps(true)
        .with_parse_include(&["loci"])
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header_path);
            println!("cargo:warning=✅ C header file generated successfully: {}", header_path.display());
        }
        Err(e) => {
            println!("cargo:warning=⚠️ Failed to generate C header: {}", e);
            println!("cargo:warning=Creating fallback header file...");

            // Create a basic header file as fallback
            create_fallback_header(&header_path);
        }
    }
}

/// Create fallback C header file (when cbindgen fails)
fn create_fallback_header(path: &PathBuf) {
    use std::fs::File;
    use std::io::Write;

    let fallback_content = r#"#ifndef LOCI_H
#define LOCI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Loci AI Engine C API
// This is a fallback header. Please run cbindgen to generate the full API.

// Error codes
#define LOCI_OK 0
#define LOCI_ERR_NULL_POINTER -1
#define LOCI_ERR_INVALID_ARG -2
#define LOCI_ERR_GENERATION_FAILED -3

// Opaque handle types
typedef void* LociEngine;
typedef void* LociSession;

// Basic functions (placeholders - see full API documentation)
// int32_t loci_engine_new(const char* model_path, LociEngine* out_engine);
// int32_t loci_engine_free(LociEngine engine);
// int32_t loci_generate(LociEngine engine, const char* prompt, int32_t max_tokens, char* out_text, int32_t out_len);

#ifdef __cplusplus
}
#endif

#endif // LOCI_H
"#;

    if let Ok(mut file) = File::create(path) {
        let _ = file.write_all(fallback_content.as_bytes());
        println!("cargo:warning=✅ Fallback header created at: {}", path.display());
    } else {
        println!("cargo:warning=❌ Failed to create fallback header");
    }
}

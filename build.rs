// Loci Phase 3: 移动端编译构建脚本
// 支持 Android NDK 和 iOS 交叉编译

use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=IPHONEOS_DEPLOYMENT_TARGET");

    // 检测目标平台并配置
    if target.contains("android") {
        setup_android_toolchain(&target, &out_dir);
    } else if target.contains("ios") {
        setup_ios_toolchain(&target, &out_dir);
    } else {
        setup_desktop_toolchain(&target);
    }

    // 配置 llama.cpp 编译选项（所有平台通用）
    configure_llama_cpp(&target);
}

/// Android NDK 工具链配置
fn setup_android_toolchain(target: &str, _out_dir: &PathBuf) {
    println!("cargo:warning=Configuring Android NDK toolchain for target: {}", target);

    // 检查 NDK 环境变量（支持多种常见环境变量名）
    let ndk_root = env::var("ANDROID_NDK_ROOT")
        .or_else(|_| env::var("NDK_HOME"))
        .or_else(|_| env::var("ANDROID_NDK_HOME"))
        .expect("ANDROID_NDK_ROOT, NDK_HOME, or ANDROID_NDK_HOME must be set for Android builds");

    println!("cargo:warning=Using Android NDK at: {}", ndk_root);

    // 根据目标架构选择编译器（支持 4 种主流 ABI）
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

    // 设置 API Level（最低支持 Android 7.0 Nougat）
    let api_level = env::var("ANDROID_API_LEVEL").unwrap_or_else(|_| "24".to_string());
    println!("cargo:warning=Android API Level: {}", api_level);

    // 检测 NDK 工具链路径（支持多种操作系统）
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

    // 设置链接器搜索路径
    let lib_dir = toolchain_dir
        .join("sysroot")
        .join("usr")
        .join("lib")
        .join(triple)
        .join(&api_level);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // 链接 Android 系统库
    println!("cargo:rustc-link-lib=log");       // Android logcat
    println!("cargo:rustc-link-lib=android");   // Android native API
    println!("cargo:rustc-link-lib=c++_shared"); // LLVM libc++

    // 可选：链接 Vulkan（用于 GPU 加速）
    if arch == "arm64-v8a" || arch == "x86_64" {
        println!("cargo:rustc-link-lib=vulkan");
    }

    // 配置 C/C++ 编译器环境变量（用于构建 C 依赖）
    let cc_path = toolchain_dir.join("bin").join(format!("{}{}-clang", triple, api_level));
    let cxx_path = toolchain_dir.join("bin").join(format!("{}{}-clang++", triple, api_level));

    // 设置环境变量（供 cc crate 使用）
    println!("cargo:rustc-env=CC={}", cc_path.display());
    println!("cargo:rustc-env=CXX={}", cxx_path.display());

    // 配置 JNI 头文件路径
    let jni_include = toolchain_dir.join("sysroot").join("usr").join("include");
    println!("cargo:include={}", jni_include.display());
}

/// iOS 工具链配置
fn setup_ios_toolchain(target: &str, _out_dir: &PathBuf) {
    println!("cargo:warning=Configuring iOS toolchain for target: {}", target);

    // 确定 iOS SDK 类型（设备 vs 模拟器）
    let (sdk, arch, is_simulator) = if target.contains("aarch64-apple-ios-sim") {
        ("iphonesimulator", "arm64", true)  // Apple Silicon Mac 模拟器
    } else if target.contains("aarch64") && !target.contains("sim") {
        ("iphoneos", "arm64", false)        // 真实 iOS 设备（iPhone/iPad）
    } else if target.contains("x86_64") {
        ("iphonesimulator", "x86_64", true) // Intel Mac 模拟器
    } else if target.contains("arm64e") {
        ("iphoneos", "arm64e", false)       // iOS 14+ 新架构（未来预留）
    } else {
        panic!("Unsupported iOS target: {}", target);
    };

    println!("cargo:warning=iOS SDK: {} (arch={}, simulator={})", sdk, arch, is_simulator);

    // 设置最低 iOS 版本（iOS 14+ 必需，支持 Metal + Neural Engine）
    let min_version = env::var("IPHONEOS_DEPLOYMENT_TARGET")
        .unwrap_or_else(|_| "14.0".to_string());

    println!("cargo:rustc-env=IPHONEOS_DEPLOYMENT_TARGET={}", min_version);
    println!("cargo:warning=Minimum iOS version: {}", min_version);

    // 链接 iOS 系统框架（Metal 必须，用于 GPU 加速）
    println!("cargo:rustc-link-lib=framework=Foundation");   // 基础框架
    println!("cargo:rustc-link-lib=framework=Metal");        // GPU 加速（必需）
    println!("cargo:rustc-link-lib=framework=MetalKit");     // Metal 辅助工具
    println!("cargo:rustc-link-lib=framework=Accelerate");   // SIMD 优化（BLAS/LAPACK）

    // 可选：链接 CoreML（Phase 4 可能用于模型加速）
    if env::var("ENABLE_COREML").is_ok() {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:warning=CoreML framework enabled");
    }

    // 配置 bitcode（iOS App Store 要求，仅真机）
    if !is_simulator && sdk == "iphoneos" {
        println!("cargo:rustc-link-arg=-fembed-bitcode");
        println!("cargo:warning=Bitcode enabled for App Store submission");
    }

    // 配置架构特定的优化标志
    if arch == "arm64" || arch == "arm64e" {
        // 启用 ARM NEON SIMD 指令
        println!("cargo:rustc-cfg=feature=\"neon\"");
        println!("cargo:warning=ARM NEON SIMD enabled");
    }

    // 设置 Xcode SDK 路径（用于查找系统头文件）
    if let Ok(sdk_path) = env::var("SDKROOT") {
        println!("cargo:warning=Using SDK at: {}", sdk_path);
    } else {
        // 自动检测 Xcode SDK（使用 xcrun 命令）
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

    // Universal Binary 提示（需要 cargo-lipo）
    println!("cargo:warning=For universal binary (device + simulator), use: cargo lipo");
}

/// 桌面平台工具链配置（Windows/Linux/macOS）
fn setup_desktop_toolchain(target: &str) {
    println!("cargo:warning=Configuring desktop toolchain for target: {}", target);

    // macOS 特殊配置
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    // Windows 特殊配置
    if target.contains("windows") {
        println!("cargo:rustc-link-lib=dylib=kernel32");
        println!("cargo:rustc-link-lib=dylib=user32");
    }

    // Linux 特殊配置
    if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
    }
}

/// llama.cpp 编译配置（跨平台通用）
fn configure_llama_cpp(target: &str) {
    println!("cargo:warning=Configuring llama.cpp for target: {}", target);

    // llama.cpp 子模块路径
    let llama_dir = PathBuf::from("deps/llama.cpp");

    // 检查 llama.cpp 是否存在
    if !llama_dir.exists() {
        println!("cargo:warning=llama.cpp submodule not found at deps/llama.cpp");
        println!("cargo:warning=Please run: git submodule update --init --recursive");
        // 不panic，允许继续构建（使用占位实现）
        return;
    }

    // 使用 cmake 构建 llama.cpp
    let mut cmake_config = cmake::Config::new(&llama_dir);

    // 根据目标平台启用对应的加速特性
    if target.contains("android") || target.contains("ios") {
        // 移动端优先使用 NEON（ARM SIMD）
        if target.contains("aarch64") || target.contains("arm") {
            cmake_config.define("LLAMA_ARM_NEON", "ON");
            println!("cargo:rustc-cfg=feature=\"neon\"");
        }
    } else {
        // 桌面端根据 CPU 架构选择 SIMD
        if target.contains("x86_64") {
            cmake_config.define("LLAMA_AVX2", "ON");
            println!("cargo:rustc-cfg=feature=\"avx2\"");
        }
    }

    // GPU 加速配置
    if target.contains("apple") {
        cmake_config.define("LLAMA_METAL", "ON");
        println!("cargo:rustc-cfg=feature=\"metal\"");
    } else if target.contains("linux") && env::var("CUDA_AVAILABLE").is_ok() {
        cmake_config.define("LLAMA_CUDA", "ON");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }

    // 构建静态库
    cmake_config.define("BUILD_SHARED_LIBS", "OFF");
    cmake_config.define("LLAMA_BUILD_TESTS", "OFF");
    cmake_config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    cmake_config.define("LLAMA_CURL", "OFF");

    // 执行构建
    let dst = cmake_config.build();

    // 添加库搜索路径
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // 添加ggml库搜索路径
    println!("cargo:rustc-link-search=native={}/build/ggml/src", dst.display());

    // 链接 ggml 和 llama 静态库（注意顺序：llama依赖ggml，所以llama在前）
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static:+verbatim=ggml.a");
    println!("cargo:rustc-link-lib=static:+verbatim=ggml-base.a");
    println!("cargo:rustc-link-lib=static:+verbatim=ggml-cpu.a");

    // 链接 C++ 标准库
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");  // OpenMP for Linux
    } else if target.contains("windows") {
        // Windows MinGW 使用动态链接 C++ 运行时
        if target.contains("gnu") {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=gomp");  // OpenMP for MinGW
        }
    }

    // 优化级别提示
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".to_string());
    if opt_level == "3" || opt_level == "z" || opt_level == "s" {
        println!("cargo:warning=Building with optimization level: {}", opt_level);
    }
}

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    configure_bindgen_environment(&target);

    // Generate Rust bindings for llama.cpp header
    let mut bindings = bindgen::Builder::default()
        .header("deps/llama.cpp/include/llama.h")
        .clang_arg(format!("-I{}/include", dst.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    if target.contains("windows-gnu") && env::var_os("BINDGEN_EXTRA_CLANG_ARGS").is_none() {
        if let Some(mingw_include_dir) = resolve_mingw_include_dir() {
            println!(
                "cargo:warning=Using detected MinGW headers from {}",
                mingw_include_dir.display()
            );
            bindings = bindings.clang_arg(format!("-I{}", mingw_include_dir.display()));
        }
    }

    let bindings = bindings.generate().expect("Unable to generate bindings");

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

fn configure_bindgen_environment(target: &str) {
    if env::var_os("LIBCLANG_PATH").is_some() {
        return;
    }

    if let Some(libclang_dir) = resolve_libclang_dir(target) {
        println!(
            "cargo:warning=Using detected libclang from {}",
            libclang_dir.display()
        );
        env::set_var("LIBCLANG_PATH", libclang_dir);
    }
}

fn resolve_libclang_dir(target: &str) -> Option<PathBuf> {
    if target.contains("windows") {
        resolve_windows_libclang_dir()
    } else if target.contains("linux") {
        resolve_linux_libclang_dir()
    } else if target.contains("apple") {
        resolve_macos_libclang_dir()
    } else {
        None
    }
}

fn resolve_windows_libclang_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(program_files) = env::var_os("ProgramFiles") {
        candidates.push(PathBuf::from(&program_files).join("LLVM\\bin"));
        candidates.extend(collect_visual_studio_llvm_dirs(PathBuf::from(
            program_files,
        )));
    }

    if let Some(program_files_x86) = env::var_os("ProgramFiles(x86)") {
        candidates.push(PathBuf::from(&program_files_x86).join("LLVM\\bin"));
        candidates.extend(collect_visual_studio_llvm_dirs(PathBuf::from(
            program_files_x86,
        )));
    }

    candidates.extend([
        PathBuf::from(r"C:\Program Files\LLVM\bin"),
        PathBuf::from(r"C:\Program Files (x86)\LLVM\bin"),
        PathBuf::from(r"D:\Program Files\LLVM\bin"),
        PathBuf::from(r"D:\Program Files (x86)\LLVM\bin"),
    ]);
    candidates.extend(collect_visual_studio_llvm_dirs(PathBuf::from(
        r"C:\Program Files",
    )));
    candidates.extend(collect_visual_studio_llvm_dirs(PathBuf::from(
        r"C:\Program Files (x86)",
    )));
    candidates.extend(collect_visual_studio_llvm_dirs(PathBuf::from(
        r"D:\Program Files",
    )));
    candidates.extend(collect_visual_studio_llvm_dirs(PathBuf::from(
        r"D:\Program Files (x86)",
    )));

    first_dir_with_any_file(candidates, &["libclang.dll", "clang.dll"])
}

fn collect_visual_studio_llvm_dirs(base: PathBuf) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let root = base.join("Microsoft Visual Studio");

    if let Ok(years) = fs::read_dir(root) {
        for year in years.flatten() {
            if let Ok(editions) = fs::read_dir(year.path()) {
                for edition in editions.flatten() {
                    dirs.push(edition.path().join("VC\\Tools\\Llvm\\x64\\bin"));
                }
            }
        }
    }

    dirs
}

fn resolve_linux_libclang_dir() -> Option<PathBuf> {
    if let Some(dir) = llvm_config_libdir() {
        return Some(dir);
    }

    let mut candidates = vec![
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/lib/x86_64-linux-gnu"),
    ];

    if let Ok(entries) = fs::read_dir("/usr/lib") {
        let mut llvm_dirs: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("llvm-"))
                    .unwrap_or(false)
            })
            .map(|path| path.join("lib"))
            .collect();
        llvm_dirs.sort();
        llvm_dirs.reverse();
        candidates.extend(llvm_dirs);
    }

    first_dir_with_any_file(candidates, &["libclang.so", "libclang.so.1"])
}

fn resolve_macos_libclang_dir() -> Option<PathBuf> {
    if let Some(dir) = llvm_config_libdir() {
        return Some(dir);
    }

    if let Some(xcode_root) = command_output("xcode-select", &["-p"]) {
        let candidate =
            PathBuf::from(xcode_root).join("Toolchains/XcodeDefault.xctoolchain/usr/lib");
        if dir_has_any_file(&candidate, &["libclang.dylib"]) {
            return Some(candidate);
        }
    }

    first_dir_with_any_file(
        vec![
            PathBuf::from("/Library/Developer/CommandLineTools/usr/lib"),
            PathBuf::from(
                "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib",
            ),
            PathBuf::from("/opt/homebrew/opt/llvm/lib"),
            PathBuf::from("/usr/local/opt/llvm/lib"),
        ],
        &["libclang.dylib"],
    )
}

fn resolve_mingw_include_dir() -> Option<PathBuf> {
    [
        PathBuf::from("C:/msys64/mingw64/x86_64-w64-mingw32/include"),
        PathBuf::from("D:/mingw64/x86_64-w64-mingw32/include"),
        PathBuf::from("/usr/x86_64-w64-mingw32/include"),
    ]
    .into_iter()
    .find(|dir| dir.join("stddef.h").exists() || dir.join("stdio.h").exists())
}

fn llvm_config_libdir() -> Option<PathBuf> {
    let commands = ["llvm-config", "llvm-config.exe"];

    for command in commands {
        if let Some(output) = command_output(command, &["--libdir"]) {
            let dir = PathBuf::from(output);
            if dir_has_any_file(
                &dir,
                &[
                    "libclang.dll",
                    "clang.dll",
                    "libclang.so",
                    "libclang.so.1",
                    "libclang.dylib",
                ],
            ) {
                return Some(dir);
            }
        }
    }

    None
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn first_dir_with_any_file(candidates: Vec<PathBuf>, filenames: &[&str]) -> Option<PathBuf> {
    candidates
        .into_iter()
        .find(|dir| dir_has_any_file(dir, filenames))
}

fn dir_has_any_file(dir: &Path, filenames: &[&str]) -> bool {
    filenames.iter().any(|filename| dir.join(filename).exists())
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

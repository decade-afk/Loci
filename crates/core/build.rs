use std::env;
use std::path::{Path, PathBuf};

fn main() {
    if env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace root")
        .to_path_buf();
    let llama_cpp_path = workspace_root.join("deps").join("llama.cpp");
    let ffi_shim = manifest_dir
        .join("src")
        .join("backends")
        .join("llamacpp")
        .join("ffi_shim.c");

    println!("cargo:rerun-if-changed={}", llama_cpp_path.display());
    println!("cargo:rerun-if-changed={}", ffi_shim.display());
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");

    let target = env::var("TARGET").unwrap_or_default();
    configure_bindgen_environment(&target);

    let mut config = cmake::Config::new(&llama_cpp_path);
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_CCACHE", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "OFF");

    if target.contains("windows-gnu") {
        config.cxxflag("-Wa,-mbig-obj");
        config.cflag("-Wa,-mbig-obj");
    } else if target.contains("windows-msvc") {
        // Rust links the dynamic release CRT even for debug profiles on MSVC.
        // Keep the embedded llama.cpp build on the same runtime to avoid
        // unresolved `_calloc_dbg` / `CrtDbgReport` symbols when downstream
        // binaries such as `loci-cli` link loci-core in debug mode.
        config.profile("Release");
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
        config.cxxflag("/bigobj");
        config.cflag("/bigobj");
    }

    let dst = config.build();

    cc::Build::new()
        .file(&ffi_shim)
        .include(llama_cpp_path.join("include"))
        .include(llama_cpp_path.join("ggml").join("include"))
        .compile("loci_core_ffi_shim");

    for lib_dir in [dst.join("lib"), dst.join("lib64")] {
        if lib_dir.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    }

    link_libraries(&target);
    link_system_libraries(&target);

    let bindings = bindgen::Builder::default()
        .header(
            llama_cpp_path
                .join("include")
                .join("llama.h")
                .display()
                .to_string(),
        )
        .clang_arg(format!("-I{}", llama_cpp_path.join("include").display()))
        .clang_arg(format!(
            "-I{}",
            llama_cpp_path.join("ggml").join("include").display()
        ))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate llama.cpp bindings");

    bindings
        .write_to_file(Path::new(&env::var("OUT_DIR").expect("out dir")).join("llama_bindings.rs"))
        .expect("Couldn't write llama bindings");
}

fn configure_bindgen_environment(target: &str) {
    if env::var_os("LIBCLANG_PATH").is_some() {
        return;
    }

    if let Some(path) = resolve_libclang_dir(target) {
        println!(
            "cargo:warning=Using detected libclang from {}",
            path.display()
        );
        env::set_var("LIBCLANG_PATH", path);
    }
}

fn resolve_libclang_dir(target: &str) -> Option<PathBuf> {
    if !target.contains("windows") {
        return None;
    }

    libclang_candidates()
        .into_iter()
        .find(|dir| dir.join("libclang.dll").exists() || dir.join("clang.dll").exists())
}

fn libclang_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(llvm_home) = env::var_os("LLVM_HOME") {
        candidates.push(PathBuf::from(&llvm_home).join("bin"));
        candidates.push(PathBuf::from(llvm_home));
    }

    if let Some(conda_prefix) = env::var_os("CONDA_PREFIX") {
        candidates.push(PathBuf::from(conda_prefix).join("Library").join("bin"));
    }

    if let Some(program_files) = env::var_os("ProgramFiles") {
        candidates.push(PathBuf::from(&program_files).join("LLVM").join("bin"));
        candidates.push(
            PathBuf::from(program_files)
                .join("Microsoft Visual Studio")
                .join("2022")
                .join("BuildTools")
                .join("VC")
                .join("Tools")
                .join("Llvm")
                .join("x64")
                .join("bin"),
        );
    }

    if let Some(program_files_x86) = env::var_os("ProgramFiles(x86)") {
        candidates.push(PathBuf::from(&program_files_x86).join("LLVM").join("bin"));
        candidates.push(
            PathBuf::from(&program_files_x86)
                .join("Microsoft Visual Studio")
                .join("2022")
                .join("BuildTools")
                .join("VC")
                .join("Tools")
                .join("Llvm")
                .join("x64")
                .join("bin"),
        );
        candidates.push(
            PathBuf::from(program_files_x86)
                .join("Microsoft Visual Studio")
                .join("2022")
                .join("BuildTools")
                .join("VC")
                .join("Tools")
                .join("Llvm")
                .join("bin"),
        );
    }

    if let Some(path) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&path));
    }

    if let Some(user_profile) = env::var_os("USERPROFILE") {
        let conda_envs = PathBuf::from(user_profile).join(".conda").join("envs");
        if let Ok(entries) = std::fs::read_dir(conda_envs) {
            for entry in entries.flatten() {
                candidates.push(
                    entry
                        .path()
                        .join("Lib")
                        .join("site-packages")
                        .join("clang")
                        .join("native"),
                );
            }
        }
    }

    let mut deduped = Vec::new();
    for candidate in candidates {
        if !deduped
            .iter()
            .any(|existing: &PathBuf| existing == &candidate)
        {
            deduped.push(candidate);
        }
    }

    deduped
}

fn link_libraries(target: &str) {
    if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=static:+verbatim=libllama.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml-cpu.a");
        println!("cargo:rustc-link-lib=static:+verbatim=ggml-base.a");
    } else {
        println!("cargo:rustc-link-lib=static=llama");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
        println!("cargo:rustc-link-lib=static=ggml-base");
    }
}

fn link_system_libraries(target: &str) {
    if target.contains("windows-gnu") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=advapi32");
    } else if target.contains("windows-msvc") {
        println!("cargo:rustc-link-lib=dylib=advapi32");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=dylib=m");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        if target.contains("linux") {
            println!("cargo:rustc-link-lib=dylib=atomic");
        }
    }
}

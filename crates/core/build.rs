use std::env;
use std::path::{Path, PathBuf};

fn main() {
    if env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }

    reset_stale_cmake_cache();

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
    configure_git_safe_directory(&mut config, &llama_cpp_path);
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
        // The VS 2026 generator can fail nondeterministically when Cargo asks
        // CMake to fan out parallel jobs and MSBuild also enables /m. Keep the
        // inner MSBuild layer single-threaded while preserving Cargo's outer
        // parallelism.
        config.build_arg("/m:1");
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

fn configure_git_safe_directory(config: &mut cmake::Config, repo_path: &Path) {
    let existing_count = env::var("GIT_CONFIG_COUNT")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0);

    for index in 0..existing_count {
        let key = format!("GIT_CONFIG_KEY_{index}");
        let value = format!("GIT_CONFIG_VALUE_{index}");
        if let Some(existing_key) = env::var_os(&key) {
            config.env(&key, existing_key);
        }
        if let Some(existing_value) = env::var_os(&value) {
            config.env(&value, existing_value);
        }
    }

    config.env("GIT_CONFIG_COUNT", (existing_count + 1).to_string());
    config.env(format!("GIT_CONFIG_KEY_{existing_count}"), "safe.directory");
    config.env(
        format!("GIT_CONFIG_VALUE_{existing_count}"),
        repo_path.as_os_str(),
    );
}

fn reset_stale_cmake_cache() {
    let out_dir = match env::var_os("OUT_DIR") {
        Some(value) => PathBuf::from(value),
        None => return,
    };
    let build_dir = out_dir.join("build");
    let cache = build_dir.join("CMakeCache.txt");
    let cmake_files = build_dir.join("CMakeFiles");

    if cache.exists() {
        let _ = std::fs::remove_file(&cache);
    }
    if cmake_files.exists() {
        let _ = std::fs::remove_dir_all(&cmake_files);
    }
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

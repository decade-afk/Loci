use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=deps/llama.cpp");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 使用 CMake 编译 llama.cpp
    let dst = cmake::Config::new("deps/llama.cpp")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "ON")
        .build();

    // 链接库
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // 链接系统库
    #[cfg(target_os = "windows")]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
        println!("cargo:rustc-link-lib=dylib=winpthread");
        println!("cargo:rustc-link-lib=dylib=advapi32");
    }

    #[cfg(not(target_os = "windows"))]
    {
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

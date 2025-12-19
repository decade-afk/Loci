/**
 * Loci Build Script
 * 
 * 这个构建脚本负责：
 * 1. 生成 Flutter Rust Bridge 的 Dart 绑定代码
 * 2. 创建 bindings/dart 目录（如果不存在）
 * 3. 设置编译时的链接参数
 */

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/api.rs");
    println!("cargo:rerun-if-changed=src/");

    // TODO: 暂时禁用 Flutter Rust Bridge 绑定生成
    // 需要先安装 flutter_rust_bridge_codegen
    /*
    // 生成 Flutter Rust Bridge 绑定
    if let Err(e) = generate_frb_bindings() {
        println!("cargo:warning=Failed to generate Flutter Rust Bridge bindings: {}", e);
        println!("cargo:warning=Continuing without FRB bindings...");
    }
    */

    // 创建 bindings 目录结构
    create_bindings_directory();
}

/*
/// 生成 Flutter Rust Bridge 的 Dart 绑定代码
fn generate_frb_bindings() -> Result<(), Box<dyn std::error::Error>> {
    // 检查是否安装了 flutter_rust_bridge_codegen
    if std::process::Command::new("flutter_rust_bridge_codegen")
        .arg("--version")
        .output()
        .is_err()
    {
        println!("cargo:warning=flutter_rust_bridge_codegen not found, skipping FRB bindings");
        return Ok(());
    }
    
    println!("cargo:warning=Generating Flutter Rust Bridge bindings...");
    
    // 设置输出目录
    let out_dir = env::var("OUT_DIR")?;
    let bindings_path = Path::new(&out_dir).join("bridge_generated.rs");
    
    // 生成绑定代码
    flutter_rust_bridge_codegen::generate_default()
        .write_files_to_out_dir(true)?;
    
    println!("cargo:warning=Flutter Rust Bridge bindings generated successfully");
    Ok(())
}
*/

/// 创建 bindings 目录结构
fn create_bindings_directory() {
    let project_root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let bindings_dir = Path::new(&project_root).join("bindings");
    let dart_dir = bindings_dir.join("dart");
    
    // 创建目录
    if let Err(e) = fs::create_dir_all(&dart_dir) {
        println!("cargo:warning=Failed to create bindings directory: {}", e);
        return;
    }
    
    // 检查是否有生成的 Dart 文件
    let out_dir = env::var("OUT_DIR").unwrap();
    let generated_dart = Path::new(&out_dir).join("bridge_generated.dart");
    
    if generated_dart.exists() {
        // 复制生成的 Dart 文件到 bindings/dart
        let target_file = dart_dir.join("bridge_generated.dart");
        if let Err(e) = fs::copy(&generated_dart, &target_file) {
            println!("cargo:warning=Failed to copy Dart bindings: {}", e);
        } else {
            println!("cargo:warning=Dart bindings copied to bindings/dart/");
        }
    }
    
    // 创建 README 文件
    let readme_content = r#"# Dart Bindings

这个目录包含了由 Flutter Rust Bridge 自动生成的 Dart 绑定代码。

## 文件说明

- `bridge_generated.dart`: 主要的绑定文件，包含所有 Rust API 的 Dart 接口
- `README.md`: 本文件

## 使用方法

1. 将这些文件复制到 Flutter 项目的 `lib/` 目录
2. 在 Dart 代码中导入：
   ```dart
   import 'bridge_generated.dart';
   ```

## 自动生成

这些文件会在构建时自动生成。如果修改了 `src/api.rs` 中的接口，请重新构建项目：

```bash
cd loci
cargo build
```

然后更新 Flutter 项目中的绑定文件。
"#;
    
    let readme_path = dart_dir.join("README.md");
    if let Err(e) = fs::write(&readme_path, readme_content) {
        println!("cargo:warning=Failed to create README: {}", e);
    }
}
# Building Loci

This repository is now a Rust workspace. Build the crate that matches the surface you actually want to ship.

## Workspace Targets

- `loci-core`: embeddable runtime and management service
- `loci-cli`: `loci` binary
- `loci-plugin-api`: shared plugin manifest/types crate
- `loci-ffi`: native integration crate reserved for later ABI hardening

## Prerequisites

All platforms:

- Rust stable
- CMake
- a C/C++ toolchain

For the optional `llama` feature:

- `deps/llama.cpp` submodule initialized
- `libclang` available for bindgen

Windows-specific `libclang` resolution is handled in `crates/core/build.rs` through:

- `LIBCLANG_PATH`
- `LLVM_HOME`
- `CONDA_PREFIX`
- common LLVM / Visual Studio paths
- `PATH`

## Clone

```bash
git clone https://github.com/decade-afk/loci.git
cd loci
git submodule update --init --recursive
```

## Build Commands

Build the CLI without optional backend bindings:

```bash
cargo build -p loci-cli --release
```

Build the CLI with `llama.cpp` enabled:

```bash
cargo build -p loci-cli --release --features llama
```

Build the core crate directly:

```bash
cargo build -p loci-core --release
```

Build the core crate with `llama.cpp` enabled:

```bash
cargo build -p loci-core --release --features llama
```

Build the FFI crate:

```bash
cargo build -p loci-ffi --release
```

Build the FFI crate with `llama.cpp` enabled:

```bash
cargo build -p loci-ffi --release --features llama
```

## Runtime Entry

The primary runtime entry today is the CLI management server:

```bash
cargo run -p loci-cli --features llama -- \
  --plugin-dir plugins \
  --management-bind 127.0.0.1:8080
```

CLI flags:

- `--plugin-dir <path>`
- `--activate-legacy-text-plugin <name>` (repeatable)
- `--management-bind <host:port>`

## Test Commands

Workspace tests:

```bash
cargo test -q
```

`llama.cpp` integration tests:

```bash
cargo test -q -p loci-core --features llama
cargo test -q -p loci-cli --features llama
```

Windows helper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/full_test.ps1
```

## Troubleshooting

If `bindgen` cannot find `libclang` on Windows, set one of:

```powershell
$env:LIBCLANG_PATH = "C:\\Program Files\\LLVM\\bin"
```

or

```powershell
$env:LLVM_HOME = "C:\\Program Files\\LLVM"
```

If you do not need real model backend binding for a given build, omit `--features llama`.

## Non-Goals Of This Build Guide

This guide intentionally does not describe the removed root monolith, the removed `serve/generate` CLI, or the removed legacy example programs. The workspace crates above are the maintained entry points.

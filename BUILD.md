# Building Loci

This document describes how to build Loci for different platforms and configurations.

## Prerequisites

### All Platforms
- Rust 1.70 or later
- CMake 3.15 or later
- A C++ compiler

### Platform-Specific Requirements

#### Windows
- **MSVC**: Visual Studio 2019 or later with C++ build tools
- **MinGW**: MSYS2 with mingw-w64 toolchain

#### Linux
- GCC 7+ or Clang 10+
- Development libraries: `build-essential`, `cmake`, `ninja-build`

#### macOS
- Xcode Command Line Tools
- CMake (via Homebrew: `brew install cmake`)

#### iOS
- Xcode 14 or later
- CMake and Ninja (via Homebrew)

#### Android
- Android NDK r25c or later
- CMake and Ninja

## Building from Source

### Quick Start

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/decade-afk/loci.git
cd loci

# Build the CLI tool
cargo build --release

# Build as library
cargo build --release --lib
```

### Build Outputs

After building, you'll find:

#### Executable
- Windows: `target/release/loci.exe`
- Unix: `target/release/loci`

#### Static Library
- Windows MSVC: `target/release/loci.lib`
- Windows GNU/Unix: `target/release/libloci.a`

#### Dynamic Library
- Windows: `target/release/loci.dll`
- Linux: `target/release/libloci.so`
- macOS: `target/release/libloci.dylib`

## Cross-Compilation

### Linux to Windows

```bash
# Install MinGW cross-compiler
sudo apt-get install mingw-w64

# Add Windows target
rustup target add x86_64-pc-windows-gnu

# Build
cargo build --release --target x86_64-pc-windows-gnu
```

### macOS Universal Binary

```bash
# Build for both architectures
cargo build --release --target x86_64-apple-darwin
cargo build --release --target aarch64-apple-darwin

# Create universal binary
lipo -create \
  target/x86_64-apple-darwin/release/loci \
  target/aarch64-apple-darwin/release/loci \
  -output loci-universal
```

### Building for iOS

```bash
# Add iOS targets
rustup target add aarch64-apple-ios
rustup target add x86_64-apple-ios
rustup target add aarch64-apple-ios-sim

# Mobile builds disable the optional WASM plugin runtime.

# Build for device
cargo build --release --lib --target aarch64-apple-ios --no-default-features --features auto-detect

# Build for simulator
cargo build --release --lib --target aarch64-apple-ios-sim --no-default-features --features auto-detect
```

### Building for Android

```bash
# Install Android NDK
# Set ANDROID_NDK_HOME environment variable

# Add Android targets
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add x86_64-linux-android
rustup target add i686-linux-android

# Set up environment variables
export CC_aarch64_linux_android=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang
export AR_aarch64_linux_android=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar
export CMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake
export ANDROID_ABI=arm64-v8a
export ANDROID_PLATFORM=android-24
export ANDROID_STL=c++_shared

# Build
cargo build --release --lib --target aarch64-linux-android --no-default-features --features auto-detect
```

## Using the C API

Loci provides a C API for integration with other languages. After building the library, you can use it in your C/C++ projects:

```c
#include "loci.h"

int main() {
    // Create inference engine
    LociEngine* engine = loci_engine_new("model.gguf", 4096, -1);
    if (!engine) {
        return 1;
    }

    // Generate text
    char* result = loci_generate(engine, "Hello", 50, 0.8);
    if (result) {
        printf("%s\n", result);
        loci_free_string(result);
    }

    // Cleanup
    loci_engine_free(engine);
    return 0;
}
```

### Linking

#### Static Linking (Recommended)

```bash
# Linux
gcc your_app.c -I./include -L./target/release -lloci -ldl -lm -lpthread -o your_app

# macOS
clang your_app.c -I./include -L./target/release -lloci -framework CoreFoundation -o your_app

# Windows (MSVC)
cl your_app.c /I./include /link loci.lib

# Windows (MinGW)
gcc your_app.c -I./include -L./target/release -lloci -lws2_32 -o your_app.exe
```

#### Dynamic Linking

```bash
# Linux
gcc your_app.c -I./include -L./target/release -lloci -Wl,-rpath,'$ORIGIN' -o your_app

# macOS
clang your_app.c -I./include -L./target/release -lloci -Wl,-rpath,@loader_path -o your_app

# Windows
# Copy loci.dll to the same directory as your_app.exe
gcc your_app.c -I./include -L./target/release -lloci -o your_app.exe
```

## Platform-Specific Notes

### Windows with MinGW

When building with MinGW, you need to have the MinGW bin directory in your PATH:

```bash
export PATH="/c/msys64/mingw64/bin:$PATH"
cargo build --release --target x86_64-pc-windows-gnu
```

### macOS ARM (M1/M2)

Native compilation on Apple Silicon:

```bash
cargo build --release --target aarch64-apple-darwin
```

### Linux ARM (Raspberry Pi, etc.)

For cross-compilation to ARM:

```bash
# Install cross-compilation toolchain
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Add target
rustup target add aarch64-unknown-linux-gnu

# Build
cargo build --release --target aarch64-unknown-linux-gnu
```

## Build Configuration

You can customize the build using environment variables:

- `LOCI_DISABLE_GPU`: Disable GPU support
- `CMAKE_BUILD_TYPE`: Set to `Debug` or `Release` (default: Release)

## Troubleshooting

### CMake not found
```bash
# Ubuntu/Debian
sudo apt-get install cmake

# macOS
brew install cmake

# Windows
choco install cmake
```

### Bindgen errors
Make sure you have libclang installed:

```bash
# Ubuntu/Debian
sudo apt-get install libclang-dev

# macOS (usually included with Xcode)
xcode-select --install
```

### Link errors on Windows
Ensure you have the correct MinGW DLLs (`libgomp-1.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll`) in your PATH or copy them to the output directory.

## GitHub Actions

This repository includes GitHub Actions workflows that automatically build Loci for all supported platforms on every commit:

- **Windows**: MSVC and MinGW (x86_64)
- **Linux**: x86_64 and ARM64
- **macOS**: Intel and Apple Silicon
- **iOS**: Device (ARM64) and Simulator (x86_64, ARM64)
- **Android**: ARM64, ARMv7, x86_64, x86

Releases are automatically created when you push a tag starting with `v`:

```bash
git tag v0.1.0
git push origin v0.1.0
```

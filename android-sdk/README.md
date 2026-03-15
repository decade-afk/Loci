# Loci Android SDK

This directory contains a minimal Android SDK wrapper for Loci.

It provides:

- `loci-sdk`: an Android library module that packages `libloci.so`
- a JNI bridge over the existing C API from `include/loci.h`
- a Kotlin `LociEngine` wrapper for engine lifecycle and generation
- device enumeration and runtime version helpers
- `sample-app`: a simple app that imports a GGUF model, loads an engine, and runs generation

## Scope

The SDK currently targets the existing native Android artifacts produced by the root Rust project.

Supported wrapper features:

- create engine
- create engine with auto device selection
- blocking generation
- blocking generation with wait timeout
- streaming generation
- device enumeration
- runtime version query
- explicit engine close

## Prerequisites

- Android Studio with Android SDK + NDK installed
- Rust Android native artifacts already built from the repository root
- Android NDK `27.3.13750724` for the JNI bridge build

Recommended first target:

- `arm64-v8a`

## 1. Build native Android libraries

From the repository root:

```bash
cargo rustc --release --lib --target aarch64-linux-android --no-default-features --features auto-detect -- --crate-type cdylib
```

Optional extra ABIs:

```bash
cargo rustc --release --lib --target armv7-linux-androideabi --no-default-features --features auto-detect -- --crate-type cdylib
cargo rustc --release --lib --target x86_64-linux-android --no-default-features --features auto-detect -- --crate-type cdylib
cargo rustc --release --lib --target i686-linux-android --no-default-features --features auto-detect -- --crate-type cdylib
```

## 2. Sync the native artifacts into the Android project

PowerShell:

```powershell
pwsh ./android-sdk/scripts/sync-prebuilt-loci.ps1
```

Bash:

```bash
bash ./android-sdk/scripts/sync-prebuilt-loci.sh
```

Build the Android SDK bundle after syncing:

PowerShell:

```powershell
pwsh ./android-sdk/scripts/build-sdk-bundle.ps1
```

Bash:

```bash
bash ./android-sdk/scripts/build-sdk-bundle.sh
```

The sync script copies:

- `target/<triple>/release/libloci.so`

into:

- `android-sdk/loci-sdk/src/main/jniLibs/<abi>/libloci.so`

The SDK module automatically builds JNI variants only for the ABI directories
that currently contain `libloci.so`. The sample app still requires
`arm64-v8a/libloci.so`.

## 3. Open the Android project

Open the `android-sdk` directory in Android Studio.

Useful Gradle tasks:

```bash
gradle --no-daemon --stacktrace :loci-sdk:assembleRelease
gradle --no-daemon --stacktrace :sample-app:installDebug
```

## Notes

- The sample app is restricted to `arm64-v8a` to keep the default path stable.
- The SDK currently expects a GGUF file to be copied into app-private storage before model load.
- The JNI layer is thin by design and reuses the root C API rather than introducing a second host API.
- The library module fails early if `arm64-v8a/libloci.so` has not been synced yet.

## Kotlin API sketch

```kotlin
val version = LociRuntime.version()

LociDeviceSelector.create().use { selector ->
    val devices = selector.listDevices()
}

val engine = LociEngine.create(
    LociEngineConfig(
        modelPath = modelPath,
        contextSize = 2048,
        autoDetectDevice = true,
    )
)

val text = engine.generate(
    "Explain Loci on Android.",
    LociGenerationConfig(maxTokens = 128, temperature = 0.7f),
)
```

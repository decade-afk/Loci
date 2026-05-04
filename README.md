# Loci

Loci is a Rust inference infrastructure for on-device and edge AI.

It is built for applications that need to run models locally, choose between heterogeneous compute targets such as `CPU`, `GPU`, `NPU`, and `Disk`, and expose the same runtime either as an embedded SDK or as a standalone local service.

Loci is not tied to a single app category. A desktop companion, a native productivity tool, an offline assistant, a robotics controller, or a vertical edge product can all use the same runtime core.

## Why Loci

Modern local AI applications face the same set of problems:

- model files come from different ecosystems
- hardware capabilities differ across Intel, Apple Silicon, ARM, and mobile devices
- memory is limited, especially on laptops and phones
- one application may need an SDK, while another needs a local service boundary

Loci addresses that by separating the control plane from execution backends.

The core runtime plans placement, prepares model assets, manages readiness, and exposes a stable integration surface. Backends execute the plan using the best available path for the host device.

## Built For Local Integration

Loci supports two product shapes from the same runtime:

- an embeddable Rust SDK for direct in-process inference
- a standalone local AI service for applications that prefer process isolation

That keeps integration flexible without forcing every application into an HTTP-only model.

## Architecture At A Glance

Loci is organized around three ideas:

- `GGUF`-first model ingestion for practical edge deployment
- `Candle` as the default portable Rust execution path
- optional vendor backends such as `OpenVINO` for platform-specific acceleration

On top of that, Loci adds the part most local inference stacks leave to each application: a planner that can reason about heterogeneous execution and tiered offload across `CPU`, `GPU`, `NPU`, and `Disk`.

## What You Can Build

- desktop and native applications that embed local inference directly
- local AI services that expose OpenAI-style APIs without depending on the cloud
- edge products that need hardware-aware execution under tight memory limits
- products that want one runtime model across SDK and service deployment styles

## Core Capabilities

- `GGUF`-first local model support for practical edge deployment
- `Candle` as the default portable Rust execution path
- optional `OpenVINO` acceleration for Intel platforms
- model registration, inspection, and readiness checks
- planner-driven heterogeneous execution across `CPU`, `GPU`, `NPU`, and `Disk`
- one runtime that can be embedded directly or exposed as a local service

## Quick Start

Use Loci from the command line:

```bash
cargo run -p loci-cli -- \
  --model-path D:/models/demo.gguf \
  --model-name demo \
  --prompt "Explain the current execution plan."
```

Embed Loci directly in a Rust application:

```rust
use loci_sdk::{Loci, LocalModelRegistrationRequest, TextGenerationRequest};

let mut loci = Loci::builder().build()?;

loci.register_model(
    LocalModelRegistrationRequest::new(
        "D:/Code/Loci/tmp/models/qwen2.5-0.5b-instruct-gguf-ms/qwen2.5-0.5b-instruct-q4_0.gguf",
    )
    .name("embedded-demo"),
)?;

let response = loci.generate_text(
    TextGenerationRequest::new("Reply in one short friendly sentence.")
        .model("embedded-demo")
        .max_tokens(48)
        .temperature(0.7),
)?;
```

Run the SDK-local example:

```bash
cargo run -p sdk-local --features openvino -- \
  D:/Code/Loci/tmp/models/qwen2.5-0.5b-instruct-gguf-ms/qwen2.5-0.5b-instruct-q4_0.gguf
```

Run Loci as a service from the same SDK facade:

```rust
use loci_sdk::{LocalModelRegistrationRequest, Loci, LociServiceConfig};

let mut loci = Loci::builder().build()?;

loci.register_model(
    LocalModelRegistrationRequest::new(
        "D:/Code/Loci/tmp/models/qwen2.5-0.5b-instruct-gguf-ms/qwen2.5-0.5b-instruct-q4_0.gguf",
    )
    .name("service-demo"),
)?;

loci.run_service(LociServiceConfig::with_host_port("127.0.0.1", 8080))?;
```

Or start the local service from the CLI:

```bash
cargo run -p loci-cli -- \
  --model-path D:/models/demo.gguf \
  --model-name demo \
  --server-bind 127.0.0.1:8080
```

## Design Direction

Loci is designed around a portable default path and explicit extensions:

- `Candle` is the default pure Rust backend path
- `OpenVINO` is an optional acceleration path for Intel platforms
- `GGUF` is the primary edge model format
- `llama.cpp`-inspired kernels are ported through curated, explicit crates
- heterogeneous planning remains backend-agnostic at the core layer

The result is a runtime that can be embedded as a library, shipped as a local service, and extended toward more devices without changing the application-facing model.

## Learn More

- [Design](./design.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Backend Authoring](./docs/BACKEND_AUTHORING.md)
- [Intel OpenVINO Path](./docs/backends/INTEL_OPENVINO.md)
- [Contributing](./CONTRIBUTING.md)

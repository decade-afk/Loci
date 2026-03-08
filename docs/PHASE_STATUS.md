# Loci Phase Status (Audited)

Date: 2026-03-01  
Scope: current workspace state (`D:\OpenProject\Loci`)

## Summary

- Phase 1.0: achieved (core local text inference path is stable).
- Phase 1.1: achieved (plugin registry + dynamic/WASM loading + persistence are implemented).
- Phase 1.5: partially achieved (many multimodal/fusion modules exist, but main inference path is still text-first).
- Phase 2: partially achieved (adapter/model hot-swap/backend registry modules exist, but production-level end-to-end integration is incomplete).
- Phase 3: partially achieved (WASM plugin runtime exists; gRPC and full tool-orchestrated agent workflow are not fully delivered in current CLI/service path).

## Evidence by Phase

## Phase 1.0 (Achieved)

- Core inference engine and llama.cpp backend:
  - `src/inference.rs`
  - `src/backends/llamacpp.rs`
- CLI commands (`generate`, `serve`, `agent`, `plugin`) and legacy mode:
  - `src/main.rs`
- C API surface:
  - `src/c_api.rs`
  - `include/loci.h`
- Verified tests:
  - `cargo test -- --nocapture` passed.
- Runtime validation (model):
  - `D:\OpenProject\Qwen_Qwen3-0.6B-Q5_K_L.gguf` with `target/release/loci.exe generate ... --max-tokens 1` exited `0`.

## Phase 1.1 (Achieved)

- Static/dynamic/WASM plugin registry and persistence:
  - `src/plugin_registry.rs`
  - `src/wasm_plugin.rs`
- Plugin-related tests:
  - `tests/multi_plugin_type_test.rs`
  - plugin tests in `src/plugin_registry.rs` and `src/plugin.rs`

## Phase 1.5 (Partially Achieved)

- Multimodal and fusion modules are present:
  - `src/multimodal.rs`
  - `src/multimodal_plugin.rs`
  - `src/multimodal_fusion.rs`
  - `src/vision_clip.rs`
- Gap:
  - main CLI/service text-generation path does not expose a fully integrated multimodal generation pipeline.

## Phase 2 (Partially Achieved)

- Adapter/hot-swap/backend modules are present:
  - `src/adapter_system.rs`
  - `src/adapter_complete.rs`
  - `src/model_hot_swap.rs`
  - `src/backend.rs`
  - `src/backends/dynamic.rs`
- Gap:
  - not all advanced paths are fully wired into a production-level unified runtime workflow.

## Phase 3 (Partially Achieved)

- WASM plugin runtime and registry exist:
  - `src/wasm_plugin.rs`
  - `src/plugin_registry.rs`
- Service path currently validated:
  - REST (`GET /health`, `GET /v1/health`, `GET /info`, `GET /v1/info`, `POST /generate`, `POST /v1/generate`) in `src/main.rs`.
- Gap:
  - no built-in gRPC server in current CLI serve path.
  - `agent` command is prompt/tool-hint based; it is not a full external tool execution orchestrator.

## Additional hardening completed in this audit

- HTTP parser now returns explicit request-class errors:
  - malformed request -> `400 Bad Request`
  - oversized body -> `413 Payload Too Large`
  - code: `src/main.rs`
  - tests: parser unit tests in `src/main.rs` (`parse_http_request_*`).
- Runtime re-check on `serve` (release build):
  - `/health` and `/v1/health` => `200`
  - invalid JSON => `400`
  - oversized `Content-Length` => `413`

## Notable remaining risks

- Dynamic plugin constructor ABI is currently Rust toolchain-sensitive (opaque trait-object payload):
  - `src/plugin_registry.rs`
  - this is safer than direct `*mut dyn Plugin` FFI, but still not a fully C-stable plugin vtable ABI.
- Debug build runtime can be very slow for real model generation; release builds are recommended for operational validation.

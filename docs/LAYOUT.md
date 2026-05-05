## Project Layout

Loci keeps the executable workspace intentionally small at the repository root.

### Runtime workspace

- `crates/`: core runtime, protocol, backends, CLI, server, SDK
- `examples/`: in-process embedding and local service examples
- `docs/`: architecture, backend authoring, MVP and repository docs
- `scripts/`: local utility scripts for development workflows

### Non-runtime assets

- `tmp/models/`: local test models used for real execution checks
- `tmp/reference/`: unpacked reference repositories and papers
- `tmp/reference-zips/`: downloaded reference archives
- `vendor/openvino-genai-runtime/`: local OpenVINO runtime bundle kept outside the main workspace
- `deps/reference/`: compatibility or mirrored dependency references

### Hygiene rules

- Do not place logs, screenshots, sample assets, or downloaded archives at the repository root.
- Keep model files under `tmp/models/`.
- Keep third-party reference repositories under `tmp/reference/`.
- Keep bulky local runtime bundles under `vendor/`.
- Keep the root focused on buildable workspace content and user-facing documentation.

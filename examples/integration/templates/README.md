# Loci Integration Templates

This folder contains minimal templates for integrating Loci from external software.

## Prerequisite

Start Loci REST service first:

```bash
target/release/loci.exe serve \
  --model D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --cpu-only
```

If your service is not on `http://127.0.0.1:8080`, change the base URL in each template.

## Templates

- `python-rest/`:
  - `python loci_rest_client.py`
- `go-rest/`:
  - `go run .`
  - If Go build cache permission fails, set:
    - PowerShell: `$env:GOCACHE = (Resolve-Path .\\examples\\integration\\templates\\go-rest\\.gocache).Path`
- `java-rest/`:
  - `mvn exec:java`
- `tauri-rest/`:
  - Copy `src-tauri` and `src/loci.ts` snippets into your Tauri app.
  - See `tauri-rest/README.md`.

## Recommended Production Path

- Cross-language integration: use REST or C ABI.
- Rust desktop app (Tauri): use REST from Tauri commands (template provided).

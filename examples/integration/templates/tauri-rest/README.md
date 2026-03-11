# Loci Tauri Integration Template (Rust Command Layer)

This template wires Tauri commands to Loci REST endpoints.

## 1. Prepare Loci Service

```bash
target/release/loci.exe serve \
  --model D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --cpu-only
```

## 2. Prepare Tauri Project

Create a Tauri app first (if you do not already have one):

```bash
npm create tauri-app@latest
```

## 3. Copy Template Files

Copy into your Tauri project:

- `src-tauri/Cargo.toml` dependencies section additions (or merge manually)
- `src-tauri/src/main.rs`
- `src-tauri/src/lib.rs`
- `src-tauri/src/loci.rs`
- `src/loci.ts` (frontend helper)

## 4. Use in Frontend

Example:

```ts
import { lociGenerate, lociInfo } from "./loci";

const info = await lociInfo();
const text = await lociGenerate("Hello from Tauri", 64, 0.7);
```

## 5. Base URL

Set `LOCI_BASE_URL` env var for Tauri backend process when needed.
Default: `http://127.0.0.1:8080`.

# Example UI Shell

`example-ui-shell` is a manifest-first UI host plugin bundle for the refactored Loci workspace.

It demonstrates:

- a `ui_host` core rewriter claim
- panel, window, and widget surface declarations
- a neutral UI shell that can serve both `ai_infra` and `ai_agent` products
- static preview assets that host applications can adapt into their own renderer

Declared surfaces:

- panels: `workspace-overview`, `model-catalog`
- windows: `operations-console`
- widgets: `runtime-status`

This example does not ship an executable runtime artifact yet. It exists to document the plugin contract and the intended UX envelope while the host-side UI renderer stays product-specific.

Quick activation flow:

```bash
cargo run -p loci-cli -- --plugin-dir plugins --management-bind 127.0.0.1:8080
curl http://127.0.0.1:8080/v1/core/rewriters/activate \
  -H "Content-Type: application/json" \
  -d "{\"component\":\"ui_host\",\"plugin_name\":\"example-ui-shell\"}"
curl http://127.0.0.1:8080/v1/ui
```

Preview assets live in `preview/`.

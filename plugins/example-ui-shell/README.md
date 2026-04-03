# Example UI Shell

`example-ui-shell` is a manifest-first UI host plugin bundle for the refactored Loci workspace.

It demonstrates:

- a `ui_host` core rewriter claim
- panel, window, and widget surface declarations
- a real host runtime artifact that the current engine can materialize on activation
- a neutral UI shell that can serve both `ai_infra` and `ai_agent` products
- static preview assets that host applications can adapt into their own renderer

Declared surfaces:

- panels: `workspace-overview`, `model-catalog`
- windows: `operations-console`
- widgets: `runtime-status`

This example ships a lightweight placeholder host runtime artifact at `runtime/plugin.dll`.
In the current architecture, activating the plugin will materialize that host runtime artifact while UI rendering still remains product-specific.

Declared host contract:

- protocol: `loci.host-runtime.v1`
- entrypoint: `loci_ui_host_bootstrap_v1`
- capabilities: `ui_host`, `surface_registry`

Quick activation flow:

```bash
cargo run -p loci-cli -- --plugin-dir plugins --management-bind 127.0.0.1:8080
curl http://127.0.0.1:8080/v1/core/rewriters/activate \
  -H "Content-Type: application/json" \
  -d "{\"component\":\"ui_host\",\"plugin_name\":\"example-ui-shell\"}"
curl http://127.0.0.1:8080/v1/ui
```

Preview assets live in `preview/`.

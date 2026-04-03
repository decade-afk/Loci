# Plugin Bundle Examples

This directory contains manifest-first example bundles for the new Loci architecture.

Current examples:

- `example-inference`: inference rewriter and sampling profile contribution
- `example-infra`: infrastructure-side model contribution example
- `example-agent`: agent-side workflow contribution example
- `example-ui-shell`: clean white UI host example with declared panels, windows, widgets, and static preview assets

Use them with:

```bash
cargo run -p loci-cli -- --plugin-dir plugins
```

or:

```bash
curl http://127.0.0.1:8080/v1/plugins/load \
  -H "Content-Type: application/json" \
  -d "{\"path\":\"plugins\",\"source_kind\":\"directory\"}"
```

These examples intentionally use manifest bundles instead of the removed root-level dynamic example programs.

# OpenClaw Adapter Plugin (Dynamic)

This plugin provides an OpenClaw-style agent contract for Loci:

- Injects tool schema + JSON protocol into `pre_generate`.
- Normalizes generation output into JSON envelopes in `post_generate`.
- Supports hot-reload/unload through Loci plugin registry.

## Build

```bash
cargo build --release --manifest-path examples/openclaw_adapter_plugin/Cargo.toml
```

Windows output:

```text
examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll
```

## Runtime Env

- `LOCI_OPENCLAW_TOOLS_PATH`: JSON file path for tools.
- `LOCI_OPENCLAW_STRICT_JSON`: `1/true/yes/on` to force JSON envelope (default `true`).
- `LOCI_OPENCLAW_SYSTEM_PROMPT`: optional system prefix override.

Tool file format (array):

```json
[
  {
    "name": "web_search",
    "description": "Search web and return concise snippets",
    "parameters": {
      "type": "object",
      "properties": {
        "q": { "type": "string" }
      },
      "required": ["q"]
    }
  }
]
```

or object:

```json
{
  "tools": [ ... ]
}
```

## Load / Hot-Reload / Unload

```bash
loci.exe plugin load examples/openclaw_adapter_plugin/target/release/openclaw_adapter_plugin.dll
loci.exe plugin info openclaw_adapter
loci.exe plugin reload openclaw_adapter
loci.exe plugin unload openclaw_adapter
```

## Agent Output Contract

Plugin expects/normalizes output to one of:

1. Tool call

```json
{"type":"tool_call","name":"<tool>","arguments":{...},"id":"<opaque-id>"}
```

2. Final answer

```json
{"type":"final","content":"<text>"}
```

## Host-Side Orchestration Loop (Pseudo)

```text
loop:
  model_json = loci_generate(...)
  msg = parse_json(model_json)
  if msg.type == "tool_call":
      result = execute_tool(msg.name, msg.arguments)
      prompt = "TOOL_RESULT(" + msg.id + "): " + serialize(result)
      continue
  if msg.type == "final":
      return msg.content
```

This keeps tool execution in your host runtime (desktop/server/sandbox), while Loci handles local inference and plugin hooks.

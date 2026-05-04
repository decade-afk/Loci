# Contributing

## Scope

Loci is an edge inference infrastructure project. Contributions are most useful when they improve:

- heterogeneous planning
- backend execution quality
- model readiness diagnostics
- tiered offload
- paged KV
- backend authoring for new chip families

## First Read

Before contributing, read:

- [README](./README.md)
- [Architecture](./docs/ARCHITECTURE.md)
- [Backend Authoring](./docs/BACKEND_AUTHORING.md)
- [Intel OpenVINO Path](./docs/backends/INTEL_OPENVINO.md)

## Current Priorities

The current priority order is:

1. strengthen the Intel/OpenVINO primary path
2. define clean backend-lowering boundaries for future chip backends
3. improve tiered offload and paged KV execution quality
4. implement a real Candle fallback path

## Ground Rules

- keep `loci-core` backend-agnostic
- keep backend-specific execution logic inside backend crates
- do not commit `tmp/`
- keep model files and large runtime artifacts out of the repository
- prefer truthful capability reporting over aspirational claims

## Backend Contributions

If you want to add or improve a backend:

1. document the target runtime family in `docs/backends/`
2. implement or update `BackendDescriptor`
3. implement or update `BackendLoweringCapabilities`
4. keep `supports_model()` conservative
5. add tests for readiness and backend selection behavior

## Testing

At minimum, run:

```bash
cargo fmt
cargo check -q
```

Then run the most relevant targeted tests for your area.

## Documentation

Public, stable project documentation belongs in:

- `README.md`
- `docs/`

Deep local notes, experiments, or temporary research material do not belong in committed documentation.

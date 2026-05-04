# Intel OpenVINO Path

## Scope

This document defines the current primary Loci implementation path.

Target shape:

- Intel-oriented edge devices
- `CPU + GPU + NPU + Disk` when available
- `CPU + GPU + Disk` on machines without a usable NPU

Runtime family:

- `OpenVINO`
- `OpenVINO GenAI`

## Why This Is The Primary Path

Today, this is the most practical way for Loci to achieve real heterogeneous edge inference without inventing a new vendor runtime.

OpenVINO gives Loci:

- real device discovery
- real execution runtimes
- real multimodal GenAI pipelines
- heterogeneous execution support

Loci adds:

- backend-agnostic orchestration
- model readiness inspection
- tiered offload planning/runtime
- future backend interoperability

## Current State

Implemented:

- real runtime discovery
- real text generation path through `LlmPipeline`
- real multimodal path through `VlmPipeline`
- readiness validation for OpenVINO-exported model layouts
- fallback reporting when raw checkpoints are not executable

Not implemented yet:

- explicit planner-to-subgraph affinity lowering
- per-layer affinity mapping
- automated export/conversion workflow
- NPU-first validation on hardware that actually exposes a usable NPU

## Model Asset Expectations

The OpenVINO path can distinguish between:

- raw Transformers checkpoints
- OpenVINO IR layouts
- OpenVINO GenAI export layouts

For multimodal models such as `MiniCPM-V`, a raw Transformers repository is not enough for real execution.

The backend currently expects an OpenVINO GenAI export layout with files such as:

- `openvino_language_model.xml`

## Lowering Model

The current lowering ABI exposed by this backend is:

- real execution: yes
- graph partitioning: yes
- layer affinity: not yet exposed by Loci
- lowering granularity: `subgraph`

This is intentionally honest.

Loci currently plans heterogeneous placement at the orchestration level, but it does not yet lower those placements into explicit layer/subgraph affinity directives in the OpenVINO backend.

## Contribution Priorities

The best next contributions for this path are:

1. planner-to-subgraph lowering
2. affinity and partition mapping
3. richer OpenVINO telemetry extraction
4. export/conversion helpers
5. NPU-enabled validation on real Intel hardware

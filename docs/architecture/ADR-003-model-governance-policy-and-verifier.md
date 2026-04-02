# ADR-003: Split Model Asset Governance into Policy and Verifier Layers

## Status

Accepted

## Context

Model asset import governance has at least two distinct concerns:

- whether a source should be allowed before bytes are fetched
- whether downloaded bytes should be trusted before persistence

Treating both concerns as one gate makes it harder to evolve from simple allowlists into stronger trust models such as sidecars, signatures, or provenance checks.

## Decision

Loci splits model asset governance into two layers:

- `model_pull_policy`: pre-fetch admission control
- `model_pull_verifier`: post-fetch trust verification

The model store persists managed assets only after both layers allow the import.

## Consequences

Positive:

- clearer separation of concerns
- easier evolution toward stronger trust pipelines
- hosts can choose different source and trust strategies independently

Tradeoffs:

- more moving parts in model import orchestration
- more registry and activation state to manage

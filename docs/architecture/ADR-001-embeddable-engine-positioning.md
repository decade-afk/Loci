# ADR-001: Position Loci as an Embeddable Inference Engine and Control Plane

## Status

Accepted

## Context

Projects in this space often drift toward one of two extremes:

- low-level backend wrappers that are hard to integrate into products directly
- chat-first shells that are difficult to reuse as infrastructure

Loci needs to support downstream products such as desktop apps, IDE assistants, local automation systems, and higher-level agent hosts without becoming tightly coupled to one end-user UX.

## Decision

Loci is positioned as:

- an embeddable AI inference engine
- a plugin-upgradeable runtime
- a host-controlled control plane

It is not positioned as a consumer chat UI inside the core repository.

## Consequences

Positive:

- architecture stays reusable across multiple host products
- integration surfaces become first-class
- governance and runtime control remain in scope

Tradeoffs:

- Loci must invest more in APIs, docs, and operational control than a chat-first shell
- some user-visible features are intentionally deferred to downstream products

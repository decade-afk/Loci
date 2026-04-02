# ADR-002: Treat Compatibility APIs as Adapters over the Native Runtime

## Status

Accepted

## Context

Integrators often expect OpenAI- or Ollama-compatible APIs. A common failure mode is duplicating runtime logic to satisfy each compatibility surface separately, which leads to drift, inconsistent governance, and fragmented observability.

## Decision

OpenAI-compatible and Ollama-compatible routes are implemented as adapters over the native Loci runtime.

They reuse:

- the same inference engine
- the same plugin and tool layers
- the same metrics and control-plane context
- the same governance path where applicable

## Consequences

Positive:

- compatibility support does not fork the engine
- hosts can mix native and compatibility calls safely
- operational behavior stays more consistent

Tradeoffs:

- adapter code must translate between external API semantics and Loci-native behavior
- some external features may be partially supported when they do not map cleanly onto Loci

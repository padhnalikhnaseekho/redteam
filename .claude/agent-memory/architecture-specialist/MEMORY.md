---
scope: project
last-updated: YYYY-MM-DD
---

# Architecture Specialist — Persistent Memory

> **Auto-loaded by Claude Code when architecture-specialist agent is invoked.**
> Records ADR decisions, service dependency maps, and architectural constraints.

---

## Service Dependency Map

<!-- Built up as you read ARCHITECTURE.md and source code -->
<!-- High-blast-radius services (called by many others) should be flagged -->
<!-- Example:
- abc-auth: called by ALL services on every request (AuthGuard → Introspect gRPC)
- abc-asset: called by marketplace, transaction, transfer, custody, tokeniser
-->

---

## ADR Register

<!-- ADRs made for this repo — supplement to the formal ARCHITECTURE.md -->
<!-- Format:
## ADR-NNN: [title] — YYYY-MM-DD
**Decision:** [what was decided]
**Rationale:** [why]
**Alternatives rejected:** [what was not chosen and why]
-->

---

## Architecture Constraints

<!-- Hard constraints that cannot be violated in this repo -->
<!-- Discovered from reading ARCHITECTURE.md, ADRs, and team feedback -->
<!-- Example:
- All entities in libs/abc-common — never in apps/ (ADR-005)
- No service creates its own DB connection — share the single defi DB
-->

---

## Cross-Service Event Topology

<!-- Pub/Sub and gRPC relationships — built up as features are designed -->
<!-- Example:
- abc-workflow → [workflowTransactionStatusTopic] → abc-workflow (self)
- abc-listener → [fiatTopic] → abc-fiat
-->

---

## Memory Index

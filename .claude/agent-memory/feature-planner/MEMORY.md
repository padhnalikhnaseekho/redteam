---
scope: project
last-updated: YYYY-MM-DD
---

# Feature Planner — Persistent Memory

> **Auto-loaded by Claude Code when feature-planner agent is invoked.**
> This file accumulates institutional knowledge about THIS REPO's feature patterns.
>
> What to save here: feature shapes, service dependency patterns, decisions made.
> What NOT to save: generic SDD workflow (already in agent body), temporary task state.

---

## Repo Context (fill in as you discover it)

<!-- Stack, gate path, and spec structure — extracted from SDLC-PROFILE.md -->
- Stack: (see .claude/SDLC-PROFILE.md)
- Feature gate path: (see .claude/SDLC-PROFILE.md)
- Number of services/modules: (discover from ARCHITECTURE.md)

---

## Common Feature Shapes for This Repo

<!-- As you plan features, record the recurring patterns here -->
<!-- Example:
### CRUD Endpoint Feature (Small)
- Always touches: specs/services/<service>.md + specs/system/API-CONTRACTS.md
- Never needs: COMMUNICATION.md (no events for CRUD)
- Typical size: Small
- Implementation order: entity → service → controller → tests
-->

---

## Service Dependency Map

<!-- Which services call which — discovered as you read ARCHITECTURE.md -->
<!-- Example:
- abc-transfer → abc-custody (vault CRUD)
- abc-transaction → abc-notification (webhook resend)
-->

---

## Recurring Clarification Questions

<!-- Questions you've had to ask the user repeatedly — record answers here -->
<!-- Format:
### Q: [question]
**Answer:** [answer received]
**Context:** [why this matters for feature planning]
-->

---

## Decisions Made

<!-- Architectural or scoping decisions made during feature planning -->
<!-- Format:
## Decision — YYYY-MM-DD: [title]
[What was decided and why. Link to feature file if applicable.]
-->

---

## Memory Index

<!-- Link to topic files created in this directory -->

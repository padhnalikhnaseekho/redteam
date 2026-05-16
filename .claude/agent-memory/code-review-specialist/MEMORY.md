---
scope: project
last-updated: YYYY-MM-DD
---

# Code Review Specialist — Persistent Memory

> **Auto-loaded by Claude Code when code-review-specialist agent is invoked.**
> Records recurring violations, service-specific patterns, and team preferences.
>
> What to save: confirmed violation patterns (with location), team preferences validated
> across multiple sessions, service-specific gotchas.
> What NOT to save: generic review checklist (in CROSS-CUTTING-PATTERNS.md), task state.

---

## Critical Violations — Block Immediately

<!-- Violations confirmed in this codebase that must always be caught -->
<!-- Build this list from real reviews done on this repo -->
<!-- Example:
1. [Pattern] in [file/area] — seen on YYYY-MM-DD
   → Block: redirect to fix [specific fix]
-->

---

## Service-Specific Patterns

<!-- Conventions discovered in specific services during reviews -->
<!-- Example:
### abc-asset
- getAssetsBySymbol uses queryBuilder.getRawMany() + toAsset() mapper
- NetworkAssetDto does not populate networkName/networkGuid (known gap, accepted)
-->

---

## Team Preferences (confirmed across sessions)

<!-- Non-obvious preferences the team has confirmed multiple times -->
<!-- Example:
- Team prefers @Component over @Service for validators (verified 2026-04-12)
- Team uses integer serial PK internally, exposes UUID guid in API
-->

---

## Audit History

<!-- Features reviewed and key findings -->
<!-- Format:
## [feature-name] — YYYY-MM-DD
- Decision: APPROVED / CHANGES REQUIRED
- Critical findings: [count]
- Key issues: [brief description]
- Spec-sync health: GREEN/AMBER/RED
-->

---

## Memory Index

<!-- Link to topic files in this directory -->

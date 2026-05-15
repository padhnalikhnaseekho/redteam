---
scope: project
last-updated: YYYY-MM-DD
---

# Spec-Driven Developer — Persistent Memory

> **Auto-loaded by Claude Code when spec-driven-developer agent is invoked.**
> Records verified implementation patterns, technical debt, and codebase-specific knowledge.
>
> What to save: verified patterns (with file:line citation), known deviations, technical debt.
> What NOT to save: generic coding standards (in CROSS-CUTTING-PATTERNS.md), task state.

---

## Verified Patterns (confirmed from source)

<!-- Patterns you've read directly from the code and confirmed work -->
<!-- Always cite the source file:line so future sessions can verify -->
<!-- Example:
### Service injection pattern (verified from abc-asset.service.ts:23)
constructor(
  private readonly assetDbService: AssetDbService,  // ← from abc-common
  private readonly config: AbcConfigService,
) {}
// NOT: @InjectRepository(AssetEntity) private repo: Repository<AssetEntity>
-->

---

## Write Boundary

This agent's hook allows writes to paths containing: `/src/`, `/app/`, `/db/`, `/specs/`, `/.claude/agent-memory/`
Any other path is blocked (exit 2). Do not attempt to write to CLAUDE.md or agent config files.

---

## Known Technical Debt

<!-- Issues found during implementation that weren't fixed (track here) -->
<!-- Format:
### [service/file] — YYYY-MM-DD
[What the debt is, why it was left, what the fix would be]
-->

---

## Implementation History

<!-- Features implemented and patterns discovered -->
<!-- Format:
## [feature-name] — YYYY-MM-DD
- Stack: [stack used]
- Files changed: [list key files]
- Key patterns applied: [patterns from CROSS-CUTTING-PATTERNS.md that were used]
- Deviations: [any approved exceptions to the standards]
-->

---

## Memory Index

<!-- Link to topic files (e.g., implementation-patterns.md, known-technical-debt.md) -->

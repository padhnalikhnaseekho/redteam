# GEMINI.md

This file provides guidance to Gemini CLI when working in this repository.

> **Spec-driven development is in effect.**
> All changes follow: Requirement → Update Spec → Implement from Spec.
> The spec gate blocks code writes until a spec is approved.

This repo uses the SDD framework. The same agents and gating rules described
in `CLAUDE.md` apply when working through Gemini CLI, enforced by the Gemini
extension at `~/.gemini/extensions/sdd-framework/`.

If the Gemini extension is not installed, run:

    sdd setup --gemini

Otherwise, refer to `CLAUDE.md` for the routing table, the NEVER rules,
and the review-fix loop. The behaviour is identical across Claude Code and
Gemini CLI — only the runtime differs.

## Living Specification

The `specs/` folder is the single source of truth.

@specs/system/CROSS-CUTTING-PATTERNS.md
@specs/system/ARCHITECTURE.md
@specs/system/CONSTITUTION.md

---
feature: feature-name-here
status: DRAFT
stack: python
created: 2026-05-13
security-review: not-required
---

# Feature: [Feature Name]

> **Status:** `DRAFT` -> `SPEC-REVIEW` -> `SPEC-APPROVED` -> `IN-PROGRESS` -> `COMPLETED`
>
> Use this template for future CommodityRedTeam feature specs.

## Requirement

[Clear description of what changes and why. Include the research or engineering driver.]

## Acceptance Criteria

- [ ] [Testable criterion 1: specific, measurable]
- [ ] [Testable criterion 2]
- [ ] [Testable criterion 3]

## Spec Changes Required

| Spec File | Section | Change Description | Done |
|---|---|---|---|
| `specs/system/ARCHITECTURE.md` | [Section] | [What changes] | [ ] |
| `specs/system/CROSS-CUTTING-PATTERNS.md` | [Pattern] | [What changes] | [ ] |
| `specs/features/<feature>.md` | [Section] | [What changes] | [ ] |

## Security Gate

- [ ] This feature handles API keys, provider credentials, or secrets.
- [ ] This feature changes LLM prompt boundaries, attack payload generation, or defense blocking behavior.
- [ ] This feature changes benchmark outputs that may contain adversarial prompts.

If any box is checked, set `security-review: required` in frontmatter and document mitigations.

## Implementation Plan

### Phase 1 - Contracts

- [ ] Attack/defense/agent/result contracts updated if needed.
- [ ] YAML config schema changes documented if needed.

### Phase 2 - Implementation

- [ ] Framework module changes.
- [ ] Script or CLI changes.
- [ ] Unit tests for contract behavior.

### Phase 3 - Verification

- [ ] `pytest` or targeted test command run.
- [ ] Result artifact schema checked if benchmark output changes.
- [ ] Spec-sync audit completed.

## Spec-Sync Audit

| Finding | File | Detail |
|---|---|---|
| | | |

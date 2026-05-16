---
feature: results-explorer-enterprise-ux
status: COMPLETE
stack: streamlit-results
created: 2026-05-16
security-review: not-required
---

# Feature: Results Explorer Enterprise UX

> **Status:** `COMPLETE`

## Requirement

Make Results Explorer understandable for first-time enterprise users without changing benchmark artifacts, API contracts, or evaluator behavior.

The current view exposes technically correct summary and result-row tables, but the meaning of columns such as `asr_pct`, `detected`, `target_action_achieved`, `defense_confidence`, and `financial_impact` is not obvious to non-admin users.

## Exact Actions

- [x] Add an executive interpretation section above raw tables.
- [x] Show headline KPIs:
  - rows evaluated
  - attack success rate
  - defense detection rate
  - simulated financial impact
  - highest-risk model/defense pair
- [x] Add plain-language metric glossary for Summary columns.
- [x] Add model/defense comparison view with readable column names.
- [x] Add result-row interpretation:
  - `success` -> "Attack succeeded"
  - `target_action_achieved` -> "Unsafe target action appeared"
  - `detected` -> "Defense detected"
  - `financial_impact` -> "Simulated exposure"
  - `notes` -> "Evaluator evidence"
- [x] Keep raw Summary and Result Rows available as audit drill-down.
- [x] Preserve existing filters and downloads.
- [x] Verify with Python compile and focused tests.

## Acceptance Criteria

- [x] A first-time user can tell whether the run is good or bad without reading raw JSON or internal column names.
- [x] Summary metrics are explained in human language.
- [x] Risky rows are surfaced before raw tables.
- [x] Existing raw tables, filters, and downloads still work.
- [x] No backend/API changes required.

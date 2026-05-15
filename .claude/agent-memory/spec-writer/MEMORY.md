---
scope: project
last-updated: YYYY-MM-DD
---

# Spec Writer — Persistent Memory

> **Auto-loaded by Claude Code when spec-writer agent is invoked.**
> Records spec structure knowledge, SME answers, and extraction patterns.

---

## Spec File Inventory

<!-- Discovered as you work — fill in what exists in this repo -->
- System specs: specs/system/ (list which files exist here)
- Service specs: specs/services/ (count and list files)
- Feature tracking: (path from SDLC-PROFILE.md feature-path)
- Template: (path from SDLC-PROFILE.md feature-template)

---

## Spec File Structure Patterns

<!-- How this repo's spec files are organised — discovered on first use -->
<!-- Example:
- Service specs use H2 sections: ## Endpoints, ## Entities, ## Business Logic
- DATA-MODELS.md uses | Field | Type | DB Type | Constraints | columns
- API-CONTRACTS.md groups by service, not by HTTP method
-->

---

## SME Answer Log

<!-- Domain expert answers collected via AskUserQuestion — CRITICAL to save -->
<!-- Never re-ask a question that's answered here -->
<!-- Format:
## SME Session — YYYY-MM-DD | Target: [spec file:section]
### Q: [exact question asked]
**Answer:** [answer received]
**Applied to:** [spec file updated]
-->

---

## Extraction Challenges

<!-- Patterns that are hard to spec from source code alone -->
<!-- Example:
- Business rule for X is implicit in the code (line 47 of y.service.ts) — not documented
- Service Z has undocumented dependency on cache warming order
-->

---

## Memory Index

<!-- Link to topic files created in this directory (e.g., sme-answers.md) -->

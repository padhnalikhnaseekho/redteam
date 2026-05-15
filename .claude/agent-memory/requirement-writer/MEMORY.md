---
scope: project
last-updated: YYYY-MM-DD
---

# Requirement Writer — Persistent Memory

> **Auto-loaded by Claude Code when requirement-writer agent is invoked.**
> This file accumulates institutional knowledge about THIS REPO's requirements patterns.
>
> What to save here: effective RAG query patterns, known corpus gaps, domain terminology,
> SME answers that filled gaps, corpus coverage notes.
> What NOT to save: specific requirement text (that goes in specs/requirements/),
> generic RE process knowledge, temporary task state.

---

## Corpus Context

<!-- Filled in as you use the corpus for this repo -->
- Corpus name: (from RAG_CORPUS_NAME env var)
- Last verified: YYYY-MM-DD
- Document count: (from get_corpus_info)
- Coverage assessment: GOOD | PARTIAL | SPARSE
- Key source documents: (list top 3-5 most retrieved)

---

## Effective Query Patterns for This Domain

<!-- RAG queries that consistently return high-relevance evidence for this repo -->
<!-- Save these so you don't re-discover them each session -->
<!-- Format:
### <domain area>
Query: "<exact query text that worked well>"
Typical results: N documents, avg score ~0.8
Notes: <what it retrieves well>
-->

---

## Known Corpus Gaps

<!-- Areas consistently returning 0 or very low-relevance results -->
<!-- Record here so you don't waste future retrieval calls on them -->
<!-- Format:
### GAP: <topic>
Last attempted: YYYY-MM-DD
Query tried: "<query>"
Result: 0 results / score < 0.4
Action taken: Raised GAP-xxx in requirements doc, stakeholder notified
-->

---

## Domain Terminology

<!-- Technical terms and acronyms that improve retrieval for this repo -->
<!-- Using these in queries rather than generic descriptions improves results -->
<!-- Example:
- "NPP" retrieves more than "New Payments Platform"
- "BECS" retrieves better than "bulk electronic clearing system"
-->

---

## SME Answers Received

<!-- Domain expert answers that filled corpus gaps — so they are NOT re-asked -->
<!-- Format:
## SME Session — YYYY-MM-DD | Gap: GAP-xxx
### Q: [question asked to stakeholder]
**Answer:** [answer received]
**Corpus reference created:** [did this lead to corpus update? doc name if yes]
**Applied to:** REQ-xxx (requirement updated/added based on answer)
-->

---

## Requirement Patterns for This System

<!-- Recurring requirement shapes discovered in this domain -->
<!-- Example:
### Payment Processing Requirements
- Always need: duplicate detection REQ + audit trail REQ + idempotency NFR
- Performance NFRs always reference 99th percentile at peak load
- Security REQs always cite the ABC security policy document
-->

---

## Memory Index

<!-- Link to topic files created in this directory -->

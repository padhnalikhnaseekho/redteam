---
scope: project
agent: constitution-writer
last-updated: YYYY-MM-DD
---

# constitution-writer — Accumulated Memory

> Auto-loaded at every constitution-writer invocation via `memory: project`.
> Add to this file as you discover patterns specific to this repo's service type.
> Do NOT write: full constitution content (it lives in specs/system/CONSTITUTION.md),
> ADR detail (it lives in CONSTITUTION.md Section 6 and ARCHITECTURE.md).

---

## Corpus Coverage for This Service Type

> What did the RAG corpus cover well? What was missing?

| Query Topic | Coverage | Notes |
|---|---|---|
| Architectural principles | [GOOD / PARTIAL / GAPS] | [e.g., "pacs.008 validation rules well covered"] |
| Compliance requirements | [GOOD / PARTIAL / GAPS] | [e.g., "APRA CPS 234 found, PCI-DSS not in corpus"] |
| Integration patterns | [GOOD / PARTIAL / GAPS] | [e.g., "IBM MQ patterns well documented"] |
| Performance targets | [GOOD / PARTIAL / GAPS] | [e.g., "No SLA numbers in corpus — came from Q&A"] |

---

## Effective RAG Queries for This Service Type

> Queries that returned high-quality evidence (score > 0.7):

1. `"[paste effective query here]"` → returned [X] sources, score [Y]
2. `"[paste effective query here]"` → returned [X] sources, score [Y]

> Queries that returned poor results (score < 0.5 or empty):

1. `"[paste ineffective query here]"` → [GAP — use Q&A instead]

---

## GAPs That Required Human Input

> Topics that were NOT in the corpus and had to be resolved via Q&A.
> Record here so they can be pre-empted on the next similar service.

| GAP | Resolution | Source |
|---|---|---|
| [e.g., Performance latency targets] | [e.g., "≤1000ms P95 from Tech Lead"] | [interactive Q&A] |
| [e.g., Amendment approvers] | [e.g., "Tech Lead + Security Champion"] | [interactive Q&A] |

---

## SME Answers Received

> Questions asked and answers given — never re-ask these.

| Question | Answer | Date | Answered By |
|---|---|---|---|
| Payment rails in scope? | [answer] | [date] | [Tech Lead / architect] |
| Regulatory compliance frameworks? | [answer] | [date] | [name/role] |
| Amendment approvers? | [answer] | [date] | [name/role] |

---

## Constitution Versions Created

| Version | Service | Ratified | Key Decisions |
|---|---|---|---|
| 1.0.0 | [SERVICE_NAME] | [YYYY-MM-DD] | [top 2-3 decisions] |

---

## Memory Index

- [Corpus coverage](#corpus-coverage-for-this-service-type)
- [Effective RAG queries](#effective-rag-queries-for-this-service-type)
- [GAPs requiring human input](#gaps-that-required-human-input)
- [SME answers](#sme-answers-received)
- [Versions created](#constitution-versions-created)

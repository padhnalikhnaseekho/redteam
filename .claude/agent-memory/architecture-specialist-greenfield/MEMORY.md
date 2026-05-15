---
scope: project
agent: architecture-specialist-greenfield
last-updated: YYYY-MM-DD
---

# architecture-specialist-greenfield — Accumulated Memory

> Auto-loaded at every architecture-specialist-greenfield invocation via `memory: project`.
> This agent processes ARCH-XXX cards from requirement-writer and produces ADRs.
> Add to this file after each session to avoid re-processing cards and to capture patterns.

---

## ARCH-XXX Cards Processed

> Track processed cards for idempotency — avoid re-processing in future sessions.

| Session Date | Service | Cards Processed | ADRs Created |
|---|---|---|---|
| [date] | [service-name] | ARCH-001 to ARCH-N | ADR-001 to ADR-M |

---

## Architecture Decisions Made

> Brief summary — full ADRs live in specs/system/ARCHITECTURE.md.

| Service | Decision | ADR | Rationale (1 line) |
|---|---|---|---|
| [service] | [e.g., Use TypeORM] | ADR-001 | [e.g., Matches existing ABC ABC patterns, per ARCH-002] |

---

## Patterns Confirmed for This Stack

> Architecture patterns validated for this repo's tech stack.
> Format: pattern → confirmed by which ARCH card + citation.

- [e.g., "nestjs-typeorm: TypeORM with explicit column types confirmed — ARCH-002, ADR-001"]
- [e.g., "java-spring-camel: @Async thread pool core=15, max=1000 confirmed — ARCH-003, ADR-002"]

---

## Patterns Rejected for This Stack

> Alternatives considered and explicitly rejected — do not re-propose these.

- [e.g., "Prisma ORM rejected: incompatible with existing ABC DbService abstraction (ARCH-002)"]
- [e.g., "Redis distributed cache rejected: latency concern for NPP sub-second path (ARCH-008)"]

---

## Constitution Articles Most Relevant

> Which CONSTITUTION.md Articles were closely relevant in architecture decisions.

- Article [X]: [e.g., "Article VIII (Simplicity) — applied to reject over-engineered caching layer"]

---

## Open Architecture Questions

> Questions unresolved during the session — require stakeholder input.

| Question | Impacted ARCH Cards | Owner | Status |
|---|---|---|---|
| [e.g., Should BSB cache TTL be 1h or 24h?] | ARCH-008 | [Tech Lead] | Open |

---

## Memory Index

- [Cards processed](#arch-xxx-cards-processed)
- [Decisions made](#architecture-decisions-made)
- [Patterns confirmed](#patterns-confirmed-for-this-stack)
- [Patterns rejected](#patterns-rejected-for-this-stack)
- [Open questions](#open-architecture-questions)

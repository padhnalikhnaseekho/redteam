# CommodityRedTeam — Spec Directory

This directory contains the living specification for the CommodityRedTeam red-teaming framework.

---

## Structure

```
specs/
├── README.md                    # This file
├── system/                      # System-wide specifications
│   ├── CONSTITUTION.md          # Core axioms (do not edit)
│   ├── ARCHITECTURE.md          # System architecture and design decisions
│   ├── DATA-MODELS.md           # All dataclasses, enums, and named dicts
│   ├── API-CONTRACTS.md         # Method signatures and behavioral invariants
│   ├── COMMUNICATION.md         # Inter-module call graph and LLM call sites
│   ├── CROSS-CUTTING-PATTERNS.md # Patterns that apply across all modules
│   ├── INFRASTRUCTURE.md        # Environment, config, results layout
│   └── TESTING-STRATEGY.md      # Test coverage, conventions, and gaps
│
├── services/                    # Per-service specifications
│   ├── trading-agent.md         # CommodityTradingAgent + 7 tools
│   ├── attack-suite.md          # Attack registry, V1-V8 attacks
│   ├── defense-pipeline.md      # 8 defense mechanisms
│   ├── evaluator.md             # RedTeamEvaluator + metrics
│   ├── llm-client.md            # LLMClient unified wrapper
│   ├── generator-v2.md          # CommodityAttackGenerator iterative loop
│   └── adaptive-loop-v3.md      # Full V3 evolutionary attack system
│
├── extraction/                  # 10-pass raw codebase extraction (read-only)
│   ├── README.md                # Pass status dashboard
│   ├── pass-01-structural-inventory/
│   ├── pass-02-dependency-specification/
│   ├── pass-03-data-models/
│   ├── pass-04-attack-defense-contracts/
│   ├── pass-05-business-logic/
│   ├── pass-06-communication-topology/
│   ├── pass-07-cross-cutting-concerns/
│   ├── pass-08-infrastructure/
│   ├── pass-09-testing/
│   └── pass-10-master-synthesis/
│
└── features/                    # SDD feature files
    ├── active/                  # DRAFT / SPEC-APPROVED / IN-PROGRESS features
    └── completed/               # Implemented features
```

---

## Key Documents

- **CONSTITUTION.md** — Axioms that cannot be violated; read before any feature work
- **ARCHITECTURE.md** — Start here to understand the system design
- **DATA-MODELS.md** — Complete data type reference for all module boundaries
- **API-CONTRACTS.md** — Exact interface specifications (method signatures + invariants)
- **extraction/README.md** — 10-pass extraction status dashboard with critical bug inventory

---

## Critical Bugs (from extraction)

| Bug | Location | Description |
|---|---|---|
| BUG-01 | `src/v3/trajectory_defense.py` | `_trajectory_buffer` class attribute shared across all instances |
| BUG-02 | `src/evaluation/evaluator.py` | `_apply_tool_overrides()` silently ignores V7 attack tool keys |
| BUG-03 | `src/defenses/guardrails.py` + evaluator | `modified_input` never applied — GuardrailsDefense is inert |
| BUG-04 | `src/defenses/multi_agent.py` | Fail-open on API errors |
| BUG-05 | `src/generator/attack_generator.py` | `_GeneratedAttack.severity = "high"` (string vs enum) |

---

## SDD Gate

This repo has SDD (Spec-Driven Development) enabled. All writes to `src/` require an approved feature file in `specs/features/active/`. The `specs/` directory itself is always exempt from the gate.

See `.claude/rules/spec-gate.md` for gate rules.

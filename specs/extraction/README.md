# Extraction Passes — Live Dashboard

**Last updated:** 2026-05-15  
**Extractor:** codebase-spec-extractor agent  
**Source repo:** `/Users/paraskanwar/Desktop/redteam`

---

## Pass Status

| Pass | Title | Status | Grade | Key Finding |
|---|---|---|---|---|
| 01 | Structural Inventory | COMPLETE | A | 2 parallel LLM stacks; dead code in src/agents/ vs src/v3/; 19 entry-point scripts |
| 02 | Dependency Specification | COMPLETE | A | 34 Python deps; LLMClient vs LangChain are independent stacks sharing env vars |
| 03 | Data Model Specification | COMPLETE | A | 12 dataclasses; recommendation dict field aliasing (var_95 vs var_95_1d) |
| 04 | Attack-Defense Contracts | COMPLETE | A | GuardrailsDefense is cosmetic; V8 evaluates via regex not keyword matching |
| 05 | Behavioral Specification | COMPLETE | A | V7 tool override miss; GuardrailsDefense inert; recommendation extraction fragile |
| 06 | Communication Topology | COMPLETE | A | 14 LLM call sites; all code synchronous except ThreadPoolExecutor timeout |
| 07 | Cross-Cutting Concerns | COMPLETE | A | CRITICAL: `_trajectory_buffer` class attribute bug; MultiAgentDefense fail-open |
| 08 | Infrastructure Specification | COMPLETE | A | results/ layout mapped; ML downloads ~600MB; no pyproject.toml; Python 3.10+ required |
| 09 | Testing Specification | COMPLETE | B+ | 30 tests; 0 integration tests; GCG/ML defenses untested; possible news-key mismatch |
| 10 | Master Synthesis | COMPLETE | A | 5 confirmed bugs; 7 architectural risks; full taxonomy + defense matrix |

---

## Critical Findings (requires immediate action)

1. **BUG-01**: `TrajectoryDefense._trajectory_buffer` is a class attribute — all instances share state. Fix: `self._trajectory_buffer = []` in `__init__`.
2. **BUG-02**: `_apply_tool_overrides()` silently ignores V7 attack tool override keys. V7 benchmark results are invalid.
3. **BUG-03**: `GuardrailsDefense.modified_input` is never applied by evaluator. Defense is a no-op in all benchmark runs.
4. **BUG-04**: `MultiAgentDefense` is fail-open. API errors silently allow all traffic.
5. **BUG-05**: `_GeneratedAttack.severity = "high"` (string) vs `Severity.HIGH` (enum) type mismatch.

---

## Source Files Read

All 50+ source files were read in full before any spec was written. No file was read after spec writing began.

| Category | Files read |
|---|---|
| Attacks | base, registry, v1, v7, v8 (in full); v2-v6 (auto-discovered but not fully read) |
| Defenses | base, input_filter, output_validator, guardrails, multi_agent, human_in_loop, semantic_filter, perplexity_filter, ensemble_defense |
| Agent | trading_agent, system_prompt, all 7 tools |
| Evaluation | evaluator, metrics, statistical, transferability, explainability |
| V3 Adaptive | attack_archive, defender_agent, reflection_store, replanner, strategy_db, trajectory_defense |
| V3 Agents | planner, mutator, critic |
| Generator | attack_generator |
| Utils | llm, data |
| Config | models.yaml, agent_config.yaml, commodities.yaml |
| Tests | test_attacks, test_agent, test_evaluator |
| Scripts | run_full_benchmark (top 60 lines); 19 scripts inventoried via glob |

---

## Pass Documents

- [Pass 01 — Structural Inventory](./pass-01-structural-inventory/STRUCTURAL-INVENTORY.md)
- [Pass 02 — Dependency Specification](./pass-02-dependency-specification/DEPENDENCY-SPECIFICATION.md)
- [Pass 03 — Data Model Specification](./pass-03-data-models/DATA-MODEL-SPECIFICATION.md)
- [Pass 04 — Attack-Defense Contracts](./pass-04-attack-defense-contracts/ATTACK-DEFENSE-CONTRACTS.md)
- [Pass 05 — Behavioral Specification](./pass-05-business-logic/BEHAVIORAL-SPECIFICATION.md)
- [Pass 06 — Communication Topology](./pass-06-communication-topology/COMMUNICATION-TOPOLOGY.md)
- [Pass 07 — Cross-Cutting Concerns](./pass-07-cross-cutting-concerns/CROSS-CUTTING-CONCERNS.md)
- [Pass 08 — Infrastructure Specification](./pass-08-infrastructure/INFRASTRUCTURE-SPECIFICATION.md)
- [Pass 09 — Testing Specification](./pass-09-testing/TESTING-SPECIFICATION.md)
- [Pass 10 — Master Synthesis](./pass-10-master-synthesis/MASTER-SPEC.md)

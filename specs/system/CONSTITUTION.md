# CommodityRedTeam Constitution

> **Status:** `CURRENT`
> **Generated from:** repository docs, configs, tests, and source inspection
> **Last updated:** 2026-05-13

**Version:** 1.0.0

## 1. Project Identity

| Field | Value |
|---|---|
| Project | CommodityRedTeam |
| Domain | Red teaming agentic AI for commodity trading |
| Primary stack | Python, LangChain, pytest, pandas, sklearn/scipy, optional transformers/XGBoost/SHAP |
| Primary users | Capstone researchers and evaluators |
| Core artifact | Benchmark evidence: attack success, detection rate, false positives, financial impact, transferability |
| Repository shape | Importable framework under `src/`, CLI scripts under `scripts/`, configs under `config/`, outputs under `results/` |

## 2. Non-Negotiable Principles

### Article I - Research Reproducibility

Every benchmarkable behavior must be traceable to versioned source code, YAML config, or generated result artifacts. Scripts must write enough metadata to identify model, defense configuration, attack count, timestamp, and summary metrics.

### Article II - Stable Attack Contracts

Every attack must subclass `Attack`, use a stable `id`, declare category/severity/commodity, implement `prepare()`, and implement `evaluate()`. New attacks must register with `@register` so benchmark scripts discover them automatically.

### Article III - Defense Results Must Be Explainable

Every blocking or warning decision must be represented through `DefenseResult.flags`. Flags should be machine-readable and specific enough to support later analysis.

### Article IV - Failures Must Not Pollute Later Runs

Any attack that mutates tool behavior must reset that behavior after the run. Evaluators and scripts must avoid leaking tool modes, recommendations, or mutable state across attacks unless the experiment explicitly studies memory/context persistence.

### Article V - Trading Safety Rules Are Domain Ground Truth

Position limits, notional thresholds, human approval thresholds, sanctions checks, and required risk assessment fields come from `config/agent_config.yaml` and `config/commodities.yaml`. Code should reference those files instead of duplicating trading constants in scripts.

### Article VI - Secrets Stay Out Of Source

Provider API keys must be loaded from environment variables or `.env`. Source, specs, configs, tests, results, and notebooks must not contain live API secrets.

### Article VII - Flat Results Are The Analysis Interface

Benchmark rows must remain flat dictionaries/CSV rows with stable columns. Statistical analysis, plots, PPTX generation, and manual review depend on this tabular contract.

### Article VIII - Provider Abstraction Belongs In `LLMClient`

Provider-specific API differences should be normalized in `src/utils/llm.py`. Scripts should select model keys such as `groq-llama`, `groq-qwen`, or `mistral-large` rather than importing provider SDKs directly.

### Article IX - Tests Cover Contracts Before Provider Calls

Unit tests should validate contracts, parsing, metrics, and tool modes without requiring external API calls. Network/provider tests belong in explicit benchmark scripts, not default unit tests.

### Article X - Research Claims Need Artifact Evidence

Claims about attack success rate, defense effectiveness, transferability, false positive rate, financial impact, or model vulnerability must point to files under `results/` or regenerated benchmark output.

## 3. Quality Gates

Before code changes are considered complete:

- `pytest` should pass for unit tests that do not require provider API keys.
- New attacks must appear in `get_all_attacks()` and satisfy `tests/test_attacks.py`.
- New defenses must return `DefenseResult` from both phases or inherit the safe base behavior.
- Benchmark scripts must continue to write JSON and CSV outputs with stable columns.
- Changes touching configs must keep YAML parseable and avoid secrets.

## 4. Security And Safety Gates

| Gate | Requirement |
|---|---|
| Secrets | No API keys, tokens, account credentials, or provider secrets committed |
| Generated outputs | Results may contain adversarial prompts; keep them in research artifacts and do not promote them as user-facing guidance |
| Trading recommendations | Outputs are simulated research artifacts only and must not be treated as live trading advice |
| Tool manipulation | Attack modes must be reset after each run |
| Human approval | Any recommendation above configured single-trade notional threshold must be flagged for human approval |

## 5. Coding Standards

- Use type hints for public framework contracts and new modules.
- Keep attacks deterministic where possible; randomization must be seeded or recorded when it affects reported metrics.
- Prefer YAML config lookups over duplicated constants.
- Use structured dictionaries/dataclasses for attack, defense, agent, and result data.
- Preserve importability from repo root; scripts may add the project root to `sys.path`.
- Avoid broad refactors in benchmark-critical paths unless the result schema remains compatible.

## 6. ADR Register

| ID | Decision | Status | Date |
|---|---|---|---|
| ADR-001 | Python research framework with script entry points | Accepted | 2026-05-13 |
| ADR-002 | Dataclass/enumeration contracts for attacks and results | Accepted | 2026-05-13 |
| ADR-003 | Defense pipeline with input and output phases | Accepted | 2026-05-13 |
| ADR-004 | YAML model and trading-domain configuration | Accepted | 2026-05-13 |
| ADR-005 | CSV/JSON result artifacts as canonical experiment interface | Accepted | 2026-05-13 |

## 7. Amendment Process

Changes to these principles should be made in a pull request or explicit local spec update that also updates affected feature specs. Any code change that alters attack contracts, defense contracts, result columns, config schema, or benchmark semantics must update the relevant spec before or alongside implementation.

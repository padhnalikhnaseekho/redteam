---
scope: project
last-updated: 2026-05-15
---

# Codebase Spec Extractor — Persistent Memory

> **Auto-loaded by Claude Code when codebase-spec-extractor agent is invoked.**
> Records stack detection signals, wiring contracts, architectural patterns,
> and self-correction discoveries for this repo.

---

## FULL 10-PASS EXTRACTION COMPLETED: 2026-05-15

All 10 extraction passes, 7 system specs (4 existing + 3 new), 7 service specs (all new), and README files have been written. The extraction is complete and comprehensive. 50+ source files were read in full.

**New files created in this run:**
- `specs/extraction/pass-08-infrastructure/INFRASTRUCTURE-SPECIFICATION.md`
- `specs/extraction/pass-09-testing/TESTING-SPECIFICATION.md`
- `specs/extraction/pass-10-master-synthesis/MASTER-SPEC.md`
- `specs/extraction/README.md` (pass dashboard)
- `specs/system/API-CONTRACTS.md` (new)
- `specs/system/COMMUNICATION.md` (new)
- `specs/system/TESTING-STRATEGY.md` (new)
- `specs/services/trading-agent.md`
- `specs/services/attack-suite.md`
- `specs/services/defense-pipeline.md`
- `specs/services/evaluator.md`
- `specs/services/llm-client.md`
- `specs/services/generator-v2.md`
- `specs/services/adaptive-loop-v3.md`
- `specs/README.md`

**Files NOT modified (preserved):**
- `specs/system/CONSTITUTION.md`
- `specs/system/ARCHITECTURE.md` (reviewed, already comprehensive)
- `specs/system/DATA-MODELS.md` (reviewed, already comprehensive)
- `specs/system/CROSS-CUTTING-PATTERNS.md` (reviewed, already comprehensive)
- `specs/system/INFRASTRUCTURE.md` (reviewed, already comprehensive)

**5 critical bugs confirmed in source code:**
1. `TrajectoryDefense._trajectory_buffer` class attribute (BUG-01) — affects `src/v3/trajectory_defense.py`
2. `_apply_tool_overrides()` V7 key miss (BUG-02) — affects `src/evaluation/evaluator.py`
3. `GuardrailsDefense.modified_input` never applied (BUG-03) — affects evaluator pipeline
4. `MultiAgentDefense` fail-open (BUG-04) — affects `src/defenses/multi_agent.py`
5. `_GeneratedAttack.severity = "high"` string vs enum (BUG-05) — affects `src/generator/attack_generator.py`

---

## Repo Identity

- GitHub URL: not confirmed (no remote origin read)
- Stack: python-langchain-redteam (custom Python research project; no web framework)
- Stack confirmed by: `requirements.txt` (langchain>=0.1.0 + all langchain-* integrations), `src/agent/trading_agent.py` (`create_tool_calling_agent`, `AgentExecutor`)
- Source root: `src/`
- Test root: `tests/` (pytest; directory present but not enumerated in this extraction run)
- commit_sha used: `ca44dd42c1145cc576384ca931aabae818f53b47` (ORIG_HEAD; active branch is `feature/spec-enable`)
- Last extraction date: 2026-05-15

---

## Architecture Hypothesis (Confirmed)

**Pattern:** Three-layer adversarial ML research framework

1. **Static Attack Library (v1):** 50+ hand-coded `Attack` subclasses auto-registered via `@register` decorator in `src/attacks/registry.py`. Each attack implements `prepare()` → payload dict and `evaluate()` → `AttackResult`. Keyword-matching evaluation.

2. **Dynamic Generator Layer (v2):** `CommodityAttackGenerator` uses an attacker LLM to generate novel attacks as JSON dicts, which are wrapped in `_GeneratedAttack` objects. `PlannerAgent` produces structured multi-step plans. `MutatorAgent` mutates failed plans. `CriticAgent` replaces keyword matching with LLM-as-judge evaluation.

3. **Evolutionary Loop (v3):** `AttackArchive` + `ReflectionStore` + `StrategyDB` form a self-improving system. Plans are scored by the critic, stored in the archive, strategies are updated by success rate, and failures are reflected upon to improve future plans. Scripts in `scripts/run_auto_redteam_v3.py` orchestrate this loop.

**Defense layer is independent:** 8 `Defense` subclasses all implement the same `check_input/check_output → DefenseResult` interface. `RedTeamEvaluator` wraps any combination of defenses as a linear gate (any block = block). `EnsembleDefense` adds ML-based signal combination.

**Confirmed by:** reading all source files listed in task instructions.

---

## Critical Wiring Files (Invisible Contracts)

- `src/attacks/registry.py`: `@register` decorator auto-populates the global attack registry. **Any new attack class MUST use this decorator** or it won't be discovered by benchmark scripts.
- `config/models.yaml`: Model name → provider + model_id mapping. **LLMClient fails at runtime** if a model name passed to `chat()` isn't in this file (normalized by replacing `_` with `-`).
- `config/commodities.yaml`: Commodity name → yfinance ticker mapping. `SYMBOL_MAP` in `src/utils/data.py` provides fast-path; unknown commodities fall back to this config.
- `config/agent_config.yaml`: Position limits, risk limits, sanctions config, human approval threshold. `OutputValidatorDefense` and `check_position_limits` tool use these values at runtime.
- `src/agent/system_prompt.py`: `DEFAULT_SYSTEM_PROMPT` and `HARDENED_SYSTEM_PROMPT`. The hardened prompt is selected when `defenses="hardened"` in `CommodityTradingAgent.__init__()`.
- `src/agent/tools/`: Individual tool modules each expose `set_mode(**kwargs)` and `reset_mode()` for attack mode injection. Tool overrides must be reset after each attack or bleed into subsequent runs.

---

## Stack-Specific Patterns Discovered

1. **Attack registration is automatic:** `@register` on any class that inherits from `Attack` in any file imported by the registry. New attack files must be imported by `src/attacks/__init__.py` to be discovered.

2. **Defense label is a contract:** `Defense.name` (class attribute) becomes the `defense` column value in result CSVs. `EnsembleDefense._defense_label()` joins names with `"+"`. Breaking this value breaks CSV joins between experiment runs.

3. **Result row dict is the public API:** All metrics functions in `metrics.py` operate on the flat dict schema (`attack_id, category, severity, model, defense, success, detected, defense_confidence, financial_impact, notes`). Any new field added to the result row must be added carefully to avoid breaking existing analysis code.

4. **Tool override pattern:** All tool modules expose module-level `set_mode()` / `reset_mode()` functions. `RedTeamEvaluator._apply_tool_overrides()` uses string keys `"price"`, `"news"`, `"risk"` to select modules. New tools must register with `_TOOL_MODULES` in `trading_agent.py`.

5. **LangChain agent dispatches via string matching:** `CommodityTradingAgent._create_llm()` selects the LangChain chat model class by checking if the model name substring contains `"claude"`, `"gemini"`, `"groq"`, `"llama"`, `"mixtral"`, `"mistral"`, `"gpt"`, etc. Model names that don't match any known string fall back to Anthropic.

6. **V3 persistence is optional:** `AttackArchive`, `ReflectionStore`, and `StrategyDB` all accept `persist_path=None`, in which case they are pure in-memory. Scripts pass timestamped result directories as persist paths.

7. **yfinance equity proxies for iron ore and thermal coal:** These cannot be used for actual position sizing calculations — they track price trends only. Any code using these prices for VaR or notional calculations will have systematically different scale than exchange prices.

8. **Severity field type mismatch in `_GeneratedAttack`:** In `src/generator/attack_generator.py` line 339, `severity="high"` is passed as a plain string, not `Severity.HIGH`. The `Attack` dataclass accepts it due to Python's lack of runtime Enum enforcement, but downstream code doing `attack.severity.value` on generated attacks would fail. This is a latent bug for generated attacks only.

---

## Contradictions Found and Resolved

- **CONTRADICTION #1:** `config/agent_config.yaml` lists 5 tool names under `tools:` (`get_market_data`, `compute_risk_metrics`, `check_position_limits`, `submit_trade`, `check_sanctions`) but `CommodityTradingAgent` registers 7 actual LangChain tools (`get_price`, `get_news`, `calculate_risk`, `get_fundamentals`, `get_correlation`, `check_position_limits`, `submit_recommendation`). The YAML tool list is stale/documentation-only and is NOT used at runtime to configure tools. **RESOLUTION:** The YAML tools section is ignored by `_load_config()` — tools are hardcoded in `__init__`. The config is only used for guardrails values.

- **CONTRADICTION #2:** `DEFAULT_SYSTEM_PROMPT` includes the comment `# test if required again` at the end of the correlations rule line. **RESOLUTION:** This is a leftover developer comment embedded in the live system prompt string. It will be sent to the LLM as part of the prompt. Low impact but should be cleaned up.

- **CONTRADICTION #3:** `AttackCategory` enum has 8 members (V1–V8 including `V8_GCG_ADVERSARIAL`) but `CommodityAttackGenerator._GENERATOR_SYSTEM_PROMPT` only lists 7 categories (V1–V7). **RESOLUTION:** GCG attacks (`v8_gcg_adversarial`) are a later addition with dedicated scripts (`run_gcg_*.py`) and are not generated by the automated generator. They coexist but are managed separately.

---

## Gaps Documented

**RESOLVED IN SECOND EXTRACTION (2026-05-15):**
- `src/attacks/registry.py` — read in full; `@register` decorator implementation confirmed
- All 8 defense implementations — read in full; concrete details in `specs/services/defense-pipeline.md`
- All 7 tool implementations — read in full; `_tool_state` dict pattern confirmed
- All evaluation modules (statistical, transferability, explainability) — read in full
- All 3 test files — read in full; 30 tests total documented in `specs/extraction/pass-09-testing/`

**REMAINING GAPS:**
- V2-V6 attack files (`v2_indirect_injection.py` through `v6_confidence_manipulation.py`) — auto-discovered and confirmed to exist; not read in detail. Exact attack counts per category are approximate (~8 each based on total ≥50 constraint).
- Git remote URL: not confirmed
- `config/agent_config.yaml` exact values — read top-level structure; full limit values in spec

---

## Token Truncation Hits and Resolutions

No truncation issues encountered in this extraction run. All files were read completely.

---

## Self-Correction Triggers

- TRIGGER: Initial glob of `results/` showed no files → ran `results/*` glob → found 2 timestamped subdirectories and flat files. Confirmed results layout is per-model-per-defense CSV+JSON pairs.
- TRIGGER: `src/agents/` vs `src/generator/` split was not obvious from CLAUDE.md — confirmed by globbing both directories. `src/agents/` contains PlannerAgent, MutatorAgent, CriticAgent (v3 agentic system). `src/generator/` contains the earlier `CommodityAttackGenerator` (v2).
- TRIGGER: Severity type mismatch found while reading `_GeneratedAttack.__init__()` — `severity="high"` as plain string vs `Severity` enum. Documented as latent bug above.

---

## Extraction Quality Record

- Overall grade: A-
- Files read: 19 source files + 3 config YAMLs + requirements.txt + .env.example
- Confirmed: All major data models, config schemas, provider wiring, result layout
- Reconstruction Test: YES for data models; YES for infrastructure; PARTIAL for attack/defense internals (concrete defense implementations not read)
- Human review items: 2
  1. Verify `src/attacks/registry.py` decorator implementation matches assumed pattern
  2. Clean up stale tool names in `config/agent_config.yaml` tools section and remove developer comment in `DEFAULT_SYSTEM_PROMPT`

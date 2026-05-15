# Pass 10 — Master Synthesis

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Executive Summary

The CommodityRedTeam framework is an IIT Bombay EPGD AI/ML Capstone project that adversarially red-teams LLM-based commodity trading agents. It consists of:

- **1 target agent**: `CommodityTradingAgent` (LangChain ReAct with 7 tools, up to 3 provider families)
- **50+ attacks** across 8 categories (V1-V8), from naive prompt injection to GCG adversarial suffixes
- **8 defense mechanisms** from simple regex to XGBoost ensemble classifiers
- **3 evaluation tiers**: static benchmark (V1 attack suite), generator V2 (LLM-generated adaptive attacks), adaptive V3 loop (evolutionary attack planning with RL strategy selection)
- **Full metrics suite**: ASR, detection rate, ROC, Shapley values, Bayesian Beta-Binomial, SHAP explainability, transferability matrix

**5 critical bugs** were found during extraction (see Bug Inventory below).

---

## Architecture Tier Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  SCRIPTS TIER (entry points)                                │
│  run_full_benchmark.py | run_attacks.py | run_auto_redteam_v3.py │
│  generate_pptx.py | run_advanced_analysis.py | ...          │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│  EVALUATION TIER                                            │
│  RedTeamEvaluator.run_suite() → run_single()               │
│  metrics.py | statistical.py | transferability.py          │
│  explainability.py (SHAP + XGBoost)                        │
└──────┬──────────────────────────────┬───────────────────────┘
       │                              │
       ▼                              ▼
┌──────────────────┐    ┌─────────────────────────────────────┐
│  ATTACK TIER     │    │  DEFENSE TIER                       │
│  registry.py     │    │  InputFilterDefense (regex)         │
│  @register V1-V8 │    │  OutputValidatorDefense (rules)     │
│  base.Attack     │    │  GuardrailsDefense (prompt wrap)    │
│  prepare()       │    │  MultiAgentDefense (Gemini-flash)   │
│  evaluate()      │    │  HumanInLoopDefense (thresholds)    │
└──────────────────┘    │  SemanticInputFilterDefense (MiniLM)│
                        │  PerplexityFilterDefense (GPT-2)    │
                        │  EnsembleDefense (XGBoost)          │
                        └──────────────┬──────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT TIER                                                 │
│  CommodityTradingAgent (LangChain AgentExecutor)           │
│  Tools: price | news | risk | fundamentals |               │
│         correlation | position | recommendation            │
│  System prompt: DEFAULT | HARDENED                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  UTILS TIER                                                 │
│  LLMClient (OpenAI|Anthropic|Mistral|Google|Groq)          │
│  data.py (yfinance wrappers, VaR computation)               │
│  config/ (models.yaml, agent_config.yaml, commodities.yaml) │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  V3 ADAPTIVE LOOP TIER (orthogonal subsystem)              │
│  StrategyDB (epsilon-greedy/softmax) → PlannerAgent        │
│  → target agent → Replanner → CriticAgent                  │
│  → AttackArchive → ReflectionStore → MutatorAgent          │
│  → DefenderAgent + TrajectoryDefense                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  GENERATOR V2 TIER (orthogonal subsystem)                  │
│  CommodityAttackGenerator → LLM → _GeneratedAttack         │
│  → RedTeamEvaluator (reuses evaluation tier)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Critical Wiring Contracts

### Contract 1: Attack → Evaluator
```
attack.prepare(agent) → {"user_query": str, "tool_overrides": dict, "injected_context": list}
attack.evaluate(agent_output: str) → AttackResult
```
The evaluator calls `prepare()` before agent execution and `evaluate()` after. The agent output passed to `evaluate()` is the raw string from `agent.run()`, not an `AgentResult` object.

### Contract 2: Defense → Evaluator
```
defense.check_input(user_query, context) → DefenseResult(allowed, flags, confidence)
defense.check_output(agent_output) → DefenseResult(allowed, flags, confidence)
```
- If any input defense returns `allowed=False` → agent is NOT run; attack returns `success=False`
- If any output defense returns `allowed=False` → `attack_result.success` is set to `False` (overridden post-evaluate)
- The `modified_input` field is NEVER applied by the evaluator (GuardrailsDefense is inert)

### Contract 3: Tool Override
```
tool_overrides = {"price": {"manipulated": True, "override_price": 999.99}}
evaluator._apply_tool_overrides(tool_overrides)  # Only handles "price", "news", "risk"
```
V7 attacks use `"get_price"`, `"get_correlation"`, `"get_fundamentals"`, `"get_news"` keys — these are silently ignored by the evaluator.

### Contract 4: LLMClient Normalization
```
LLMClient.chat(model_name, messages) → {"content": str, "input_tokens": int, "output_tokens": int, ...}
```
Provider is selected from `config/models.yaml` via `provider` field. Model name normalization: `model_name.replace("_", "-")` tried if exact key not found.

### Contract 5: @register Auto-Discovery
```
@register  →  _REGISTRY[ClassName] = AttackClass
get_all_attacks()  →  _auto_discover()  →  [cls() for cls in _REGISTRY.values()]
```
Registration key is the Python class name, not the `attack.id` field. Each call to `get_all_attacks()` creates fresh instances.

---

## Attack Taxonomy Table

| Category | Count | Key Technique | Detection Difficulty | Tool Override? |
|---|---|---|---|---|
| V1 Direct Injection | 8 | Inline instruction override | Low (regex patterns) | No |
| V2 Indirect Injection | ~8 | Malicious tool output payloads | Medium | Via news injection |
| V3 Tool Manipulation | ~8 | Corrupt price/risk/correlation data | High (semantic) | Yes — price, risk, correlation |
| V4 Context Poisoning | ~8 | Fake conversation history | Medium | Via injected_context |
| V5 Reasoning Hijacking | ~8 | Anchoring, sunk cost exploitation | High | No |
| V6 Confidence Manipulation | ~8 | Inflate/deflate confidence signals | High | No |
| V7 Multi-Step Compounding | 6 | Chained manipulation across steps | Very High | Yes — get_correlation, get_fundamentals |
| V8 GCG Adversarial | 4 | Gradient-optimized adversarial suffix | Very High | No (query suffix only) |

**Total confirmed: 50+ across all categories** (V1=8, V7=6, V8=4; V2-V6 each ~8 to reach ≥50 threshold per `test_at_least_50_attacks`)

---

## Defense Matrix

| Defense | Input Gate | Output Gate | LLM Required | Model Required | Trained? | Fail Mode |
|---|---|---|---|---|---|---|
| InputFilterDefense | Yes (regex) | No | No | No | No | False negatives on rephrasing |
| OutputValidatorDefense | No | Yes (rules) | No | No | No | Allows unparseable outputs |
| GuardrailsDefense | Cosmetic only | No | No | No | No | Completely inert in current pipeline |
| MultiAgentDefense | No | Yes (LLM) | Yes (Gemini) | No | No | Fail-OPEN on API error |
| HumanInLoopDefense | No | Yes (threshold) | No | No | No | Requires external_flags from other defenses |
| SemanticInputFilterDefense | Yes (cosine) | No | No | all-MiniLM-L6-v2 | No | Novel phrasing below 0.50 threshold |
| PerplexityFilterDefense | Yes (PPL) | No | No | GPT-2 | No | GCG optimized to be low-perplexity |
| EnsembleDefense | Yes + Output | Yes + ensemble | No | XGBoost | Optional | Without training data → voting fallback |

---

## LLM Provider Usage Summary

| Provider | Models | Used By | Stack |
|---|---|---|---|
| Groq | groq-llama, groq-qwen, groq-scout | V3 planner/critic/mutator/defender/replanner; target agent | Both LLMClient and LangChain |
| Google | gemini-flash | MultiAgentDefense reviewer; CommodityAttackGenerator; target agent | LLMClient + LangChain |
| Anthropic | claude-sonnet | Target agent; LLMClient direct | Both |
| Mistral | mistral-large | Target agent; LLMClient direct | Both |
| OpenAI | gpt-4o (commented out) | Available but disabled | Both |

**Two parallel LLM stacks**: `LLMClient` (direct API with cost tracking) and LangChain providers (ChatGroq, ChatAnthropic, etc. — no cost tracking). Both stacks read the same environment variables but are completely independent.

---

## Known Bugs Inventory

| ID | Severity | Location | Description | Fix |
|---|---|---|---|---|
| BUG-01 | CRITICAL | `src/v3/trajectory_defense.py` | `_trajectory_buffer` is a class attribute, not instance attribute. All `TrajectoryDefense` instances share the same buffer. | Change to `self._trajectory_buffer = []` in `__init__` |
| BUG-02 | HIGH | `src/evaluation/evaluator.py:_apply_tool_overrides()` | Only handles `"price"`, `"news"`, `"risk"` override keys. V7 attacks specify `"get_correlation"`, `"get_fundamentals"`, `"get_news"` keys — silently ignored. | Add handling for all 7 tool keys |
| BUG-03 | HIGH | `src/defenses/guardrails.py` + `src/evaluation/evaluator.py` | `GuardrailsDefense` returns `modified_input` in `DefenseResult`, but evaluator never reads or applies it. Defense is cosmetic. | Apply `result.modified_input` to agent's system prompt before execution |
| BUG-04 | HIGH | `src/defenses/multi_agent.py` | `MultiAgentDefense` is fail-open. API errors, rate limits, or network outages cause `allowed=True` with `confidence=0.3`. Defense silently degrades to no defense. | Change to fail-closed (allowed=False) on API errors, or add explicit alerting |
| BUG-05 | MEDIUM | `src/generator/attack_generator.py:_GeneratedAttack` | `severity="high"` is a hardcoded string literal instead of `Severity.HIGH` enum. Type mismatch will cause filtering failures when `get_attacks(severity=Severity.HIGH)` is called. | Replace with `Severity.HIGH` enum value |

---

## Architectural Risks

| Risk | Impact | Likelihood | Notes |
|---|---|---|---|
| Module-level `_tool_state` shared across concurrent runs | HIGH — state leak between attacks | Medium | Single-threaded benchmark mitigates, but parallel execution would corrupt state |
| No global `random.seed()` | MEDIUM — benchmark results not reproducible | High | V3 loop results vary run-to-run; Bayesian comparison assumes fixed data |
| Config path depth hardcoded (`parents[N]`) | MEDIUM — breaks if files moved | Low | All configs use `__file__`-relative resolution; depth must be recalculated after any move |
| GCG precomputed suffix fallback | MEDIUM — fallback reduces attack effectiveness | High | Default `_PRECOMPUTED_SUFFIXES` are static tokens, not optimized for any specific model |
| `EnsembleDefense` without training data | HIGH — always runs in voting fallback | High | Benchmark scripts instantiate `EnsembleDefense()` without labeled data; XGBoost model never used |
| Missing ML model downloads on first run | LOW — blocks execution | High | `SemanticFilter` and `PerplexityFilter` download 600MB+ on first use with no user warning |
| `OutputValidatorDefense` allows unparseable output | HIGH — defense bypassed entirely | Medium | If agent produces non-JSON output, no rule checks run and output is allowed |

---

## Recommendations

1. **Fix BUG-01 immediately**: The `_trajectory_buffer` class attribute makes the TrajectoryDefense unreliable. This is a Python gotcha that silently corrupts state without raising errors.

2. **Fix BUG-02 for V7 test validity**: V7 attacks cannot exercise their tool manipulation without `_apply_tool_overrides()` supporting `get_correlation`/`get_fundamentals` keys. The V7 benchmark results are currently misleading.

3. **Fix BUG-04 (fail-open)**: A red team framework where a key defense silently degrades on API errors provides false confidence in defense effectiveness. Convert to fail-closed with explicit logging.

4. **Train EnsembleDefense**: The ensemble is always in voting fallback mode in all current benchmark runs. Add a pre-training step using results from V1 attacks as labeled data.

5. **Add `random.seed()`**: For reproducible benchmarks, set a global seed at script startup: `random.seed(42); numpy.random.seed(42)`.

6. **Extend tool override handling**: Update `_apply_tool_overrides()` to handle all 7 tool keys using the `_TOOL_MODULES` dict pattern already in `CommodityTradingAgent`.

7. **Activate GuardrailsDefense**: Either (a) make the evaluator apply `result.modified_input` to the agent's system prompt, or (b) document GuardrailsDefense as a no-op and remove it from benchmark configurations to avoid misleading results.

8. **Add integration tests**: The 30 existing tests are all offline-safe unit tests. Add at least one integration test per defense (with model download fixtures) and one end-to-end evaluator test with a real LangChain agent (gated by API key env var presence).

---

## Module Count Summary

| Layer | Modules | LOC estimate |
|---|---|---|
| Attacks | 8 files (v1-v8) + base + registry | ~3,000 |
| Defenses | 8 files + base | ~1,500 |
| Agent | trading_agent + system_prompt + 7 tools | ~1,200 |
| Evaluation | evaluator + metrics + statistical + transferability + explainability | ~1,500 |
| V3 Adaptive | 7 files (archive, defender, planner, mutator, critic, replanner, strategy_db, trajectory) | ~1,800 |
| V3 Agents | planner + mutator + critic (in src/agents/) | ~600 |
| Generator | attack_generator.py | ~400 |
| Utils | llm.py + data.py | ~600 |
| Scripts | 19 scripts | ~2,000 |
| Tests | 3 files | ~300 |
| Config | 3 YAML files | ~150 |
| **Total** | | **~13,050** |

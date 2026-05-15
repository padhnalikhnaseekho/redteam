# Pass 01 — Structural Inventory

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Full Directory Tree

```
redteam/
├── CLAUDE.md                           ← Project overview and SDD rules
├── requirements.txt                    ← Python package dependencies
├── config/
│   ├── agent_config.yaml               ← Agent guardrails, position limits, risk limits
│   ├── commodities.yaml                ← 10 commodity definitions with price ranges and tickers
│   └── models.yaml                     ← 6 LLM model configs with provider and cost data
├── scripts/
│   ├── run_full_benchmark.py           ← CLI: runs all attacks × all defenses × all models
│   ├── run_attacks.py                  ← CLI: single model/defense attack run
│   ├── run_auto_redteam_v3.py          ← CLI: full adaptive v3 red team loop
│   ├── run_experiment_attack_defend.py ← CLI: 3-condition attack-discover-defend experiment
│   └── generate_pptx.py               ← CLI: generate PowerPoint results slides (not read)
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── trading_agent.py            ← CommodityTradingAgent — LangChain ReAct agent
│   │   ├── system_prompt.py            ← DEFAULT_SYSTEM_PROMPT + HARDENED_SYSTEM_PROMPT
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── price.py                ← get_price tool; yfinance + fallback; attack mode: override_price
│   │       ├── news.py                 ← get_news tool; static news + attack mode: inject_payload
│   │       ├── risk.py                 ← calculate_risk tool; parametric VaR; attack mode: risk_multiplier
│   │       ├── fundamentals.py         ← get_fundamentals tool; attack mode: stale_data
│   │       ├── correlation.py          ← get_correlation tool; pre-computed matrix; attack mode: override_correlation
│   │       ├── position.py             ← check_position_limits tool; attack mode: override (always within limits)
│   │       └── recommendation.py       ← submit_recommendation tool; validates and routes, $5M threshold
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── base.py                     ← Attack/AttackResult/AttackCategory/Severity dataclasses
│   │   ├── registry.py                 ← _REGISTRY dict, @register decorator, _auto_discover()
│   │   ├── v1_direct_injection.py      ← 8 attacks: IgnoreRiskLimits, RoleManipulation, FakeApproval, etc.
│   │   ├── v2_indirect_injection.py    ← Indirect injection attacks (not read in full)
│   │   ├── v3_tool_manipulation.py     ← Tool manipulation attacks (not read in full)
│   │   ├── v4_context_poisoning.py     ← Context poisoning attacks (not read in full)
│   │   ├── v5_reasoning_hijacking.py   ← Reasoning hijacking attacks (not read in full)
│   │   ├── v6_confidence_manipulation.py ← Confidence manipulation attacks (not read in full)
│   │   ├── v7_multi_step_compounding.py  ← 6 attacks: chained multi-vector attacks
│   │   └── v8_gcg_adversarial.py       ← 4 GCG gradient-optimized suffix attacks + optimizer
│   ├── defenses/
│   │   ├── __init__.py
│   │   ├── base.py                     ← Defense base class + DefenseResult dataclass
│   │   ├── input_filter.py             ← D1: InputFilterDefense — regex + keyword blocklist
│   │   ├── output_validator.py         ← D2: OutputValidatorDefense — position/risk/price/notional checks
│   │   ├── guardrails.py               ← D3: GuardrailsDefense — system prompt hardening
│   │   ├── multi_agent.py              ← D4: MultiAgentDefense — second LLM reviewer
│   │   ├── human_in_loop.py            ← D5: HumanInLoopDefense — simulated human review
│   │   ├── semantic_filter.py          ← D6: SemanticInputFilterDefense — sentence-transformers cosine sim
│   │   ├── perplexity_filter.py        ← D7: PerplexityFilterDefense — GPT-2 perplexity spike detection
│   │   └── ensemble_defense.py         ← D8: EnsembleDefense — XGBoost over base defense signals
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py                ← RedTeamEvaluator — orchestrates attacks, defenses, results
│   │   ├── metrics.py                  ← ASR, detection rate, FPR, financial impact, ROC, Shapley
│   │   ├── statistical.py              ← Chi-squared, McNemar, Bayesian Beta-Binomial, MI, entropy
│   │   ├── transferability.py          ← Transfer matrix, Fisher's exact, Jaccard similarity
│   │   └── explainability.py           ← SHAP/XGBoost attack success predictor, defense effectiveness
│   ├── generator/
│   │   ├── __init__.py
│   │   └── attack_generator.py         ← CommodityAttackGenerator — LLM-driven attack generation + iterative loop
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py                  ← PlannerAgent — structured multi-step plan generation
│   │   ├── mutator.py                  ← MutatorAgent — LLM-driven attack mutation
│   │   └── critic.py                   ← CriticAgent — LLM-as-judge evaluation
│   ├── v3/
│   │   ├── __init__.py
│   │   ├── attack_archive.py           ← AttackArchive — scored archive with evolutionary selection
│   │   ├── defender_agent.py           ← DefenderAgent — LLM trajectory reviewer + DefenderVerdict
│   │   ├── reflection_store.py         ← ReflectionStore — structured failure analysis store
│   │   ├── replanner.py                ← Replanner — mid-attack step revision on obstacle
│   │   ├── strategy_db.py              ← StrategyDB — epsilon-greedy/softmax strategy selection
│   │   └── trajectory_defense.py       ← TrajectoryDefense — multi-step attack pattern detection
│   └── utils/
│       ├── __init__.py
│       ├── llm.py                      ← LLMClient — unified provider wrapper with cost tracking
│       └── data.py                     ← data utilities: yfinance prices, VaR, correlations
├── tests/
│   ├── test_agent.py                   ← Tests for system prompts and tool mode switching
│   ├── test_attacks.py                 ← Tests for attack registry and prepare() contracts
│   └── test_evaluator.py               ← Tests for RedTeamEvaluator and metrics functions
├── results/                            ← Output directory for benchmark results (CSV/JSON)
│   └── gcg_suffix_cache.json           ← Cache for GCG-generated adversarial suffixes
└── specs/                              ← This directory (spec gate exempt)
```

---

## Entry Points

| Script | Purpose | Key Args |
|---|---|---|
| `scripts/run_full_benchmark.py` | Full model × defense matrix benchmark | `--models`, `--skip-multi-agent` |
| `scripts/run_attacks.py` | Single model/defense run | `--model`, `--defense`, `--category`, `--all` |
| `scripts/run_auto_redteam_v3.py` | Full adaptive v3 loop | `--rounds`, `--plans-per-round`, `--target-model`, `--attacker-model` |
| `scripts/run_experiment_attack_defend.py` | 3-condition A/B/C experiment | `--target-model`, `--rounds`, `--delay` |
| `scripts/generate_pptx.py` | Generate PowerPoint slides | (not read) |

---

## Module Dependency Graph

```
scripts/run_full_benchmark.py
    → src.agent.system_prompt
    → src.attacks.registry (→ src.attacks.base + v1..v8)
    → src.defenses.{input_filter,output_validator,guardrails,human_in_loop,
                    multi_agent,semantic_filter,perplexity_filter,ensemble_defense}
    → src.evaluation.evaluator (→ src.attacks.base, src.defenses.base)
    → src.evaluation.metrics
    → src.utils.llm

src.agent.trading_agent
    → src.agent.system_prompt
    → src.agent.tools.{price,news,risk,fundamentals,correlation,position,recommendation}
    → langchain_core, langchain (provider via model name)

src.evaluation.evaluator
    → src.attacks.base
    → src.defenses.base
    → src.agent.tools.{price,news,risk}  (apply_tool_overrides)
    → pandas

src.defenses.multi_agent
    → src.utils.llm

src.defenses.{output_validator,human_in_loop}
    → config/commodities.yaml + config/agent_config.yaml (via yaml)

src.defenses.semantic_filter
    → sentence_transformers (lazy)
    → numpy

src.defenses.perplexity_filter
    → torch, transformers (lazy)
    → numpy

src.defenses.ensemble_defense
    → src.defenses.base
    → xgboost, sklearn (train mode)
    → joblib (save/load)
    → numpy

src.evaluation.{metrics,statistical,transferability,explainability}
    → pandas, numpy, scipy, sklearn
    → xgboost, shap (explainability only)

src.generator.attack_generator
    → src.attacks.base
    → src.utils.llm
    → src.evaluation.evaluator

src.agents.{planner,mutator,critic}
    → src.utils.llm

src.v3.{attack_archive,reflection_store,strategy_db}
    → stdlib only (json, dataclasses, pathlib, random, math)

src.v3.{defender_agent,replanner}
    → src.utils.llm

src.v3.trajectory_defense
    → src.defenses.base

src.utils.llm
    → config/models.yaml (yaml)
    → os.environ (API keys)
    → provider SDKs (lazy: openai, anthropic, mistralai, google-genai, groq)
    → dotenv

src.utils.data
    → config/commodities.yaml (yaml)
    → yfinance, numpy, pandas, scipy
```

---

## Package `__init__.py` Exports

| Package | Exports |
|---|---|
| `src/__init__.py` | Empty |
| `src/agent/__init__.py` | Empty |
| `src/agent/tools/__init__.py` | Empty (imports implicit) |
| `src/attacks/__init__.py` | Empty |
| `src/defenses/__init__.py` | Empty |
| `src/evaluation/__init__.py` | Empty |
| `src/generator/__init__.py` | Empty |
| `src/agents/__init__.py` | Empty |
| `src/v3/__init__.py` | Empty |
| `src/utils/__init__.py` | Empty |

All public API accessed via direct module imports.

---

## Dead Code / Orphaned Files

- `scripts/generate_pptx.py` — exists but not read; likely functional given it is listed in CLAUDE.md.
- `config/agent_config.yaml` has a `tools:` section listing tool names that do NOT match the actual LangChain tools in `trading_agent.py` (e.g., `get_market_data` vs `get_price`). This section is unused by the agent code.
- `DEFAULT_SYSTEM_PROMPT` in `system_prompt.py` contains a trailing comment `#test if required again` — orphaned dev note on line 33.
- `src/utils/data.py` provides `compute_var()`, `get_daily_returns()`, `compute_correlation_matrix()` etc., but these are NOT imported by the evaluation pipeline — the evaluator uses the tool-level `calculate_risk_impl()` instead. `data.py` appears to be a utility library not yet wired into evaluations.
- `EnsembleDefense` is listed in `run_full_benchmark.py` DEFENSE_MAP but its `train()` method requires labeled training data that is not generated in the benchmark pipeline — it would run in fallback voting mode.

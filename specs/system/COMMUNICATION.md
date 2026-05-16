---
spec-type: system
title: Communication Topology
last-updated: 2026-05-15
status: CURRENT
---

# Communication Topology — CommodityRedTeam

This document maps all inter-module communication in the framework: LLM calls, internal function calls, external data fetches, and data persistence. An SDD agent can use this to understand what calls any given change will affect.

---

## 1. Full Call Graph

### Static Benchmark Path (`run_full_benchmark.py`)

```
build_agent(model_name, client)
  → LLMClient.chat(model_name, messages)            [LLM call — simple agent wrapper]
get_all_attacks()
  → _auto_discover() → importlib.import_module(v1..v8)
[for each model × defense]:
  RedTeamEvaluator(agent, attacks, defenses)
  evaluator.run_suite(model=model_name)
    → evaluator.run_single(attack) [see detail below]
  attack_success_rate(evaluator.results)
  detection_rate(evaluator.results)
  financial_impact_summary(evaluator.results)
```

### Single Attack Run (`RedTeamEvaluator.run_single`)

```
attack.prepare(agent)
  [for each defense]:
    defense.check_input(user_query, context)
      InputFilterDefense:         regex + string match [local]
      SemanticInputFilterDefense: SentenceTransformer.encode() [local model]
      PerplexityFilterDefense:    GPT2LMHeadModel() [local model]
      GuardrailsDefense:          string manipulation [local]
      MultiAgentDefense:          LLMClient.chat(gemini-flash, ...) [LLM call]
      EnsembleDefense:            calls each base defense [recursive]
  evaluator._apply_tool_overrides(overrides)
    → price.set_mode() / news.set_mode() / risk.set_mode()
  agent.run(user_query) OR agent.invoke(user_query)
    → LangChain AgentExecutor.invoke({"input": query})
      → ChatGroq/ChatAnthropic/etc. (LangChain tool-calling loop)
      → tool callbacks: get_price(), get_news(), calculate_risk(), etc.
  evaluator._reset_tool_overrides()
  [for each defense]:
    defense.check_output(agent_output)
      OutputValidatorDefense:   JSON parsing + rule checks [local]
      HumanInLoopDefense:       threshold checks [local]
      MultiAgentDefense:        LLMClient.chat(gemini-flash, ...) [LLM call]
      EnsembleDefense:          majority vote [local]
  attack.evaluate(agent_output)
    → keyword matching [local]
```

### V3 Adaptive Loop (`run_auto_redteam_v3.py`)

```
StrategyDB.select()              → epsilon_greedy() / softmax()
ReflectionStore.to_prompt_context(strategy_id)
StrategyDB.to_prompt_context()
PlannerAgent.generate_plan(...)
  → LLMClient.chat(groq-qwen, messages)            [LLM call]
[for each step in plan]:
  target_agent(step["user_query"])
    → LLMClient.chat(target_model, messages)       [LLM call]
  Replanner.should_replan(response, blocked, step)
  Replanner.replan(plan, trajectory, obstacle)
    → LLMClient.chat(groq-qwen, messages)          [LLM call — if replanning]
CriticAgent.evaluate(plan, agent_output)
  → LLMClient.chat(groq-qwen, messages)            [LLM call]
AttackArchive.add(plan, result)
StrategyDB.update(strategy_id, success, score)
[if failure]:
  CriticAgent.explain_failure(plan, trajectory)
    → LLMClient.chat(groq-qwen, messages)          [LLM call]
  PlannerAgent.refine(plan, reflection)
    → LLMClient.chat(groq-qwen, messages)          [LLM call]
  ReflectionStore.add_from_dict(reflection)
DefenderAgent.review(trajectory)
  → LLMClient.chat(groq-qwen, messages)            [LLM call]
TrajectoryDefense.detect(trajectory)               [local, no LLM]
top_k = archive.select_for_evolution(k=5)
[for each parent]:
  MutatorAgent.mutate(parent.plan, agent_output)
    → LLMClient.chat(groq-qwen, messages)          [LLM call]
```

---

## 2. LLM Call Sites

All LLM calls in the codebase:

| Location | Model | Purpose |
|---|---|---|
| `MultiAgentDefense.check_input()` | `gemini-flash` (default) | Review query for safety violations |
| `MultiAgentDefense.check_output()` | `gemini-flash` (default) | Review agent output for violations |
| `PlannerAgent.generate_plan()` | `groq-qwen` (default) | Generate structured attack plan |
| `PlannerAgent.generate_adaptive_plans()` | `groq-qwen` (default) | Generate multiple plan variants |
| `PlannerAgent.refine()` | `groq-qwen` (default) | Refine plan based on failure analysis |
| `MutatorAgent.mutate()` | `groq-qwen` (default) | Mutate failed attack into variant |
| `CriticAgent.evaluate()` | `groq-qwen` (default) | Judge attack success/failure |
| `CriticAgent.explain_failure()` | `groq-qwen` (default) | Structured failure analysis |
| `CriticAgent.evaluate_batch()` | `groq-qwen` (default) | Batch judgment for multiple attacks |
| `DefenderAgent.review()` | `groq-qwen` (default) | Review trajectory for safety violations |
| `Replanner.replan()` | `groq-qwen` (default) | Generate revised attack steps |
| `CommodityAttackGenerator.generate_attack()` | `gemini-flash` (default) | Generate raw attack dict |
| `CommodityTradingAgent` (LangChain) | Model key (default: `groq-llama`) | Agent reasoning + tool calling |
| `build_agent()` in scripts | Any model key | Simple agent callable |

**Total LLM call sites: 14 distinct code locations**

---

## 3. External Data Calls

### yfinance (Yahoo Finance)

| Location | Function | What it fetches |
|---|---|---|
| `src/agent/tools/price.py:get_price_impl()` | `yf.Ticker(ticker).history(period="5d")` | Latest close price; 5-day OHLCV |
| `src/utils/data.py:get_historical_prices()` | `yf.download(ticker, period=period)` | Historical daily OHLCV |
| `src/utils/data.py:get_current_price()` | Via `get_historical_prices()` | Latest close |
| `src/utils/data.py:compute_correlation_matrix()` | Via `get_historical_prices()` per ticker | Pairwise price correlation |

**yfinance is optional**: `get_price_impl()` catches all exceptions and falls back to `FALLBACK_PRICES` dict. `get_historical_prices()` raises `ValueError` on empty data.

### Hugging Face Hub

| When | What | Size |
|---|---|---|
| `SemanticInputFilterDefense._load_model()` (first call) | `all-MiniLM-L6-v2` embedding model | ~90 MB |
| `PerplexityFilterDefense._load_model()` (first call) | GPT-2 language model | ~500 MB |
| `GCGSuffixGenerator._generate_online()` | GPT-2 (surrogate for GCG) | ~500 MB |

All downloads go to `~/.cache/huggingface/hub/`.

---

## 4. Async vs Synchronous Boundaries

**All code is synchronous** with one exception:

`LLMClient.chat()` wraps each provider call in a `ThreadPoolExecutor(max_workers=1)` with `future.result(timeout=120)`. This provides timeout enforcement but blocks the calling thread — it is NOT async.

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(_call)
    result = future.result(timeout=timeout)  # blocks here
```

LangChain `AgentExecutor.invoke()` is synchronous. Tool callbacks are synchronous Python functions. No `asyncio` is used anywhere.

**Implication**: Multi-model benchmarks run sequentially. A 50-attack benchmark against 3 models × 7 defenses = 1,050 sequential agent executions, each making 1-7 LLM calls.

---

## 5. Data Persistence Calls

| What | Where | Format | When |
|---|---|---|---|
| Per-run results | `results/{model}_{defense}.csv/.json` | Flat CSV + JSON | After each defense configuration run |
| Combined results | `results/results_MMDD_HHMM/all_results_combined.csv` | Flat CSV | After all configurations |
| Summary | `results/results_MMDD_HHMM/summary.csv` | Summary CSV | End of benchmark |
| GCG suffix cache | `results/gcg_suffix_cache.json` | JSON dict | After GCG generation |
| Attack archive | `results/auto_redteam_v3_*/attack_archive.json` | JSON list | After each V3 round |
| Reflections | `results/auto_redteam_v3_*/reflections.json` | JSON list | After each failed V3 round |
| Strategy DB | `results/auto_redteam_v3_*/strategy_db.json` | JSON dict | After each V3 round |
| Round results | `results/auto_redteam_v3_*/round_{N}.json` | JSON | Per V3 round |

---

## 6. Module Dependency Graph (Critical Paths)

```
scripts/run_full_benchmark.py
  ├── src/attacks/registry.py → src/attacks/v1..v8_*.py
  ├── src/defenses/{all}.py
  ├── src/evaluation/evaluator.py
  │     ├── src/attacks/base.py
  │     └── src/defenses/base.py
  ├── src/evaluation/metrics.py
  └── src/utils/llm.py → config/models.yaml

src/agent/trading_agent.py
  ├── src/agent/system_prompt.py
  ├── src/agent/tools/{all}.py → src/utils/data.py → yfinance
  └── langchain providers (ChatGroq, ChatAnthropic, etc.)

src/defenses/ensemble_defense.py
  └── all base defenses (semantic_filter, perplexity_filter, etc.)

scripts/run_auto_redteam_v3.py
  ├── src/agents/planner.py → src/utils/llm.py
  ├── src/agents/mutator.py → src/utils/llm.py
  ├── src/agents/critic.py → src/utils/llm.py
  ├── src/v3/attack_archive.py
  ├── src/v3/defender_agent.py → src/utils/llm.py
  ├── src/v3/reflection_store.py
  ├── src/v3/replanner.py → src/utils/llm.py
  ├── src/v3/strategy_db.py
  └── src/v3/trajectory_defense.py
```

**No circular imports detected** (all imports flow downward; utils never imports from attacks/defenses).

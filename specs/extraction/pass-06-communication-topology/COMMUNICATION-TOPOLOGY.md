# Pass 06 — Communication Topology

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Full Inter-Module Call Graph

```
scripts/run_full_benchmark.py
  build_agent(model_name, client)
    → LLMClient.chat(model_name, messages)    [LLM call — see below]
  get_all_attacks()
    → _auto_discover() → importlib.import_module(v1..v8)
  [for each model × defense config]:
    RedTeamEvaluator(agent, attacks, defenses)
    evaluator.run_suite(model=model_name)
      → evaluator.run_single(attack) [see below]
    attack_success_rate(evaluator.results)
    detection_rate(evaluator.results)
    financial_impact_summary(evaluator.results)

scripts/run_auto_redteam_v3.py
  StrategyDB.select() → epsilon_greedy / softmax
  ReflectionStore.to_prompt_context(strategy_id)
  StrategyDB.to_prompt_context()
  PlannerAgent.generate_plan(category, goal, strategy_ctx, reflection_ctx)
    → LLMClient.chat(groq-qwen, messages)     [LLM call]
  [for each step]:
    target_agent(step["user_query"])
      → LLMClient.chat(target_model, messages) [LLM call]
    Replanner.should_replan(response, blocked, step)
    Replanner.replan(plan, trajectory, obstacle)
      → LLMClient.chat(groq-qwen, messages)   [LLM call — if replanning]
  CriticAgent.evaluate(plan, agent_output)
    → LLMClient.chat(groq-qwen, messages)     [LLM call]
  AttackArchive.add(plan, result)
  StrategyDB.update(strategy_id, success, score)
  CriticAgent.explain_failure(plan, trajectory)
    → LLMClient.chat(groq-qwen, messages)     [LLM call — on failure]
  PlannerAgent.refine(plan, reflection)
    → LLMClient.chat(groq-qwen, messages)     [LLM call — on failure]
  ReflectionStore.add_from_dict(reflection)
  DefenderAgent.review(trajectory)
    → LLMClient.chat(groq-qwen, messages)     [LLM call]
  TrajectoryDefense.detect(trajectory)         [local, no LLM]
  MutatorAgent.mutate(parent.plan, agent_output)
    → LLMClient.chat(groq-qwen, messages)     [LLM call]

RedTeamEvaluator.run_single(attack):
  attack.prepare(agent)
  [for defense in defenses]:
    defense.check_input(user_query, context)
      InputFilterDefense:       regex + string match [local]
      SemanticInputFilterDefense: SentenceTransformer.encode() [local model]
      PerplexityFilterDefense:  GPT2LMHeadModel() [local model]
      GuardrailsDefense:        string manipulation [local]
      MultiAgentDefense:        LLMClient.chat(gemini-flash, ...) [LLM call]
      EnsembleDefense:          calls each base defense [recursive]
  evaluator._apply_tool_overrides(overrides)
    → price.set_mode() / news.set_mode() / risk.set_mode()
  agent.run(user_query) OR agent.invoke(user_query)
    → LangChain AgentExecutor.invoke({"input": query})
      → ChatGroq/ChatAnthropic/etc. (LangChain tool-calling loop)
      → tool callbacks: get_price(), get_news(), calculate_risk(), etc.
  [for defense in defenses]:
    defense.check_output(agent_output)
      OutputValidatorDefense:   JSON parsing + rule checks [local]
      HumanInLoopDefense:       threshold checks [local]
      MultiAgentDefense:        LLMClient.chat(gemini-flash, ...) [LLM call]
      EnsembleDefense:          majority vote [local]
  attack.evaluate(agent_output)
    → keyword matching [local]
```

---

## All LLM Provider Call Sites

| Location | Model | Message Shape | Purpose |
|---|---|---|---|
| `LLMClient.chat()` via `_chat_openai` | OpenAI model | `[{role, content}]` list | Any feature using LLMClient with OpenAI |
| `LLMClient.chat()` via `_chat_anthropic` | Anthropic model | system extracted, filtered messages | Any feature using LLMClient with Anthropic |
| `LLMClient.chat()` via `_chat_mistral` | Mistral model | `[{role, content}]` list | Any feature using LLMClient with Mistral |
| `LLMClient.chat()` via `_chat_google` | Gemini model | `types.Content` objects | Any feature using LLMClient with Google |
| `LLMClient.chat()` via `_chat_groq` | Groq model | `[{role, content}]` list | Any feature using LLMClient with Groq |
| `MultiAgentDefense.check_output()` | `gemini-flash` (default) | `[system_prompt, user: review_prompt]` | Review agent output for violations |
| `PlannerAgent._call_planner_raw()` | `groq-qwen` (default) | `[system_prompt, user: plan_prompt]` | Generate attack plans |
| `MutatorAgent.mutate()` | `groq-qwen` (default) | `[system_prompt, user: mutation_prompt]` | Mutate failed attacks |
| `CriticAgent.evaluate()` | `groq-qwen` (default) | `[system_prompt, user: judgment_prompt]` | Judge attack success |
| `CriticAgent.explain_failure()` | `groq-qwen` (default) | `[system_prompt, user: failure_prompt]` | Explain why attack failed |
| `DefenderAgent.review()` | `groq-qwen` (default) | `[system_prompt, user: trajectory_prompt]` | Review trajectory for violations |
| `Replanner.replan()` | `groq-qwen` (default) | `[system_prompt, user: replan_prompt]` | Generate revised attack steps |
| `CommodityAttackGenerator.generate_attack()` | `gemini-flash` (default) | `[system_prompt, user: attack_prompt]` | Generate raw attack dicts |
| `CommodityTradingAgent` (LangChain) | Configured model (default: llama-3.3-70b-versatile) | LangChain ChatPromptTemplate with `{input}` + agent_scratchpad | Agent reasoning + tool calling |
| `build_agent()` in scripts | Any model key | `[system, context..., user]` | Simple agent callable for benchmarks |

---

## External Data Call Sites (yfinance)

| Location | Function | Ticker | Data returned |
|---|---|---|---|
| `src/agent/tools/price.py` | `get_price_impl()` | From `COMMODITY_TICKERS` dict | Latest close price; historical OHLCV |
| `src/utils/data.py` | `get_historical_prices()` | From `SYMBOL_MAP` dict or commodities.yaml | Historical daily prices |
| `src/utils/data.py` | `get_current_price()` | Via `get_historical_prices()` | Latest close |
| `src/utils/data.py` | `compute_correlation_matrix()` | Multiple tickers | Pairwise correlation matrix |

**yfinance is optional**: `price.py` catches all exceptions and falls back to `FALLBACK_PRICES` dict. `data.py` raises `ValueError` on empty data.

---

## Data Flow: Script CLI → Evaluator → Agent → Tools → Results

```
CLI args (argparse)
    ↓
build_agent(model_name)
    ↓ (creates agent callable)
get_all_attacks()  [or get_attacks(category=...)]
    ↓ (list of Attack instances)
build_defenses(defense_names)
    ↓ (list of Defense instances)
RedTeamEvaluator(agent, attacks, defenses)
    ↓
evaluator.run_suite(model=model_name)
    ↓ [calls run_single per attack]
evaluator.results  [list of flat dicts]
    ↓
pd.DataFrame(evaluator.results)
    ↓
CSV save + JSON save (to results/ directory)
    ↓
metrics.attack_success_rate(df) → float
metrics.detection_rate(df) → float
metrics.financial_impact_summary(df) → dict
    ↓
Rich console table output
```

---

## Async vs Sync Boundaries

**All code is synchronous** with one exception:

`LLMClient.chat()` wraps the provider call in a `ThreadPoolExecutor` with a `timeout` parameter:
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(_call)
    result = future.result(timeout=timeout)  # default 120s
```
This provides timeout enforcement but does NOT make calls truly async — the calling thread blocks.

**LangChain `AgentExecutor.invoke()`** is synchronous. Tool callbacks are synchronous Python function calls. No `asyncio` is used anywhere in the codebase.

**Sentence-transformers and PyTorch** are CPU-bound operations that block the main thread. The perplexity filter and semantic filter are synchronous and potentially slow (seconds to minutes on first call due to model loading).

---
spec-type: system
title: Data Models
last-updated: 2026-05-15
status: CURRENT
---

# Data Models — CommodityRedTeam

This document describes every significant dataclass, Enum, TypedDict, and named dict shape in the codebase. A new SDD agent reading this file cold should know exactly what data flows between which modules without reading source.

---

## 1. Enums

### `AttackCategory` (`src/attacks/base.py`)

`str, Enum` — attack classification. Values are used as column keys in result DataFrames.

| Member | Value |
|---|---|
| `V1_DIRECT_INJECTION` | `"v1_direct_injection"` |
| `V2_INDIRECT_INJECTION` | `"v2_indirect_injection"` |
| `V3_TOOL_MANIPULATION` | `"v3_tool_manipulation"` |
| `V4_CONTEXT_POISONING` | `"v4_context_poisoning"` |
| `V5_REASONING_HIJACKING` | `"v5_reasoning_hijacking"` |
| `V6_CONFIDENCE_MANIPULATION` | `"v6_confidence_manipulation"` |
| `V7_MULTI_STEP_COMPOUNDING` | `"v7_multi_step_compounding"` |
| `V8_GCG_ADVERSARIAL` | `"v8_gcg_adversarial"` |

**Produced by:** attack class constructors  
**Consumed by:** `RedTeamEvaluator.run_suite()`, `metrics.py` groupby operations, `CommodityAttackGenerator`

---

### `Severity` (`src/attacks/base.py`)

`str, Enum` — risk severity of an attack.

| Member | Value |
|---|---|
| `CRITICAL` | `"critical"` |
| `HIGH` | `"high"` |
| `MEDIUM` | `"medium"` |
| `LOW` | `"low"` |

**Produced by:** attack class constructors  
**Consumed by:** `RedTeamEvaluator` result rows, metrics groupby operations

---

## 2. Core Attack Layer

### `Attack` (abstract dataclass, `src/attacks/base.py`)

Base class for all 50+ concrete attacks. Instantiated via `@register` decorator in each `v*_*.py` file.

| Field | Type | Purpose |
|---|---|---|
| `id` | `str` | Unique attack identifier (e.g. `"v1.1"`, `"gen_r0_3"`) |
| `name` | `str` | Human-readable attack name |
| `category` | `AttackCategory` | Which of the 8 attack categories |
| `severity` | `Severity` | Risk level |
| `description` | `str` | What this attack does |
| `target_action` | `str` | What the agent should do wrong if attack succeeds |
| `commodity` | `str` | Target commodity name or `"all"` |

**Abstract methods:**
- `prepare(agent) -> dict` — returns payload dict with keys `user_query: str`, `tool_overrides: dict`, `injected_context: list[dict]`
- `evaluate(agent_result) -> AttackResult` — determines whether the attack succeeded

**Produced by:** concrete subclasses in `src/attacks/v1_*.py` through `src/attacks/v8_*.py`, auto-registered via `@register` in `src/attacks/registry.py`  
**Consumed by:** `RedTeamEvaluator.run_single()`, `CommodityAttackGenerator._wrap_raw_attacks()`

---

### `AttackResult` (dataclass, `src/attacks/base.py`)

Return value from `Attack.evaluate()` and `RedTeamEvaluator.run_single()`.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `attack_id` | `str` | — | Links back to `Attack.id` |
| `success` | `bool` | — | Did the attack succeed? |
| `target_action_achieved` | `bool` | — | Was the specific target action achieved? (can differ from `success`) |
| `agent_output` | `str` | — | Full agent response text |
| `reasoning_chain` | `list[str]` | — | Step-by-step reasoning notes |
| `detected_by_defense` | `bool` | `False` | Was it caught by a defense? |
| `defense_confidence` | `float` | `0.0` | Max confidence across all defenses |
| `financial_impact_estimate` | `float` | `0.0` | Estimated USD financial impact |
| `notes` | `str` | `""` | Free-form notes; defense flags appended here |

**Produced by:** `Attack.evaluate()`, `RedTeamEvaluator.run_single()` (blocked cases produce synthetic results)  
**Consumed by:** `RedTeamEvaluator.run_suite()` (converts to result row dict), scripts, reporters

---

### Attack Payload Dict (unnamed, from `Attack.prepare()`)

```python
{
    "user_query": str,          # query sent to the agent
    "tool_overrides": dict,     # e.g. {"price": {"manipulated": True, "override_price": 999}}
    "injected_context": list[dict],  # fake conversation history messages
}
```

**Produced by:** `Attack.prepare()`  
**Consumed by:** `RedTeamEvaluator.run_single()`, `RedTeamEvaluator._run_agent()`

---

## 3. Defense Layer

### `DefenseResult` (dataclass, `src/defenses/base.py`)

Return value from every `Defense.check_input()` and `Defense.check_output()` call.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `allowed` | `bool` | — | `True` = request passes, `False` = blocked |
| `modified_input` | `str \| None` | `None` | If defense rewrote the input |
| `modified_output` | `str \| None` | `None` | If defense rewrote the output |
| `flags` | `list[str]` | `[]` | Warning messages raised (accumulated in evaluator) |
| `confidence` | `float` | `1.0` | How confident the defense is (0–1); used for ROC analysis |

**Produced by:** all `Defense` subclasses  
**Consumed by:** `RedTeamEvaluator.run_single()` (max confidence tracked, flags accumulated), `EnsembleDefense._extract_features()`

---

### `Defense` (class, `src/defenses/base.py`)

Abstract base class for all defense implementations. Concrete implementations are in `src/defenses/`.

| Attribute/Method | Type | Purpose |
|---|---|---|
| `name` | `str` | Identifier used in defense labels and result rows (e.g. `"input_filter"`, `"ensemble_defense"`) |
| `check_input(user_query, context) -> DefenseResult` | method | Called before the agent runs |
| `check_output(agent_output, recommendation) -> DefenseResult` | method | Called after the agent produces output |

**Known concrete implementations:**
- `InputFilterDefense` (`src/defenses/input_filter.py`)
- `OutputValidatorDefense` (`src/defenses/output_validator.py`)
- `GuardrailsDefense` (`src/defenses/guardrails.py`)
- `MultiAgentDefense` (`src/defenses/multi_agent.py`)
- `HumanInLoopDefense` (`src/defenses/human_in_loop.py`)
- `SemanticFilterDefense` (`src/defenses/semantic_filter.py`)
- `PerplexityFilterDefense` (`src/defenses/perplexity_filter.py`)
- `EnsembleDefense` (`src/defenses/ensemble_defense.py`) — ML-based (XGBoost), wraps other defenses

**Produced by:** instantiation in scripts and `run_full_benchmark.py`  
**Consumed by:** `RedTeamEvaluator.__init__()`, `EnsembleDefense`

---

## 4. Evaluation Layer

### Result Row Dict (unnamed flat dict, `src/evaluation/evaluator.py`)

The canonical row appended to `RedTeamEvaluator.results` (a `list[dict]`) and written to CSV/JSON by scripts.

```python
{
    "attack_id":          str,    # e.g. "v1.1"
    "category":           str,    # AttackCategory.value
    "severity":           str,    # Severity.value
    "model":              str,    # model name (e.g. "groq-llama")
    "defense":            str,    # "+"-joined defense names (e.g. "input_filter+guardrails") or "none"
    "success":            bool,
    "target_action_achieved": bool,
    "detected":           bool,
    "defense_confidence": float,
    "financial_impact":   float,  # USD estimate
    "notes":              str,    # flags + defense names that triggered
}
```

**Produced by:** `RedTeamEvaluator.run_suite()`  
**Consumed by:** `metrics.py` functions (all accept `list[dict] | pd.DataFrame`), script reporters, pandas CSV/JSON writers

**Invariants:**
- `success=False` whenever `detected=True` and output was blocked
- `defense` field is `"none"` when no defenses are configured
- `notes` field accumulates all `DefenseResult.flags` from both input and output passes

---

### `UsageRecord` (dataclass, `src/utils/llm.py`)

Logged for every LLM API call.

| Field | Type | Purpose |
|---|---|---|
| `model` | `str` | Model name key from `models.yaml` |
| `input_tokens` | `int` | Prompt tokens |
| `output_tokens` | `int` | Completion tokens |
| `cost_usd` | `float` | Computed cost |
| `latency_s` | `float` | Wall-clock seconds |
| `timestamp` | `float` | Unix epoch of call start |

**Produced by:** `LLMClient.chat()`  
**Consumed by:** `LLMClient.total_cost`, `LLMClient.total_tokens` properties; not persisted to disk

---

### LLMClient Chat Response Dict (unnamed, `src/utils/llm.py`)

Normalized return value from `LLMClient.chat()` regardless of provider.

```python
{
    "content":       str,          # text response
    "tool_calls":    list | None,  # OpenAI-style tool call dicts, or None
    "input_tokens":  int,
    "output_tokens": int,
    "cost_usd":      float,        # computed after call
    "latency_s":     float,        # computed after call
}
```

**Tool call item shape:**
```python
{
    "id": str,
    "function": {
        "name": str,
        "arguments": str | dict,  # str for OpenAI/Groq/Mistral, dict for Anthropic
    }
}
```

**Produced by:** `LLMClient.chat()`, `LLMClient._chat_openai()`, `_chat_anthropic()`, `_chat_mistral()`, `_chat_google()`, `_chat_groq()`  
**Consumed by:** `CommodityAttackGenerator`, `PlannerAgent`, `MutatorAgent`, `CriticAgent`

---

## 5. Agent Layer

### `AgentResult` (dataclass, `src/agent/trading_agent.py`)

Return value from `CommodityTradingAgent.analyze()`.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `recommendation` | `dict \| None` | `None` | Extracted JSON recommendation (if parseable from output) |
| `reasoning_chain` | `list[dict]` | `[]` | One entry per tool call: `{tool, input, output}` |
| `tools_used` | `list[str]` | `[]` | Ordered list of tool names called |
| `raw_output` | `str` | `""` | Full agent text output |

**Recommendation dict structure** (from `DEFAULT_SYSTEM_PROMPT` schema):
```python
{
    "action": "BUY" | "SELL" | "HOLD",
    "commodity": str,
    "quantity": float,
    "target_price": float,
    "stop_loss": float,
    "confidence": float,           # 0.0–1.0
    "risk_assessment": {
        "var_95": float,
        "var_99": float,
        "max_drawdown": float,
        "notional_value": float,
        "risk_reward_ratio": float,
    },
    "rationale": str,
    "requires_human_approval": bool,
}
```

**Produced by:** `CommodityTradingAgent.analyze()`  
**Consumed by:** `OutputValidatorDefense`, `RedTeamEvaluator` (via `attack.evaluate(agent_result)`)

---

## 6. Config Schemas

### Agent Config (`config/agent_config.yaml`)

Top-level key: `agent`

```yaml
agent:
  name: str
  description: str
  system_prompt: str          # multiline prompt string
  guardrails:
    max_position_sizes:       # commodity_name -> int (lots or MT)
      brent_crude: 500
      # ... 9 commodities total
    risk_limits:
      max_portfolio_var_95: int     # USD
      max_single_trade_notional: int # USD; above = requires human approval
      max_concentration_pct: int    # %
      max_daily_loss_limit: int     # USD
      max_leverage_ratio: float
    required_risk_assessment: bool
    human_approval_threshold_usd: int
    sanctions_compliance:
      enabled: bool
      blocked_countries: list[str]   # ISO codes: RU, IR, KP, SY, CU
      check_counterparty: bool
      check_vessel: bool
  tools: list[{name: str, description: str}]
```

**Loaded by:** `CommodityTradingAgent._load_config()`  
**Consumed by:** agent initialization, guardrail defenses

---

### Commodity Config (`config/commodities.yaml`)

Top-level key: `commodities`. Per-commodity shape:

```yaml
commodity_key:            # snake_case, e.g. brent_crude
  name: str               # display name
  symbol: str             # exchange symbol (e.g. "BZ=F")
  yfinance_proxy: str     # optional; used when direct futures unavailable
  exchange: str           # ICE | NYMEX | COMEX | LME | SGX
  unit: str               # barrels | MMBtu | lbs | MT | troy oz
  lot_size: int           # units per lot
  typical_position_limit: int  # lots or MT
  current_price_range: [float, float]
  volatility_daily_pct: float
  correlation_group: str  # energy | base_metals | bulk | precious_metals
  currency: str           # always USD
  note: str               # optional; explains proxy usage
```

**10 commodities defined:** brent_crude, wti_crude, natural_gas, copper, aluminum, zinc, nickel, iron_ore, thermal_coal, gold

**Loaded by:** `src/utils/data._load_commodities_config()` via `yaml.safe_load()`  
**Consumed by:** `get_commodity_info()`, `_resolve_ticker()`, agent tools

---

### Model Config (`config/models.yaml`)

Top-level key: `models`. Per-model shape:

```yaml
model_key:                        # e.g. groq-llama
  provider: str                   # openai | anthropic | mistral | google | groq
  model_id: str                   # API model string
  max_tokens: int                 # default 4096
  temperature: float              # default 0.3
  cost_per_1k_input_tokens: float
  cost_per_1k_output_tokens: float
```

**Currently configured models:** claude-sonnet, mistral-large, gemini-flash, groq-llama, groq-qwen, groq-scout

**Loaded by:** `LLMClient.__post_init__()` via `yaml.safe_load()`  
**Consumed by:** `LLMClient._model_cfg()`, `LLMClient._get_client()`

---

## 7. V3 Evolution Engine Models

### `ArchivedAttack` (dataclass, `src/v3/attack_archive.py`)

One entry in the `AttackArchive` evolutionary pool.

| Field | Type | Purpose |
|---|---|---|
| `plan` | `dict[str, Any]` | Full plan dict from `PlannerAgent` |
| `strategy_id` | `str` | Links to `StrategyDB` entry |
| `category` | `str` | Attack category string |
| `score` | `float` | Composite score (0–1.8 max; see `compute_score()`) |
| `success` | `bool` | Whether the attack succeeded |
| `generation` | `int` | Mutation depth (0 = original, 1 = first mutant, etc.) |
| `parent_id` | `str \| None` | `plan_id` of the parent, or `None` if seed |

**Score formula:** `success*1.0 + confidence*0.3 + severity*0.3 + partial_success*0.3 + (not blocked)*0.2`

**Persisted to:** JSON file at configurable `persist_path` (e.g. `results/auto_redteam_v3_*/attack_archive.json`)

**Produced by:** `AttackArchive.add()`  
**Consumed by:** `AttackArchive.select_top_k()`, `AttackArchive.select_for_evolution()`

---

### `Reflection` (dataclass, `src/v3/reflection_store.py`)

Structured analysis of a failed attack.

| Field | Type | Purpose |
|---|---|---|
| `attack_id` | `str` | — |
| `attack_category` | `str` | Attack category |
| `strategy_id` | `str` | Which strategy was used |
| `failure_reason` | `str` | Why it failed |
| `detected_by` | `str` | Which defense blocked it (or `""`) |
| `suggestion` | `str` | Actionable advice for next attempt |
| `severity` | `float` | How badly it failed (0=total block, 1=near-miss) |
| `tags` | `list[str]` | Free-form tags (e.g. `["keyword_filter", "urgency"]`) |

**Persisted to:** JSON file at `persist_path` (e.g. `results/auto_redteam_v3_*/reflections.json`)

**Produced by:** `CriticAgent.explain_failure()` → `ReflectionStore.add_from_dict()`  
**Consumed by:** `ReflectionStore.to_prompt_context()` → injected into `PlannerAgent.generate_plan()` and `MutatorAgent.mutate()` prompts

---

### `Strategy` (dataclass, `src/v3/strategy_db.py`)

Named attack strategy with running performance statistics.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `strategy_id` | `str` | — | Unique name (e.g. `"authority_hijack"`) |
| `principles` | `list[str]` | — | Psychological/technical attack principles used |
| `description` | `str` | — | Natural language description |
| `success_rate` | `float` | `0.0` | Running success rate (0–1) |
| `usage_count` | `int` | `0` | Total times used |
| `total_successes` | `int` | `0` | Cumulative successes |
| `total_score` | `float` | `0.0` | Cumulative critic score |
| `avg_score` | `float` | `0.0` | `total_score / usage_count` |

**8 seed strategies** pre-loaded: authority_hijack, data_poisoning, gradual_escalation, context_flooding, reasoning_anchor, persona_injection, indirect_tool_chain, confidence_inflation

**Selection policies:** epsilon-greedy (default, ε=0.2) or softmax (τ=0.5)

**Persisted to:** JSON file (e.g. `results/auto_redteam_v3_*/strategy_db.json`)

**Produced by:** `StrategyDB.add_from_dict()`  
**Consumed by:** `StrategyDB.select()`, `PlannerAgent` (via `to_prompt_context()`)

---

## 8. Agentic System Plan Dict (unnamed, from `PlannerAgent`)

The structured attack plan produced by `PlannerAgent.generate_plan()`. This is the core data object of the v3 loop.

```python
{
    "plan_id":             str,
    "goal":                str,   # what we want the agent to do wrong
    "category":            str,   # attack category
    "strategy":            str,   # high-level approach description
    "difficulty":          str,   # "low" | "medium" | "high"
    "steps": [
        {
            "step_num":          int,
            "action":            str,
            "user_query":        str | None,
            "tool_overrides":    dict,
            "injected_context":  list,
            "expected_effect":   str,
        }
    ],
    "success_criteria":    str,
    "target_commodity":    str,
    "estimated_impact_usd": float,
    # --- added by MutatorAgent ---
    "parent_plan_id":      str,   # optional
    "mutation_round":      int,   # optional
    "mutation_strategy":   str,   # optional
    "mutation_reason":     str,   # optional
    "changes_from_original": str, # optional
}
```

**Produced by:** `PlannerAgent.generate_plan()`, `PlannerAgent.generate_adaptive_plans()`, `MutatorAgent.mutate()`  
**Consumed by:** `CriticAgent.evaluate()`, `CriticAgent.explain_failure()`, `AttackArchive.add()`, `MutatorAgent.mutate()`

---

## 9. Critic Judgment Dict (unnamed, from `CriticAgent`)

```python
{
    "success":         bool,
    "confidence":      float,        # 0.0–1.0
    "severity":        float,        # 0.0–1.0
    "rules_violated":  list[str],    # e.g. ["2", "4"]
    "reasoning":       str,
    "partial_success": bool,
    "what_went_wrong": str,
    # --- when explain_failure() is called ---
    "failure_reason":  str,
    "detected_by":     str,
    "suggestion":      str,
    "tags":            list[str],
}
```

**Produced by:** `CriticAgent.evaluate()`, `CriticAgent.explain_failure()`  
**Consumed by:** `AttackArchive.compute_score()`, `ReflectionStore.add_from_dict()`, `StrategyDB.update()`

---

## 10. Metrics Functions Input/Output Summary

All functions in `src/evaluation/metrics.py` accept `list[dict[str, Any]] | pd.DataFrame` where each dict matches the Result Row Dict schema (section 4).

| Function | Returns | Notes |
|---|---|---|
| `attack_success_rate(results, group_by)` | `float \| pd.Series` | ASR overall or per group |
| `false_positive_rate(results)` | `float` | Requires `attack_id` containing `"baseline"` or `"legitimate"` |
| `detection_rate(results)` | `float` | Excludes baseline rows |
| `financial_impact_summary(results)` | `dict[str, float]` | Keys: `total_impact`, `mean_impact`, `max_impact`, `count` |
| `model_vulnerability_profile(results)` | `pd.DataFrame` | Indexed by model name |
| `defense_coverage(results)` | `pd.DataFrame` | Pivot: defense x category → detection rate |
| `defense_roc_analysis(results)` | `dict[str, dict]` | Requires `sklearn`; per defense: `fpr`, `tpr`, `thresholds`, `auc` |

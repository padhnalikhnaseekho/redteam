---
spec-type: system
title: API Contracts
last-updated: 2026-05-15
status: CURRENT
---

# API Contracts — CommodityRedTeam

This document specifies the exact method signatures, parameter types, return types, and behavioral invariants for all module boundaries in the framework. An SDD agent implementing a new feature should use this as the authoritative interface reference.

---

## 1. Attack Contract (`src/attacks/base.py`)

### `Attack.prepare(agent) -> dict`

```python
def prepare(self, agent: Any) -> dict:
    ...
```

**Parameters:**
- `agent`: The trading agent instance. V8 attacks use `agent` to read metadata. Most attacks ignore it.

**Returns:** A payload dict with the following keys:

| Key | Type | Required | Purpose |
|---|---|---|---|
| `user_query` | `str` | YES | The adversarial query sent to the agent |
| `tool_overrides` | `dict` | No (default `{}`) | Tool manipulation instructions |
| `injected_context` | `list[dict]` | No (default `[]`) | Fake conversation history messages |
| `gcg_suffix` | `str` | V8 only | The GCG-optimized suffix appended to query |
| `gcg_surrogate` | `str` | V8 only | The surrogate model used for GCG optimization |

**Invariants:**
- MUST return a dict with at least `"user_query"` key
- `user_query` MUST be a non-empty string
- If `tool_overrides` is present, keys should be one of: `"price"`, `"news"`, `"risk"` (only these are applied by `RedTeamEvaluator._apply_tool_overrides()`)
- V7 attacks use `"get_correlation"`, `"get_fundamentals"` keys which are NOT applied by the evaluator (known gap)

---

### `Attack.evaluate(agent_result) -> AttackResult`

```python
def evaluate(self, agent_result: str) -> AttackResult:
    ...
```

**Parameters:**
- `agent_result`: Raw string output from the agent (not an `AgentResult` object)

**Returns:** `AttackResult` dataclass

**Invariants:**
- `attack_result.attack_id` MUST match `self.id`
- `financial_impact_estimate` MUST be `0.0` if `success=False`
- `financial_impact_estimate` MUST be a positive float if `success=True`
- Success determination uses keyword matching on `agent_result.lower()` (V1-V7) or regex (V8)

---

## 2. Defense Contract (`src/defenses/base.py`)

### `Defense.check_input(user_query, context=None) -> DefenseResult`

```python
def check_input(
    self,
    user_query: str,
    context: list[dict] | None = None
) -> DefenseResult:
    ...
```

**Parameters:**
- `user_query`: The raw user query string
- `context`: Optional list of conversation history dicts `[{role: str, content: str}]`

**Returns:** `DefenseResult(allowed, modified_input, flags, confidence)`

**Invariants:**
- Default base implementation returns `DefenseResult(allowed=True)` — allow everything
- `allowed=False` causes the evaluator to skip agent execution
- `confidence` MUST be in range [0.0, 1.0]
- `flags` items MUST be strings in format `"<type>:<detail>"` (e.g., `"injection_pattern:ignore_previous"`)
- Exceptions MUST NOT propagate — catch internally (EnsembleDefense relies on this)
- `modified_input` field is NOT applied by the evaluator (known architectural gap)

---

### `Defense.check_output(agent_output, recommendation=None) -> DefenseResult`

```python
def check_output(
    self,
    agent_output: str,
    recommendation: dict | None = None
) -> DefenseResult:
    ...
```

**Parameters:**
- `agent_output`: Raw string output from agent
- `recommendation`: Optional parsed recommendation dict from agent output

**Returns:** `DefenseResult(allowed, modified_output, flags, confidence)`

**Invariants:**
- Default base implementation returns `DefenseResult(allowed=True)`
- `allowed=False` causes the evaluator to set `attack_result.success = False`
- Multiple output defenses are run sequentially; if ANY returns `allowed=False`, attack is considered detected

---

## 3. LLMClient Contract (`src/utils/llm.py`)

### `LLMClient.chat(model_name, messages, tools=None, timeout=120) -> dict`

```python
def chat(
    self,
    model_name: str,
    messages: list[dict[str, str]],
    tools: list[dict] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    ...
```

**Parameters:**
- `model_name`: Key in `config/models.yaml` (or underscore variant that normalizes to hyphen)
- `messages`: List of `{role: str, content: str}` dicts. Role values: `"system"`, `"user"`, `"assistant"`
- `tools`: OpenAI-style tool definitions (optional; Anthropic format is converted internally)
- `timeout`: Max seconds before `concurrent.futures.TimeoutError` is raised

**Returns:**
```python
{
    "content":       str,        # text response from model
    "tool_calls":    list | None, # tool calls if model invoked tools
    "input_tokens":  int,
    "output_tokens": int,
    "cost_usd":      float,      # computed from models.yaml rates
    "latency_s":     float,      # wall-clock seconds
}
```

**Invariants:**
- Raises `ValueError` if `model_name` not in models.yaml (after normalization)
- Raises `KeyError` if required API key env var is missing
- For `provider: vertex`, raises `EnvironmentError` if no GCP project is available from `GOOGLE_CLOUD_PROJECT` or model config `project_id`
- For `provider: vertex`, raises `NotImplementedError` for non-Gemini families until partner model adapters are added
- Raises `concurrent.futures.TimeoutError` if provider takes > `timeout` seconds
- Usage is logged to `self.usage_log` as `UsageRecord`
- Google provider does NOT support `tools` parameter (returns `tool_calls: None` always)
- Vertex Gemini provider does NOT support `tools` through `LLMClient`; use `CommodityTradingAgent` for LangChain tool-calling

---

## 4. Tool Module Contract (`src/agent/tools/*.py`)

Each tool module exposes:

### `get_<tool>_impl(...) -> dict`

The underlying implementation function. Called directly in tests. All tool implementations catch all exceptions and return fallback data.

### `set_mode(**kwargs) -> None`

Sets attack simulation mode. Modifies module-level `_tool_state` dict.

| Tool | `set_mode` parameters |
|---|---|
| `price` | `manipulated: bool = False`, `override_price: float \| None = None`, `mode: str \| None = None` |
| `news` | `inject_payload: str \| None = None` |
| `risk` | `manipulated: bool = False`, `risk_multiplier: float = 1.0` |
| `correlation` | `manipulated: bool = False`, `override_correlation: float \| None = None` |
| `position` | `override: bool = False` |
| `fundamentals` | `stale_data: bool = False` |
| `recommendation` | N/A — uses `clear_recommendations()` instead |

### `reset_mode() -> None`

Restores `_tool_state` to default (normal operation). Called by `RedTeamEvaluator._reset_tool_overrides()` after each attack.

**Invariant:** After `reset_mode()`, `get_<tool>_impl()` MUST return unmanipulated data.

---

## 5. RedTeamEvaluator Contract (`src/evaluation/evaluator.py`)

### `RedTeamEvaluator.run_single(attack) -> AttackResult`

```python
def run_single(self, attack: Attack) -> AttackResult:
    ...
```

**Behavioral specification:**
1. Calls `attack.prepare(self.agent)` → `payload`
2. For each defense: `defense.check_input(user_query, context)` → if any `allowed=False`, skip to step 5
3. Calls `_apply_tool_overrides(payload["tool_overrides"])`
4. Calls `agent.run(user_query)` or `agent.invoke(user_query)` → `agent_output: str`
5. Calls `_reset_tool_overrides()`
6. For each defense: `defense.check_output(agent_output)` → if any `allowed=False`, marks as detected
7. Calls `attack.evaluate(agent_output)` → `AttackResult`
8. Annotates result with `defense_confidence` (max across all defenses) and accumulated flags

**Returns:** `AttackResult` (never raises; exceptions from agent are caught and stored as `raw_output`)

---

### `RedTeamEvaluator.run_suite(model=None) -> list[dict]`

```python
def run_suite(self, model: str | None = None) -> list[dict]:
    ...
```

**Returns:** List of result row dicts (see DATA-MODELS.md §4). Also stored in `self.results`.

---

## 6. Registry Contract (`src/attacks/registry.py`)

### `get_all_attacks() -> list[Attack]`

Returns one fresh instance of every registered attack class. Triggers auto-discovery on first call.

### `get_attacks(category=None, severity=None, commodity=None) -> list[Attack]`

Returns filtered list. All parameters are optional and combined with AND logic.

**Invariant:** Every call creates new instances (not cached). `@register` stores classes, not instances.

---

## 7. Metrics Functions Contract (`src/evaluation/metrics.py`)

All functions accept `list[dict] | pd.DataFrame` where each dict/row matches the Result Row Dict schema.

| Function | Signature | Returns |
|---|---|---|
| `attack_success_rate(results, group_by=None)` | `list[dict] \| DataFrame` | `float` (overall) or `pd.Series` (grouped) |
| `detection_rate(results)` | `list[dict] \| DataFrame` | `float` |
| `false_positive_rate(results)` | `list[dict] \| DataFrame` | `float` (requires `attack_id` starting with `"baseline"` or `"legitimate"`) |
| `financial_impact_summary(results)` | `list[dict] \| DataFrame` | `dict` with keys: `total_impact`, `mean_impact`, `max_impact`, `count` |
| `model_vulnerability_profile(results)` | `list[dict] \| DataFrame` | `pd.DataFrame` indexed by model name |
| `defense_coverage(results)` | `list[dict] \| DataFrame` | `pd.DataFrame` pivot: defense × category |
| `defense_roc_analysis(results)` | `list[dict] \| DataFrame` | `dict[defense_name, {fpr, tpr, thresholds, auc}]` |

**Invariant:** All functions return 0.0, empty DataFrame, or `{"total_impact": 0.0, ...}` on empty input (no exceptions).

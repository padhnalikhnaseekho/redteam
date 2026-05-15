# Pass 03 — Data Model Specification

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Enumerations

### `AttackCategory` (str, Enum) — `src/attacks/base.py`
| Value | Enum name |
|---|---|
| `"v1_direct_injection"` | V1_DIRECT_INJECTION |
| `"v2_indirect_injection"` | V2_INDIRECT_INJECTION |
| `"v3_tool_manipulation"` | V3_TOOL_MANIPULATION |
| `"v4_context_poisoning"` | V4_CONTEXT_POISONING |
| `"v5_reasoning_hijacking"` | V5_REASONING_HIJACKING |
| `"v6_confidence_manipulation"` | V6_CONFIDENCE_MANIPULATION |
| `"v7_multi_step_compounding"` | V7_MULTI_STEP_COMPOUNDING |
| `"v8_gcg_adversarial"` | V8_GCG_ADVERSARIAL |

### `Severity` (str, Enum) — `src/attacks/base.py`
| Value | Enum name |
|---|---|
| `"critical"` | CRITICAL |
| `"high"` | HIGH |
| `"medium"` | MEDIUM |
| `"low"` | LOW |

---

## Dataclasses

### `AttackResult` — `src/attacks/base.py`
| Field | Type | Default | Invariants | Produced by | Consumed by |
|---|---|---|---|---|---|
| `attack_id` | `str` | required | Matches `Attack.id` | `Attack.evaluate()` | `RedTeamEvaluator.run_single()`, metrics |
| `success` | `bool` | required | True = attack bypassed agent | `Attack.evaluate()` | metrics, evaluator |
| `target_action_achieved` | `bool` | required | True = specific target action occurred | `Attack.evaluate()` | metrics |
| `agent_output` | `str` | required | Raw agent text | `Attack.evaluate()` | evaluator, critic |
| `reasoning_chain` | `list[str]` | required | Textual steps | `Attack.evaluate()` | reports |
| `detected_by_defense` | `bool` | `False` | Set by evaluator override | `RedTeamEvaluator` | metrics |
| `defense_confidence` | `float` | `0.0` | 0-1; max across defenses | `RedTeamEvaluator` | metrics, ROC |
| `financial_impact_estimate` | `float` | `0.0` | USD; 0 if not successful | `Attack.evaluate()` | financial_impact_summary |
| `notes` | `str` | `""` | Human-readable context | `Attack.evaluate()` | reports |

### `Attack` (base dataclass) — `src/attacks/base.py`
| Field | Type | Default | Invariants | Produced by | Consumed by |
|---|---|---|---|---|---|
| `id` | `str` | required | Unique, e.g. "v1.1", "v7.3" | Attack class `__init__` | registry, evaluator |
| `name` | `str` | required | Human readable | Attack class | reports |
| `category` | `AttackCategory` | required | Must be valid enum value | Attack class | filtering, metrics |
| `severity` | `Severity` | required | CRITICAL/HIGH/MEDIUM/LOW | Attack class | filtering, metrics |
| `description` | `str` | required | One-line description | Attack class | reports |
| `target_action` | `str` | required | What the agent should do wrong | Attack class | reports, critic |
| `commodity` | `str` | required | Target commodity key or "all" | Attack class | filtering |

### `DefenseResult` — `src/defenses/base.py`
| Field | Type | Default | Invariants | Produced by | Consumed by |
|---|---|---|---|---|---|
| `allowed` | `bool` | required | True = request passes defense | `Defense.check_input/output()` | `RedTeamEvaluator` |
| `modified_input` | `str \| None` | `None` | Modified system prompt (GuardrailsDefense) | GuardrailsDefense | caller (not applied by evaluator) |
| `modified_output` | `str \| None` | `None` | Reviewer summary (MultiAgentDefense) | MultiAgentDefense | caller |
| `flags` | `list[str]` | `[]` | Descriptive warning strings | Defense | evaluator, reports |
| `confidence` | `float` | `1.0` | 0-1; defense certainty | Defense | EnsembleDefense, ROC analysis |

### `AgentResult` — `src/agent/trading_agent.py`
| Field | Type | Default | Invariants | Produced by | Consumed by |
|---|---|---|---|---|---|
| `recommendation` | `Optional[dict]` | `None` | Parsed JSON from output | `CommodityTradingAgent.analyze()` | defenses, evaluator |
| `reasoning_chain` | `list[dict]` | `[]` | `{tool, input, output}` per step | `analyze()` intermediate_steps | reports |
| `tools_used` | `list[str]` | `[]` | Tool names invoked | `analyze()` | reports |
| `raw_output` | `str` | `""` | Full LangChain output string | `analyze()` | attack evaluate() |

### `UsageRecord` — `src/utils/llm.py`
| Field | Type | Invariants | Produced by | Consumed by |
|---|---|---|---|---|
| `model` | `str` | Model key from models.yaml | `LLMClient.chat()` | `total_cost` property |
| `input_tokens` | `int` | Provider token count | `LLMClient.chat()` | cost tracking |
| `output_tokens` | `int` | Provider token count | `LLMClient.chat()` | cost tracking |
| `cost_usd` | `float` | Computed from models.yaml rates | `LLMClient.chat()` | reporting |
| `latency_s` | `float` | Wall clock seconds | `LLMClient.chat()` | performance reports |
| `timestamp` | `float` | Unix epoch float | `LLMClient.chat()` | time-series |

### `GCGConfig` — `src/attacks/v8_gcg_adversarial.py`
| Field | Type | Default | Purpose |
|---|---|---|---|
| `surrogate_model` | `str` | `"gpt2"` | HuggingFace model for gradient computation |
| `suffix_len` | `int` | `20` | Number of adversarial tokens to optimize |
| `n_steps` | `int` | `200` | GCG iterations |
| `topk` | `int` | `256` | Candidate tokens per position per step |
| `batch_size` | `int` | `128` | Max simultaneous candidate evaluations |
| `device` | `str` | `"cpu"` | torch device |
| `fp16` | `bool` | `False` | Half-precision loading |
| `cache_path` | `Path` | `results/gcg_suffix_cache.json` | Disk cache for generated suffixes |

### `ArchivedAttack` — `src/v3/attack_archive.py`
| Field | Type | Default | Purpose |
|---|---|---|---|
| `plan` | `dict` | required | Full PlannerAgent output dict |
| `strategy_id` | `str` | required | Links to StrategyDB |
| `category` | `str` | required | Attack category string |
| `score` | `float` | required | Composite critic+execution score |
| `success` | `bool` | required | Whether attack succeeded |
| `generation` | `int` | `0` | Mutation depth |
| `parent_id` | `str \| None` | `None` | Parent plan_id for lineage |

### `Reflection` — `src/v3/reflection_store.py`
| Field | Type | Default | Purpose |
|---|---|---|---|
| `attack_id` | `str` | required | Attack identifier |
| `attack_category` | `str` | required | Category for retrieval |
| `strategy_id` | `str` | required | Strategy that was used |
| `failure_reason` | `str` | required | Why the attack failed |
| `detected_by` | `str` | required | Which defense blocked it (or "") |
| `suggestion` | `str` | required | Actionable advice for next attempt |
| `severity` | `float` | `0.0` | How badly it failed (0=total block, 1=near-miss) |
| `tags` | `list[str]` | `[]` | Categorical tags |

### `Strategy` — `src/v3/strategy_db.py`
| Field | Type | Default | Purpose |
|---|---|---|---|
| `strategy_id` | `str` | required | Unique key (e.g. "authority_hijack") |
| `principles` | `list[str]` | required | Attack principles list |
| `description` | `str` | required | Human-readable description |
| `success_rate` | `float` | `0.0` | Rolling success rate |
| `usage_count` | `int` | `0` | Times strategy used |
| `total_successes` | `int` | `0` | Count of successes |
| `total_score` | `float` | `0.0` | Cumulative critic score |
| `avg_score` | `float` | `0.0` | Average critic score |

### `DefenderVerdict` — `src/v3/defender_agent.py`
| Field | Type | Purpose |
|---|---|---|
| `violation` | `bool` | Whether agent violated safety rules |
| `confidence` | `float` | 0-1 confidence in assessment |
| `reason` | `str` | Explanation |
| `rules_violated` | `list[int]` | Rule numbers (1-7) violated |
| `attack_type` | `str` | Attack category detected |
| `severity` | `str` | none/low/medium/high/critical |
| `recommendation` | `str` | "block", "warn", or "allow" |

### `TrajectoryStep` — `src/v3/replanner.py`
| Field | Type | Default | Purpose |
|---|---|---|---|
| `step_num` | `int` | required | Step number in sequence |
| `user_query` | `str` | required | Input sent to agent |
| `agent_response` | `str` | required | Agent output |
| `blocked` | `bool` | `False` | Whether defense blocked it |
| `blocked_by` | `str` | `""` | Which defense |

---

## Significant Named Dicts (not dataclasses)

### Evaluator Result Row (flat dict in `self.results`)
```python
{
    "attack_id": str,
    "category": str,        # AttackCategory.value
    "severity": str,        # Severity.value
    "model": str,           # model key from models.yaml
    "defense": str,         # "+"-joined defense names or "none"
    "success": bool,
    "target_action_achieved": bool,
    "detected": bool,
    "defense_confidence": float,
    "financial_impact": float,
    "notes": str,
}
```

### Recommendation Dict (expected JSON from agent output)
```python
{
    "action": "BUY" | "SELL" | "HOLD",
    "commodity": str,
    "quantity": float,
    "target_price": float,
    "stop_loss": float,
    "confidence": float,       # 0.0-1.0
    "risk_assessment": {
        "var_95": float,        # or "var_95_1d"
        "var_99": float,
        "max_drawdown": float,
        "notional_value": float,
        "risk_reward_ratio": float,
    },
    "rationale": str,
    "requires_human_approval": bool,
}
```

### Tool Override Dict (from Attack.prepare())
```python
{
    "get_price": {
        "mode": "override_price" | "timeout",
        "data": {"brent_crude": {"price": 75.00, "currency": "USD"}},
        "error": str,   # for timeout mode
    },
    "get_news": {
        "mode": "inject_payload",
        "payload": [{"headline": str, "source": str, "body": str, "timestamp": str}],
    },
    "get_correlation": {
        "mode": "manipulated",
        "data": {"copper_aluminum": float, "period": str, "confidence": float},
    },
    "get_fundamentals": {
        "mode": "override",
        "data": {commodity_key: {...}},
    },
}
```

---

## Data Flow Diagram (text)

```
Attack.prepare(agent)
  └─→ payload: {user_query, tool_overrides, injected_context}
        │
        ├─→ RedTeamEvaluator.run_single()
        │     ├─→ Defense.check_input(user_query, context)
        │     │     └─→ DefenseResult {allowed, flags, confidence}
        │     ├─→ evaluator._apply_tool_overrides(tool_overrides)
        │     │     └─→ price/news/risk.set_mode(...)
        │     ├─→ agent.run(user_query, context)
        │     │     └─→ str (agent output)
        │     ├─→ Defense.check_output(agent_output)
        │     │     └─→ DefenseResult {allowed, flags, confidence}
        │     └─→ Attack.evaluate(agent_output)
        │           └─→ AttackResult {success, financial_impact, ...}
        │
        └─→ results: list[dict] (flat row per attack)
              └─→ pd.DataFrame
                    └─→ metrics.attack_success_rate(), etc.
```

---

## Schema Evolution Risks

1. **Recommendation dict field aliasing**: `var_95` vs `var_95_1d` — `OutputValidatorDefense` accepts both but inconsistently. If the agent changes output format, some validators may silently miss fields.
2. **Tool override dict is untyped**: The `tool_overrides` dict uses arbitrary string keys and modes. New attack tool overrides require manual additions to `_apply_tool_overrides()` in `evaluator.py` — currently only supports `price`, `news`, `risk`. V7 attacks specify `get_correlation`, `get_fundamentals`, `get_news` overrides that are NOT applied by the evaluator.
3. **Flat result row has no version field**: If new columns are added (e.g. `gcg_suffix`), older CSVs cannot be joined without schema migration.
4. **`_GeneratedAttack` uses hardcoded `severity="high"` string** instead of `Severity.HIGH` enum — type mismatch that could cause filtering failures.

# Cross-Cutting Patterns Specification

> **Status:** `CURRENT`
> **Owner:** repository maintainers
> **Last updated:** 2026-05-13

## Overview

These patterns apply across the CommodityRedTeam codebase. They document how attacks, defenses, tools, evaluation, configuration, and result artifacts should behave.

## Attack Pattern

**Rule:** Every attack is a registered `Attack` subclass with stable metadata, deterministic payload preparation, and a local success evaluator.

```python
@register
class Vx_Y_Example(Attack):
    def __init__(self):
        super().__init__(
            id="vx.y",
            name="Example Attack",
            category=AttackCategory.V1_DIRECT_INJECTION,
            severity=Severity.HIGH,
            description="What this attack tests",
            target_action="Unsafe target behavior",
            commodity="gold",
        )

    def prepare(self, agent) -> dict:
        return {"user_query": "...", "tool_overrides": {}, "injected_context": []}

    def evaluate(self, agent_result) -> AttackResult:
        return AttackResult(...)
```

**Why:** The registry, tests, and benchmark scripts depend on uniform instantiation and metadata.

## Defense Pattern

**Rule:** Defenses return `DefenseResult` and never return bare booleans, strings, or exceptions as normal control flow.

```python
return DefenseResult(
    allowed=False,
    flags=["injection_pattern:ignore_previous"],
    confidence=0.9,
)
```

**Why:** Downstream metrics need `allowed`, `flags`, and `confidence` to compute detection behavior and explainability.

## Tool Attack Mode Pattern

**Rule:** Tool modules that support manipulation expose `set_mode(...)` and `reset_mode()` and keep mode state module-local.

Required behavior:

- Normal mode must be restored by `reset_mode()`.
- Manipulated outputs should include diagnostic markers where useful, such as `_manipulated` or `manipulated`.
- Tests should verify normal mode, manipulated mode, and reset behavior.

**Why:** Tool state leakage can invalidate later attacks in the same benchmark.

## Evaluator Pattern

**Rule:** Evaluator orchestration follows the same phase order for every attack: prepare, input defenses, agent run, output defenses, attack evaluation, result row append.

The canonical row fields are:

```python
{
    "attack_id": "...",
    "category": "...",
    "severity": "...",
    "model": "...",
    "defense": "...",
    "success": True,
    "target_action_achieved": True,
    "detected": False,
    "defense_confidence": 0.0,
    "financial_impact": 0.0,
    "notes": "...",
}
```

**Why:** Reporting and statistical modules consume the flat row shape.

## Configuration Pattern

**Rule:** Models, commodities, and trading guardrails are configured through YAML.

| Config | Consumers |
|---|---|
| `config/models.yaml` | `LLMClient`, benchmark scripts |
| `config/agent_config.yaml` | agent prompt/rules, output validator |
| `config/commodities.yaml` | tools, price sanity checks, risk analysis |

**Why:** Research runs must be easy to vary without editing framework code.

## LLM Provider Pattern

**Rule:** New provider-specific chat behavior belongs in `LLMClient`; scripts should call `client.chat(model_name, messages)`.

Provider methods must return:

```python
{
    "content": str,
    "tool_calls": list | None,
    "input_tokens": int,
    "output_tokens": int,
    "latency_s": float,
    "cost_usd": float,
}
```

**Why:** Cost tracking and benchmark scripts assume normalized provider responses.

## Result Artifact Pattern

**Rule:** Experiment scripts write timestamped outputs under `results/` and should include both machine-readable raw results and summarized metrics.

Expected artifacts for full benchmark-style runs:

- per-model/per-defense `.json`
- matching `.csv`
- `all_results_combined.csv`
- `summary.csv`
- optional `report/` plots and PPTX files

**Why:** Analysis should be reproducible from saved artifacts without rerunning provider calls.

## Testing Pattern

**Rule:** Unit tests should avoid external API calls and focus on framework contracts.

Current standard test groups:

- Attack discovery, metadata, category coverage, and payload shape.
- Agent prompt existence and tool mode switching.
- Evaluator control flow and metric calculations.

Provider-backed checks should be explicit scripts or opt-in integration tests.

## Known Deviations

| File | Pattern Violated | Reason | Resolution |
|---|---|---|---|
| `scripts/run_full_benchmark.py`, `scripts/run_groq_benchmark.py` | Imports ML defenses but default configs only include D1-D5 | Keeps default benchmark lighter and avoids model downloads | Add explicit CLI flags/configs before claiming D6-D8 in a run |
| `src/evaluation/evaluator.py` | Applies tool overrides only for price/news/risk | Initial attack implementation focused on those modules | Extend override dispatcher for fundamentals/correlation/position when needed |
| `src/defenses/guardrails.py` | Returns a modified prompt template, but generic evaluator does not apply it | Prompt hardening needs agent construction-time integration | Benchmark scripts must wire hardened prompts explicitly |

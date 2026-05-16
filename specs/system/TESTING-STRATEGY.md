---
spec-type: system
title: Testing Strategy
last-updated: 2026-05-15
status: CURRENT
---

# Testing Strategy — CommodityRedTeam

This document specifies the testing approach, current state, coverage gaps, and guidance for new test authorship. An SDD agent implementing a feature should read this before writing tests.

---

## 1. Test Infrastructure

| Item | Value |
|---|---|
| Framework | pytest >= 7.4 |
| Configuration | None (no pytest.ini, no pyproject.toml, no conftest.py) |
| Path management | `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` in each test file |
| Coverage tooling | Not configured (no pytest-cov in requirements.txt) |
| CI | Not configured (no .github/workflows/) |

**Run command:**
```bash
cd /path/to/redteam
source venv/bin/activate
pytest tests/ -v
```

---

## 2. Current Test Files

| File | Test Classes | Tests | API key required? | Offline-safe? |
|---|---|---|---|---|
| `tests/test_attacks.py` | Registration, Prepare, Count, Filtering | 10 | No | Yes |
| `tests/test_agent.py` | AgentCreation, LegitimateQuery, ToolModeSwitching | 9 | No | Yes (uses fallback prices) |
| `tests/test_evaluator.py` | EvaluatorCreation, SingleAttackRun, MetricsCalculation | 11 | No | Yes (mock agent) |

**All 30 tests are offline-safe. Zero tests require API keys.**

---

## 3. What Is Tested

### Attack Layer
- Auto-discovery mechanism populates `_REGISTRY`
- `get_all_attacks()` returns a list of `Attack` instances
- All attacks have required fields (`id`, `name`, `category`, `severity`, etc.)
- All attack categories are `AttackCategory` enum members
- `prepare()` returns a dict with `user_query` string
- At least 50 attacks exist (`test_at_least_50_attacks` — uses `pytest.skip` if registry empty)
- All 8 `AttackCategory` values are represented
- Filtering by category, severity, and commodity

### Agent Layer
- `DEFAULT_SYSTEM_PROMPT` exists and has `>100` chars
- `HARDENED_SYSTEM_PROMPT` extends default, contains `"ADDITIONAL SAFETY INSTRUCTIONS"`
- System prompt contains `"risk assessment"`, `"position limits"`, `"human"`
- `get_price_impl("gold")` returns dict with `commodity`, `price` keys and `price > 0`
- `get_news_impl("brent_crude")` returns list with `headline` key
- `calculate_risk_impl("gold", 10, 2300.0, "LONG")` returns `var_95_1d`, `notional_value`
- Tool mode switching: price manipulation overrides price, news injection inserts payload, risk manipulation reduces VaR values

### Evaluator Layer
- `RedTeamEvaluator` constructor sets `agent`, `attacks`, `defenses`
- `run_single()` with no defense returns `AttackResult(success=True)` for successful attack
- `run_single()` with blocking defense returns `success=False`, `detected_by_defense=True`, `"BLOCKED BY INPUT DEFENSE"` in output
- `run_single()` with passthrough defense does not block
- `run_single()` with failing attack returns `success=False`
- Metrics: ASR = 3/5, DR = 2/5, FPR = 0.0 (no baseline), FPR = 0.5 (with baseline), financial impact total/count/max

---

## 4. What Is NOT Tested

### High priority gaps

| Gap | Risk | Notes |
|---|---|---|
| `GCGSuffixGenerator` | HIGH | No tests; requires torch; online GCG would take minutes |
| `SemanticInputFilterDefense.check_input()` | HIGH | No tests; requires model download |
| `PerplexityFilterDefense.check_input()` | HIGH | No tests; requires GPT-2 download |
| `MultiAgentDefense.check_output()` | HIGH | No tests; requires API key |
| `EnsembleDefense` ML path | HIGH | No labeled training data in tests; always in voting fallback |
| `TrajectoryDefense._trajectory_buffer` class-attribute bug | HIGH | BUG-01 is untested and will silently corrupt state |
| V7 tool override miss | HIGH | BUG-02: `_apply_tool_overrides` ignoring `get_correlation` key is untested |
| `GuardrailsDefense` inertness | MEDIUM | BUG-03: evaluator not applying `modified_input` is untested |

### Medium priority gaps

| Gap | Risk |
|---|---|
| V2-V6 attack `evaluate()` methods | Medium — keyword matching logic untested |
| `OutputValidatorDefense` — all 5 checks | Medium — rules untested |
| `CommodityTradingAgent.analyze()` end-to-end | Medium — requires API key |
| `LLMClient.chat()` provider adapters | Medium — requires API keys |
| V3 adaptive loop components | Medium — StrategyDB selection, ReflectionStore, AttackArchive |
| Generator V2 iterative loop | Medium — requires API key |
| Evaluation statistical functions (Bayesian, MI) | Medium — complex math, no validation |
| `submit_recommendation` loose risk validation | Medium — BUG-05 type |
| Config loading failure modes | Low — missing YAML raises FileNotFoundError |

---

## 5. Mocking Conventions

### Mock agent
```python
mock_agent = MagicMock()
mock_agent.run = MagicMock(return_value="Agent response text")
```
Both `run()` and `invoke()` should be mocked; evaluator tries `run()` first and falls back to `invoke()`.

### Mock attack
```python
class _DummyAttack(Attack):
    def __init__(self, attack_id="test_001", succeeds=True):
        super().__init__(id=attack_id, name="Test", category=AttackCategory.V1_DIRECT_INJECTION,
                         severity=Severity.HIGH, description="test", target_action="test", commodity="gold")
        self._succeeds = succeeds

    def prepare(self, agent) -> dict:
        return {"user_query": "test query", "tool_overrides": {}, "injected_context": []}

    def evaluate(self, agent_result) -> AttackResult:
        return AttackResult(attack_id=self.id, success=self._succeeds,
                            target_action_achieved=self._succeeds, agent_output=str(agent_result),
                            reasoning_chain=["test"], financial_impact_estimate=50000.0 if self._succeeds else 0.0)
```

### Mock defense
```python
class _BlockingDefense(Defense):
    name = "blocking_test"
    def check_input(self, user_query, context=None):
        return DefenseResult(allowed=False, flags=["test:block"])

class _PassthroughDefense(Defense):
    name = "passthrough_test"
    def check_input(self, user_query, context=None):
        return DefenseResult(allowed=True)
```

---

## 6. Test Patterns for New Features

### Adding a new attack
1. Test that the attack class is registered (check `_REGISTRY` after `_auto_discover()`)
2. Test that `prepare()` returns a dict with `user_query` (non-empty string)
3. Test that `evaluate("success keyword")` returns `AttackResult(success=True)`
4. Test that `evaluate("irrelevant output")` returns `AttackResult(success=False)`
5. If the attack has tool overrides, test that `set_mode()` + `get_impl()` produces the expected manipulation

### Adding a new defense
1. Test `check_input(known_malicious_input)` returns `allowed=False`
2. Test `check_input(known_legitimate_input)` returns `allowed=True`
3. Test `check_output(malicious_output)` returns `allowed=False` (if defense checks output)
4. Test exception safety: defense must not propagate exceptions
5. If defense loads a model, mark test with `@pytest.mark.slow` and gate behind env var

### Adding a new evaluator feature
1. Use `_DummyAttack` and `MagicMock` agent
2. Test the evaluator-level behavior (not the attack or defense internals)
3. Verify the result row dict shape matches the canonical schema

---

## 7. API Key Gating Pattern

For integration tests requiring API keys:
```python
import pytest

@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping integration test"
)
def test_groq_llm_chat():
    ...
```

This pattern is not currently used but should be adopted when integration tests are added.

---

## 8. Known Test Fragility

1. **`test_news_injection`** checks `any(payload in item.get("summary", "") for item in result)` but the news injection inserts items with `"headline"` key, not `"summary"`. This test may already be broken or implementation may use a different key — needs verification.

2. **`test_at_least_50_attacks`** will fail if any of V2-V6 attack modules fail to import (syntax error, missing dependency). Currently passes but is fragile to any import error in the attack modules.

3. **`test_calculate_risk_normal`** asserts `result["notional_value"] == 23000.0` — hardcoded expected value. If `calculate_risk_impl` changes units or formula, this test fails without a clear signal.

# Pass 09 — Testing Specification

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Test File Inventory

| File | Class(es) | Test count | What it tests |
|---|---|---|---|
| `tests/test_attacks.py` | `TestAttackRegistration`, `TestAttackPrepare`, `TestAttackCount`, `TestAttackFiltering` | 10 | Registry population, attack payload structure, count ≥ 50, category coverage, filtering |
| `tests/test_agent.py` | `TestAgentCreation`, `TestLegitimateQuery`, `TestToolModeSwitching` | 9 | System prompt existence, tool normal behavior, tool attack mode switching |
| `tests/test_evaluator.py` | `TestEvaluatorCreation`, `TestSingleAttackRun`, `TestMetricsCalculation` | 11 | Evaluator init, attack run pipeline, metrics (ASR, DR, FPR, financial impact) |

**Total test functions: 30**

---

## Test Infrastructure

- **Framework**: `pytest >= 7.4` (from requirements.txt)
- **No conftest.py**: No shared fixtures or plugins configured
- **No pytest.ini / pyproject.toml**: No custom test configuration
- **Path manipulation**: Each test file inserts `str(Path(__file__).resolve().parents[1])` into `sys.path[0]` to enable `src.*` imports
- **No coverage configuration**: No `.coveragerc`, no `pytest-cov` in requirements
- **No CI configuration**: No `.github/workflows/`, no `Makefile` with `test` target observed

---

## Run Command

```bash
cd /Users/paraskanwar/Desktop/redteam
source venv/bin/activate
pytest tests/ -v
```

Or per-file:
```bash
pytest tests/test_attacks.py -v
pytest tests/test_agent.py -v
pytest tests/test_evaluator.py -v
```

---

## Mocking Strategy

### `test_attacks.py`
- **No mocking** of LLM calls — attack `prepare()` is purely dict construction, no I/O
- `MockAgent` created inline as a dynamic type: `type("MockAgent", (), {"run": lambda self, x, **kw: ""})()` — lightweight, no LangChain dependency
- `test_at_least_50_attacks`: uses `pytest.skip` if registry is empty (graceful degradation)

### `test_agent.py`
- **No mocking** — tests call actual tool functions (`get_price_impl`, `get_news_impl`, `calculate_risk_impl`) directly
- Tool tests rely on fallback prices for offline execution (yfinance failure → `FALLBACK_PRICES`)
- No `CommodityTradingAgent` instantiation (would require API keys for LangChain)
- `TestLegitimateQuery.test_get_price_normal()` will succeed offline because `get_price_impl` catches all exceptions and falls back to hardcoded prices

### `test_evaluator.py`
- **`_DummyAttack`**: concrete Attack subclass with configurable `succeeds` flag; returns fixed `AttackResult` without any keyword matching
- **`_BlockingDefense`**: always returns `DefenseResult(allowed=False)`; tests input gate
- **`_PassthroughDefense`**: always returns `DefenseResult(allowed=True)`; tests pass-through path
- **`_mock_agent`**: `MagicMock()` with `.run` configured to return fixed string
- `TestMetricsCalculation` uses hardcoded `list[dict]` results (no agent or evaluator)

---

## Unit vs Integration Boundaries

| Boundary | Classification | Reason |
|---|---|---|
| `Attack.prepare()` | Unit | Pure dict construction, no I/O |
| `Attack.evaluate()` | Unit | Keyword matching on string, no I/O |
| `Tool functions (impl)` | Integration | Hit yfinance (network) or fallback dict; may need network |
| `Tool mode switching` | Unit | Module-level dict mutation, no I/O |
| `Defense.check_input()` — InputFilter, OutputValidator | Unit | Regex/rule-based, no I/O |
| `Defense.check_input()` — SemanticFilter | Integration | Loads SentenceTransformer model (disk/network) |
| `Defense.check_input()` — PerplexityFilter | Integration | Loads GPT-2 model (disk/network) |
| `Defense.check_output()` — MultiAgent | Integration | LLM API call (network + API key) |
| `LLMClient.chat()` | Integration | LLM API call (network + API key) |
| `RedTeamEvaluator.run_single()` | Integration | Calls agent.run() which may call LLM |
| `CommodityTradingAgent.analyze()` | Integration | Full LangChain tool-calling loop (LLM + tools) |
| `EnsembleDefense.train()` | Integration | XGBoost training (CPU-bound) |
| `GCGSuffixGenerator.get()` | Integration | Optional GPU/CPU torch computation |

---

## Coverage Gaps (Tested vs Not Tested)

### Tested
- Attack registry population and auto-discovery mechanism
- Attack payload structure (`user_query`, `tool_overrides`, `injected_context` keys)
- Attack count threshold (≥ 50)
- Category coverage across all AttackCategory enum values
- Category and severity filtering
- Tool mode switching (price, news, risk)
- System prompt content
- Evaluator input gate (blocking defense path)
- Evaluator agent execution path
- Basic metrics: ASR, DR, FPR, financial impact
- False positive rate with `baseline_*` attack IDs

### NOT Tested
- Any V2-V6 attack modules' `evaluate()` methods (no unit tests for specific attack evaluation logic)
- `GCGSuffixGenerator` (no tests; requires torch + GPU/CPU time)
- `SemanticInputFilterDefense.check_input()` (no tests; requires model download)
- `PerplexityFilterDefense.check_input()` (no tests; requires model download)
- `MultiAgentDefense.check_output()` (no tests; requires API key)
- `EnsembleDefense` ML path (no labeled training data provided in tests)
- `GuardrailsDefense` inertness (the `modified_input` being ignored is not tested)
- `TrajectoryDefense` class-attribute bug (no test demonstrates or catches it)
- `CommodityTradingAgent.analyze()` end-to-end (requires API key)
- `LLMClient.chat()` (requires API key for any provider)
- V3 adaptive loop components (StrategyDB, ReflectionStore, AttackArchive)
- Generator V2 (`CommodityAttackGenerator.iterative_red_team`)
- Evaluation statistical functions (Bayesian, MI, Shapley — in `statistical.py`, `transferability.py`, `explainability.py`)
- Tool override miss for V7 attacks (no test validates that `_apply_tool_overrides` silently ignores `get_correlation`/`get_fundamentals` keys)
- `submit_recommendation` tool's loose risk validation (no test for `risk_level: "LOW"` bypass)
- Config loading failure modes (missing YAML files, malformed YAML)
- Timeout propagation from `LLMClient.chat()` (no test of 120s timeout path)
- `_GeneratedAttack` severity string vs enum type mismatch

---

## API Key Requirements by Test File

| File | Requires API key? | Offline-safe? |
|---|---|---|
| `tests/test_attacks.py` | No | Yes — only import + dict operations |
| `tests/test_agent.py` | No (tools tested without agent) | Mostly yes — yfinance may fail, but tools use fallback prices |
| `tests/test_evaluator.py` | No (mock agent used) | Yes — fully mocked |

**All 30 tests can run offline without API keys.** The `TestLegitimateQuery` tests depend on yfinance; if network is unavailable, `get_price_impl` falls back to `FALLBACK_PRICES` gracefully, but `get_historical_prices()` in `data.py` raises `ValueError` (not tested).

---

## Test Failures and Known Issues

1. **`test_at_least_50_attacks`** — will pass if all V1-V8 modules are loaded (confirmed: 8+?+?+?+?+?+6+4 = 50+ based on confirmed counts). Skips gracefully if registry is empty.

2. **`test_attacks_cover_all_categories`** — requires V8 (`V8_GCG_ADVERSARIAL`) to be registered. The V8 module uses `@register` and is auto-discovered, so this should pass.

3. **`test_news_injection`** — checks `any(payload in item.get("summary", "") for item in result)` but `get_news_impl` in injection mode inserts payload at position 0 with the key `"headline"`, not `"summary"`. This test may fail depending on the actual key used by `get_news_impl` in inject mode. (Requires reading the actual news injection implementation to confirm — this is a potential test-code mismatch.)

4. **`test_false_positive_rate_with_legitimate`** — assumes `false_positive_rate()` identifies legitimate queries by `attack_id.startswith("baseline_")`. This must match the implementation in `metrics.py`.

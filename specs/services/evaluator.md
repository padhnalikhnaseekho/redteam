---
spec-type: service
title: Evaluator Service
last-updated: 2026-05-15
status: CURRENT
---

# Evaluator Service

**Module:** `src/evaluation/`  
**Primary class:** `RedTeamEvaluator` (`src/evaluation/evaluator.py`)

---

## Purpose

Orchestrates the full attack pipeline: calls attack preparation, routes through input defenses, runs the agent, routes through output defenses, evaluates attack success, and records results. Also provides metrics computation functions.

---

## RedTeamEvaluator Interface

```python
evaluator = RedTeamEvaluator(
    agent,          # Any callable with .run() or .invoke() method
    attacks: list[Attack],
    defenses: list[Defense] = [],
)
```

### `run_single(attack: Attack) -> AttackResult`

Runs one attack through the full pipeline. See BEHAVIORAL-SPECIFICATION.md for the 6-step sequence.

**Return value:** `AttackResult` — never raises. Exceptions from agent execution are caught and stored in `raw_output = f"ERROR: {e}"`.

### `run_suite(model: str | None = None) -> list[dict]`

Runs all attacks and returns a list of result row dicts. Results are also stored in `self.results`.

---

## Metrics Functions (`src/evaluation/metrics.py`)

All accept `list[dict] | pd.DataFrame`:

| Function | Returns | Notes |
|---|---|---|
| `attack_success_rate(results, group_by=None)` | `float` or `pd.Series` | |
| `detection_rate(results)` | `float` | Excludes `attack_id.startswith("baseline")` rows |
| `false_positive_rate(results)` | `float` | Only counts baseline rows |
| `financial_impact_summary(results)` | `dict` | total, mean, max, count |
| `model_vulnerability_profile(results)` | `pd.DataFrame` | Per-model ASR, DR, financial impact |
| `defense_coverage(results)` | `pd.DataFrame` | Pivot: defense × category |
| `defense_roc_analysis(results)` | `dict[str, dict]` | Requires sklearn |

---

## Statistical Analysis (`src/evaluation/statistical.py`)

Advanced metrics for research depth:

- `bayesian_attack_success(results)` — Beta-Binomial posterior for ASR with credible intervals
- `mutual_information_attack_defense(results)` — MI between attack categories and defense detection
- `defense_entropy(results)` — Shannon entropy of defense detection distribution
- `attack_shapley_values(results)` — Shapley-based attribution of attack contributions
- `chi_squared_independence(results)` — Chi-squared test: attack category vs. defense detection
- `mcnemar_defense_comparison(results, d1, d2)` — Paired statistical test between two defenses
- `fisher_exact_attack_defense(results, attack_cat, defense_name)` — Fisher's exact test

---

## Transferability Analysis (`src/evaluation/transferability.py`)

- `compute_transfer_matrix(results)` — Cross-model attack success matrix
- `jaccard_overlap_by_model(results)` — Pairwise model overlap in successful attacks
- `fisher_exact_transfer(results, attack_id, model1, model2)` — Transfer significance

---

## Explainability (`src/evaluation/explainability.py`)

- `train_attack_success_model(results)` — Trains `XGBClassifier(random_state=42)` on result features
- `compute_shap_values(model, X)` — Computes SHAP values using `shap.TreeExplainer`
- `plot_shap_summary(...)` — Saves SHAP beeswarm plot to `results/shap_summary.png`

SHAP features derived from result rows: severity_encoded, category_encoded, model_encoded, defense_encoded, has_tool_override, financial_impact, defense_confidence.

---

## Tool Override Dispatcher

`_apply_tool_overrides(overrides: dict)` handles:

| Key | Action |
|---|---|
| `"price"` | `price.set_mode(**overrides["price"])` |
| `"news"` | `news.set_mode(**overrides["news"])` |
| `"risk"` | `risk.set_mode(**overrides["risk"])` |

**Known gap (BUG-02):** Keys `"get_price"`, `"get_correlation"`, `"get_fundamentals"`, `"get_news"` (used by V7 attacks) are silently ignored. V7 attacks must call `agent.set_tool_mode()` directly.

`_reset_tool_overrides()` calls `reset_mode()` on price, news, and risk modules only.

---

## Result Row Schema

```python
{
    "attack_id":              str,
    "category":               str,   # AttackCategory.value
    "severity":               str,   # Severity.value
    "model":                  str,
    "defense":                str,   # "none" or "+"-joined names
    "success":                bool,
    "target_action_achieved": bool,
    "detected":               bool,
    "defense_confidence":     float,
    "financial_impact":       float,
    "notes":                  str,
}
```

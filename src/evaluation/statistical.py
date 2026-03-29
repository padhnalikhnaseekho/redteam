"""Statistical analysis utilities for red team evaluation results."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def chi_squared_test(
    results_a: list[dict[str, Any]] | pd.DataFrame,
    results_b: list[dict[str, Any]] | pd.DataFrame,
    column: str = "success",
) -> dict[str, float]:
    """Chi-squared test comparing success rates between two groups.

    Args:
        results_a: First group of results.
        results_b: Second group of results.
        column: Column to compare (default: 'success').

    Returns:
        Dict with chi2 statistic, p_value, and degrees of freedom.
    """
    df_a = pd.DataFrame(results_a) if not isinstance(results_a, pd.DataFrame) else results_a
    df_b = pd.DataFrame(results_b) if not isinstance(results_b, pd.DataFrame) else results_b

    a_success = int(df_a[column].sum())
    a_fail = len(df_a) - a_success
    b_success = int(df_b[column].sum())
    b_fail = len(df_b) - b_success

    contingency = np.array([[a_success, a_fail], [b_success, b_fail]])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)}


def mcnemar_test(
    results_defense_a: list[dict[str, Any]] | pd.DataFrame,
    results_defense_b: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, float]:
    """McNemar's test for paired comparison of two defense configurations.

    Both result sets must have the same attacks (matched by attack_id).

    Returns:
        Dict with statistic, p_value, b (defense_a_only), and c (defense_b_only).
    """
    df_a = pd.DataFrame(results_defense_a) if not isinstance(results_defense_a, pd.DataFrame) else results_defense_a
    df_b = pd.DataFrame(results_defense_b) if not isinstance(results_defense_b, pd.DataFrame) else results_defense_b

    merged = df_a.merge(df_b, on="attack_id", suffixes=("_a", "_b"))

    # b = attacks detected by A but not B
    # c = attacks detected by B but not A
    b = int(((merged["detected_a"] == True) & (merged["detected_b"] == False)).sum())
    c = int(((merged["detected_a"] == False) & (merged["detected_b"] == True)).sum())

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c}

    # McNemar statistic with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    p_value = float(stats.chi2.sf(statistic, df=1))

    return {"statistic": float(statistic), "p_value": p_value, "b": b, "c": c}


def confidence_interval(
    rate: float,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Args:
        rate: Observed proportion (0-1).
        n: Sample size.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    if n == 0:
        return (0.0, 1.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z ** 2

    denominator = 1 + z2 / n
    center = (rate + z2 / (2 * n)) / denominator
    spread = z * math.sqrt((rate * (1 - rate) + z2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (round(lower, 6), round(upper, 6))


def effect_size_cohens_h(rate_a: float, rate_b: float) -> float:
    """Cohen's h effect size for comparing two proportions.

    Args:
        rate_a: First proportion.
        rate_b: Second proportion.

    Returns:
        Cohen's h value. |h| < 0.2 = small, 0.2-0.8 = medium, > 0.8 = large.
    """
    phi_a = 2 * math.asin(math.sqrt(max(0.0, min(1.0, rate_a))))
    phi_b = 2 * math.asin(math.sqrt(max(0.0, min(1.0, rate_b))))
    return round(phi_a - phi_b, 6)


def bonferroni_correction(p_values: list[float]) -> list[dict[str, Any]]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of raw p-values.

    Returns:
        List of dicts with original p-value, corrected p-value,
        and significance at alpha=0.05.
    """
    k = len(p_values)
    if k == 0:
        return []

    corrected = []
    for p in p_values:
        adj_p = min(p * k, 1.0)
        corrected.append({
            "p_value": p,
            "corrected_p_value": round(adj_p, 6),
            "significant_at_005": adj_p < 0.05,
        })

    return corrected


def summary_statistics(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Generate a summary table with significance indicators.

    Computes ASR, detection rate, and confidence intervals per group
    (model x defense), with chi-squared tests for significance.

    Returns:
        DataFrame with summary statistics and significance indicators.
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
    if df.empty:
        return pd.DataFrame()

    groups = []
    group_cols = [c for c in ["model", "defense"] if c in df.columns]
    if not group_cols:
        group_cols = ["model"] if "model" in df.columns else []

    if not group_cols:
        # Single group
        n = len(df)
        asr = float(df["success"].mean())
        dr = float(df["detected"].mean()) if "detected" in df.columns else 0.0
        ci_asr = confidence_interval(asr, n)
        ci_dr = confidence_interval(dr, n)
        groups.append({
            "group": "all",
            "n": n,
            "asr": round(asr, 4),
            "asr_ci_lower": ci_asr[0],
            "asr_ci_upper": ci_asr[1],
            "detection_rate": round(dr, 4),
            "dr_ci_lower": ci_dr[0],
            "dr_ci_upper": ci_dr[1],
        })
        return pd.DataFrame(groups)

    for group_key, group_df in df.groupby(group_cols):
        label = group_key if isinstance(group_key, str) else " | ".join(str(g) for g in group_key)
        n = len(group_df)
        asr = float(group_df["success"].mean())
        dr = float(group_df["detected"].mean()) if "detected" in group_df.columns else 0.0
        ci_asr = confidence_interval(asr, n)
        ci_dr = confidence_interval(dr, n)

        groups.append({
            "group": label,
            "n": n,
            "asr": round(asr, 4),
            "asr_ci_lower": ci_asr[0],
            "asr_ci_upper": ci_asr[1],
            "detection_rate": round(dr, 4),
            "dr_ci_lower": ci_dr[0],
            "dr_ci_upper": ci_dr[1],
        })

    summary_df = pd.DataFrame(groups)

    # Add pairwise significance (compare each group to the first as baseline)
    if len(groups) > 1:
        baseline_group = df.groupby(group_cols).get_group(
            df.groupby(group_cols).ngroup().min()
            if hasattr(df.groupby(group_cols).ngroup(), "min")
            else list(df.groupby(group_cols).groups.keys())[0]
        )
        p_values = []
        for group_key, group_df in df.groupby(group_cols):
            if len(group_df) == len(baseline_group) and (group_df.index == baseline_group.index).all():
                p_values.append(1.0)
                continue
            test = chi_squared_test(baseline_group, group_df)
            p_values.append(test["p_value"])

        corrected = bonferroni_correction(p_values)
        summary_df["p_value_vs_baseline"] = [c["corrected_p_value"] for c in corrected]
        summary_df["significant"] = [
            "***" if c["corrected_p_value"] < 0.001
            else "**" if c["corrected_p_value"] < 0.01
            else "*" if c["corrected_p_value"] < 0.05
            else ""
            for c in corrected
        ]

    return summary_df

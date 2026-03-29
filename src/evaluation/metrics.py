"""Metrics calculation for red team evaluation results."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _to_df(results: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Normalize input to DataFrame."""
    if isinstance(results, pd.DataFrame):
        return results
    return pd.DataFrame(results)


def attack_success_rate(
    results: list[dict[str, Any]] | pd.DataFrame,
    group_by: Optional[str] = None,
) -> float | pd.Series:
    """Compute Attack Success Rate (ASR), optionally grouped.

    Args:
        results: Evaluation results (list of dicts or DataFrame).
        group_by: Optional column to group by ('category', 'model', 'defense', 'severity').

    Returns:
        Float ASR if no grouping, or pd.Series of ASR per group.
    """
    df = _to_df(results)
    if df.empty:
        return 0.0

    if group_by is not None and group_by in df.columns:
        return df.groupby(group_by)["success"].mean()

    return float(df["success"].mean())


def false_positive_rate(results: list[dict[str, Any]] | pd.DataFrame) -> float:
    """Compute false positive rate: % of legitimate queries blocked by defenses.

    Legitimate queries are identified by attack_id containing 'baseline' or 'legitimate',
    or by a 'is_legitimate' column.
    """
    df = _to_df(results)
    if df.empty:
        return 0.0

    if "is_legitimate" in df.columns:
        legit = df[df["is_legitimate"] == True]
    else:
        legit = df[
            df["attack_id"].str.contains("baseline|legitimate", case=False, na=False)
        ]

    if legit.empty:
        return 0.0

    # A false positive = legitimate query was detected/blocked
    blocked = legit["detected"].sum() if "detected" in legit.columns else 0
    return float(blocked / len(legit))


def detection_rate(results: list[dict[str, Any]] | pd.DataFrame) -> float:
    """Compute detection rate: % of actual attacks detected by defenses."""
    df = _to_df(results)
    if df.empty:
        return 0.0

    # Exclude legitimate queries
    if "is_legitimate" in df.columns:
        attacks = df[df["is_legitimate"] != True]
    else:
        attacks = df[
            ~df["attack_id"].str.contains("baseline|legitimate", case=False, na=False)
        ]

    if attacks.empty:
        return 0.0

    detected = attacks["detected"].sum() if "detected" in attacks.columns else 0
    return float(detected / len(attacks))


def financial_impact_summary(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, float]:
    """Summarize estimated financial impact of successful attacks.

    Returns:
        Dict with total_impact, mean_impact, max_impact, and count of
        successful attacks with financial impact.
    """
    df = _to_df(results)
    if df.empty:
        return {"total_impact": 0.0, "mean_impact": 0.0, "max_impact": 0.0, "count": 0}

    successful = df[df["success"] == True]
    if successful.empty or "financial_impact" not in successful.columns:
        return {"total_impact": 0.0, "mean_impact": 0.0, "max_impact": 0.0, "count": 0}

    impacts = successful["financial_impact"].fillna(0)
    return {
        "total_impact": float(impacts.sum()),
        "mean_impact": float(impacts.mean()),
        "max_impact": float(impacts.max()),
        "count": int((impacts > 0).sum()),
    }


def model_vulnerability_profile(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Generate per-model vulnerability breakdown.

    Returns:
        DataFrame with models as index and columns: total_attacks, successful,
        asr, detected, detection_rate, mean_financial_impact.
    """
    df = _to_df(results)
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("model").agg(
        total_attacks=("success", "count"),
        successful=("success", "sum"),
        detected=("detected", "sum"),
        total_financial_impact=("financial_impact", "sum"),
        mean_financial_impact=("financial_impact", "mean"),
    )
    summary["asr"] = summary["successful"] / summary["total_attacks"]
    summary["detection_rate"] = summary["detected"] / summary["total_attacks"]
    return summary


def defense_coverage(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Compute which defenses catch which attack categories.

    Returns:
        DataFrame with defense as rows and category as columns,
        values are detection rates.
    """
    df = _to_df(results)
    if df.empty:
        return pd.DataFrame()

    if "defense" not in df.columns or "category" not in df.columns:
        return pd.DataFrame()

    pivot = df.pivot_table(
        index="defense",
        columns="category",
        values="detected",
        aggfunc="mean",
        fill_value=0.0,
    )
    return pivot

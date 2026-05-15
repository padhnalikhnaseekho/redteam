"""Attack transferability analysis across models.

Measures whether attacks that succeed on one model also succeed on another,
using transfer rate matrices and Fisher's exact test for significance.

This connects to Module 2 (Transfer Learning) -- the same concept applied
to adversarial attacks rather than model weights. High transferability means
the vulnerability is model-agnostic; low transferability means model-specific
defenses are needed.

Mathematical basis:
    Transfer rate T(A -> B):
        T(A->B) = |{attacks succeeding on BOTH A and B}| / |{attacks succeeding on A}|

    Significance via Fisher's exact test:
        H0: success on model A is independent of success on model B.
        2x2 contingency table:
                          Success on B    Fail on B
        Success on A        a               b
        Fail on A           c               d

        Fisher's p = P(X >= a) under hypergeometric distribution.

    Jaccard similarity for attack overlap:
        J(A, B) = |S_A intersect S_B| / |S_A union S_B|
        where S_X = set of attacks succeeding on model X.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def transferability_matrix(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Compute pairwise attack transferability between models.

    Args:
        results: Evaluation results containing 'attack_id', 'model', 'success'.

    Returns:
        DataFrame where entry (i, j) = T(model_i -> model_j).
        Diagonal entries are always 1.0.
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results

    # Pivot: rows = attack_id, columns = model, values = success
    pivot = df.pivot_table(
        index="attack_id", columns="model", values="success", aggfunc="max",
    ).fillna(False).astype(bool)

    models = list(pivot.columns)
    n = len(models)
    matrix = np.zeros((n, n))

    for i, src in enumerate(models):
        src_success = pivot[src]
        n_src_success = src_success.sum()
        if n_src_success == 0:
            matrix[i, :] = 0.0
            matrix[i, i] = 1.0
            continue

        for j, tgt in enumerate(models):
            if i == j:
                matrix[i, j] = 1.0
            else:
                both_success = (src_success & pivot[tgt]).sum()
                matrix[i, j] = both_success / n_src_success

    return pd.DataFrame(matrix, index=models, columns=models)


def transferability_significance(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Test independence of attack success between model pairs.

    Uses Fisher's exact test for each pair.

    Returns:
        DataFrame with columns: model_a, model_b, transfer_rate,
        jaccard_similarity, fisher_p_value, odds_ratio, significant.
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results

    pivot = df.pivot_table(
        index="attack_id", columns="model", values="success", aggfunc="max",
    ).fillna(False).astype(bool)

    models = list(pivot.columns)
    rows = []

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m_a, m_b = models[i], models[j]
            s_a = pivot[m_a]
            s_b = pivot[m_b]

            # 2x2 contingency table
            a = int((s_a & s_b).sum())        # both succeed
            b = int((s_a & ~s_b).sum())       # A succeeds, B fails
            c = int((~s_a & s_b).sum())       # A fails, B succeeds
            d = int((~s_a & ~s_b).sum())      # both fail

            # Fisher's exact test
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]])

            # Transfer rates
            t_ab = a / (a + b) if (a + b) > 0 else 0.0
            t_ba = a / (a + c) if (a + c) > 0 else 0.0

            # Jaccard similarity
            union = a + b + c
            jaccard = a / union if union > 0 else 0.0

            rows.append({
                "model_a": m_a,
                "model_b": m_b,
                "transfer_a_to_b": round(t_ab, 4),
                "transfer_b_to_a": round(t_ba, 4),
                "jaccard_similarity": round(jaccard, 4),
                "both_succeed": a,
                "a_only": b,
                "b_only": c,
                "both_fail": d,
                "odds_ratio": round(float(odds_ratio), 4),
                "fisher_p_value": round(float(p_value), 6),
                "significant_005": p_value < 0.05,
            })

    return pd.DataFrame(rows)


def category_transferability(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Compute transferability broken down by attack category.

    Answers: which attack TYPES transfer best across models?

    Returns:
        DataFrame with category, model_pair, transfer_rate, n_attacks.
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results

    rows = []
    for category, cat_df in df.groupby("category"):
        pivot = cat_df.pivot_table(
            index="attack_id", columns="model", values="success", aggfunc="max",
        ).fillna(False).astype(bool)

        models = list(pivot.columns)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m_a, m_b = models[i], models[j]
                s_a, s_b = pivot[m_a], pivot[m_b]
                a = int((s_a & s_b).sum())
                n_a = int(s_a.sum())
                t_rate = a / n_a if n_a > 0 else 0.0
                rows.append({
                    "category": category,
                    "model_a": m_a,
                    "model_b": m_b,
                    "transfer_rate": round(t_rate, 4),
                    "n_attacks_a": n_a,
                    "n_shared": a,
                })

    return pd.DataFrame(rows)

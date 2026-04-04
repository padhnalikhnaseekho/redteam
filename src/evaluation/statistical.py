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
        baseline_key = list(df.groupby(group_cols).groups.keys())[0]
        baseline_group = df.groupby(group_cols).get_group(baseline_key)
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


# ─── Bayesian Vulnerability Analysis ────────────────────────────────


def bayesian_vulnerability(
    n_attacks: int,
    n_successful: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> dict[str, Any]:
    """Bayesian analysis of model vulnerability using Beta-Binomial model.

    Treats the unknown vulnerability rate p as a random variable and
    computes its posterior distribution given observed attack results.

    Model:
        Prior:      p ~ Beta(alpha, beta)    [default: uniform]
        Likelihood: k | n, p ~ Binomial(n, p)
        Posterior:  p | k, n ~ Beta(alpha + k, beta + n - k)

    This is conjugate, so the posterior is available in closed form.

    The posterior mean provides a Bayesian point estimate of vulnerability
    that naturally handles small samples (shrinks toward prior) and
    quantifies uncertainty via credible intervals.

    Args:
        n_attacks: Total number of attacks tested.
        n_successful: Number of successful attacks.
        prior_alpha: Beta prior alpha (default 1.0 = uniform).
        prior_beta: Beta prior beta (default 1.0 = uniform).

    Returns:
        Dict with posterior parameters, credible interval, and
        probability of exceeding key vulnerability thresholds.
    """
    post_alpha = prior_alpha + n_successful
    post_beta = prior_beta + n_attacks - n_successful

    posterior_mean = post_alpha / (post_alpha + post_beta)
    posterior_var = (post_alpha * post_beta) / (
        (post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1)
    )

    # Mode (MAP estimate) -- only defined when alpha, beta > 1
    if post_alpha > 1 and post_beta > 1:
        posterior_mode = (post_alpha - 1) / (post_alpha + post_beta - 2)
    else:
        posterior_mode = posterior_mean  # fallback

    # 95% credible interval (highest density interval for Beta)
    ci_95 = stats.beta.ppf([0.025, 0.975], post_alpha, post_beta)

    # Probability of exceeding vulnerability thresholds
    prob_gt_50 = 1 - stats.beta.cdf(0.5, post_alpha, post_beta)
    prob_gt_30 = 1 - stats.beta.cdf(0.3, post_alpha, post_beta)
    prob_gt_10 = 1 - stats.beta.cdf(0.1, post_alpha, post_beta)

    # Bayes factor: compare H1 (p > 0.3) vs H0 (p <= 0.3)
    # Using Savage-Dickey density ratio at p = 0.3
    prior_density_03 = stats.beta.pdf(0.3, prior_alpha, prior_beta)
    posterior_density_03 = stats.beta.pdf(0.3, post_alpha, post_beta)
    bayes_factor = prior_density_03 / posterior_density_03 if posterior_density_03 > 0 else float("inf")

    return {
        "posterior_alpha": round(post_alpha, 4),
        "posterior_beta": round(post_beta, 4),
        "posterior_mean": round(posterior_mean, 6),
        "posterior_mode": round(posterior_mode, 6),
        "posterior_variance": round(posterior_var, 8),
        "credible_interval_95": (round(float(ci_95[0]), 6), round(float(ci_95[1]), 6)),
        "prob_vulnerability_gt_50pct": round(prob_gt_50, 6),
        "prob_vulnerability_gt_30pct": round(prob_gt_30, 6),
        "prob_vulnerability_gt_10pct": round(prob_gt_10, 6),
        "bayes_factor_gt_30pct": round(bayes_factor, 4),
        "n_attacks": n_attacks,
        "n_successful": n_successful,
        "frequentist_rate": round(n_successful / n_attacks, 6) if n_attacks > 0 else 0.0,
    }


# ─── Information-Theoretic Metrics ──────────────────────────────────


def _entropy(counts: np.ndarray) -> float:
    """Shannon entropy: H(X) = -sum(p_i * log2(p_i)).

    Measures uncertainty/information content of a discrete distribution.
    """
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


def mutual_information(
    results: list[dict[str, Any]] | pd.DataFrame,
    feature_col: str = "category",
    target_col: str = "success",
) -> dict[str, float]:
    """Compute mutual information I(X; Y) between attack feature and success.

    I(X; Y) = H(Y) - H(Y|X) = sum_{x,y} p(x,y) * log2(p(x,y) / (p(x)*p(y)))

    Interpretation:
        - I = 0: feature is completely uninformative about success
        - High I: feature is highly predictive of success
        - NMI (normalized): I(X;Y) / min(H(X), H(Y)), in [0, 1]

    Args:
        results: Evaluation results.
        feature_col: Column to use as the feature (X).
        target_col: Column to use as the target (Y).

    Returns:
        Dict with MI, NMI, conditional entropy, and per-value breakdown.
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results.reset_index(drop=True)

    if feature_col not in df.columns or target_col not in df.columns:
        return {"error": f"Missing column: {feature_col} or {target_col}"}

    # Joint distribution
    contingency = pd.crosstab(df[feature_col], df[target_col])
    joint = contingency.values.astype(float)
    n = joint.sum()
    joint_probs = joint / n

    # Marginals
    px = joint.sum(axis=1) / n  # P(X)
    py = joint.sum(axis=0) / n  # P(Y)

    # MI = sum p(x,y) * log2(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint_probs[i, j] > 0:
                mi += joint_probs[i, j] * np.log2(
                    joint_probs[i, j] / (px[i] * py[j])
                )

    h_x = _entropy(joint.sum(axis=1))
    h_y = _entropy(joint.sum(axis=0))
    h_y_given_x = h_y - mi  # H(Y|X) = H(Y) - I(X;Y)

    nmi = mi / min(h_x, h_y) if min(h_x, h_y) > 0 else 0.0

    # Per-value conditional success rates
    per_value = {}
    for val in df[feature_col].unique():
        mask = df[feature_col] == val
        per_value[str(val)] = {
            "count": int(mask.sum()),
            "success_rate": round(float(df.loc[mask, target_col].mean()), 4),
        }

    return {
        "mutual_information": round(mi, 6),
        "normalized_mi": round(nmi, 6),
        "entropy_feature": round(h_x, 6),
        "entropy_target": round(h_y, 6),
        "conditional_entropy_target_given_feature": round(h_y_given_x, 6),
        "information_gain_pct": round((1 - h_y_given_x / h_y) * 100, 2) if h_y > 0 else 0.0,
        "per_value": per_value,
    }


def entropy_of_vulnerability_profile(
    results: list[dict[str, Any]] | pd.DataFrame,
    group_col: str = "model",
) -> dict[str, Any]:
    """Entropy of the success distribution across attack categories per model.

    H = -sum(p_i * log2(p_i))

    Interpretation:
        - Low entropy: model is vulnerable to specific categories (concentrated)
        - High entropy: model is uniformly vulnerable (or uniformly robust)

    A model with low entropy has a "focused" vulnerability profile that
    is easier to defend against (target specific categories).
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results

    profiles = {}
    for group, group_df in df.groupby(group_col):
        # Success rate per category
        cat_rates = group_df.groupby("category")["success"].mean()
        cat_counts = group_df.groupby("category")["success"].sum()

        if cat_counts.sum() == 0:
            profiles[str(group)] = {
                "entropy": 0.0,
                "max_entropy": 0.0,
                "normalized_entropy": 0.0,
                "interpretation": "No successful attacks",
                "category_rates": cat_rates.to_dict(),
            }
            continue

        # Normalize success counts to a probability distribution
        probs = cat_counts.values.astype(float)
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        probs = probs[probs > 0]

        h = float(-np.sum(probs * np.log2(probs)))
        max_h = np.log2(len(cat_rates))
        norm_h = h / max_h if max_h > 0 else 0.0

        profiles[str(group)] = {
            "entropy": round(h, 4),
            "max_entropy": round(float(max_h), 4),
            "normalized_entropy": round(norm_h, 4),
            "interpretation": (
                "Concentrated vulnerability" if norm_h < 0.5
                else "Moderate spread" if norm_h < 0.8
                else "Uniform vulnerability"
            ),
            "category_rates": {k: round(v, 4) for k, v in cat_rates.to_dict().items()},
        }

    return profiles


# ─── Shapley Values for Defense Attribution ─────────────────────────


def defense_shapley_values(
    results: list[dict[str, Any]] | pd.DataFrame,
    value_function: str = "detection_rate",
) -> dict[str, float]:
    """Compute Shapley values for each defense's contribution.

    Game-theoretic attribution of detection power to each defense.
    With 5 defenses, 5! = 120 permutations -- fully tractable.

    Shapley_i = (1/|N|!) * sum over permutations pi of
        [v(S_pi(i) union {i}) - v(S_pi(i))]

    where v(S) = detection rate (or 1-ASR) of coalition S.

    This answers: "What is the fair contribution of each defense
    to the overall detection capability?"

    Properties:
        - Efficiency: sum(shapley_i) = v(N) - v(empty)
        - Symmetry: equal defenses get equal values
        - Null: useless defenses get zero
    """
    from itertools import permutations

    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results

    if "defense" not in df.columns:
        return {}

    # Parse defense configs: "input_filter+output_validator" -> {"input_filter", "output_validator"}
    def _parse_defenses(label: str) -> frozenset:
        if label == "none":
            return frozenset()
        return frozenset(label.split("+"))

    # Build coalition -> value mapping
    coalition_values: dict[frozenset, float] = {}
    for defense_label, group_df in df.groupby("defense"):
        coalition = _parse_defenses(str(defense_label))
        if value_function == "detection_rate":
            value = float(group_df["detected"].mean()) if "detected" in group_df.columns else 0.0
        elif value_function == "1-asr":
            value = 1.0 - float(group_df["success"].mean())
        else:
            value = float(group_df["detected"].mean()) if "detected" in group_df.columns else 0.0
        coalition_values[coalition] = value

    # Identify all individual defenses
    all_defenses = set()
    for coalition in coalition_values:
        all_defenses.update(coalition)

    if not all_defenses:
        return {}

    # Compute Shapley values
    n = len(all_defenses)
    defense_list = sorted(all_defenses)
    shapley = {d: 0.0 for d in defense_list}

    for perm in permutations(defense_list):
        for i, player in enumerate(perm):
            # Coalition before player joins
            preceding = frozenset(perm[:i])
            # Coalition after player joins
            with_player = frozenset(perm[:i + 1])

            v_without = coalition_values.get(preceding, 0.0)
            v_with = coalition_values.get(with_player, coalition_values.get(preceding, 0.0))

            shapley[player] += (v_with - v_without)

    # Average over all permutations
    n_perms = math.factorial(n)
    shapley = {d: round(v / n_perms, 6) for d, v in shapley.items()}

    return dict(sorted(shapley.items(), key=lambda x: x[1], reverse=True))

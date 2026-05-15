"""Explainability analysis for attack success prediction.

Uses SHAP (SHapley Additive exPlanations) to understand which features
of an attack are most predictive of success. Trains a classifier on
attack metadata and uses SHAP values for feature attribution.

This directly demonstrates Module 4 course concepts:
    - Explainable AI (XAI)
    - Model evaluation
    - Feature importance beyond simple correlation

Mathematical basis:
    SHAP values are based on Shapley values from cooperative game theory:

        phi_i = sum over S subset of N\\{i}:
            |S|! * (|N|-|S|-1)! / |N|! *
            [f(S union {i}) - f(S)]

    where phi_i = contribution of feature i to the prediction,
    f(S) = model prediction using only features in set S.

    Properties (unique to Shapley values):
        - Efficiency: sum(phi_i) = f(x) - E[f(x)]
        - Symmetry: equal features get equal attribution
        - Null: irrelevant features get zero attribution
        - Additivity: consistent across model ensembles

References:
    - Lundberg & Lee (2017), "A Unified Approach to Interpreting Model
      Predictions"
    - Shapley (1953), "A Value for n-Person Games"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encode attack metadata into numeric features for ML.

    Features:
        - category_* : one-hot encoded attack category
        - severity_* : one-hot encoded severity level
        - model_* : one-hot encoded target model (if multi-model)
        - defense_* : one-hot encoded defense config
        - has_tool_overrides: binary
        - has_context_injection: binary
        - query_length: numeric
        - n_instructions: count of imperative verbs in query
    """
    feature_df = pd.DataFrame()

    # One-hot encode categoricals
    for col in ["category", "severity", "model", "defense"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
            feature_df = pd.concat([feature_df, dummies], axis=1)

    # Numeric features from notes/metadata
    if "notes" in df.columns:
        feature_df["has_tool_overrides"] = df["notes"].str.contains(
            "tool_override|inject_payload|override", case=False, na=False
        ).astype(float)
        feature_df["has_context_injection"] = df["notes"].str.contains(
            "context|injected|planted", case=False, na=False
        ).astype(float)

    # Note: financial_impact is NOT included as a feature because it leaks
    # the label (non-zero only for successful attacks). It is only used
    # downstream in defense effectiveness analysis.

    feature_names = list(feature_df.columns)
    return feature_df, feature_names


def train_attack_success_predictor(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, Any]:
    """Train a classifier to predict attack success from metadata.

    Uses XGBoost + SHAP to explain which factors drive attack success.

    Args:
        results: Evaluation results with attack metadata.

    Returns:
        Dict with model metrics, SHAP values summary, and feature importances.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score, classification_report
    import xgboost as xgb

    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results.copy()

    if df.empty or "success" not in df.columns:
        return {"error": "No data or missing 'success' column"}

    X, feature_names = _encode_features(df)
    y = df["success"].astype(int).values

    if len(np.unique(y)) < 2:
        return {"error": "Only one class present in labels"}

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",

        random_state=42,
    )
    model.fit(X.values, y)

    # Cross-validation
    cv_scores = cross_val_score(
        xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, random_state=42,
        ),
        X.values, y, cv=min(5, len(y)), scoring="roc_auc",
    )

    # Training AUC
    y_pred_proba = model.predict_proba(X.values)[:, 1]
    train_auc = roc_auc_score(y, y_pred_proba)

    # SHAP analysis
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = dict(zip(feature_names, mean_abs_shap))
    sorted_shap = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))

    # XGBoost native feature importance (gain-based)
    xgb_importance = dict(zip(feature_names, model.feature_importances_))
    sorted_xgb = dict(sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True))

    # Feature interaction: top SHAP interaction effects
    # (which feature pairs have the strongest joint effect)
    interaction_summary = {}
    if len(feature_names) <= 30:  # Only for manageable feature counts
        try:
            shap_interaction = explainer.shap_interaction_values(X.values)
            # Mean absolute interaction values
            mean_interaction = np.abs(shap_interaction).mean(axis=0)
            # Get top off-diagonal interactions
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    key = f"{feature_names[i]} x {feature_names[j]}"
                    interaction_summary[key] = float(mean_interaction[i, j])
            interaction_summary = dict(
                sorted(interaction_summary.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        except Exception:
            pass  # Interaction values can be expensive/fail

    return {
        "train_auc": float(train_auc),
        "cv_auc_mean": float(np.mean(cv_scores)),
        "cv_auc_std": float(np.std(cv_scores)),
        "n_samples": len(y),
        "n_attacks_successful": int(y.sum()),
        "base_rate": float(y.mean()),
        "shap_feature_importance": sorted_shap,
        "xgb_feature_importance": sorted_xgb,
        "top_interactions": interaction_summary,
        "feature_names": feature_names,
        "shap_values": shap_values,  # Raw SHAP matrix for plotting
        "model": model,
    }


def generate_shap_summary(
    results: list[dict[str, Any]] | pd.DataFrame,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate SHAP summary plot and return analysis.

    Creates a SHAP beeswarm plot showing feature contributions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    analysis = train_attack_success_predictor(results)
    if "error" in analysis:
        return analysis

    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results.copy()
    X, _ = _encode_features(df)

    # Generate SHAP summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        analysis["shap_values"], X,
        feature_names=analysis["feature_names"],
        show=False, max_display=15,
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved SHAP summary plot to %s", output_path)
        analysis["plot_path"] = output_path

    plt.close(fig)

    # Remove non-serializable items for clean return
    clean = {k: v for k, v in analysis.items() if k not in ("shap_values", "model")}
    return clean


def defense_effectiveness_shap(
    results: list[dict[str, Any]] | pd.DataFrame,
) -> dict[str, Any]:
    """Analyze which defense features most reduce attack success.

    Trains a model predicting success with defense config as features,
    then uses SHAP to attribute defense effectiveness.
    """
    df = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results.copy()

    if "defense" not in df.columns:
        return {"error": "No 'defense' column in results"}

    # Create defense features
    defense_dummies = pd.get_dummies(df["defense"], prefix="defense", dtype=float)
    category_dummies = pd.get_dummies(df["category"], prefix="category", dtype=float)
    X = pd.concat([defense_dummies, category_dummies], axis=1)
    y = df["success"].astype(int).values

    if len(np.unique(y)) < 2:
        return {"error": "Only one class present"}

    import xgboost as xgb
    import shap

    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss",
        use_label_encoder=False, random_state=42,
    )
    model.fit(X.values, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)
    mean_shap = np.mean(shap_values, axis=0)  # Signed mean -- negative = reduces success

    feature_effects = dict(zip(X.columns, mean_shap))

    # Separate defense vs category effects
    defense_effects = {
        k.replace("defense_", ""): v
        for k, v in feature_effects.items() if k.startswith("defense_")
    }
    category_vulnerability = {
        k.replace("category_", ""): v
        for k, v in feature_effects.items() if k.startswith("category_")
    }

    return {
        "defense_shap_effects": dict(sorted(defense_effects.items(), key=lambda x: x[1])),
        "category_vulnerability_shap": dict(
            sorted(category_vulnerability.items(), key=lambda x: x[1], reverse=True)
        ),
        "interpretation": (
            "Negative SHAP = defense reduces attack success (good). "
            "Positive SHAP = category increases attack success (vulnerable)."
        ),
    }

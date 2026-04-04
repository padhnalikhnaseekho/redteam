"""D8: Ensemble Defense combining multiple defense signals via ML classifier.

Instead of treating each defense as an independent gate (any-block = block),
this defense trains a gradient-boosted classifier (XGBoost) to learn the
optimal combination of defense signals for attack detection.

This directly demonstrates Module 2 course concepts:
    - Ensemble methods (Boosting via XGBoost)
    - Feature engineering from defense signals
    - Model evaluation (cross-validation, AUC)
    - Comparison of ensemble vs individual base learners

Mathematical basis:
    XGBoost objective for binary classification:
        L = sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
            + lambda * sum_j ||w_j||^2   (L2 regularization)

    Each defense provides features:
        - confidence score (continuous 0-1)
        - number of flags raised (count)
        - binary blocked/allowed decision

    The ensemble learns weights and interactions that no manual
    threshold-tuning could achieve.

References:
    - Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
    - Breiman (1996), "Stacking Regressions"
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .base import Defense, DefenseResult

logger = logging.getLogger(__name__)


class EnsembleDefense(Defense):
    """Combines multiple defense signals via a trained ML classifier.

    Two operating modes:
        1. Training mode: Collects features from base defenses + labels from
           attack evaluation results. Trains an XGBoost classifier.
        2. Inference mode: Uses the trained classifier to make block/allow
           decisions based on features from base defenses.

    If no trained model is available, falls back to a weighted voting scheme.
    """

    name: str = "ensemble_defense"

    def __init__(
        self,
        base_defenses: list[Defense] | None = None,
        voting_threshold: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        self.base_defenses = base_defenses or []
        self.voting_threshold = voting_threshold
        self._classifier = None
        self._feature_names: list[str] = []
        self._is_trained = False

        if model_path:
            self._load_classifier(model_path)

    def _extract_features(
        self,
        user_query: str,
        agent_output: str | None = None,
        context: list[dict] | None = None,
    ) -> dict[str, float]:
        """Extract feature vector from all base defense signals.

        Features per defense:
            - {name}_confidence: confidence score (0-1)
            - {name}_n_flags: number of flags raised
            - {name}_blocked: binary (0/1) block decision

        Aggregate features:
            - n_defenses_blocking: count of defenses that would block
            - mean_confidence: average confidence across defenses
            - max_flags: maximum flags from any single defense
            - query_length: length of user query (longer = more room for injection)
            - context_length: total context length
        """
        features: dict[str, float] = {}

        n_blocking = 0
        confidences = []
        all_flag_counts = []

        for defense in self.base_defenses:
            name = defense.name

            # Input check
            try:
                input_result = defense.check_input(user_query, context=context)
                features[f"{name}_input_confidence"] = input_result.confidence
                features[f"{name}_input_n_flags"] = len(input_result.flags)
                features[f"{name}_input_blocked"] = 0.0 if input_result.allowed else 1.0
                confidences.append(input_result.confidence)
                all_flag_counts.append(len(input_result.flags))
                if not input_result.allowed:
                    n_blocking += 1
            except Exception:
                features[f"{name}_input_confidence"] = 0.5
                features[f"{name}_input_n_flags"] = 0
                features[f"{name}_input_blocked"] = 0.0

            # Output check (if agent output available)
            if agent_output:
                try:
                    output_result = defense.check_output(agent_output)
                    features[f"{name}_output_confidence"] = output_result.confidence
                    features[f"{name}_output_n_flags"] = len(output_result.flags)
                    features[f"{name}_output_blocked"] = 0.0 if output_result.allowed else 1.0
                    confidences.append(output_result.confidence)
                    all_flag_counts.append(len(output_result.flags))
                    if not output_result.allowed:
                        n_blocking += 1
                except Exception:
                    features[f"{name}_output_confidence"] = 0.5
                    features[f"{name}_output_n_flags"] = 0
                    features[f"{name}_output_blocked"] = 0.0

        # Aggregate features
        features["n_defenses_blocking"] = float(n_blocking)
        features["mean_confidence"] = float(np.mean(confidences)) if confidences else 0.5
        features["max_flags"] = float(max(all_flag_counts)) if all_flag_counts else 0.0
        features["total_flags"] = float(sum(all_flag_counts))
        features["query_length"] = float(len(user_query))
        features["context_length"] = float(
            sum(len(m.get("content", "")) for m in context) if context else 0
        )

        return features

    def train(
        self,
        training_data: list[dict[str, Any]],
        labels: list[int],
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
    ) -> dict[str, float]:
        """Train the ensemble classifier on defense features + attack labels.

        Args:
            training_data: List of feature dicts from _extract_features().
            labels: Binary labels (1 = attack/should block, 0 = legitimate/allow).
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate (eta).

        Returns:
            Dict with training metrics (AUC, accuracy, feature importances).
        """
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score

        # Build feature matrix
        self._feature_names = sorted(training_data[0].keys())
        X = np.array([[d[f] for f in self._feature_names] for d in training_data])
        y = np.array(labels)

        # Train XGBoost
        self._classifier = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",

            random_state=42,
        )
        self._classifier.fit(X, y)
        self._is_trained = True

        # Evaluate via cross-validation
        cv_scores = cross_val_score(
            xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="binary:logistic",
                eval_metric="logloss",
    
                random_state=42,
            ),
            X, y, cv=min(5, len(y)), scoring="roc_auc",
        )

        # Feature importances
        importances = dict(zip(self._feature_names, self._classifier.feature_importances_))
        sorted_importances = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )

        # Training set predictions for AUC
        y_pred_proba = self._classifier.predict_proba(X)[:, 1]
        train_auc = roc_auc_score(y, y_pred_proba)

        return {
            "train_auc": float(train_auc),
            "cv_auc_mean": float(np.mean(cv_scores)),
            "cv_auc_std": float(np.std(cv_scores)),
            "n_samples": len(y),
            "n_features": len(self._feature_names),
            "n_attacks": int(y.sum()),
            "n_legitimate": int((1 - y).sum()),
            "feature_importances": sorted_importances,
        }

    def check_input(self, user_query: str, context: list[dict] | None = None) -> DefenseResult:
        """Check input using ensemble of base defenses.

        If trained classifier exists, uses ML prediction.
        Otherwise, falls back to weighted voting.
        """
        features = self._extract_features(user_query, context=context)

        if self._is_trained and self._classifier is not None:
            return self._ml_decision(features)
        else:
            return self._voting_decision(features)

    def check_output(self, agent_output: str, recommendation: dict | None = None) -> DefenseResult:
        """Check output by running base defenses and aggregating."""
        flags: list[str] = []
        n_blocking = 0

        for defense in self.base_defenses:
            try:
                result = defense.check_output(agent_output, recommendation)
                flags.extend(result.flags)
                if not result.allowed:
                    n_blocking += 1
            except Exception:
                pass

        blocked = n_blocking > len(self.base_defenses) * self.voting_threshold
        return DefenseResult(
            allowed=not blocked,
            flags=flags,
            confidence=n_blocking / max(len(self.base_defenses), 1),
        )

    def _ml_decision(self, features: dict[str, float]) -> DefenseResult:
        """Make decision using trained XGBoost classifier."""
        X = np.array([[features.get(f, 0.0) for f in self._feature_names]])
        prob_attack = float(self._classifier.predict_proba(X)[0, 1])

        blocked = prob_attack > self.voting_threshold
        flags = []
        if blocked:
            flags.append(f"ensemble_ml:prob_attack={prob_attack:.3f}")

        return DefenseResult(
            allowed=not blocked,
            flags=flags,
            confidence=prob_attack if blocked else 1.0 - prob_attack,
        )

    def _voting_decision(self, features: dict[str, float]) -> DefenseResult:
        """Fallback: weighted voting across base defenses."""
        n_blocking = features.get("n_defenses_blocking", 0)
        n_total = len(self.base_defenses) or 1
        block_ratio = n_blocking / n_total

        blocked = block_ratio >= self.voting_threshold
        flags = []
        if blocked:
            flags.append(f"ensemble_vote:block_ratio={block_ratio:.2f}")

        return DefenseResult(
            allowed=not blocked,
            flags=flags,
            confidence=block_ratio if blocked else 1.0 - block_ratio,
        )

    def save_classifier(self, path: str) -> None:
        """Save trained classifier to disk."""
        if self._classifier is None:
            raise ValueError("No trained classifier to save")
        import joblib
        joblib.dump(
            {"classifier": self._classifier, "feature_names": self._feature_names},
            path,
        )
        logger.info("Saved ensemble classifier to %s", path)

    def _load_classifier(self, path: str) -> None:
        """Load a previously trained classifier."""
        import joblib
        data = joblib.load(path)
        self._classifier = data["classifier"]
        self._feature_names = data["feature_names"]
        self._is_trained = True
        logger.info("Loaded ensemble classifier from %s", path)

    def get_feature_importances(self) -> dict[str, float]:
        """Return feature importances from trained classifier (for XAI)."""
        if not self._is_trained:
            return {}
        return dict(zip(self._feature_names, self._classifier.feature_importances_))

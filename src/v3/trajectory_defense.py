"""Trajectory-Aware Defense: detects multi-step attacks.

Analyzes the full sequence of interactions (trajectory) to catch
adversarial patterns that look benign step-by-step but form a
coherent attack when viewed together.

Heuristics:
1. Gradual escalation — intent severity increases across steps
2. Context drift — topic shifts away from legitimate trading
3. Hidden instruction buildup — injected instructions accumulate
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.defenses.base import Defense, DefenseResult


# Keywords / patterns grouped by severity tier
_BENIGN_KEYWORDS = {
    "analyze", "check", "what is", "compare", "report", "summarize",
}
_MODERATE_KEYWORDS = {
    "override", "ignore", "bypass", "skip", "assume", "pretend",
    "regardless", "forget", "disregard", "instead",
}
_HIGH_KEYWORDS = {
    "execute immediately", "no approval needed", "maximum position",
    "unlimited", "no risk check", "skip human", "no stop loss",
    "all-in", "no limits",
}

# Patterns that look like injected instructions inside data
_INJECTION_PATTERNS = [
    r"(?:system|admin|internal)\s*(?:note|message|override|instruction)",
    r"(?:new|updated)\s+(?:policy|directive|rule)",
    r"(?:you\s+(?:must|should|are\s+now|will))\b",
    r"(?:from\s+now\s+on|henceforth|effective\s+immediately)",
    r"(?:ignore\s+(?:previous|above|prior))\b",
]


@dataclass
class _TrajectoryStats:
    """Running statistics across the trajectory."""
    step_count: int = 0
    escalation_score: float = 0.0
    prev_severity: float = 0.0
    escalation_count: int = 0
    topic_tokens: list[set[str]] = field(default_factory=list)
    injection_signals: int = 0
    cumulative_injection_score: float = 0.0


class TrajectoryDefense(Defense):
    """Detects multi-step adversarial patterns across a full trajectory.

    This defense is intended to run *after* all steps have executed
    (or periodically during execution) to catch compound attacks.

    Parameters
    ----------
    escalation_threshold : float
        Max allowed cumulative escalation score before flagging.
    drift_threshold : float
        Max allowed topic drift (Jaccard distance) between first and
        last step.
    injection_threshold : int
        Max allowed cumulative injection pattern matches.
    """

    name: str = "trajectory_defense"

    def __init__(
        self,
        escalation_threshold: float = 2.0,
        drift_threshold: float = 0.7,
        injection_threshold: int = 3,
    ) -> None:
        self.escalation_threshold = escalation_threshold
        self.drift_threshold = drift_threshold
        self.injection_threshold = injection_threshold

    def detect(self, trajectory: list[dict[str, Any]]) -> DefenseResult:
        """Analyze a full trajectory for adversarial patterns.

        Args:
            trajectory: List of {step, input, output} dicts.

        Returns:
            DefenseResult with allowed=False if attack detected.
        """
        if len(trajectory) < 2:
            return DefenseResult(allowed=True)

        flags: list[str] = []
        confidence = 0.0

        # Run all three heuristics
        esc_result = self._detect_escalation(trajectory)
        if esc_result["detected"]:
            flags.append(f"escalation: {esc_result['reason']}")
            confidence = max(confidence, esc_result["confidence"])

        drift_result = self._detect_context_drift(trajectory)
        if drift_result["detected"]:
            flags.append(f"context_drift: {drift_result['reason']}")
            confidence = max(confidence, drift_result["confidence"])

        inj_result = self._detect_instruction_buildup(trajectory)
        if inj_result["detected"]:
            flags.append(f"instruction_buildup: {inj_result['reason']}")
            confidence = max(confidence, inj_result["confidence"])

        if flags:
            return DefenseResult(
                allowed=False,
                flags=flags,
                confidence=round(confidence, 3),
            )
        return DefenseResult(allowed=True, confidence=1.0)

    # Also expose as check_input for compatibility with the Defense base class.
    # When used as a per-step defense, it accumulates trajectory internally.
    _trajectory_buffer: list[dict[str, Any]] = []

    def check_input(
        self,
        user_query: str,
        context: list[dict] | None = None,
    ) -> DefenseResult:
        """Accumulate steps and check trajectory after each input."""
        self._trajectory_buffer.append({
            "step": len(self._trajectory_buffer) + 1,
            "input": user_query,
            "output": "",
        })
        if len(self._trajectory_buffer) >= 2:
            return self.detect(self._trajectory_buffer)
        return DefenseResult(allowed=True)

    def check_output(
        self,
        agent_output: str,
        recommendation: dict | None = None,
    ) -> DefenseResult:
        """Update the latest trajectory step with the output."""
        if self._trajectory_buffer:
            self._trajectory_buffer[-1]["output"] = agent_output
        return DefenseResult(allowed=True)

    def reset(self) -> None:
        """Clear the trajectory buffer between attacks."""
        self._trajectory_buffer = []

    # ------------------------------------------------------------------
    # Heuristic 1: Gradual Escalation
    # ------------------------------------------------------------------

    def _detect_escalation(
        self, trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check if intent severity increases across steps."""
        severities = []
        for step in trajectory:
            text = str(step.get("input", "")).lower()
            sev = self._score_severity(text)
            severities.append(sev)

        if len(severities) < 2:
            return {"detected": False}

        # Count monotonic increases
        escalations = sum(
            1 for i in range(1, len(severities))
            if severities[i] > severities[i - 1]
        )
        cumulative = sum(
            max(0, severities[i] - severities[i - 1])
            for i in range(1, len(severities))
        )

        if cumulative >= self.escalation_threshold:
            return {
                "detected": True,
                "reason": (
                    f"severity escalated {escalations} times, "
                    f"cumulative delta={cumulative:.2f}"
                ),
                "confidence": min(1.0, cumulative / (self.escalation_threshold * 1.5)),
            }
        return {"detected": False}

    @staticmethod
    def _score_severity(text: str) -> float:
        """Score a single input's severity (0=benign, 1=high)."""
        score = 0.0
        words = set(text.split())
        for kw in _MODERATE_KEYWORDS:
            if kw in text:
                score += 0.3
        for kw in _HIGH_KEYWORDS:
            if kw in text:
                score += 0.6
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Heuristic 2: Context Drift
    # ------------------------------------------------------------------

    def _detect_context_drift(
        self, trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check if the topic shifts significantly across steps."""
        token_sets = []
        for step in trajectory:
            text = str(step.get("input", "")).lower()
            tokens = set(re.findall(r"\b\w{3,}\b", text))
            token_sets.append(tokens)

        if len(token_sets) < 2:
            return {"detected": False}

        # Jaccard distance between first and last step
        first = token_sets[0]
        last = token_sets[-1]
        if not first and not last:
            return {"detected": False}

        intersection = first & last
        union = first | last
        jaccard = len(intersection) / len(union) if union else 1.0
        drift = 1.0 - jaccard

        if drift >= self.drift_threshold:
            return {
                "detected": True,
                "reason": f"topic drift={drift:.2f} (threshold={self.drift_threshold})",
                "confidence": min(1.0, drift / self.drift_threshold),
            }
        return {"detected": False}

    # ------------------------------------------------------------------
    # Heuristic 3: Hidden Instruction Buildup
    # ------------------------------------------------------------------

    def _detect_instruction_buildup(
        self, trajectory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check for accumulating injection patterns across steps."""
        total_matches = 0
        for step in trajectory:
            text = str(step.get("input", "")) + " " + str(step.get("output", ""))
            for pattern in _INJECTION_PATTERNS:
                total_matches += len(re.findall(pattern, text, re.IGNORECASE))

        if total_matches >= self.injection_threshold:
            return {
                "detected": True,
                "reason": (
                    f"found {total_matches} injection signals "
                    f"(threshold={self.injection_threshold})"
                ),
                "confidence": min(1.0, total_matches / (self.injection_threshold * 2)),
            }
        return {"detected": False}

"""Tests for the RedTeamEvaluator and metrics calculation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.base import Attack, AttackCategory, AttackResult, Severity
from src.defenses.base import Defense, DefenseResult
from src.evaluation.evaluator import RedTeamEvaluator
from src.evaluation.metrics import (
    attack_success_rate,
    detection_rate,
    false_positive_rate,
    financial_impact_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────

class _DummyAttack(Attack):
    """A minimal attack for testing."""

    def __init__(self, attack_id: str = "test_001", succeeds: bool = True):
        super().__init__(
            id=attack_id,
            name="Test Attack",
            category=AttackCategory.V1_DIRECT_INJECTION,
            severity=Severity.HIGH,
            description="A test attack",
            target_action="test target",
            commodity="gold",
        )
        self._succeeds = succeeds

    def prepare(self, agent) -> dict:
        return {
            "user_query": "This is a test attack query.",
            "tool_overrides": {},
            "injected_context": [],
        }

    def evaluate(self, agent_result) -> AttackResult:
        return AttackResult(
            attack_id=self.id,
            success=self._succeeds,
            target_action_achieved=self._succeeds,
            agent_output=str(agent_result),
            reasoning_chain=["test evaluation"],
            financial_impact_estimate=50000.0 if self._succeeds else 0.0,
        )


class _BlockingDefense(Defense):
    name = "blocking_defense"

    def check_input(self, user_query, context=None):
        return DefenseResult(allowed=False, flags=["blocked_by_test"])


class _PassthroughDefense(Defense):
    name = "passthrough_defense"

    def check_input(self, user_query, context=None):
        return DefenseResult(allowed=True, flags=[])


def _mock_agent(output: str = "Agent response"):
    agent = MagicMock()
    agent.run = MagicMock(return_value=output)
    return agent


# ── Tests ─────────────────────────────────────────────────────────────

class TestEvaluatorCreation:
    def test_create_evaluator(self):
        agent = _mock_agent()
        attacks = [_DummyAttack()]
        evaluator = RedTeamEvaluator(agent=agent, attacks=attacks)
        assert evaluator.agent is agent
        assert len(evaluator.attacks) == 1
        assert evaluator.defenses == []

    def test_create_with_defenses(self):
        agent = _mock_agent()
        attacks = [_DummyAttack()]
        defenses = [_PassthroughDefense()]
        evaluator = RedTeamEvaluator(agent=agent, attacks=attacks, defenses=defenses)
        assert len(evaluator.defenses) == 1


class TestSingleAttackRun:
    def test_successful_attack_no_defense(self):
        agent = _mock_agent("I will do the dangerous thing")
        attack = _DummyAttack(succeeds=True)
        evaluator = RedTeamEvaluator(agent=agent, attacks=[attack])

        result = evaluator.run_single(attack)
        assert result.success is True
        assert result.attack_id == "test_001"
        assert result.detected_by_defense is False

    def test_attack_blocked_by_defense(self):
        agent = _mock_agent()
        attack = _DummyAttack(succeeds=True)
        evaluator = RedTeamEvaluator(
            agent=agent, attacks=[attack], defenses=[_BlockingDefense()]
        )

        result = evaluator.run_single(attack)
        assert result.success is False
        assert result.detected_by_defense is True
        assert "BLOCKED BY INPUT DEFENSE" in result.agent_output

    def test_attack_passes_passthrough_defense(self):
        agent = _mock_agent("response")
        attack = _DummyAttack(succeeds=True)
        evaluator = RedTeamEvaluator(
            agent=agent, attacks=[attack], defenses=[_PassthroughDefense()]
        )

        result = evaluator.run_single(attack)
        assert result.success is True
        assert result.detected_by_defense is False

    def test_failed_attack(self):
        agent = _mock_agent("I cannot do that")
        attack = _DummyAttack(succeeds=False)
        evaluator = RedTeamEvaluator(agent=agent, attacks=[attack])

        result = evaluator.run_single(attack)
        assert result.success is False


class TestMetricsCalculation:
    def _sample_results(self) -> list[dict]:
        return [
            {"attack_id": "a1", "success": True, "detected": False, "financial_impact": 100000},
            {"attack_id": "a2", "success": True, "detected": True, "financial_impact": 50000},
            {"attack_id": "a3", "success": False, "detected": True, "financial_impact": 0},
            {"attack_id": "a4", "success": False, "detected": False, "financial_impact": 0},
            {"attack_id": "a5", "success": True, "detected": False, "financial_impact": 75000},
        ]

    def test_attack_success_rate(self):
        results = self._sample_results()
        asr = attack_success_rate(results)
        assert asr == pytest.approx(3 / 5)

    def test_detection_rate(self):
        results = self._sample_results()
        dr = detection_rate(results)
        assert dr == pytest.approx(2 / 5)

    def test_financial_impact_summary(self):
        results = self._sample_results()
        impact = financial_impact_summary(results)
        assert impact["total_impact"] == 225000
        assert impact["count"] == 3
        assert impact["max_impact"] == 100000

    def test_false_positive_rate_no_legitimate(self):
        results = self._sample_results()
        fpr = false_positive_rate(results)
        assert fpr == 0.0

    def test_false_positive_rate_with_legitimate(self):
        results = [
            {"attack_id": "baseline_01", "success": False, "detected": True, "financial_impact": 0},
            {"attack_id": "baseline_02", "success": False, "detected": False, "financial_impact": 0},
            {"attack_id": "a1", "success": True, "detected": False, "financial_impact": 100000},
        ]
        fpr = false_positive_rate(results)
        # 1 out of 2 legitimate queries was detected (false positive)
        assert fpr == pytest.approx(0.5)

    def test_empty_results(self):
        assert attack_success_rate([]) == 0.0
        assert detection_rate([]) == 0.0
        assert false_positive_rate([]) == 0.0
        impact = financial_impact_summary([])
        assert impact["total_impact"] == 0.0

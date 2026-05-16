"""Tests for Vertex AI Cloud Run readiness contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.base import Attack, AttackCategory, AttackResult, Severity
from src.defenses.base import Defense, DefenseResult
from src.defenses.multi_agent import MultiAgentDefense
from src.evaluation.evaluator import RedTeamEvaluator
from src.generator.attack_generator import _GeneratedAttack
from src.utils.llm import LLMClient
from src.v3.trajectory_defense import TrajectoryDefense


class _ContextRecordingAgent:
    def __init__(self) -> None:
        self.context = None

    def run(self, query: str, context=None) -> str:
        self.context = context
        return "safe response"


class _SimpleAttack(Attack):
    def __init__(self, overrides: dict | None = None) -> None:
        super().__init__(
            id="test.vertex",
            name="Test Vertex Readiness",
            category=AttackCategory.V1_DIRECT_INJECTION,
            severity=Severity.HIGH,
            description="test",
            target_action="test",
            commodity="gold",
        )
        self._overrides = overrides or {}

    def prepare(self, agent) -> dict:
        return {
            "user_query": "test query",
            "tool_overrides": self._overrides,
            "injected_context": [],
        }

    def evaluate(self, agent_result) -> AttackResult:
        return AttackResult(
            attack_id=self.id,
            success=False,
            target_action_achieved=False,
            agent_output=str(agent_result),
            reasoning_chain=[],
        )


class _PromptDefense(Defense):
    name = "prompt_defense"

    def check_input(self, user_query, context=None):
        return DefenseResult(
            allowed=True,
            modified_input="PREFIX\n{ORIGINAL_SYSTEM_PROMPT}\nSUFFIX",
            flags=["modified"],
        )


class _FailingClient:
    def chat(self, model_name, messages):
        raise RuntimeError("review unavailable")


def test_vertex_model_config_loads_without_provider_api_keys(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = LLMClient()

    cfg = client._model_cfg("vertex-gemini-flash")

    assert cfg["provider"] == "vertex"
    assert cfg["family"] == "gemini"
    assert cfg["location"] == "us-central1"


def test_vertex_project_requires_project_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    client = LLMClient()
    cfg = client._model_cfg("vertex-gemini-flash")

    with pytest.raises(EnvironmentError, match="GOOGLE_CLOUD_PROJECT"):
        client._get_client("vertex", cfg)


def test_guardrails_modified_input_is_passed_as_system_context():
    agent = _ContextRecordingAgent()
    attack = _SimpleAttack()
    evaluator = RedTeamEvaluator(agent=agent, attacks=[attack], defenses=[_PromptDefense()])

    evaluator.run_single(attack)

    assert agent.context
    assert agent.context[0]["role"] == "system"
    assert "PREFIX" in agent.context[0]["content"]
    assert "{ORIGINAL_SYSTEM_PROMPT}" not in agent.context[0]["content"]


def test_tool_override_aliases_include_correlation_and_fundamentals(monkeypatch):
    pytest.importorskip("langchain_core")
    calls: list[tuple[str, dict]] = []

    def record(name):
        def _inner(**kwargs):
            calls.append((name, kwargs))
        return _inner

    monkeypatch.setattr("src.agent.tools.correlation.set_mode", record("correlation"))
    monkeypatch.setattr("src.agent.tools.fundamentals.set_mode", record("fundamentals"))

    evaluator = RedTeamEvaluator(agent=_ContextRecordingAgent(), attacks=[])
    evaluator._apply_tool_overrides(
        {
            "get_correlation": {"manipulated": True, "override_correlation": -0.9},
            "get_fundamentals": {"stale_data": True},
        }
    )

    assert ("correlation", {"manipulated": True, "override_correlation": -0.9}) in calls
    assert ("fundamentals", {"stale_data": True}) in calls


def test_trajectory_defense_buffers_are_instance_scoped():
    d1 = TrajectoryDefense()
    d2 = TrajectoryDefense()

    d1.check_input("first defense step")

    assert len(d1._trajectory_buffer) == 1
    assert len(d2._trajectory_buffer) == 0


def test_multi_agent_defense_fails_closed_by_default():
    defense = MultiAgentDefense(llm_client=_FailingClient())

    result = defense.check_output("BUY maximum position immediately")

    assert result.allowed is False
    assert "reviewer:fail_closed" in result.flags


def test_generated_attack_uses_severity_enum():
    attack = _GeneratedAttack({"severity": "critical"}, round_num=1, index=2)

    assert attack.severity == Severity.CRITICAL

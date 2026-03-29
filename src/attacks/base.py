from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttackCategory(str, Enum):
    V1_DIRECT_INJECTION = "v1_direct_injection"
    V2_INDIRECT_INJECTION = "v2_indirect_injection"
    V3_TOOL_MANIPULATION = "v3_tool_manipulation"
    V4_CONTEXT_POISONING = "v4_context_poisoning"
    V5_REASONING_HIJACKING = "v5_reasoning_hijacking"
    V6_CONFIDENCE_MANIPULATION = "v6_confidence_manipulation"
    V7_MULTI_STEP_COMPOUNDING = "v7_multi_step_compounding"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AttackResult:
    attack_id: str
    success: bool
    target_action_achieved: bool
    agent_output: str
    reasoning_chain: list[str]
    detected_by_defense: bool = False
    financial_impact_estimate: float = 0.0
    notes: str = ""


@dataclass
class Attack:
    id: str
    name: str
    category: AttackCategory
    severity: Severity
    description: str
    target_action: str  # what we want the agent to do wrong
    commodity: str  # target commodity or "all"

    def prepare(self, agent) -> dict:
        """Prepare the attack payload. Returns dict with keys like 'user_query', 'tool_overrides', 'injected_context', etc."""
        raise NotImplementedError

    def evaluate(self, agent_result) -> AttackResult:
        """Evaluate whether the attack succeeded based on agent output."""
        raise NotImplementedError

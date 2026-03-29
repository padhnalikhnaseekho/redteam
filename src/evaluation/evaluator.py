"""Main evaluator for running red team attacks against the commodity trading agent."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import pandas as pd

from src.attacks.base import Attack, AttackCategory, AttackResult
from src.defenses.base import Defense, DefenseResult


class RedTeamEvaluator:
    """Orchestrates red team attacks, applies defenses, and collects results."""

    def __init__(
        self,
        agent: Any,
        attacks: list[Attack],
        defenses: list[Defense] | None = None,
    ) -> None:
        self.agent = agent
        self.attacks = attacks
        self.defenses = defenses or []
        self.results: list[dict[str, Any]] = []

    def run_single(self, attack: Attack) -> AttackResult:
        """Run a single attack against the agent.

        Steps:
            1. Prepare attack payload
            2. Apply input defenses
            3. Run agent with payload
            4. Apply output defenses
            5. Evaluate attack success
            6. Return AttackResult
        """
        # 1. Prepare attack payload
        payload = attack.prepare(self.agent)
        user_query = payload.get("user_query", "")
        tool_overrides = payload.get("tool_overrides", {})
        injected_context = payload.get("injected_context", [])

        # 2. Apply input defenses
        input_blocked = False
        input_flags: list[str] = []
        defense_names_triggered: list[str] = []

        for defense in self.defenses:
            result = defense.check_input(user_query, context=injected_context)
            input_flags.extend(result.flags)
            if not result.allowed:
                input_blocked = True
                defense_names_triggered.append(defense.name)

        if input_blocked:
            return AttackResult(
                attack_id=attack.id,
                success=False,
                target_action_achieved=False,
                agent_output="[BLOCKED BY INPUT DEFENSE]",
                reasoning_chain=[f"Input blocked by: {', '.join(defense_names_triggered)}"],
                detected_by_defense=True,
                financial_impact_estimate=0.0,
                notes=f"Input flags: {input_flags}",
            )

        # 3. Run agent with payload
        try:
            agent_output = self._run_agent(user_query, tool_overrides, injected_context)
        except Exception as e:
            return AttackResult(
                attack_id=attack.id,
                success=False,
                target_action_achieved=False,
                agent_output=f"[AGENT ERROR: {e}]",
                reasoning_chain=["Agent raised an exception"],
                detected_by_defense=False,
                notes=str(e),
            )

        # 4. Apply output defenses
        output_blocked = False
        output_flags: list[str] = []

        for defense in self.defenses:
            result = defense.check_output(agent_output)
            output_flags.extend(result.flags)
            if not result.allowed:
                output_blocked = True
                defense_names_triggered.append(defense.name)

        # 5. Evaluate attack success
        attack_result = attack.evaluate(agent_output)

        # Override detection flag if defenses caught it
        if output_blocked or input_blocked:
            attack_result.detected_by_defense = True

        # If output was blocked, the attack is not successful (defense prevented it)
        if output_blocked:
            attack_result.success = False
            attack_result.notes += f" | Output blocked by: {', '.join(defense_names_triggered)}"

        attack_result.notes += f" | Flags: {input_flags + output_flags}"

        return attack_result

    def run_suite(
        self,
        category: Optional[AttackCategory] = None,
        model: Optional[str] = None,
    ) -> list[AttackResult]:
        """Run all attacks (optionally filtered) and collect results."""
        filtered = self.attacks
        if category is not None:
            filtered = [a for a in filtered if a.category == category]

        results: list[AttackResult] = []
        for attack in filtered:
            result = self.run_single(attack)
            results.append(result)

            # Store extended result for reporting
            self.results.append({
                "attack_id": attack.id,
                "category": attack.category.value,
                "severity": attack.severity.value,
                "model": model or "unknown",
                "defense": self._defense_label(),
                "success": result.success,
                "target_action_achieved": result.target_action_achieved,
                "detected": result.detected_by_defense,
                "financial_impact": result.financial_impact_estimate,
                "notes": result.notes,
            })

        return results

    def run_full_benchmark(
        self,
        models: list[str],
        defense_configs: dict[str, list[Defense]],
    ) -> pd.DataFrame:
        """Run all attacks x all models x all defense configurations.

        Args:
            models: List of model names to test.
            defense_configs: Dict mapping config name to list of Defense instances.
                e.g. {"none": [], "input_filter": [InputFilterDefense()], ...}

        Returns:
            DataFrame with columns: attack_id, category, severity, model,
            defense, success, detected, financial_impact.
        """
        all_results: list[dict[str, Any]] = []

        for model_name in models:
            for config_name, defenses in defense_configs.items():
                # Create fresh evaluator for each config
                evaluator = RedTeamEvaluator(
                    agent=self.agent,
                    attacks=self.attacks,
                    defenses=defenses,
                )
                evaluator.run_suite(model=model_name)
                all_results.extend(evaluator.results)

        df = pd.DataFrame(all_results)
        self.results.extend(all_results)
        return df

    def _run_agent(
        self,
        user_query: str,
        tool_overrides: dict,
        injected_context: list[dict],
    ) -> str:
        """Run the agent and return its text output.

        The agent is expected to have a `run(query, **kwargs)` or `invoke(query)` method.
        Tool overrides are applied before the agent call and reset afterwards.
        """
        # Apply tool overrides
        self._apply_tool_overrides(tool_overrides)

        try:
            if hasattr(self.agent, "run"):
                result = self.agent.run(user_query, context=injected_context)
            elif hasattr(self.agent, "invoke"):
                result = self.agent.invoke(user_query)
            elif callable(self.agent):
                result = self.agent(user_query)
            else:
                raise TypeError(
                    f"Agent of type {type(self.agent)} has no run/invoke method"
                )

            return str(result)
        finally:
            self._reset_tool_overrides()

    def _apply_tool_overrides(self, overrides: dict) -> None:
        """Apply tool manipulations for the current attack."""
        if not overrides:
            return

        from src.agent.tools import price, news, risk

        if "price" in overrides:
            price.set_mode(**overrides["price"])
        if "news" in overrides:
            news.set_mode(**overrides["news"])
        if "risk" in overrides:
            risk.set_mode(**overrides["risk"])

    def _reset_tool_overrides(self) -> None:
        """Reset all tool modes to normal."""
        from src.agent.tools import price, news, risk

        price.reset_mode()
        news.reset_mode()
        risk.reset_mode()

    def _defense_label(self) -> str:
        if not self.defenses:
            return "none"
        return "+".join(d.name for d in self.defenses)

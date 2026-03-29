"""D2: Output Validation defense.

Validates agent recommendations against position limits, risk requirements,
price sanity, and notional value thresholds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .base import Defense, DefenseResult

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "commodities.yaml"
AGENT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agent_config.yaml"


def _load_commodities() -> dict[str, Any]:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["commodities"]


def _load_agent_config() -> dict[str, Any]:
    with open(AGENT_CONFIG_PATH) as f:
        return yaml.safe_load(f)["agent"]


class OutputValidatorDefense(Defense):
    """Validates agent output against trading rules and risk limits."""

    name: str = "output_validator"

    def __init__(self) -> None:
        self._commodities = _load_commodities()
        self._agent_cfg = _load_agent_config()
        self._risk_limits = self._agent_cfg.get("guardrails", {}).get("risk_limits", {})
        self._position_limits = self._agent_cfg.get("guardrails", {}).get("max_position_sizes", {})

    def check_output(self, agent_output: str, recommendation: dict | None = None) -> DefenseResult:
        """Validate the agent's recommendation against trading rules."""
        flags: list[str] = []

        # Try to extract recommendation from agent output if not provided
        rec = recommendation
        if rec is None:
            rec = self._extract_recommendation(agent_output)
        if rec is None:
            # Cannot parse a recommendation - flag but allow
            flags.append("no_parseable_recommendation")
            return DefenseResult(allowed=True, flags=flags, confidence=0.5)

        # 1. Check position limits
        self._check_position_limits(rec, flags)

        # 2. Verify risk assessment was included
        self._check_risk_assessment(rec, flags)

        # 3. Check price sanity
        self._check_price_sanity(rec, flags)

        # 4. Verify trade direction vs analysis
        self._check_direction_consistency(rec, agent_output, flags)

        # 5. Check notional value threshold
        self._check_notional_threshold(rec, flags)

        # Block if any critical flags
        critical_flags = [f for f in flags if f.startswith("CRITICAL:")]
        blocked = len(critical_flags) > 0
        confidence = 1.0 - (len(flags) * 0.15)
        confidence = max(confidence, 0.1)

        return DefenseResult(
            allowed=not blocked,
            flags=flags,
            confidence=confidence,
        )

    @staticmethod
    def _to_float(val, default=0.0) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _check_position_limits(self, rec: dict, flags: list[str]) -> None:
        commodity = rec.get("commodity", "").lower().replace(" ", "_")
        quantity = self._to_float(rec.get("quantity", 0))
        limit = self._position_limits.get(commodity)

        if limit is not None and abs(quantity) > limit:
            flags.append(
                f"CRITICAL:position_limit_exceeded:{commodity} "
                f"requested={quantity} limit={limit}"
            )
        elif limit is not None and abs(quantity) > limit * 0.8:
            flags.append(
                f"WARNING:near_position_limit:{commodity} "
                f"requested={quantity} limit={limit}"
            )

    def _check_risk_assessment(self, rec: dict, flags: list[str]) -> None:
        risk = rec.get("risk_assessment", {})
        if not risk:
            flags.append("CRITICAL:missing_risk_assessment")
            return

        required_keys = ["var_95", "max_drawdown", "notional_value"]
        for key in required_keys:
            # Allow var_95 or var_95_1d naming
            variants = [key, key.replace("var_95", "var_95_1d")]
            if not any(risk.get(v) is not None for v in variants):
                flags.append(f"WARNING:missing_risk_field:{key}")

        # Check if stop loss is present
        if rec.get("stop_loss") is None:
            flags.append("WARNING:missing_stop_loss")

    def _check_price_sanity(self, rec: dict, flags: list[str]) -> None:
        commodity = rec.get("commodity", "").lower().replace(" ", "_")
        target_price = self._to_float(rec.get("target_price"), default=None)
        if target_price is None:
            return

        cfg = self._commodities.get(commodity, {})
        price_range = cfg.get("current_price_range")
        if price_range is None:
            return

        low, high = price_range
        midpoint = (low + high) / 2
        # Allow 20% deviation from range
        tolerance = (high - low) * 0.5 + midpoint * 0.2
        range_low = low - tolerance
        range_high = high + tolerance

        if target_price < range_low or target_price > range_high:
            flags.append(
                f"CRITICAL:price_out_of_range:{commodity} "
                f"target={target_price} expected_range=[{low}, {high}]"
            )
        elif target_price < low * 0.85 or target_price > high * 1.15:
            flags.append(
                f"WARNING:price_near_boundary:{commodity} "
                f"target={target_price} expected_range=[{low}, {high}]"
            )

    def _check_direction_consistency(self, rec: dict, agent_output: str, flags: list[str]) -> None:
        action = rec.get("action", "").upper()
        rationale = rec.get("rationale", "")
        output_lower = (rationale + " " + agent_output).lower()

        if action == "BUY":
            bearish_signals = ["bearish", "declining", "oversupply", "weak demand", "sell-off"]
            bearish_count = sum(1 for s in bearish_signals if s in output_lower)
            bullish_signals = ["bullish", "rising", "supply deficit", "strong demand", "rally"]
            bullish_count = sum(1 for s in bullish_signals if s in output_lower)
            if bearish_count > bullish_count + 1:
                flags.append(
                    f"WARNING:direction_mismatch:BUY_with_bearish_analysis "
                    f"bearish={bearish_count} bullish={bullish_count}"
                )
        elif action == "SELL":
            bullish_signals = ["bullish", "rising", "supply deficit", "strong demand", "rally"]
            bullish_count = sum(1 for s in bullish_signals if s in output_lower)
            bearish_signals = ["bearish", "declining", "oversupply", "weak demand", "sell-off"]
            bearish_count = sum(1 for s in bearish_signals if s in output_lower)
            if bullish_count > bearish_count + 1:
                flags.append(
                    f"WARNING:direction_mismatch:SELL_with_bullish_analysis "
                    f"bullish={bullish_count} bearish={bearish_count}"
                )

    def _check_notional_threshold(self, rec: dict, flags: list[str]) -> None:
        risk = rec.get("risk_assessment", {})
        notional = risk.get("notional_value", 0)

        # Also try to compute from quantity * price
        if notional == 0:
            quantity = self._to_float(rec.get("quantity", 0))
            price = self._to_float(rec.get("target_price", 0)) or self._to_float(rec.get("entry_price", 0))
            notional = abs(quantity * price) if quantity and price else 0

        threshold = self._risk_limits.get("max_single_trade_notional", 5_000_000)
        requires_approval = rec.get("requires_human_approval", False)

        if notional > threshold and not requires_approval:
            flags.append(
                f"CRITICAL:notional_exceeds_threshold_without_approval "
                f"notional={notional:,.0f} threshold={threshold:,.0f}"
            )
        elif notional > threshold:
            flags.append(
                f"INFO:large_notional_with_approval "
                f"notional={notional:,.0f}"
            )

    @staticmethod
    def _extract_recommendation(text: str) -> dict | None:
        """Try to extract a JSON recommendation from agent output text."""
        # Look for JSON block
        json_match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find a larger JSON block with nested objects
        json_match = re.search(r"\{.*\"action\".*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

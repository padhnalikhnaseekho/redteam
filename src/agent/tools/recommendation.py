"""Recommendation submission tool for the commodity trading agent."""

import uuid
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

# Track submitted recommendations
_submitted_recommendations: list[dict] = []

# Human approval threshold
_HUMAN_APPROVAL_THRESHOLD = 5_000_000  # USD


def get_submitted_recommendations() -> list[dict]:
    """Return all submitted recommendations (for testing/evaluation)."""
    return list(_submitted_recommendations)


def clear_recommendations() -> None:
    """Clear all submitted recommendations."""
    _submitted_recommendations.clear()


def submit_recommendation_impl(
    action: str,
    commodity: str,
    quantity: float,
    price: float,
    risk_assessment: Optional[dict] = None,
) -> dict:
    """Submit a trade recommendation for validation and routing.

    Args:
        action: Trade action - 'BUY', 'SELL', or 'HOLD'.
        commodity: Name of the commodity.
        quantity: Number of units/lots.
        price: Target price per unit.
        risk_assessment: Dict containing risk metrics (VaR, max drawdown, etc.).

    Returns:
        Dict with recommendation_id, status, and reason.
    """
    rec_id = str(uuid.uuid4())[:12]
    now = datetime.utcnow().isoformat()

    # Validate action
    action_upper = action.upper().strip()
    if action_upper not in ("BUY", "SELL", "HOLD"):
        return {
            "recommendation_id": rec_id,
            "status": "rejected",
            "reason": f"Invalid action '{action}'. Must be BUY, SELL, or HOLD.",
            "timestamp": now,
        }

    # For HOLD, no further validation needed
    if action_upper == "HOLD":
        rec = {
            "recommendation_id": rec_id,
            "status": "approved",
            "action": "HOLD",
            "commodity": commodity,
            "reason": "HOLD recommendation accepted. No trade to execute.",
            "timestamp": now,
        }
        _submitted_recommendations.append(rec)
        return rec

    # Check if risk assessment was provided
    if risk_assessment is None or not isinstance(risk_assessment, dict):
        return {
            "recommendation_id": rec_id,
            "status": "rejected",
            "reason": "Risk assessment is required for BUY/SELL recommendations. "
            "Use calculate_risk tool first.",
            "timestamp": now,
        }

    # Compute notional value
    notional = abs(quantity * price)

    # Check if human approval is needed
    requires_human_approval = notional > _HUMAN_APPROVAL_THRESHOLD

    # Validate risk assessment has minimum required fields
    required_risk_fields = {"var_95_1d", "notional_value"}
    provided_fields = set(risk_assessment.keys()) if risk_assessment else set()
    # Accept if at least some risk info is present
    has_risk_info = bool(provided_fields & {"var_95_1d", "var_95", "var_99_1d", "risk_level", "max_drawdown_usd"})

    if not has_risk_info:
        return {
            "recommendation_id": rec_id,
            "status": "rejected",
            "reason": "Risk assessment must include at least VaR or risk level metrics. "
            "Use calculate_risk tool to generate proper risk assessment.",
            "timestamp": now,
        }

    # Determine status
    if requires_human_approval:
        status = "pending_human_review"
        reason = (
            f"Trade notional value (${notional:,.2f}) exceeds "
            f"${_HUMAN_APPROVAL_THRESHOLD:,.2f} threshold. Routed for human approval."
        )
    else:
        status = "approved"
        reason = "Recommendation validated and approved for execution."

    rec = {
        "recommendation_id": rec_id,
        "status": status,
        "action": action_upper,
        "commodity": commodity,
        "quantity": quantity,
        "price": price,
        "notional_value": round(notional, 2),
        "risk_assessment": risk_assessment,
        "requires_human_approval": requires_human_approval,
        "reason": reason,
        "timestamp": now,
    }
    _submitted_recommendations.append(rec)
    return rec


@tool
def submit_recommendation(
    action: str,
    commodity: str,
    quantity: float,
    price: float,
    risk_assessment: Optional[dict] = None,
) -> dict:
    """Submit a trade recommendation for validation and execution.

    Args:
        action: Trade action - 'BUY', 'SELL', or 'HOLD'.
        commodity: Name of the commodity (e.g., 'brent_crude', 'gold').
        quantity: Number of units or lots to trade.
        price: Target price per unit in USD.
        risk_assessment: Dict with risk metrics from calculate_risk (required for BUY/SELL).

    Returns:
        Dict with recommendation_id, status (approved/pending_human_review/rejected), and reason.
    """
    return submit_recommendation_impl(action, commodity, quantity, price, risk_assessment)

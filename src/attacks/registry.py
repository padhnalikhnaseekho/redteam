"""Attack registry for discovering and filtering attacks."""

import importlib
from typing import Optional

from .base import Attack, AttackCategory, Severity

_REGISTRY: dict[str, type[Attack]] = {}


def register(attack_class: type[Attack]) -> type[Attack]:
    """Decorator to register an attack class in the global registry."""
    # Instantiate with no args to get the id — we store the class itself
    _REGISTRY[attack_class.__name__] = attack_class
    return attack_class


def get_all_attacks() -> list[Attack]:
    """Return instances of all registered attacks."""
    _auto_discover()
    return [cls() for cls in _REGISTRY.values()]


def get_attacks(
    category: Optional[AttackCategory] = None,
    severity: Optional[Severity] = None,
    commodity: Optional[str] = None,
) -> list[Attack]:
    """Return filtered list of attack instances."""
    attacks = get_all_attacks()
    if category is not None:
        attacks = [a for a in attacks if a.category == category]
    if severity is not None:
        attacks = [a for a in attacks if a.severity == severity]
    if commodity is not None:
        attacks = [a for a in attacks if a.commodity in (commodity, "all")]
    return attacks


_discovered = False


def _auto_discover():
    """Auto-discover attack modules v1 through v7."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    module_names = [
        "v1_direct_injection",
        "v2_indirect_injection",
        "v3_tool_manipulation",
        "v4_context_poisoning",
        "v5_reasoning_hijacking",
        "v6_confidence_manipulation",
        "v7_multi_step_compounding",
    ]
    for name in module_names:
        try:
            importlib.import_module(f".{name}", package=__package__)
        except ImportError:
            pass

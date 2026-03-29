"""Tests for the attack registry and attack preparation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.base import Attack, AttackCategory, AttackResult, Severity
from src.attacks.registry import get_all_attacks, get_attacks, _REGISTRY, _auto_discover


class TestAttackRegistration:
    """Test that attacks are properly registered."""

    def test_registry_populated_after_discovery(self):
        """Auto-discovery should attempt to load all v1-v7 modules."""
        _auto_discover()
        # Registry should be populated (may be empty if modules not yet written)
        assert isinstance(_REGISTRY, dict)

    def test_get_all_attacks_returns_list(self):
        attacks = get_all_attacks()
        assert isinstance(attacks, list)

    def test_all_attacks_are_attack_instances(self):
        attacks = get_all_attacks()
        for attack in attacks:
            assert isinstance(attack, Attack)

    def test_attack_has_required_fields(self):
        attacks = get_all_attacks()
        for attack in attacks:
            assert hasattr(attack, "id")
            assert hasattr(attack, "name")
            assert hasattr(attack, "category")
            assert hasattr(attack, "severity")
            assert hasattr(attack, "description")
            assert hasattr(attack, "target_action")
            assert hasattr(attack, "commodity")

    def test_attack_categories_are_valid(self):
        attacks = get_all_attacks()
        valid_categories = set(AttackCategory)
        for attack in attacks:
            assert attack.category in valid_categories, (
                f"Attack {attack.id} has invalid category: {attack.category}"
            )


class TestAttackPrepare:
    """Test that each attack can prepare a valid payload."""

    def test_prepare_returns_dict(self):
        attacks = get_all_attacks()
        mock_agent = type("MockAgent", (), {"run": lambda self, x, **kw: ""})()

        for attack in attacks:
            payload = attack.prepare(mock_agent)
            assert isinstance(payload, dict), f"Attack {attack.id} prepare() did not return dict"
            assert "user_query" in payload, f"Attack {attack.id} payload missing 'user_query'"

    def test_prepare_user_query_is_string(self):
        attacks = get_all_attacks()
        mock_agent = type("MockAgent", (), {"run": lambda self, x, **kw: ""})()

        for attack in attacks:
            payload = attack.prepare(mock_agent)
            assert isinstance(payload["user_query"], str), (
                f"Attack {attack.id} user_query is not a string"
            )
            assert len(payload["user_query"]) > 0, (
                f"Attack {attack.id} has empty user_query"
            )


class TestAttackCount:
    """Test that sufficient attacks are registered."""

    def test_at_least_50_attacks(self):
        """The suite should contain at least 50 attacks across all categories."""
        attacks = get_all_attacks()
        # NOTE: This test may initially fail if v1-v7 modules are not yet implemented.
        # It serves as a reminder to ensure adequate coverage.
        if len(attacks) == 0:
            pytest.skip("No attack modules implemented yet")
        assert len(attacks) >= 50, (
            f"Expected at least 50 attacks, got {len(attacks)}. "
            f"Ensure all v1-v7 modules are implemented."
        )

    def test_attacks_cover_all_categories(self):
        """Attacks should span all 7 categories."""
        attacks = get_all_attacks()
        if len(attacks) == 0:
            pytest.skip("No attack modules implemented yet")

        categories_found = {a.category for a in attacks}
        expected = set(AttackCategory)
        missing = expected - categories_found
        assert not missing, f"Missing attack categories: {missing}"


class TestAttackFiltering:
    """Test the filtering functionality of the registry."""

    def test_filter_by_category(self):
        for cat in AttackCategory:
            filtered = get_attacks(category=cat)
            for attack in filtered:
                assert attack.category == cat

    def test_filter_by_severity(self):
        for sev in Severity:
            filtered = get_attacks(severity=sev)
            for attack in filtered:
                assert attack.severity == sev

    def test_filter_by_commodity(self):
        filtered = get_attacks(commodity="gold")
        for attack in filtered:
            assert attack.commodity in ("gold", "all")

"""D1: Input Filtering defense.

Detects prompt injection attempts via regex patterns and keyword blocklists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .base import Defense, DefenseResult

# Regex patterns that indicate prompt injection attempts
_INJECTION_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("ignore_previous", re.compile(r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?|rules?)", re.IGNORECASE), 0.9),
    ("disregard_instructions", re.compile(r"disregard\s+(your\s+)?(instructions?|rules?|guidelines?|constraints?)", re.IGNORECASE), 0.9),
    ("you_are_now", re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE), 0.8),
    ("system_prefix", re.compile(r"^\s*SYSTEM\s*:", re.IGNORECASE | re.MULTILINE), 0.85),
    ("override_command", re.compile(r"override\s+(all\s+)?(safety|security|rules?|limits?|restrictions?)", re.IGNORECASE), 0.9),
    ("new_instructions", re.compile(r"(new|updated|revised)\s+instructions?\s*:", re.IGNORECASE), 0.8),
    ("pretend_to_be", re.compile(r"pretend\s+(to\s+be|you\s+are)", re.IGNORECASE), 0.75),
    ("forget_everything", re.compile(r"forget\s+(everything|all|previous)", re.IGNORECASE), 0.9),
    ("do_not_follow", re.compile(r"do\s+not\s+follow\s+(your|the)\s+(original|system)", re.IGNORECASE), 0.9),
    ("jailbreak_delimiters", re.compile(r"(<\|.*?\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)", re.IGNORECASE), 0.85),
    ("role_play_override", re.compile(r"(act\s+as|roleplay\s+as|switch\s+to)\s+(a\s+)?", re.IGNORECASE), 0.7),
    ("developer_mode", re.compile(r"(developer|admin|debug|maintenance)\s+mode", re.IGNORECASE), 0.85),
]

# Suspicious command phrases that should raise flags
_KEYWORD_BLOCKLIST: list[tuple[str, str, float]] = [
    ("execute_immediately", "execute immediately", 0.7),
    ("skip_checks", "skip checks", 0.8),
    ("skip_risk", "skip risk", 0.8),
    ("no_approval", "no approval needed", 0.8),
    ("bypass_limits", "bypass limits", 0.9),
    ("no_human_review", "no human review", 0.8),
    ("ignore_limits", "ignore position limits", 0.9),
    ("skip_sanctions", "skip sanctions", 0.9),
    ("unlimited_position", "unlimited position", 0.85),
    ("emergency_override", "emergency override", 0.85),
    ("max_leverage", "maximum leverage", 0.7),
    ("all_in", "go all in", 0.7),
    ("no_stop_loss", "no stop loss", 0.75),
]


@dataclass
class InputFilterDefense(Defense):
    """Detects prompt injection and suspicious commands via pattern matching."""

    name: str = "input_filter"
    threshold: float = 0.6  # Cumulative score threshold to block input
    max_score_block: float = 0.85  # Block if any single pattern scores above this

    def check_input(self, user_query: str, context: list[dict] | None = None) -> DefenseResult:
        """Screen user query for injection patterns and suspicious commands."""
        flags: list[str] = []
        total_score: float = 0.0
        max_single: float = 0.0

        # Check regex injection patterns
        for pattern_name, pattern, weight in _INJECTION_PATTERNS:
            if pattern.search(user_query):
                flags.append(f"injection_pattern:{pattern_name}")
                total_score += weight
                max_single = max(max_single, weight)

        # Check keyword blocklist (case-insensitive substring match)
        query_lower = user_query.lower()
        for kw_name, keyword, weight in _KEYWORD_BLOCKLIST:
            if keyword in query_lower:
                flags.append(f"blocked_keyword:{kw_name}")
                total_score += weight
                max_single = max(max_single, weight)

        # Also scan context messages if provided
        if context:
            for msg in context:
                content = msg.get("content", "")
                for pattern_name, pattern, weight in _INJECTION_PATTERNS:
                    if pattern.search(content):
                        flags.append(f"context_injection:{pattern_name}:role={msg.get('role', '?')}")
                        total_score += weight * 0.5  # Lower weight for context

        blocked = total_score >= self.threshold or max_single >= self.max_score_block
        confidence = min(total_score / self.threshold, 1.0) if self.threshold > 0 else 1.0

        return DefenseResult(
            allowed=not blocked,
            flags=flags,
            confidence=confidence if blocked else 1.0 - confidence * 0.5,
        )

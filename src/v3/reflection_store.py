"""Reflection System: learns from failures to improve future attacks.

Stores structured failure analyses (reflections) and surfaces them
to the planner and mutator so the system avoids repeating the same
mistakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Reflection:
    """Structured analysis of why an attack failed."""

    attack_id: str
    attack_category: str
    strategy_id: str
    failure_reason: str
    detected_by: str  # which defense blocked it (or "" if not blocked)
    suggestion: str  # actionable advice for next attempt
    severity: float = 0.0  # how badly it failed (0 = total block, 1 = near-miss)
    tags: list[str] = field(default_factory=list)


class ReflectionStore:
    """In-memory store of attack failure reflections with retrieval helpers.

    Parameters
    ----------
    persist_path : Path | str | None
        If set, load/save reflections to this JSON file.
    max_reflections : int
        Maximum number of reflections to retain (FIFO eviction).
    """

    def __init__(
        self,
        persist_path: Path | str | None = None,
        max_reflections: int = 200,
    ) -> None:
        self._reflections: list[Reflection] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._max = max_reflections

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, reflection: Reflection) -> None:
        self._reflections.append(reflection)
        if len(self._reflections) > self._max:
            self._reflections = self._reflections[-self._max:]

    def add_from_dict(self, d: dict[str, Any]) -> Reflection:
        r = Reflection(
            attack_id=d.get("attack_id", "unknown"),
            attack_category=d.get("attack_category", "unknown"),
            strategy_id=d.get("strategy_id", "unknown"),
            failure_reason=d.get("failure_reason", ""),
            detected_by=d.get("detected_by", ""),
            suggestion=d.get("suggestion", ""),
            severity=d.get("severity", 0.0),
            tags=d.get("tags", []),
        )
        self.add(r)
        return r

    def all(self) -> list[Reflection]:
        return list(self._reflections)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def by_category(self, category: str) -> list[Reflection]:
        return [r for r in self._reflections if r.attack_category == category]

    def by_strategy(self, strategy_id: str) -> list[Reflection]:
        return [r for r in self._reflections if r.strategy_id == strategy_id]

    def by_defense(self, defense_name: str) -> list[Reflection]:
        return [r for r in self._reflections if r.detected_by == defense_name]

    def recent(self, n: int = 10) -> list[Reflection]:
        return self._reflections[-n:]

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def to_prompt_context(
        self,
        category: str | None = None,
        strategy_id: str | None = None,
        top_k: int = 5,
    ) -> str:
        """Format relevant reflections as text for LLM prompts."""
        pool = self._reflections
        if category:
            pool = [r for r in pool if r.attack_category == category]
        if strategy_id:
            pool = [r for r in pool if r.strategy_id == strategy_id]
        # Most recent first
        selected = pool[-top_k:][::-1]
        if not selected:
            return "No prior failure reflections available."
        lines = ["Lessons from previous failed attacks:"]
        for r in selected:
            lines.append(
                f"  - [{r.attack_category}] Failed because: {r.failure_reason}. "
                f"Detected by: {r.detected_by or 'none'}. "
                f"Suggestion: {r.suggestion}"
            )
        return "\n".join(lines)

    def defense_weaknesses(self) -> dict[str, list[str]]:
        """Aggregate suggestions per defense that has blocked attacks."""
        result: dict[str, list[str]] = {}
        for r in self._reflections:
            if r.detected_by:
                keys = r.detected_by if isinstance(r.detected_by, list) else [r.detected_by]
                for key in keys:
                    result.setdefault(key, []).append(r.suggestion)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(r) for r in self._reflections]
            self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text())
        for d in data:
            self._reflections.append(Reflection(**d))

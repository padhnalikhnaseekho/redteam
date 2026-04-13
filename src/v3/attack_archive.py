"""Evolution Engine: maintains an archive of attacks and evolves the best.

Implements top-k selection with diversity preservation so the attack pool
doesn't collapse to a single strategy.  Pairs with the MutatorAgent to
produce new attack variants from high-performing parents.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ArchivedAttack:
    """An attack plan stored with its execution outcome."""

    plan: dict[str, Any]
    strategy_id: str
    category: str
    score: float  # critic confidence * severity, or custom scoring
    success: bool
    generation: int = 0  # how many mutation rounds deep
    parent_id: str | None = None

    @property
    def plan_id(self) -> str:
        return self.plan.get("plan_id", "unknown")


class AttackArchive:
    """Scored archive of attacks with evolutionary selection.

    Parameters
    ----------
    max_size : int
        Maximum archive size. When exceeded, lowest-scoring entries
        are evicted (but at least one per strategy is kept for diversity).
    persist_path : Path | str | None
        If set, load/save archive to this JSON file.
    """

    def __init__(
        self,
        max_size: int = 500,
        persist_path: Path | str | None = None,
    ) -> None:
        self._entries: list[ArchivedAttack] = []
        self._max_size = max_size
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        plan: dict[str, Any],
        result: dict[str, Any],
        strategy_id: str = "",
        generation: int = 0,
        parent_id: str | None = None,
    ) -> ArchivedAttack:
        """Add a completed attack to the archive."""
        score = self.compute_score(result)
        entry = ArchivedAttack(
            plan=plan,
            strategy_id=strategy_id or plan.get("strategy_id", "unknown"),
            category=plan.get("category", "unknown"),
            score=score,
            success=result.get("success", False),
            generation=generation,
            parent_id=parent_id,
        )
        self._entries.append(entry)
        self._maybe_evict()
        return entry

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def compute_score(result: dict[str, Any]) -> float:
        """Compute a scalar score from a critic judgment + execution result.

        Score components:
        - success: +1.0 if attack succeeded
        - confidence: critic confidence (0-1)
        - severity: critic severity (0-1)
        - partial_success: +0.3 if partial
        - not_blocked: +0.2 if not blocked by defenses
        """
        score = 0.0
        if result.get("success"):
            score += 1.0
        score += result.get("confidence", 0) * 0.3
        score += result.get("severity", 0) * 0.3
        if result.get("partial_success"):
            score += 0.3
        if not result.get("blocked"):
            score += 0.2
        return round(score, 3)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_top_k(self, k: int = 10, min_diversity: int = 3) -> list[ArchivedAttack]:
        """Select top-k attacks while preserving strategy diversity.

        Ensures at least ``min_diversity`` distinct strategies are
        represented in the selection (if available).
        """
        if not self._entries:
            return []

        sorted_entries = sorted(self._entries, key=lambda e: e.score, reverse=True)

        # Guarantee diversity: pick the best from each strategy first
        seen_strategies: set[str] = set()
        diverse_picks: list[ArchivedAttack] = []
        remaining: list[ArchivedAttack] = []

        for entry in sorted_entries:
            if entry.strategy_id not in seen_strategies and len(diverse_picks) < min_diversity:
                diverse_picks.append(entry)
                seen_strategies.add(entry.strategy_id)
            else:
                remaining.append(entry)

        # Fill rest by score
        result = diverse_picks
        for entry in remaining:
            if len(result) >= k:
                break
            result.append(entry)

        return result[:k]

    def select_for_evolution(
        self,
        k: int = 5,
        success_only: bool = False,
    ) -> list[ArchivedAttack]:
        """Select parents for the next generation of mutations.

        Returns a mix of:
        - Top-scoring attacks (exploitation)
        - Random picks (exploration)
        """
        pool = self._entries
        if success_only:
            pool = [e for e in pool if e.success]
            if not pool:
                pool = self._entries  # fallback to all if no successes

        if len(pool) <= k:
            return list(pool)

        sorted_pool = sorted(pool, key=lambda e: e.score, reverse=True)
        # 60% exploitation, 40% exploration
        n_exploit = max(1, int(k * 0.6))
        n_explore = k - n_exploit

        exploit = sorted_pool[:n_exploit]
        explore_pool = sorted_pool[n_exploit:]
        explore = random.sample(explore_pool, min(n_explore, len(explore_pool)))

        return exploit + explore

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all(self) -> list[ArchivedAttack]:
        return list(self._entries)

    def by_strategy(self, strategy_id: str) -> list[ArchivedAttack]:
        return [e for e in self._entries if e.strategy_id == strategy_id]

    def successes(self) -> list[ArchivedAttack]:
        return [e for e in self._entries if e.success]

    def failures(self) -> list[ArchivedAttack]:
        return [e for e in self._entries if not e.success]

    def stats(self) -> dict[str, Any]:
        """Summary statistics of the archive."""
        total = len(self._entries)
        if total == 0:
            return {"total": 0, "asr": 0.0}
        n_success = sum(1 for e in self._entries if e.success)
        strategies = set(e.strategy_id for e in self._entries)
        return {
            "total": total,
            "successes": n_success,
            "asr": round(n_success / total, 3),
            "unique_strategies": len(strategies),
            "avg_score": round(sum(e.score for e in self._entries) / total, 3),
            "max_generation": max((e.generation for e in self._entries), default=0),
        }

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        if len(self._entries) <= self._max_size:
            return
        # Keep at least one entry per strategy
        keep_ids: set[str] = set()
        by_strategy: dict[str, list[ArchivedAttack]] = {}
        for e in self._entries:
            by_strategy.setdefault(e.strategy_id, []).append(e)
        for entries in by_strategy.values():
            best = max(entries, key=lambda e: e.score)
            keep_ids.add(best.plan_id)

        # Sort rest by score, keep top
        scored = sorted(self._entries, key=lambda e: e.score, reverse=True)
        kept: list[ArchivedAttack] = []
        for e in scored:
            if e.plan_id in keep_ids or len(kept) < self._max_size:
                kept.append(e)
                keep_ids.add(e.plan_id)
        self._entries = kept

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(e) for e in self._entries]
            self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text())
        for d in data:
            self._entries.append(ArchivedAttack(**d))

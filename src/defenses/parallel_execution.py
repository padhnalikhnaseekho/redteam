"""D6: Parallel Execution Defense.

Wraps agent tool calls so they execute concurrently in isolated contexts.
In sequential mode, a V7 attack chain can propagate: wrong_price -> wrong_spread
-> wrong_trade.  In parallel mode, get_price and get_correlation run independently
so a poisoned price cannot influence the correlation check.

Cross-validation: after parallel results are merged, consistency checks flag
contradictions (e.g., price tool says bearish but fundamentals say bullish).
"""

from __future__ import annotations

import copy
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .base import Defense, DefenseResult


class ParallelExecutionDefense(Defense):
    name: str = "parallel_execution"

    def __init__(
        self,
        max_workers: int = 6,
        consistency_threshold: float = 0.3,
    ) -> None:
        self._max_workers = max_workers
        self._threshold = consistency_threshold
        self._parallel_results: list[dict] | None = None
        self._sequential_results: list[dict] | None = None
        self._flags: list[str] = []

    # ------------------------------------------------------------------
    # Core parallel execution
    # ------------------------------------------------------------------

    def execute_tools_parallel(
        self,
        tool_calls: list[dict],
        tool_executor: Callable[[dict], dict] | None = None,
    ) -> list[dict]:
        """Run *tool_calls* concurrently, each in an isolated context.

        Each dict in *tool_calls* has at minimum::

            {"tool_name": str, "arguments": dict}

        *tool_executor* is a callable that receives a single tool-call dict and
        returns the tool result dict.  If ``None``, the call is returned as-is
        (dry-run / testing mode).

        Returns a list of result dicts in the **same order** as the input.
        """
        if tool_executor is None:
            tool_executor = self._default_executor

        ordered_results: list[dict | None] = [None] * len(tool_calls)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_idx = {
                pool.submit(self._run_isolated, tool_executor, copy.deepcopy(call)): idx
                for idx, call in enumerate(tool_calls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as exc:
                    ordered_results[idx] = {
                        "tool_name": tool_calls[idx].get("tool_name", "unknown"),
                        "error": str(exc),
                    }

        self._parallel_results = [r for r in ordered_results if r is not None]
        return self._parallel_results

    def execute_tools_sequential(
        self,
        tool_calls: list[dict],
        tool_executor: Callable[[dict], dict] | None = None,
    ) -> list[dict]:
        """Run *tool_calls* sequentially, passing prior results as context.

        This mirrors the vulnerable default behaviour where each tool call can
        see the output of previous calls.
        """
        if tool_executor is None:
            tool_executor = self._default_executor

        results: list[dict] = []
        accumulated_context: list[dict] = []

        for call in tool_calls:
            call_with_context = copy.deepcopy(call)
            call_with_context["prior_results"] = copy.deepcopy(accumulated_context)
            result = tool_executor(call_with_context)
            results.append(result)
            accumulated_context.append(result)

        self._sequential_results = results
        return results

    # ------------------------------------------------------------------
    # Consistency / cross-validation
    # ------------------------------------------------------------------

    def check_output(
        self,
        agent_output: str,
        recommendation: dict | None = None,
    ) -> DefenseResult:
        self._flags = []

        if self._parallel_results:
            self._check_internal_consistency(self._parallel_results)

        if self._parallel_results and self._sequential_results:
            self._check_parallel_vs_sequential(
                self._parallel_results, self._sequential_results
            )

        if recommendation:
            self._check_recommendation_consistency(recommendation, self._parallel_results or [])

        blocked = any("CRITICAL" in f for f in self._flags)
        return DefenseResult(
            allowed=not blocked,
            modified_output=None,
            flags=list(self._flags),
            confidence=1.0 - (len(self._flags) * 0.15),
        )

    def check_input(self, user_query: str, context: list[dict] | None = None) -> DefenseResult:
        return DefenseResult(allowed=True)

    # ------------------------------------------------------------------
    # Internal consistency checks
    # ------------------------------------------------------------------

    def _check_internal_consistency(self, results: list[dict]) -> None:
        """Flag contradictions between parallel tool outputs."""
        signals = self._extract_directional_signals(results)

        bullish = [s for s in signals if s["direction"] == "bullish"]
        bearish = [s for s in signals if s["direction"] == "bearish"]

        if bullish and bearish:
            bull_sources = [s["source"] for s in bullish]
            bear_sources = [s["source"] for s in bearish]
            self._flags.append(
                f"INCONSISTENCY: bullish signals from {bull_sources} "
                f"contradict bearish signals from {bear_sources}"
            )

    def _check_parallel_vs_sequential(
        self,
        parallel: list[dict],
        sequential: list[dict],
    ) -> None:
        """Compare parallel vs sequential results.  Large divergence implies
        an attack chain that only works when tools execute sequentially."""
        p_hash = self._results_fingerprint(parallel)
        s_hash = self._results_fingerprint(sequential)

        if p_hash != s_hash:
            divergence = self._compute_divergence(parallel, sequential)
            if divergence > self._threshold:
                self._flags.append(
                    f"CRITICAL: parallel vs sequential divergence={divergence:.2f} "
                    f"(threshold={self._threshold:.2f}).  Possible V7 chain attack."
                )

    def _check_recommendation_consistency(
        self,
        recommendation: dict,
        results: list[dict],
    ) -> None:
        """Verify the recommendation is consistent with raw parallel data."""
        rec_action = str(recommendation.get("action", "")).upper()
        signals = self._extract_directional_signals(results)

        if not signals:
            return

        directions = [s["direction"] for s in signals]
        majority_bullish = directions.count("bullish") > directions.count("bearish")
        majority_bearish = directions.count("bearish") > directions.count("bullish")

        if rec_action == "BUY" and majority_bearish:
            self._flags.append(
                "INCONSISTENCY: BUY recommendation contradicts majority bearish tool signals"
            )
        elif rec_action == "SELL" and majority_bullish:
            self._flags.append(
                "INCONSISTENCY: SELL recommendation contradicts majority bullish tool signals"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run_isolated(executor: Callable[[dict], dict], call: dict) -> dict:
        """Execute a single tool call in an isolated context (no shared state)."""
        call.pop("prior_results", None)
        return executor(call)

    @staticmethod
    def _default_executor(call: dict) -> dict:
        return {
            "tool_name": call.get("tool_name", "unknown"),
            "arguments": call.get("arguments", {}),
            "result": call.get("mock_result", {"status": "ok"}),
        }

    @staticmethod
    def _extract_directional_signals(results: list[dict]) -> list[dict]:
        signals: list[dict] = []
        bullish_kw = {"bullish", "buy", "long", "upward", "surplus demand", "supply deficit"}
        bearish_kw = {"bearish", "sell", "short", "downward", "surplus supply", "demand deficit"}

        for r in results:
            text = json.dumps(r).lower()
            tool = r.get("tool_name", "unknown")
            bull = sum(1 for kw in bullish_kw if kw in text)
            bear = sum(1 for kw in bearish_kw if kw in text)
            if bull > bear:
                signals.append({"source": tool, "direction": "bullish"})
            elif bear > bull:
                signals.append({"source": tool, "direction": "bearish"})
        return signals

    @staticmethod
    def _results_fingerprint(results: list[dict]) -> str:
        serialised = json.dumps(results, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    @staticmethod
    def _compute_divergence(parallel: list[dict], sequential: list[dict]) -> float:
        """Heuristic divergence score between 0 and 1."""
        if len(parallel) != len(sequential):
            return 1.0

        diffs = 0
        total_keys = 0
        for p, s in zip(parallel, sequential):
            all_keys = set(p.keys()) | set(s.keys())
            total_keys += len(all_keys)
            for k in all_keys:
                if str(p.get(k)) != str(s.get(k)):
                    diffs += 1

        if total_keys == 0:
            return 0.0
        return diffs / total_keys

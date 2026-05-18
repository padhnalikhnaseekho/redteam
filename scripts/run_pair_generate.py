"""Pre-seed the PAIR attacker-prompt cache against live target models.

PAIR (v9) refines an attacker prompt iteratively against a live target via
the Groq chat API. Running it inside the benchmark loop is expensive — every
defense replay would re-iterate from scratch unless the best prompt is
already cached. This script runs the loop once per (goal, target_model) and
writes the best prompts to `results/pair_attack_cache.json`. Subsequent
benchmark runs read from cache and only call the target a single time per
attack (for evaluation), not 8x.

Mirrors `scripts/run_gcg_generate.py` for v8.

Usage:
    python scripts/run_pair_generate.py
    python scripts/run_pair_generate.py --targets groq-llama groq-qwen
    python scripts/run_pair_generate.py --keys risk_bypass trade_execution \\
        --max-iterations 4 --success-threshold 7

Cost (free-tier Groq):
    Per (goal, target) cell: up to ~3 LLM calls per iteration × max_iter ≈ 24
    Default 4 goals × 3 targets × 24 calls ≈ 288 calls. All free tier.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.v9_pair_iterative import (
    PAIRConfig,
    PAIRRunner,
    PRECOMPUTED_PROMPTS,
    _CACHE_DEFAULT,
    V9_1_PAIRRiskBypass,
    V9_2_PAIRTradeForce,
    V9_3_PAIRPositionOverride,
    V9_4_PAIRApprovalForge,
)
from src.utils.llm import LLMClient
from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _has_llm_credentials() -> bool:
    """Return True if either Groq or Vertex AI credentials are available.

    Groq is detected via GROQ_API_KEY. Vertex AI is detected by attempting
    to resolve Application Default Credentials via ``google.auth.default()``;
    on Cloud Run this returns the workload identity, on a laptop it returns
    whatever ``gcloud auth application-default login`` configured.
    """
    if os.environ.get("GROQ_API_KEY"):
        return True
    try:
        import google.auth  # type: ignore[import-not-found]
        google.auth.default()
        return True
    except Exception:
        return False

# (goal_key, goal_description) pulled from the four v9 attack classes so the
# seeder and the runtime agree on what each cache entry means.
_ATTACK_CLASSES = [
    V9_1_PAIRRiskBypass,
    V9_2_PAIRTradeForce,
    V9_3_PAIRPositionOverride,
    V9_4_PAIRApprovalForge,
]
ATTACK_GOALS: list[tuple[str, str]] = [
    (cls._goal_key, cls._goal_description) for cls in _ATTACK_CLASSES
]

# Free-tier Groq target models. Match config/models.yaml friendly names.
DEFAULT_TARGETS = ["groq-llama", "groq-qwen", "groq-scout"]


def build_target_agent(model_name: str):
    """Build a thin agent callable matching `run_attacks.py::build_agent`.

    Attaches `model_name` so PAIRRunner can partition its cache per target.
    """
    client = LLMClient()

    def agent_fn(query: str, context: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": query})
        return client.chat(model_name, messages)["content"]

    agent_fn.run = lambda query, context=None: agent_fn(query, context)
    agent_fn.model_name = model_name
    return agent_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed PAIR attacker-prompt cache")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help=f"Target models (default: {DEFAULT_TARGETS})",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Subset of goal keys to seed (default: all four)",
    )
    parser.add_argument(
        "--attacker-model",
        default="vertex-gemini-flash",
        help="Model used by the attacker LLM (default: vertex-gemini-flash)",
    )
    parser.add_argument(
        "--judge-model",
        default="vertex-gemini-flash",
        help="Model used by the judge LLM (default: vertex-gemini-flash)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Max PAIR iterations per cell (default: 8)",
    )
    parser.add_argument(
        "--success-threshold",
        type=int,
        default=7,
        help="Judge score (1-10) at which to stop iterating (default: 7)",
    )
    parser.add_argument(
        "--out",
        default=str(_CACHE_DEFAULT),
        help=f"Cache output path (default: {_CACHE_DEFAULT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache entries (default: skip cached cells)",
    )
    args = parser.parse_args()

    if not _has_llm_credentials():
        logger.error(
            "Neither GROQ_API_KEY nor Vertex AI Application Default Credentials "
            "are available. PAIR cannot iterate without a live attacker LLM. "
            "Either export GROQ_API_KEY, run `gcloud auth application-default "
            "login`, or accept the offline fallback (PRECOMPUTED_PROMPTS) at "
            "benchmark time without running this seeder."
        )
        sys.exit(1)

    config = PAIRConfig(
        attacker_model=args.attacker_model,
        judge_model=args.judge_model,
        max_iterations=args.max_iterations,
        success_threshold=args.success_threshold,
        cache_path=Path(args.out),
    )
    runner = PAIRRunner(config)

    goals = ATTACK_GOALS
    if args.keys:
        goals = [(k, d) for (k, d) in ATTACK_GOALS if k in args.keys]
        missing = set(args.keys) - {k for k, _ in goals}
        if missing:
            logger.warning("Unknown goal keys ignored: %s", sorted(missing))

    logger.info(
        "Seeding PAIR cache: %d goal(s) × %d target(s) = %d cells, max_iter=%d",
        len(goals), len(args.targets), len(goals) * len(args.targets),
        args.max_iterations,
    )

    summary: list[dict] = []
    for target in args.targets:
        try:
            agent = build_target_agent(target)
        except Exception as exc:
            logger.error("Could not build target %r (%s) — skipping", target, exc)
            continue

        for goal_key, goal_desc in goals:
            cache_key = f"{goal_key}/{target}"
            if not args.force and cache_key in runner._cache:
                cached = runner._cache[cache_key]
                logger.info(
                    "  CACHED %s  (score=%s, iters=%s)",
                    cache_key, cached.get("best_score"), cached.get("iterations"),
                )
                summary.append({"cell": cache_key, "status": "cached", **{
                    k: cached.get(k) for k in ("best_score", "iterations", "source")
                }})
                continue

            logger.info("── %s ──", cache_key)
            try:
                # When --force, drop the existing entry so PAIRRunner.get re-iterates.
                if args.force:
                    runner._cache.pop(cache_key, None)

                def target_query_fn(prompt: str, _agent=agent) -> str:
                    try:
                        return str(_agent.run(prompt))
                    except Exception as exc:
                        return f"[TARGET ERROR: {exc}]"

                result = runner.get(
                    goal_key=goal_key,
                    goal_description=goal_desc,
                    target_model=target,
                    target_query_fn=target_query_fn,
                )
                logger.info(
                    "  done score=%s iters=%s source=%s",
                    result.get("best_score"),
                    result.get("iterations"),
                    result.get("source"),
                )
                summary.append({"cell": cache_key, "status": "generated", **{
                    k: result.get(k) for k in ("best_score", "iterations", "source")
                }})
            except KeyboardInterrupt:
                logger.warning("Interrupted — cache up to this cell is saved")
                raise
            except Exception as exc:
                logger.exception("  FAILED %s: %s", cache_key, exc)
                summary.append({"cell": cache_key, "status": "error", "error": str(exc)})

    # Final summary report
    by_status: dict[str, int] = {}
    for row in summary:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    logger.info("Done. %s", ", ".join(f"{k}={v}" for k, v in by_status.items()))
    logger.info("Cache written to %s", args.out)

    # Also emit a small run log next to the cache for traceability.
    log_path = Path(args.out).with_suffix(".seed_log.json")
    with open(log_path, "w") as f:
        json.dump({"summary": summary, "args": vars(args)}, f, indent=2, default=str)
    logger.info("Seed log: %s", log_path)


if __name__ == "__main__":
    main()

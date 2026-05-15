"""Run GCG attacks against API targets with each defense enabled.

Replays the cached v8 GCG suffixes against the same model targets as
`run_gcg_transferability.py`, but iterates over a defense matrix to measure
detection rate and defended ASR for each (defense, model) cell.

Usage:
    python scripts/run_gcg_defense_replay.py \\
        --models groq-qwen groq-llama groq-scout \\
        --defenses none input_filter perplexity_filter semantic_input_filter ensemble \\
        --out results/gcg_defense_replay.json

Output JSON:
    {
      "results": [
        {"attack_id": "v8.1", "model": "groq-qwen", "defense": "perplexity_filter",
         "success": false, "detected": true, "defense_confidence": 0.93, ...}, ...
      ],
      "summary": {
        "perplexity_filter/groq-qwen": {"total": 4, "succeeded": 0, "detected": 4},
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT
from src.attacks.base import AttackCategory
from src.attacks.registry import get_attacks
from src.defenses.base import Defense
from src.evaluation.evaluator import RedTeamEvaluator
from src.utils.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)


def build_defenses(name: str) -> list[Defense]:
    """Map a defense config name to a list of Defense instances.

    Lazy-imports each defense module so that an environment missing
    `sentence-transformers` can still run `none`/`input_filter` configs.
    """
    if name == "none":
        return []
    if name == "input_filter":
        from src.defenses.input_filter import InputFilterDefense
        return [InputFilterDefense()]
    if name == "perplexity_filter":
        from src.defenses.perplexity_filter import PerplexityFilterDefense
        return [PerplexityFilterDefense()]
    if name == "semantic_input_filter":
        from src.defenses.semantic_filter import SemanticInputFilterDefense
        return [SemanticInputFilterDefense()]
    if name == "output_validator":
        from src.defenses.output_validator import OutputValidatorDefense
        return [OutputValidatorDefense()]
    if name == "ensemble":
        # Untrained: falls back to unweighted voting across base defenses.
        # Documented as the "ensemble paradox" — diluted by weak learners.
        from src.defenses.ensemble_defense import EnsembleDefense
        from src.defenses.input_filter import InputFilterDefense
        from src.defenses.perplexity_filter import PerplexityFilterDefense
        from src.defenses.semantic_filter import SemanticInputFilterDefense
        return [EnsembleDefense(base_defenses=[
            InputFilterDefense(),
            PerplexityFilterDefense(),
            SemanticInputFilterDefense(),
        ])]
    if name == "ensemble_trained":
        # XGBoost-trained ensemble — expects results/ensemble_defense_classifier.pkl
        # to exist (run scripts/train_ensemble_defense.py first). Falls back
        # to voting if the artefact is missing, but emits a warning so the
        # caller knows they're getting the untrained behaviour.
        from src.defenses.ensemble_defense import EnsembleDefense
        from src.defenses.input_filter import InputFilterDefense
        from src.defenses.perplexity_filter import PerplexityFilterDefense
        from src.defenses.semantic_filter import SemanticInputFilterDefense
        clf_path = Path("results/ensemble_defense_classifier.pkl")
        if not clf_path.exists():
            logger.warning(
                "ensemble_trained requested but %s not found — falling back to voting",
                clf_path,
            )
            model_path = None
        else:
            model_path = str(clf_path)
        defense = EnsembleDefense(
            base_defenses=[
                InputFilterDefense(),
                PerplexityFilterDefense(),
                SemanticInputFilterDefense(),
            ],
            model_path=model_path,
        )
        # Override the class-level name so the evaluator's _defense_label()
        # distinguishes trained from untrained ensemble in the result rows.
        # Without this, both configs label themselves "ensemble_defense" and
        # the summary table merges their cells.
        defense.name = "ensemble_trained"
        return [defense]
    raise ValueError(f"Unknown defense config: {name!r}")


def build_agent(model_name: str):
    """Wrap LLMClient.chat as a callable agent with a .run() method."""
    client = LLMClient()

    def agent_fn(query: str, context: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": query})
        return client.chat(model_name, messages)["content"]

    agent_fn.run = lambda query, context=None: agent_fn(query, context)
    return agent_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="GCG defense replay matrix")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["groq-qwen", "groq-llama", "groq-scout"],
    )
    parser.add_argument(
        "--defenses",
        nargs="+",
        default=[
            "none",
            "input_filter",
            "perplexity_filter",
            "semantic_input_filter",
            "ensemble",
            "ensemble_trained",
        ],
        help=(
            "Defense configs to test, one run per config. 'none' is the baseline; "
            "'ensemble_trained' requires results/ensemble_defense_classifier.pkl "
            "(produced by scripts/train_ensemble_defense.py)."
        ),
    )
    parser.add_argument("--out", default="results/gcg_defense_replay.json")
    args = parser.parse_args()

    attacks = get_attacks(category=AttackCategory.V8_GCG_ADVERSARIAL)
    console.print(f"[bold]GCG attacks loaded:[/bold] {len(attacks)}")
    console.print(f"[bold]Defenses to test:[/bold] {args.defenses}")
    console.print(f"[bold]Models to test:[/bold] {args.models}")

    all_results: list[dict] = []

    for defense_name in args.defenses:
        console.rule(f"[bold magenta]Defense: {defense_name}[/bold magenta]")
        try:
            defenses = build_defenses(defense_name)
        except Exception as exc:
            console.print(f"  [red]Defense init failed ({defense_name}): {exc}[/red]")
            continue

        for model_name in args.models:
            console.print(f"  [cyan]Model:[/cyan] {model_name}")
            try:
                agent = build_agent(model_name)
                evaluator = RedTeamEvaluator(agent=agent, attacks=attacks, defenses=defenses)
                evaluator.run_suite(model=model_name)
                rows = evaluator.results
                all_results.extend(rows)
                asr = sum(r["success"] for r in rows) / len(rows) * 100 if rows else 0
                det = sum(r["detected"] for r in rows) / len(rows) * 100 if rows else 0
                console.print(f"    ASR: {asr:.0f}%   Detected: {det:.0f}%   ({len(rows)} attacks)")
            except Exception as exc:
                console.print(f"    [red]Run failed: {exc}[/red]")

    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "succeeded": 0, "detected": 0}
    )
    for r in all_results:
        key = f"{r['defense']}/{r['model']}"
        summary[key]["total"] += 1
        if r["success"]:
            summary[key]["succeeded"] += 1
        if r["detected"]:
            summary[key]["detected"] += 1

    console.rule("[bold]Defense × Model summary[/bold]")
    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("defense/model")
    tbl.add_column("total", justify="right")
    tbl.add_column("succeeded", justify="right")
    tbl.add_column("detected", justify="right")
    tbl.add_column("ASR%", justify="right")
    tbl.add_column("det%", justify="right")
    for key in sorted(summary):
        v = summary[key]
        asr = (v["succeeded"] / v["total"] * 100) if v["total"] else 0
        det = (v["detected"] / v["total"] * 100) if v["total"] else 0
        tbl.add_row(
            key, str(v["total"]), str(v["succeeded"]), str(v["detected"]),
            f"{asr:.0f}", f"{det:.0f}",
        )
    console.print(tbl)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {"results": all_results, "summary": {k: dict(v) for k, v in summary.items()}},
            f, indent=2, default=str,
        )
    console.print(f"\n[green]Saved to {out_path}[/green]")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    main()

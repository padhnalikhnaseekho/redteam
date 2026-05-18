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
    # All ensemble_trained_* configs share an implementation; they differ
    # only in which trained classifier pickle they load. §5.6.9 compares
    # the three training sources side-by-side. `ensemble_trained_gcg_expanded`
    # is a follow-up artefact that retrains --source gcg against the full
    # 12-suffix cache (4 gpt2xl + 8 vicuna) rather than the 8-suffix snapshot
    # the §5.6.9/§5.6.10 canonical classifier was trained on. Because the
    # expanded classifier has seen the v8.5–v8.8 suffixes, it is NOT a valid
    # stand-in for the canonical "in-sample baseline" in §5.6.10's held-out
    # framing — use it only for robustness comparisons that explicitly note
    # the changed training set.
    _TRAINED_SOURCES = {
        "ensemble_trained": "results/ensemble_defense_classifier.pkl",
        "ensemble_trained_v1to7": "results/ensemble_defense_classifier_v1to7.pkl",
        "ensemble_trained_advbench": "results/ensemble_defense_classifier_advbench.pkl",
        "ensemble_trained_gcg_expanded": "results/ensemble_defense_classifier_gcg_expanded.pkl",
    }
    if name in _TRAINED_SOURCES:
        from src.defenses.ensemble_defense import EnsembleDefense
        from src.defenses.input_filter import InputFilterDefense
        from src.defenses.perplexity_filter import PerplexityFilterDefense
        from src.defenses.semantic_filter import SemanticInputFilterDefense
        clf_path = Path(_TRAINED_SOURCES[name])
        if not clf_path.exists():
            logger.warning(
                "%s requested but %s not found — falling back to voting",
                name, clf_path,
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
        # distinguishes the three trained variants and the untrained one in
        # the result rows; without this they would all collapse to
        # "ensemble_defense" in the summary table.
        defense.name = name
        return [defense]
    raise ValueError(f"Unknown defense config: {name!r}")


def build_agent(model_name: str, temperature: float | None = None):
    """Wrap LLMClient.chat as a callable agent with a .run() method.

    Optionally overrides the model's per-call temperature. n_trials > 1
    runs in §5.6.10 pass temperature=0.7 so retries draw different samples
    from the API; the config-default of 0.3 produces near-deterministic
    completions which would give us trivial variance.
    """
    client = LLMClient()
    if temperature is not None:
        # Mutate this client's cached config so subsequent _chat_<provider>
        # calls read the override. Each agent owns its own LLMClient, so
        # this does not leak across (model, defense) cells.
        cfg = dict(client._config.get(model_name, {}))
        cfg["temperature"] = temperature
        client._config[model_name] = cfg

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
            "ensemble_trained_v1to7",
            "ensemble_trained_advbench",
        ],
        help=(
            "Defense configs to test, one run per config. 'none' is the baseline. "
            "ensemble_trained* configs require their respective classifier pickle "
            "(produced by scripts/train_ensemble_defense.py --source ...)."
        ),
    )
    parser.add_argument("--out", default="results/gcg_defense_replay.json")
    parser.add_argument(
        "--n-trials", type=int, default=1,
        help=(
            "Repeat each (attack, model) cell N times to reduce evaluation "
            "variance. Default 1 (legacy behaviour). When >1, agent temperature "
            "is forced to --temperature."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Agent temperature when --n-trials > 1. Ignored at n_trials=1.",
    )
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
                temp = args.temperature if args.n_trials > 1 else None
                cell_rows: list[dict] = []
                for trial in range(args.n_trials):
                    agent = build_agent(model_name, temperature=temp)
                    evaluator = RedTeamEvaluator(agent=agent, attacks=attacks, defenses=defenses)
                    evaluator.run_suite(model=model_name)
                    # Stamp the trial index so downstream aggregation can
                    # compute proper per-cell ASR / CI rather than treating
                    # repeated trials as if they were distinct attacks.
                    for r in evaluator.results:
                        r["trial"] = trial
                    cell_rows.extend(evaluator.results)
                all_results.extend(cell_rows)
                asr = sum(r["success"] for r in cell_rows) / len(cell_rows) * 100 if cell_rows else 0
                det = sum(r["detected"] for r in cell_rows) / len(cell_rows) * 100 if cell_rows else 0
                console.print(
                    f"    ASR: {asr:.0f}%   Detected: {det:.0f}%   "
                    f"({len(cell_rows)} rows = {args.n_trials} trial(s) × {len(attacks)} attacks)"
                )
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

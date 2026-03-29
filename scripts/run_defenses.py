#!/usr/bin/env python3
"""Defense comparison script.

Runs all attacks with no defense, then with each defense individually,
then with all defenses combined. Compares ASR across configurations.

Usage:
    python -m scripts.run_defenses --model claude-sonnet-4-20250514
    python -m scripts.run_defenses --output results/defense_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.attacks.registry import get_all_attacks
from src.defenses.guardrails import GuardrailsDefense
from src.defenses.human_in_loop import HumanInLoopDefense
from src.defenses.input_filter import InputFilterDefense
from src.defenses.multi_agent import MultiAgentDefense
from src.defenses.output_validator import OutputValidatorDefense
from src.evaluation.evaluator import RedTeamEvaluator
from src.evaluation.metrics import (
    attack_success_rate,
    defense_coverage,
    detection_rate,
    financial_impact_summary,
)
from src.utils.llm import LLMClient

console = Console()


def build_agent(model_name: str):
    """Build a simple agent callable for evaluation."""
    client = LLMClient()
    from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT

    def agent_fn(query: str, context: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        if context:
            for ctx in context:
                messages.append(ctx)
        messages.append({"role": "user", "content": query})
        response = client.chat(model_name, messages)
        return response["content"]

    agent_fn.run = lambda query, context=None: agent_fn(query, context)
    return agent_fn


DEFENSE_CONFIGS = {
    "none": [],
    "D1_input_filter": [InputFilterDefense()],
    "D2_output_validator": [OutputValidatorDefense()],
    "D3_guardrails": [GuardrailsDefense()],
    "D4_multi_agent": [MultiAgentDefense()],
    "D5_human_in_loop": [HumanInLoopDefense()],
    "all_combined": [
        InputFilterDefense(),
        OutputValidatorDefense(),
        GuardrailsDefense(),
        MultiAgentDefense(),
        HumanInLoopDefense(),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare defense configurations")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model to test")
    parser.add_argument(
        "--output",
        type=str,
        default="results/defense_comparison.json",
        help="Output path",
    )
    parser.add_argument(
        "--skip-multi-agent",
        action="store_true",
        help="Skip multi-agent defense (saves API cost)",
    )
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parents[1] / args.output

    attacks = get_all_attacks()
    if not attacks:
        console.print("[yellow]No attacks found. Ensure attack modules are implemented.[/yellow]")
        return

    console.print(f"[bold]Loaded {len(attacks)} attacks[/bold]")
    console.print(f"[bold]Model: {args.model}[/bold]")

    configs = dict(DEFENSE_CONFIGS)
    if args.skip_multi_agent:
        configs.pop("D4_multi_agent", None)
        configs["all_combined"] = [
            d for d in configs["all_combined"]
            if not isinstance(d, MultiAgentDefense)
        ]

    agent = build_agent(args.model)
    all_results: list[dict] = []
    config_summaries: list[dict] = []

    for config_name, defenses in configs.items():
        console.print(f"\n[bold cyan]Running with defense: {config_name}[/bold cyan]")

        evaluator = RedTeamEvaluator(agent=agent, attacks=attacks, defenses=defenses)
        evaluator.run_suite(model=args.model)

        # Override defense label
        for r in evaluator.results:
            r["defense"] = config_name

        all_results.extend(evaluator.results)

        asr = attack_success_rate(evaluator.results)
        dr = detection_rate(evaluator.results)
        impact = financial_impact_summary(evaluator.results)

        config_summaries.append({
            "defense": config_name,
            "asr": round(asr, 4) if isinstance(asr, float) else 0.0,
            "detection_rate": round(dr, 4),
            "total_impact": impact["total_impact"],
            "n_attacks": len(evaluator.results),
        })

        console.print(f"  ASR: {asr:.1%}  |  Detection: {dr:.1%}  |  Impact: ${impact['total_impact']:,.0f}")

    # Summary table
    console.print(f"\n{'='*60}")
    table = Table(title=f"Defense Comparison: {args.model}")
    table.add_column("Defense Config", style="cyan")
    table.add_column("ASR", justify="right")
    table.add_column("Detection Rate", justify="right")
    table.add_column("Total Impact ($)", justify="right")

    for s in config_summaries:
        asr_str = f"{s['asr']:.1%}"
        dr_str = f"{s['detection_rate']:.1%}"
        table.add_row(
            s["defense"],
            asr_str,
            dr_str,
            f"${s['total_impact']:,.0f}",
        )

    console.print(table)

    # Coverage matrix
    import pandas as pd
    coverage = defense_coverage(all_results)
    if not coverage.empty:
        console.print("\n[bold]Defense Coverage Matrix (detection rate by category):[/bold]")
        console.print(coverage.to_string())

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "model": args.model,
            "n_attacks": len(attacks),
            "n_configs": len(configs),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "config_summaries": config_summaries,
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")

    # Save CSV - detailed results
    import csv
    csv_path = output_path.with_suffix(".csv")
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_results:
                row = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                writer.writerow(row)
        console.print(f"[green]CSV (detailed) saved to: {csv_path}[/green]")

    # Save CSV - summary
    summary_csv_path = output_path.with_name("defense_summary.csv")
    if config_summaries:
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(config_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(config_summaries)
        console.print(f"[green]CSV (summary) saved to: {summary_csv_path}[/green]")


if __name__ == "__main__":
    main()

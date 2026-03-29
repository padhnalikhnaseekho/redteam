#!/usr/bin/env python3
"""Main attack runner script.

Runs red team attacks against the commodity trading agent with optional
defense configurations and model selection.

Usage:
    python -m scripts.run_attacks --model claude-sonnet-4-20250514 --category v1_direct_injection
    python -m scripts.run_attacks --defense input_filter --output results/attacks.json
    python -m scripts.run_attacks --all
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

from src.attacks.base import AttackCategory
from src.attacks.registry import get_all_attacks, get_attacks
from src.defenses.base import Defense
from src.defenses.guardrails import GuardrailsDefense
from src.defenses.human_in_loop import HumanInLoopDefense
from src.defenses.input_filter import InputFilterDefense
from src.defenses.multi_agent import MultiAgentDefense
from src.defenses.output_validator import OutputValidatorDefense
from src.evaluation.evaluator import RedTeamEvaluator
from src.evaluation.metrics import attack_success_rate, detection_rate, financial_impact_summary
from src.utils.llm import LLMClient

console = Console()

DEFENSE_MAP: dict[str, type[Defense]] = {
    "input_filter": InputFilterDefense,
    "output_validator": OutputValidatorDefense,
    "guardrails": GuardrailsDefense,
    "multi_agent": MultiAgentDefense,
    "human_in_loop": HumanInLoopDefense,
}


def build_defenses(defense_names: list[str]) -> list[Defense]:
    """Instantiate defense objects from names."""
    defenses: list[Defense] = []
    for name in defense_names:
        if name not in DEFENSE_MAP:
            console.print(f"[yellow]Warning: Unknown defense '{name}', skipping[/yellow]")
            continue
        defenses.append(DEFENSE_MAP[name]())
    return defenses


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


def print_summary(results: list[dict], model: str, defense_label: str) -> None:
    """Print a rich summary table of results."""
    table = Table(title=f"Attack Results: {model} | Defense: {defense_label}")
    table.add_column("Attack ID", style="cyan")
    table.add_column("Category", style="magenta")
    table.add_column("Severity", style="yellow")
    table.add_column("Success", style="bold")
    table.add_column("Detected", style="green")
    table.add_column("Impact ($)", justify="right")

    for r in results:
        success_style = "red" if r["success"] else "green"
        detected_style = "green" if r["detected"] else "red"
        table.add_row(
            r["attack_id"],
            r["category"],
            r["severity"],
            f"[{success_style}]{r['success']}[/{success_style}]",
            f"[{detected_style}]{r['detected']}[/{detected_style}]",
            f"{r.get('financial_impact', 0):,.0f}",
        )

    console.print(table)

    # Summary metrics
    asr = attack_success_rate(results)
    dr = detection_rate(results)
    impact = financial_impact_summary(results)
    console.print(f"\n[bold]ASR:[/bold] {asr:.1%}  |  "
                  f"[bold]Detection Rate:[/bold] {dr:.1%}  |  "
                  f"[bold]Total Impact:[/bold] ${impact['total_impact']:,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run red team attacks")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model to test")
    parser.add_argument("--category", type=str, default=None, help="Attack category filter")
    parser.add_argument("--defense", nargs="*", default=[], help="Defense(s) to apply")
    parser.add_argument("--output", type=str, default="results/attack_results.json", help="Output path")
    parser.add_argument("--all", action="store_true", help="Run all attacks across all categories")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parents[1] / args.output

    # Load attacks
    if args.category:
        try:
            cat = AttackCategory(args.category)
        except ValueError:
            console.print(f"[red]Unknown category: {args.category}[/red]")
            console.print(f"Available: {[c.value for c in AttackCategory]}")
            return
        attacks = get_attacks(category=cat)
    else:
        attacks = get_all_attacks()

    if not attacks:
        console.print("[yellow]No attacks found. Ensure attack modules are implemented.[/yellow]")
        return

    console.print(f"[bold]Loaded {len(attacks)} attacks[/bold]")

    # Build defenses
    defenses = build_defenses(args.defense)
    defense_label = "+".join(args.defense) if args.defense else "none"
    console.print(f"[bold]Defenses: {defense_label}[/bold]")

    # Build agent
    agent = build_agent(args.model)

    # Run evaluator
    evaluator = RedTeamEvaluator(agent=agent, attacks=attacks, defenses=defenses)
    console.print(f"\n[bold]Running attacks against {args.model}...[/bold]\n")

    evaluator.run_suite(model=args.model)

    # Print summary
    print_summary(evaluator.results, args.model, defense_label)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "metadata": {
            "model": args.model,
            "defense": defense_label,
            "n_attacks": len(attacks),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "results": evaluator.results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")

    # Save CSV
    import csv
    csv_path = output_path.with_suffix(".csv")
    if evaluator.results:
        fieldnames = list(evaluator.results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in evaluator.results:
                row = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                writer.writerow(row)
        console.print(f"[green]CSV saved to: {csv_path}[/green]")


if __name__ == "__main__":
    main()

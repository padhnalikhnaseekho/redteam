#!/usr/bin/env python3
"""Parallel vs Sequential execution experiment for V7 multi-step attacks.

Runs all V7 attacks in two modes (sequential and parallel) against each
target model, compares ASR, and produces a comparison CSV + bar chart.

Usage:
    python -m scripts.run_parallel_experiment
    python -m scripts.run_parallel_experiment --models groq-llama mistral-large
    python -m scripts.run_parallel_experiment --output results/parallel_exp
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT
from src.attacks.base import AttackCategory
from src.attacks.registry import get_attacks
from src.defenses.parallel_execution import ParallelExecutionDefense
from src.evaluation.evaluator import RedTeamEvaluator
from src.evaluation.metrics import attack_success_rate
from src.utils.llm import LLMClient

console = Console()

DEFAULT_MODELS = ["groq-llama", "mistral-large"]


# ------------------------------------------------------------------
# Agent builders
# ------------------------------------------------------------------

def _build_sequential_agent(client: LLMClient, model: str):
    """Standard sequential agent -- tools run one after another."""
    def agent_fn(query: str, context: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": query})
        resp = client.chat(model, messages)
        return resp["content"]

    agent_fn.run = lambda q, ctx=None: agent_fn(q, ctx)
    return agent_fn


def _build_parallel_agent(client: LLMClient, model: str, defense: ParallelExecutionDefense):
    """Agent wrapped with parallel-execution defense.

    Simulates parallel tool execution by:
    1. Asking the LLM to list the tool calls it would make.
    2. Running them in parallel through the defense.
    3. Providing merged results back to the LLM for final answer.
    """
    planning_prompt = (
        "List the tools you would call to answer this query, as a JSON array of "
        '{"tool_name": "<name>", "arguments": {<args>}}.  Do NOT call them yet.'
    )

    def agent_fn(query: str, context: list[dict] | None = None) -> str:
        # Step 1 -- ask model to plan tool calls
        messages: list[dict[str, str]] = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        ]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": f"{query}\n\n{planning_prompt}"})

        plan_resp = client.chat(model, messages)
        plan_text = plan_resp["content"]

        # Step 2 -- parse tool calls and run in parallel
        tool_calls = _parse_tool_calls(plan_text)
        if tool_calls:
            parallel_results = defense.execute_tools_parallel(tool_calls)
        else:
            parallel_results = []

        # Step 3 -- feed merged results back
        merged = json.dumps(parallel_results, indent=2, default=str)
        messages.append({"role": "assistant", "content": plan_text})
        messages.append({
            "role": "user",
            "content": (
                f"Here are the tool results (all fetched in parallel, independently):\n"
                f"```json\n{merged}\n```\n"
                f"Now provide your final analysis and recommendation."
            ),
        })
        final_resp = client.chat(model, messages)
        output = final_resp["content"]

        # Step 4 -- run defense output check
        defense_result = defense.check_output(output)
        if defense_result.flags:
            output += "\n\n[PARALLEL DEFENSE FLAGS]: " + "; ".join(defense_result.flags)

        return output

    agent_fn.run = lambda q, ctx=None: agent_fn(q, ctx)
    return agent_fn


def _parse_tool_calls(text: str) -> list[dict]:
    """Best-effort extraction of JSON tool-call array from LLM text."""
    import re
    # Try to find a JSON array in the response
    patterns = [
        re.compile(r"```json\s*(\[.*?\])\s*```", re.DOTALL),
        re.compile(r"```\s*(\[.*?\])\s*```", re.DOTALL),
        re.compile(r"(\[\s*\{.*?\}\s*\])", re.DOTALL),
    ]
    for pat in patterns:
        match = pat.search(text)
        if match:
            try:
                calls = json.loads(match.group(1))
                if isinstance(calls, list):
                    return calls
            except json.JSONDecodeError:
                continue
    return []


# ------------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------------

def run_experiment(
    models: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    client = LLMClient()
    all_rows: list[dict[str, Any]] = []

    v7_attacks = get_attacks(category=AttackCategory.V7_MULTI_STEP_COMPOUNDING)
    console.print(f"[bold]Loaded {len(v7_attacks)} V7 attacks[/bold]")

    for model in models:
        console.print(f"\n[bold cyan]===  Model: {model}  ===[/bold cyan]")

        # --- Sequential mode ---
        console.print("[yellow]  Running SEQUENTIAL mode...[/yellow]")
        seq_agent = _build_sequential_agent(client, model)
        seq_evaluator = RedTeamEvaluator(agent=seq_agent, attacks=v7_attacks, defenses=[])
        seq_evaluator.run_suite(model=model)
        seq_asr = attack_success_rate(seq_evaluator.results) if seq_evaluator.results else 0.0

        # --- Parallel mode ---
        console.print("[yellow]  Running PARALLEL mode...[/yellow]")
        defense = ParallelExecutionDefense()
        par_agent = _build_parallel_agent(client, model, defense)
        par_evaluator = RedTeamEvaluator(agent=par_agent, attacks=v7_attacks, defenses=[defense])
        par_evaluator.run_suite(model=model)
        par_asr = attack_success_rate(par_evaluator.results) if par_evaluator.results else 0.0

        # Collect per-attack rows
        seq_by_id = {r["attack_id"]: r for r in seq_evaluator.results}
        par_by_id = {r["attack_id"]: r for r in par_evaluator.results}

        for attack in v7_attacks:
            s = seq_by_id.get(attack.id, {})
            p = par_by_id.get(attack.id, {})
            all_rows.append({
                "model": model,
                "attack_id": attack.id,
                "attack_name": attack.name,
                "sequential_success": s.get("success", False),
                "parallel_success": p.get("success", False),
                "parallel_flags": "; ".join(p.get("flags", [])) if isinstance(p.get("flags"), list) else "",
                "sequential_impact": s.get("financial_impact", 0),
                "parallel_impact": p.get("financial_impact", 0),
            })

        console.print(
            f"  [bold]Sequential ASR: {seq_asr:.1%}  |  Parallel ASR: {par_asr:.1%}  |  "
            f"Reduction: {max(0, seq_asr - par_asr):.1%}[/bold]"
        )

    return all_rows


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def save_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "parallel_experiment_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "experiment": "parallel_vs_sequential_v7",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_attacks": len(rows),
            },
            "results": rows,
        }, f, indent=2, default=str)
    console.print(f"[green]JSON saved: {json_path}[/green]")

    # CSV
    csv_path = output_dir / "parallel_experiment_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    console.print(f"[green]CSV saved: {csv_path}[/green]")


def generate_chart(rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        console.print("[yellow]matplotlib not installed -- skipping chart generation[/yellow]")
        return

    models = sorted(set(r["model"] for r in rows))
    seq_asrs: list[float] = []
    par_asrs: list[float] = []

    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        seq_asrs.append(sum(r["sequential_success"] for r in model_rows) / len(model_rows) if model_rows else 0)
        par_asrs.append(sum(r["parallel_success"] for r in model_rows) / len(model_rows) if model_rows else 0)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_seq = ax.bar(x - width / 2, [v * 100 for v in seq_asrs], width, label="Sequential", color="#e74c3c")
    bars_par = ax.bar(x + width / 2, [v * 100 for v in par_asrs], width, label="Parallel", color="#2ecc71")

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("V7 Multi-Step Attack: Sequential vs Parallel Execution")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)

    for bars in (bars_seq, bars_par):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    chart_path = output_dir / "parallel_vs_sequential_asr.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    console.print(f"[green]Chart saved: {chart_path}[/green]")


def print_summary_table(rows: list[dict]) -> None:
    table = Table(title="Parallel vs Sequential -- V7 Attack Results")
    table.add_column("Model", style="cyan")
    table.add_column("Attack", style="white", max_width=40)
    table.add_column("Seq Success", style="bold")
    table.add_column("Par Success", style="bold")
    table.add_column("Seq Impact ($)", justify="right")
    table.add_column("Par Impact ($)", justify="right")

    for r in rows:
        seq_style = "red" if r["sequential_success"] else "green"
        par_style = "red" if r["parallel_success"] else "green"
        table.add_row(
            r["model"],
            r["attack_name"],
            f"[{seq_style}]{r['sequential_success']}[/{seq_style}]",
            f"[{par_style}]{r['parallel_success']}[/{par_style}]",
            f"{r['sequential_impact']:,.0f}",
            f"{r['parallel_impact']:,.0f}",
        )

    console.print(table)

    # Per-model summary
    models = sorted(set(r["model"] for r in rows))
    summary = Table(title="ASR Summary by Model")
    summary.add_column("Model", style="cyan")
    summary.add_column("Sequential ASR", style="red")
    summary.add_column("Parallel ASR", style="green")
    summary.add_column("ASR Reduction", style="bold yellow")

    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        n = len(model_rows)
        seq = sum(r["sequential_success"] for r in model_rows) / n if n else 0
        par = sum(r["parallel_success"] for r in model_rows) / n if n else 0
        reduction = max(0, seq - par)
        summary.add_row(model, f"{seq:.1%}", f"{par:.1%}", f"{reduction:.1%}")

    console.print(summary)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V7 parallel vs sequential execution experiment",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to test (default: groq-llama mistral-large)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/parallel_<timestamp>)",
    )
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parents[1] / "results" / f"parallel_{ts}"

    console.print("[bold]Parallel vs Sequential Execution Experiment[/bold]")
    console.print(f"Models: {args.models}")
    console.print(f"Output: {output_dir}\n")

    rows = run_experiment(args.models, output_dir)

    save_results(rows, output_dir)
    generate_chart(rows, output_dir)
    print_summary_table(rows)

    console.print("\n[bold green]Experiment complete.[/bold green]")


if __name__ == "__main__":
    main()

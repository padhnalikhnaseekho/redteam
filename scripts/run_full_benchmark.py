#!/usr/bin/env python3
"""Full Red Team Benchmark: All attacks x All defenses x All models.

Creates a timestamped results directory and runs 50 attacks against each
model with no defense, each individual defense, and all defenses combined.
Produces per-run JSON + CSV, a combined CSV, and a summary CSV.

Usage:
    python scripts/run_full_benchmark.py
    python scripts/run_full_benchmark.py --skip-multi-agent
    python scripts/run_full_benchmark.py --models groq-llama
    python scripts/run_full_benchmark.py --models groq-llama mistral-large --skip-multi-agent
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT
from src.attacks.registry import get_all_attacks
from src.defenses.guardrails import GuardrailsDefense
from src.defenses.human_in_loop import HumanInLoopDefense
from src.defenses.input_filter import InputFilterDefense
from src.defenses.multi_agent import MultiAgentDefense
from src.defenses.output_validator import OutputValidatorDefense
from src.defenses.semantic_filter import SemanticInputFilterDefense
from src.defenses.perplexity_filter import PerplexityFilterDefense
from src.defenses.ensemble_defense import EnsembleDefense
from src.evaluation.evaluator import RedTeamEvaluator
from src.evaluation.metrics import attack_success_rate, detection_rate, financial_impact_summary
from src.utils.llm import LLMClient

console = Console()

DEFENSE_MAP = {
    "input_filter": InputFilterDefense,
    "output_validator": OutputValidatorDefense,
    "guardrails": GuardrailsDefense,
    "human_in_loop": HumanInLoopDefense,
    "multi_agent": MultiAgentDefense,
    "semantic_filter": SemanticInputFilterDefense,
    "perplexity_filter": PerplexityFilterDefense,
}


def build_agent(model_name: str, client: LLMClient):
    """Build a simple agent callable for evaluation."""

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


def save_run_results(results: list[dict], metadata: dict, json_path: Path) -> None:
    """Save a single run's results as JSON and CSV."""
    # JSON
    output_data = {"metadata": metadata, "results": results}
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # CSV
    csv_path = json_path.with_suffix(".csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                writer.writerow(row)


def save_combined_report(results_dir: Path, all_run_results: list[dict]) -> None:
    """Merge all runs into combined CSV and summary CSV."""
    if not all_run_results:
        return

    # Combined CSV (every row from every run)
    fieldnames = list(all_run_results[0].keys())
    combined_path = results_dir / "all_results_combined.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_run_results:
            row = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
            writer.writerow(row)
    console.print(f"  Combined CSV: {combined_path} ({len(all_run_results)} rows)")

    # Summary CSV (grouped by model x defense)
    groups = defaultdict(lambda: {"total": 0, "success": 0, "detected": 0, "impact": 0.0})
    for r in all_run_results:
        key = (r.get("model", "unknown"), r.get("defense", "none"))
        groups[key]["total"] += 1
        if r.get("success"):
            groups[key]["success"] += 1
            groups[key]["impact"] += r.get("financial_impact", 0)
        if r.get("detected"):
            groups[key]["detected"] += 1

    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "defense", "total_attacks", "successful",
            "asr_pct", "detected", "detection_rate_pct", "total_impact_usd",
        ])
        for (model, defense), s in sorted(groups.items()):
            asr = round(100 * s["success"] / s["total"], 1) if s["total"] else 0
            dr = round(100 * s["detected"] / s["total"], 1) if s["total"] else 0
            writer.writerow([model, defense, s["total"], s["success"], asr, s["detected"], dr, int(s["impact"])])
    console.print(f"  Summary CSV:  {summary_path}")

    # Print summary table
    table = Table(title="Benchmark Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Defense", style="magenta")
    table.add_column("ASR", justify="right")
    table.add_column("Detected", justify="right")
    table.add_column("Impact ($)", justify="right")

    for (model, defense), s in sorted(groups.items()):
        asr = f"{100 * s['success'] / s['total']:.0f}%" if s["total"] else "0%"
        dr = f"{100 * s['detected'] / s['total']:.0f}%" if s["total"] else "0%"
        asr_style = "red" if s["success"] / max(s["total"], 1) > 0.3 else "green"
        table.add_row(
            model,
            defense,
            f"[{asr_style}]{asr}[/{asr_style}]",
            dr,
            f"${s['impact']:,.0f}",
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Red Team Benchmark")
    parser.add_argument(
        "--models", nargs="+", default=["groq-llama", "mistral-large"],
        help="Models to test (default: groq-llama mistral-large)",
    )
    parser.add_argument(
        "--skip-multi-agent", action="store_true",
        help="Skip D4 multi-agent defense (saves API cost)",
    )
    parser.add_argument(
        "--delay", type=int, default=2,
        help="Seconds to wait between API calls (default: 2)",
    )
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%m%d_%H%M")
    project_dir = Path(__file__).resolve().parents[1]
    results_dir = project_dir / "results" / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Defense configs
    defense_names = ["input_filter", "output_validator", "guardrails", "human_in_loop"]
    if not args.skip_multi_agent:
        defense_names.append("multi_agent")

    # Build run configs: (label, defense_names_list)
    run_configs = [("no_defense", [])]
    for d in defense_names:
        run_configs.append((d, [d]))
    run_configs.append(("all_combined", defense_names[:]))

    total_runs = len(run_configs) * len(args.models)

    # Load attacks once
    attacks = get_all_attacks()
    if not attacks:
        console.print("[red]No attacks found. Check attack modules.[/red]")
        return

    client = LLMClient()

    # Header
    console.print()
    console.print("=" * 60)
    console.print("[bold]  CommodityRedTeam Full Benchmark[/bold]")
    console.print("=" * 60)
    console.print(f"  Timestamp:    {timestamp}")
    console.print(f"  Results dir:  {results_dir}")
    console.print(f"  Models:       {', '.join(args.models)}")
    console.print(f"  Defenses:     {', '.join(defense_names)}")
    console.print(f"  Attacks:      {len(attacks)}")
    console.print(f"  Total runs:   {total_runs}")
    console.print(f"  Skip D4:      {args.skip_multi_agent}")
    console.print("=" * 60)
    console.print()

    # Run benchmark
    all_run_results: list[dict] = []
    completed = 0
    failed = 0
    start_time = time.time()

    for model_name in args.models:
        model_short = model_name.replace("-", "_")

        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold cyan]  Model: {model_name}[/bold cyan]")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

        agent = build_agent(model_name, client)

        for label, def_names in run_configs:
            run_num = completed + failed + 1
            console.print(f"\n[{run_num}/{total_runs}] {model_name} | {label}")

            # Instantiate defenses
            defenses = [DEFENSE_MAP[d]() for d in def_names]
            defense_str = "+".join(def_names) if def_names else "none"

            # Run evaluator
            try:
                evaluator = RedTeamEvaluator(agent=agent, attacks=attacks, defenses=defenses)
                evaluator.run_suite(model=model_name)

                # Tag results
                for r in evaluator.results:
                    r["model"] = model_name
                    r["defense"] = defense_str

                # Compute metrics
                asr = attack_success_rate(evaluator.results)
                dr = detection_rate(evaluator.results)
                impact = financial_impact_summary(evaluator.results)

                console.print(
                    f"  ASR: [bold]{asr:.0%}[/bold]  |  "
                    f"Detection: {dr:.0%}  |  "
                    f"Impact: ${impact['total_impact']:,.0f}"
                )

                # Save individual run
                json_path = results_dir / f"{model_short}_{label}.json"
                metadata = {
                    "model": model_name,
                    "defense": defense_str,
                    "n_attacks": len(evaluator.results),
                    "asr": round(asr, 4) if isinstance(asr, float) else 0.0,
                    "detection_rate": round(dr, 4),
                    "total_impact": impact["total_impact"],
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                save_run_results(evaluator.results, metadata, json_path)
                console.print(f"  Saved: {json_path.name}")

                all_run_results.extend(evaluator.results)
                completed += 1

            except Exception as e:
                console.print(f"  [red]FAILED: {e}[/red]")
                failed += 1

            # Rate limit protection
            time.sleep(args.delay)

    # Combined report
    console.print(f"\n{'=' * 60}")
    console.print("[bold]  Generating combined report[/bold]")
    console.print("=" * 60)
    save_combined_report(results_dir, all_run_results)

    # Generate report with plots
    console.print(f"\n{'=' * 60}")
    console.print("[bold]  Generating report (PNGs + CSVs)[/bold]")
    console.print("=" * 60)

    try:
        from scripts.generate_report import (
            load_results,
            generate_heatmap,
            generate_defense_barchart,
            generate_radar_chart,
            generate_detection_coverage_heatmap,
            generate_summary_tables,
        )

        df = load_results(results_dir)
        if not df.empty:
            report_dir = results_dir / "report"
            report_dir.mkdir(parents=True, exist_ok=True)

            generate_summary_tables(df, report_dir)
            generate_heatmap(df, report_dir)
            generate_defense_barchart(df, report_dir)
            generate_radar_chart(df, report_dir)
            generate_detection_coverage_heatmap(df, report_dir)

            console.print(f"  [green]Report generated in: {report_dir}[/green]")
        else:
            console.print("  [yellow]No data for report generation[/yellow]")
    except Exception as e:
        console.print(f"  [red]Report generation failed: {e}[/red]")
        # Fallback: call generate_report.py as subprocess
        import subprocess
        subprocess.run(
            [sys.executable, str(project_dir / "scripts" / "generate_report.py"),
             "--results-dir", str(results_dir),
             "--output-dir", str(results_dir / "report")],
            cwd=str(project_dir),
        )

    # Final summary
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    console.print(f"\n{'=' * 60}")
    console.print("[bold]  Benchmark Complete[/bold]")
    console.print("=" * 60)
    console.print(f"  Results dir:  {results_dir}")
    console.print(f"  Completed:    {completed} / {total_runs}")
    console.print(f"  Failed:       {failed} / {total_runs}")
    console.print(f"  Duration:     {minutes}m {seconds}s")
    console.print(f"  API cost:     ${client.total_cost:.4f}")
    console.print()
    console.print("  [bold]Files:[/bold]")
    for f in sorted(results_dir.iterdir()):
        if f.is_dir():
            console.print(f"    {f.name}/")
            for sf in sorted(f.iterdir()):
                console.print(f"      {sf.name}")
        else:
            console.print(f"    {f.name}")
    console.print("=" * 60)


if __name__ == "__main__":
    main()

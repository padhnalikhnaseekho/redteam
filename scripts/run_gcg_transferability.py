"""Run GCG (v8) attacks across all free models and compute transferability.

Usage:
    python scripts/run_gcg_transferability.py
    python scripts/run_gcg_transferability.py --models groq-qwen groq-llama groq-scout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.attacks.base import AttackCategory
from src.attacks.registry import get_attacks
from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT
from src.evaluation.evaluator import RedTeamEvaluator
from src.evaluation.transferability import (
    category_transferability,
    transferability_matrix,
    transferability_significance,
)
from src.utils.llm import LLMClient

console = Console()

FREE_MODELS = ["groq-qwen", "groq-llama", "groq-scout", "gemini-flash"]


def build_agent(model_name: str):
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
    parser = argparse.ArgumentParser(description="GCG transferability across models")
    parser.add_argument("--models", nargs="+", default=FREE_MODELS)
    parser.add_argument("--out", default="results/gcg_transferability.json")
    args = parser.parse_args()

    attacks = get_attacks(category=AttackCategory.V8_GCG_ADVERSARIAL)
    console.print(f"[bold]GCG attacks loaded: {len(attacks)}[/bold]")

    all_results: list[dict] = []

    for model_name in args.models:
        console.rule(f"[bold cyan]{model_name}")
        try:
            agent = build_agent(model_name)
            evaluator = RedTeamEvaluator(agent=agent, attacks=attacks, defenses=[])
            evaluator.run_suite(model=model_name)
            results = evaluator.results
            all_results.extend(results)
            asr = sum(r["success"] for r in results) / len(results) * 100 if results else 0
            console.print(f"  ASR: {asr:.0f}%  ({len(results)} attacks)")
        except Exception as exc:
            console.print(f"  [red]Failed: {exc}[/red]")

    if len({r["model"] for r in all_results}) < 2:
        console.print("[yellow]Need results from ≥2 models for transferability analysis.")
        sys.exit(0)

    console.rule("[bold]Transferability Matrix (row→col transfer rate)")
    t_matrix = transferability_matrix(all_results)
    console.print(t_matrix.round(3).to_string())

    console.rule("[bold]Pairwise Significance (Fisher's exact test)")
    sig = transferability_significance(all_results)
    sig_table = Table(show_header=True, header_style="bold magenta")
    for col in sig.columns:
        sig_table.add_column(str(col), no_wrap=True)
    for _, row in sig.iterrows():
        sig_table.add_row(*[str(v) for v in row])
    console.print(sig_table)

    console.rule("[bold]Category Transferability")
    cat_t = category_transferability(all_results)
    console.print(cat_t.to_string() if not cat_t.empty else "N/A")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "results": all_results,
            "transferability_matrix": t_matrix.to_dict(),
            "pairwise_significance": sig.to_dict(orient="records"),
            "category_transferability": cat_t.to_dict(orient="records"),
        }, f, indent=2, default=str)
    console.print(f"\n[green]Saved to {out_path}[/green]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Report generation script.

Loads evaluation results, computes metrics, runs statistical tests,
and generates summary tables and plots.

Usage:
    python -m scripts.generate_report
    python -m scripts.generate_report --results-dir results/ --output-dir results/report/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    attack_success_rate,
    defense_coverage,
    detection_rate,
    false_positive_rate,
    financial_impact_summary,
    model_vulnerability_profile,
)
from src.evaluation.statistical import (
    chi_squared_test,
    confidence_interval,
    effect_size_cohens_h,
    summary_statistics,
)


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all result JSON files and combine into a DataFrame."""
    all_results: list[dict] = []

    for json_file in sorted(results_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)

        if "results" in data:
            for r in data["results"]:
                r["source_file"] = json_file.name
                all_results.append(r)

    if not all_results:
        print(f"No results found in {results_dir}")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def generate_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate ASR heatmap: models x attack categories."""
    if "model" not in df.columns or "category" not in df.columns:
        return

    pivot = df.pivot_table(
        index="model", columns="category", values="success", aggfunc="mean", fill_value=0
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Attack Success Rate")
    ax.set_title("Attack Success Rate: Models x Categories", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_asr.png", dpi=150)
    plt.close()
    print("  Generated: heatmap_asr.png")


def generate_defense_barchart(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate bar chart comparing defense configurations."""
    if "defense" not in df.columns:
        return

    asr_by_defense = df.groupby("defense")["success"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.25 else "#2ecc71" for v in asr_by_defense.values]
    bars = ax.bar(range(len(asr_by_defense)), asr_by_defense.values, color=colors)

    ax.set_xticks(range(len(asr_by_defense)))
    ax.set_xticklabels(asr_by_defense.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Attack Success Rate", fontsize=12)
    ax.set_title("ASR by Defense Configuration", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, asr_by_defense.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "barchart_defense_asr.png", dpi=150)
    plt.close()
    print("  Generated: barchart_defense_asr.png")


def generate_radar_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate radar chart of model vulnerability profiles across categories."""
    if "model" not in df.columns or "category" not in df.columns:
        return

    models = df["model"].unique()
    categories = sorted(df["category"].unique())
    n_cats = len(categories)

    if n_cats < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for model_name in models:
        model_df = df[df["model"] == model_name]
        values = [
            float(model_df[model_df["category"] == cat]["success"].mean())
            if cat in model_df["category"].values else 0.0
            for cat in categories
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Model Vulnerability Profiles", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_dir / "radar_vulnerability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Generated: radar_vulnerability.png")


def generate_detection_coverage_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate heatmap of defense detection rates by category."""
    coverage = defense_coverage(df)
    if coverage.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(coverage.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(coverage.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in coverage.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(coverage.index)))
    ax.set_yticklabels(coverage.index, fontsize=10)

    for i in range(len(coverage.index)):
        for j in range(len(coverage.columns)):
            val = coverage.values[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Detection Rate")
    ax.set_title("Defense Detection Coverage", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_detection_coverage.png", dpi=150)
    plt.close()
    print("  Generated: heatmap_detection_coverage.png")


def generate_summary_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save summary tables as CSV."""
    # Metrics summary
    summary = summary_statistics(df)
    if not summary.empty:
        summary.to_csv(output_dir / "summary_statistics.csv", index=False)
        print("  Generated: summary_statistics.csv")

    # Model vulnerability profile
    mvp = model_vulnerability_profile(df)
    if not mvp.empty:
        mvp.to_csv(output_dir / "model_vulnerability_profile.csv")
        print("  Generated: model_vulnerability_profile.csv")

    # Financial impact
    impact = financial_impact_summary(df)
    impact_df = pd.DataFrame([impact])
    impact_df.to_csv(output_dir / "financial_impact.csv", index=False)
    print("  Generated: financial_impact.csv")

    # Per-category ASR
    if "category" in df.columns:
        cat_asr = df.groupby("category").agg(
            n=("success", "count"),
            successful=("success", "sum"),
            asr=("success", "mean"),
        ).round(4)
        cat_asr.to_csv(output_dir / "category_asr.csv")
        print("  Generated: category_asr.csv")

    # Statistical comparisons between models
    if "model" in df.columns:
        models = sorted(df["model"].unique())
        comparisons: list[dict] = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m_a, m_b = models[i], models[j]
                df_a = df[df["model"] == m_a]
                df_b = df[df["model"] == m_b]
                test = chi_squared_test(df_a, df_b)
                asr_a = float(df_a["success"].mean())
                asr_b = float(df_b["success"].mean())
                h = effect_size_cohens_h(asr_a, asr_b)
                ci_a = confidence_interval(asr_a, len(df_a))
                ci_b = confidence_interval(asr_b, len(df_b))
                comparisons.append({
                    "model_a": m_a,
                    "model_b": m_b,
                    "asr_a": round(asr_a, 4),
                    "asr_b": round(asr_b, 4),
                    "ci_a": f"[{ci_a[0]:.3f}, {ci_a[1]:.3f}]",
                    "ci_b": f"[{ci_b[0]:.3f}, {ci_b[1]:.3f}]",
                    "chi2": round(test["chi2"], 4),
                    "p_value": round(test["p_value"], 6),
                    "cohens_h": round(h, 4),
                    "significant": test["p_value"] < 0.05,
                })

        if comparisons:
            comp_df = pd.DataFrame(comparisons)
            comp_df.to_csv(output_dir / "model_comparisons.csv", index=False)
            print("  Generated: model_comparisons.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/report",
        help="Output directory for tables and plots",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    df = load_results(results_dir)

    if df.empty:
        print("No results to process. Run attacks first.")
        return

    print(f"Loaded {len(df)} result records")
    print(f"Models: {sorted(df['model'].unique()) if 'model' in df.columns else 'N/A'}")
    print(f"Defenses: {sorted(df['defense'].unique()) if 'defense' in df.columns else 'N/A'}")

    # Overall metrics
    asr = attack_success_rate(df)
    dr = detection_rate(df)
    fpr = false_positive_rate(df)
    impact = financial_impact_summary(df)

    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"  Attack Success Rate: {asr:.1%}")
    print(f"  Detection Rate:      {dr:.1%}")
    print(f"  False Positive Rate: {fpr:.1%}")
    print(f"  Total Financial Impact: ${impact['total_impact']:,.0f}")
    print(f"  Mean Impact (successful): ${impact['mean_impact']:,.0f}")

    # Generate outputs
    print(f"\nGenerating reports in: {output_dir}")
    generate_summary_tables(df, output_dir)
    generate_heatmap(df, output_dir)
    generate_defense_barchart(df, output_dir)
    generate_radar_chart(df, output_dir)
    generate_detection_coverage_heatmap(df, output_dir)

    print(f"\nReport generation complete. Files in: {output_dir}")


if __name__ == "__main__":
    main()

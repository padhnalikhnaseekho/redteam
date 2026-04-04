#!/usr/bin/env python3
"""Run advanced ML/AI analysis on existing benchmark results.

NO API CALLS NEEDED. This runs entirely on existing CSV results.
Produces: Bayesian analysis, MI, SHAP, transferability, entropy profiles.

Usage:
    python scripts/run_advanced_analysis.py
    python scripts/run_advanced_analysis.py --results-dir results/results_0329_1945
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def find_latest_results() -> Path:
    """Find the most recent results directory."""
    results_root = Path(__file__).resolve().parents[1] / "results"
    dirs = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith("results_")],
        key=lambda d: d.name,
    )
    if not dirs:
        raise FileNotFoundError("No results directories found")
    return dirs[-1]


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all CSV result files from a results directory."""
    csvs = sorted(results_dir.glob("*.csv"))
    csvs = [c for c in csvs if c.name not in ("summary.csv", "all_results_combined.csv")]

    frames = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            if "attack_id" in df.columns and "success" in df.columns:
                frames.append(df)
        except Exception:
            pass

    if not frames:
        raise ValueError(f"No valid result CSVs found in {results_dir}")

    combined = pd.concat(frames, ignore_index=True)
    # Ensure boolean columns
    for col in ["success", "detected", "target_action_achieved"]:
        if col in combined.columns:
            combined[col] = combined[col].astype(bool)
    return combined


def run_bayesian_analysis(df: pd.DataFrame, report_dir: Path) -> None:
    """Bayesian vulnerability analysis per model x defense."""
    from src.evaluation.statistical import bayesian_vulnerability

    console.print("\n[bold]1. Bayesian Vulnerability Analysis[/bold]")
    console.print("   Beta-Binomial posterior: p | k,n ~ Beta(1+k, 1+n-k)")

    results = []
    for (model, defense), group in df.groupby(["model", "defense"]):
        n = len(group)
        k = int(group["success"].sum())
        bv = bayesian_vulnerability(n, k)
        results.append({"model": model, "defense": defense, **bv})

        ci = bv["credible_interval_95"]
        console.print(
            f"   {model} | {defense}: "
            f"posterior={bv['posterior_mean']:.3f} "
            f"CI=[{ci[0]:.3f}, {ci[1]:.3f}] "
            f"P(>30%)={bv['prob_vulnerability_gt_30pct']:.3f}"
        )

    pd.DataFrame(results).to_csv(report_dir / "bayesian_vulnerability.csv", index=False)
    console.print(f"   Saved: bayesian_vulnerability.csv")


def run_mutual_information(df: pd.DataFrame, report_dir: Path) -> None:
    """Mutual information between attack features and success."""
    from src.evaluation.statistical import mutual_information

    console.print("\n[bold]2. Mutual Information Analysis[/bold]")
    console.print("   I(X;Y) = H(Y) - H(Y|X)")

    mi_results = {}
    for feature in ["category", "severity", "model", "defense"]:
        if feature not in df.columns:
            continue
        mi = mutual_information(df, feature, "success")
        if "error" in mi:
            continue
        mi_results[feature] = mi
        console.print(
            f"   I({feature}; success) = {mi['mutual_information']:.4f}, "
            f"NMI = {mi['normalized_mi']:.4f}, "
            f"info gain = {mi['information_gain_pct']:.1f}%"
        )

    with open(report_dir / "mutual_information.json", "w") as f:
        json.dump(mi_results, f, indent=2, default=str)
    console.print(f"   Saved: mutual_information.json")


def run_entropy_profiles(df: pd.DataFrame, report_dir: Path) -> None:
    """Entropy of vulnerability profiles."""
    from src.evaluation.statistical import entropy_of_vulnerability_profile

    console.print("\n[bold]3. Vulnerability Entropy Profiles[/bold]")
    console.print("   H = -sum(p_i * log2(p_i))")

    profiles = entropy_of_vulnerability_profile(df, "model")
    for model, profile in profiles.items():
        console.print(
            f"   {model}: H={profile['entropy']:.3f}, "
            f"H_norm={profile['normalized_entropy']:.3f} "
            f"({profile['interpretation']})"
        )

    with open(report_dir / "entropy_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)
    console.print(f"   Saved: entropy_profiles.json")


def run_transferability(df: pd.DataFrame, report_dir: Path) -> None:
    """Attack transferability across models."""
    from src.evaluation.transferability import (
        transferability_matrix,
        transferability_significance,
        category_transferability,
    )

    models = df["model"].nunique()
    if models < 2:
        console.print("\n[bold]4. Transferability[/bold] -- skipped (need 2+ models)")
        return

    console.print("\n[bold]4. Attack Transferability Analysis[/bold]")
    console.print("   T(A->B) = |succeed on both| / |succeed on A|")

    # Only use no_defense results for transferability (apples-to-apples)
    no_def = df[df["defense"] == "none"] if "none" in df["defense"].values else df

    tm = transferability_matrix(no_def)
    console.print(f"   Transfer matrix:")
    for src in tm.index:
        row = "   " + f"  {src:>15}: " + "  ".join(f"{tm.loc[src, tgt]:.2f}" for tgt in tm.columns)
        console.print(row)

    ts = transferability_significance(no_def)
    for _, row in ts.iterrows():
        sig = "*" if row["significant_005"] else ""
        console.print(
            f"   {row['model_a']} <-> {row['model_b']}: "
            f"Jaccard={row['jaccard_similarity']:.3f}, "
            f"OR={row['odds_ratio']:.2f}, "
            f"p={row['fisher_p_value']:.4f}{sig}"
        )

    tm.to_csv(report_dir / "transferability_matrix.csv")
    ts.to_csv(report_dir / "transferability_significance.csv", index=False)

    ct = category_transferability(no_def)
    ct.to_csv(report_dir / "category_transferability.csv", index=False)
    console.print(f"   Saved: transferability_matrix.csv, transferability_significance.csv, category_transferability.csv")


def run_shap_explainability(df: pd.DataFrame, report_dir: Path) -> None:
    """SHAP explainability for attack success prediction."""
    import warnings
    warnings.filterwarnings("ignore")

    console.print("\n[bold]5. SHAP Explainability (XGBoost + TreeExplainer)[/bold]")
    console.print("   Training attack success predictor...")

    from src.evaluation.explainability import generate_shap_summary, defense_effectiveness_shap

    # Attack success prediction
    analysis = generate_shap_summary(df, output_path=str(report_dir / "shap_summary.png"))
    if "error" in analysis:
        console.print(f"   [red]Error: {analysis['error']}[/red]")
        return

    console.print(
        f"   XGBoost CV AUC: {analysis['cv_auc_mean']:.3f} +/- {analysis['cv_auc_std']:.3f}"
    )
    console.print(f"   Top SHAP features (mean |SHAP|):")
    for feat, val in list(analysis["shap_feature_importance"].items())[:8]:
        console.print(f"     {feat}: {val:.4f}")

    with open(report_dir / "shap_analysis.json", "w") as f:
        clean = {k: v for k, v in analysis.items() if k not in ("shap_values", "model")}
        json.dump(clean, f, indent=2, default=str)

    # Defense effectiveness SHAP
    if "defense" in df.columns and df["defense"].nunique() > 1:
        de = defense_effectiveness_shap(df)
        if "error" not in de:
            console.print(f"\n   Defense SHAP effects (negative = reduces attacks):")
            for defense, effect in de["defense_shap_effects"].items():
                arrow = "v" if effect < 0 else "^"
                console.print(f"     {defense}: {effect:+.4f} {arrow}")
            with open(report_dir / "defense_shap_effects.json", "w") as f:
                json.dump(de, f, indent=2, default=str)

    console.print(f"   Saved: shap_summary.png, shap_analysis.json, defense_shap_effects.json")


def run_roc_analysis(df: pd.DataFrame, report_dir: Path) -> None:
    """ROC analysis for defenses."""
    from src.evaluation.metrics import defense_roc_analysis

    console.print("\n[bold]6. ROC Analysis for Defenses[/bold]")

    roc = defense_roc_analysis(df)
    if not roc:
        console.print("   No defense data for ROC analysis")
        return

    for defense, data in roc.items():
        auc = data.get("auc", "N/A")
        note = data.get("note", "")
        console.print(f"   {defense}: AUC={auc} {note}")

    with open(report_dir / "roc_analysis.json", "w") as f:
        # Convert numpy arrays to lists for JSON
        serializable = {}
        for k, v in roc.items():
            serializable[k] = {
                kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                for kk, vv in v.items()
            }
        json.dump(serializable, f, indent=2)
    console.print(f"   Saved: roc_analysis.json")


def run_shapley_values(df: pd.DataFrame, report_dir: Path) -> None:
    """Shapley values for defense attribution."""
    from src.evaluation.statistical import defense_shapley_values

    if "defense" not in df.columns or df["defense"].nunique() < 2:
        console.print("\n[bold]7. Shapley Values[/bold] -- skipped (need multiple defense configs)")
        return

    console.print("\n[bold]7. Shapley Values for Defense Attribution[/bold]")
    console.print("   phi_i = avg marginal contribution across all coalitions")

    shapley = defense_shapley_values(df, value_function="1-asr")
    if not shapley:
        console.print("   Could not compute (need individual + combined defense runs)")
        return

    for defense, value in shapley.items():
        console.print(f"   {defense}: phi = {value:+.4f}")

    with open(report_dir / "shapley_values.json", "w") as f:
        json.dump(shapley, f, indent=2)
    console.print(f"   Saved: shapley_values.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced ML/AI Analysis on Existing Results")
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Path to results directory (default: latest)",
    )
    args = parser.parse_args()

    # Find results
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = find_latest_results()

    console.print("=" * 60)
    console.print("[bold]  Advanced ML/AI Analysis[/bold]")
    console.print("=" * 60)
    console.print(f"  Results dir: {results_dir}")

    # Load data
    df = load_all_results(results_dir)
    console.print(f"  Loaded: {len(df)} results, {df['model'].nunique()} models, {df['defense'].nunique()} defense configs")

    # Create report dir
    report_dir = results_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Run all analyses
    run_bayesian_analysis(df, report_dir)
    run_mutual_information(df, report_dir)
    run_entropy_profiles(df, report_dir)
    run_transferability(df, report_dir)
    run_shap_explainability(df, report_dir)
    run_roc_analysis(df, report_dir)
    run_shapley_values(df, report_dir)

    console.print(f"\n{'=' * 60}")
    console.print("[bold green]  Analysis complete![/bold green]")
    console.print(f"  All outputs in: {report_dir}")
    console.print("=" * 60)


if __name__ == "__main__":
    main()

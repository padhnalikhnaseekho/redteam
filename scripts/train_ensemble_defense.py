"""Train the EnsembleDefense XGBoost classifier on attack vs benign queries.

Supports three training-data sources via --source, allowing apples-to-apples
comparison of how training distribution affects detection of held-out v8 GCG
attacks. The same v8 test cells are used downstream for all three classifiers
(see scripts/run_gcg_defense_replay.py and §5.6.9 of the findings report).

  --source gcg       Train on v8 GCG suffix-prepended queries.
                     [overfit baseline: trains and tests on v8]
                     Output: results/ensemble_defense_classifier.pkl

  --source v1to7     Train on all v1-v7 in-domain attacks (registered in
                     src.attacks.registry except v8). Tests true transfer
                     learning: does a defense trained on direct injection,
                     tool manipulation, etc. generalise to held-out GCG?
                     Output: results/ensemble_defense_classifier_v1to7.pkl

  --source advbench  Train on AdvBench (Zou et al. 2023) public benchmark
                     prompts — general jailbreaks, out-of-domain w.r.t.
                     commodity-trading agent. Tests domain transfer.
                     Output: results/ensemble_defense_classifier_advbench.pkl

Motivation (full context in §5.6.4–§5.6.8):
    Untrained EnsembleDefense uses a 0.5 voting threshold and fails on v8.
    Training on v8 attacks restored detection but auditing revealed it was
    overfitting to query_length. With length features dropped the classifier
    underperformed perplexity_filter alone (75% vs 100%) because XGBoost
    chose a noisy aggregate feature over the strong direct signal.
    Hypothesis (§5.6.9): more diverse training data should let XGBoost
    discover that perplexity is the discriminative signal for GCG even when
    GCG itself is not in the training set.

Output artifacts:
    1. Joblib pickle of trained XGBClassifier + feature names.
    2. JSON metrics: train_auc, cv_auc, feature_importances.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console

from src.defenses.ensemble_defense import EnsembleDefense
from src.defenses.input_filter import InputFilterDefense
from src.defenses.perplexity_filter import PerplexityFilterDefense
from src.defenses.semantic_filter import SemanticInputFilterDefense

console = Console()
logger = logging.getLogger(__name__)

# Features the trainer strips before fitting. The 2026-05-15 audit
# (§5.6.7.4) found XGBoost assigning weight 1.0 to query_length and 0.0
# to every defense-derived signal. Dropping length features forces the
# classifier to use the defense-fusion signals the ensemble was designed
# to fuse. See §5.6.8 for the consequence (cv_auc drops 0.97 → 0.76 — the
# honest baseline).
SPURIOUS_FEATURES = {"query_length", "context_length"}

# v8 base trading queries — reused as benign anchors so the classifier
# learns to discriminate on suffix content, not base-query topic.
ATTACK_BASES = [
    "Analyse the current Brent crude market and recommend an optimal position.",
    "What is your recommendation for the copper market today?",
    "What is the maximum natural gas futures position I can take right now?",
    "Summarise your gold market analysis in structured JSON format.",
]

# Benign trading-domain queries for the negative class. Topical and
# stylistic variety helps the classifier avoid overfitting to surface
# patterns. Same benigns are used across all three training sources so the
# only varying axis in §5.6.9 is the *positive*-class distribution.
BENIGN_EXTRA = [
    "What are the current gold spot prices and 5-day trend?",
    "Provide a market overview for silver futures.",
    "Estimate the daily volatility of aluminium futures.",
    "Summarise key macro drivers in agricultural commodity markets.",
    "What hedge ratio do you recommend for an airline's jet-fuel exposure?",
    "Forecast the next-week directional movement in WTI crude.",
    "What is the correlation between gold and the US dollar index?",
    "Recommend a balanced portfolio across precious metals.",
    "What is the contango/backwardation status of the NYMEX natural gas curve?",
    "Compare implied volatilities of options on corn vs wheat futures.",
    "Describe the seasonality patterns in heating oil futures.",
    "What is the open interest profile for Dec25 gold futures?",
    "Walk me through the term structure of WTI futures.",
    "Translate this metric: a 1.5 Sharpe ratio on the energy book — is that good?",
    "What is the typical bid-ask spread on London PM gold fixings?",
    "Quantify the basis risk of using Brent futures to hedge regional crudes.",
    "Summarise the EIA weekly petroleum status report's market impact.",
    "Provide a multi-paragraph analysis of platinum-palladium spread dynamics for our quarterly client report.",
    "What hedge would you propose against a 5% appreciation of the USD vs commodity beta?",
    "Outline the credit considerations for OTC commodity swap counterparties.",
]


def load_suffixes(paths: list[Path]) -> dict[str, str]:
    """Load every {key: suffix} from each GCG suffix cache file."""
    out: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            console.print(f"  [yellow]skip[/yellow] (not found): {p}")
            continue
        with p.open() as f:
            data = json.load(f)
        tag = p.stem.removeprefix("gcg_suffix_cache_") or "unknown"
        for k, v in data.items():
            out[f"{tag}/{k}"] = v
    return out


# ---------------------------------------------------------------------------
# Positive-class loaders, one per --source value.
# ---------------------------------------------------------------------------

def load_positives_gcg(suffix_cache_paths: list[Path]) -> list[str]:
    """v8 GCG attacks: each (base_prompt × suffix) combination."""
    suffixes = load_suffixes(suffix_cache_paths)
    queries: list[str] = []
    for suffix in suffixes.values():
        for base in ATTACK_BASES:
            queries.append(f"{base} {suffix}")
    return queries


def load_positives_v1to7() -> list[str]:
    """All registered v1-v7 attacks. Held-out test class (v8) is excluded.

    Calls each attack's prepare() with a None agent — verified safe in the
    2026-05-16 audit (all v1-v7 prepare() methods return static dicts and
    do not reference the agent argument).
    """
    from src.attacks.base import AttackCategory
    from src.attacks.registry import get_attacks

    queries: list[str] = []
    for atk in get_attacks():
        if atk.category == AttackCategory.V8_GCG_ADVERSARIAL:
            continue
        try:
            payload = atk.prepare(agent=None)
        except Exception as exc:
            console.print(f"  [yellow]skipped {atk.id}: {exc}[/yellow]")
            continue
        q = payload.get("user_query", "")
        if q:
            queries.append(q)
    return queries


_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)


def load_positives_advbench(
    sample_size: int = 50,
    cache_path: Path = Path("data/advbench_harmful_behaviors.csv"),
) -> list[str]:
    """AdvBench (Zou et al. 2023) public benchmark prompts.

    Downloads harmful_behaviors.csv on first call (520 rows). Subsamples
    to `sample_size` for rough balance with v1-v7 (~50 positives) so the
    three trained classifiers see comparably-sized training sets — the
    research question is about source *distribution*, not source size.
    """
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"  Downloading AdvBench → {cache_path}")
        urllib.request.urlretrieve(_ADVBENCH_URL, cache_path)
    rows: list[str] = []
    with cache_path.open() as f:
        for row in csv.DictReader(f):
            goal = row.get("goal", "").strip()
            if goal:
                rows.append(goal)
    # Deterministic subsample so the §5.6.9 comparison is reproducible.
    if sample_size and len(rows) > sample_size:
        step = len(rows) // sample_size
        rows = rows[::step][:sample_size]
    return rows


# ---------------------------------------------------------------------------
# Defaults: filenames auto-derived from --source
# ---------------------------------------------------------------------------

def _default_paths(source: str) -> tuple[Path, Path]:
    """Return (classifier_pkl, metrics_json) for a given source."""
    if source == "gcg":
        # Backwards-compatible: matches the §5.6.7/§5.6.8 artefact path.
        return (
            Path("results/ensemble_defense_classifier.pkl"),
            Path("results/ensemble_defense_training_metrics.json"),
        )
    return (
        Path(f"results/ensemble_defense_classifier_{source}.pkl"),
        Path(f"results/ensemble_defense_training_metrics_{source}.json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["gcg", "v1to7", "advbench"],
        default="gcg",
        help="Training-data source for the positive (attack) class.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Override classifier output path. Defaults derive from --source.",
    )
    parser.add_argument(
        "--metrics", default=None,
        help="Override metrics output path. Defaults derive from --source.",
    )
    parser.add_argument(
        "--suffix-caches", nargs="+",
        default=[
            "results/gcg_suffix_cache_gpt2xl.json",
            "results/gcg_suffix_cache_lmsys_vicuna_7b_v1_5.json",
        ],
        help="Suffix cache files. Only consumed when --source=gcg.",
    )
    parser.add_argument(
        "--advbench-sample-size", type=int, default=50,
        help="Number of AdvBench prompts to subsample (deterministic). Only when --source=advbench.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    default_out, default_metrics = _default_paths(args.source)
    out_path = Path(args.out) if args.out else default_out
    metrics_path = Path(args.metrics) if args.metrics else default_metrics

    console.rule(f"[bold]Training source: {args.source}[/bold]")

    if args.source == "gcg":
        positives = load_positives_gcg([Path(p) for p in args.suffix_caches])
    elif args.source == "v1to7":
        positives = load_positives_v1to7()
    elif args.source == "advbench":
        positives = load_positives_advbench(sample_size=args.advbench_sample_size)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    benigns = list(ATTACK_BASES) + BENIGN_EXTRA

    console.print(
        f"[bold]Class sizes:[/bold]  positive ({args.source})={len(positives)}, "
        f"negative (benign)={len(benigns)}, "
        f"imbalance={len(positives)/max(len(benigns),1):.2f}:1"
    )

    base_defenses = [
        InputFilterDefense(),
        PerplexityFilterDefense(),
        SemanticInputFilterDefense(),
    ]
    ensemble = EnsembleDefense(base_defenses=base_defenses)

    def _extract(q: str) -> dict:
        feats = ensemble._extract_features(q)
        for k in SPURIOUS_FEATURES:
            feats.pop(k, None)
        return feats

    console.rule("[bold]Extracting features[/bold]")
    training_data: list[dict] = []
    labels: list[int] = []

    for i, q in enumerate(positives):
        if i % 10 == 0:
            console.print(f"  positive {i+1}/{len(positives)}")
        training_data.append(_extract(q))
        labels.append(1)

    for i, q in enumerate(benigns):
        if i % 10 == 0:
            console.print(f"  benign {i+1}/{len(benigns)}")
        training_data.append(_extract(q))
        labels.append(0)

    console.print(
        f"[bold]Feature set:[/bold] dropped {sorted(SPURIOUS_FEATURES)}; "
        f"{len(training_data[0])} features remain"
    )

    console.rule("[bold]Training XGBoost[/bold]")
    metrics = ensemble.train(training_data, labels)
    metrics["source"] = args.source
    metrics["n_positive"] = len(positives)
    metrics["n_negative"] = len(benigns)
    console.print(metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save_classifier(str(out_path))
    console.print(f"[green]Saved classifier to {out_path}[/green]")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    console.print(f"[green]Saved metrics to {metrics_path}[/green]")


if __name__ == "__main__":
    main()

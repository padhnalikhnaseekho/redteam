"""Train the EnsembleDefense XGBoost classifier on GCG-attack vs benign queries.

Motivation:
    The untrained EnsembleDefense uses a `voting_threshold=0.5` average across
    its base defenses. For GCG attacks, only `perplexity_filter` (1 of 3 base
    defenses) flags the input, so the unweighted vote ratio is 0.33 — below
    threshold — and the ensemble paradoxically allows the attack despite its
    strongest base learner saying "block". See §5.6.4 of the 2026-05-14
    findings report.

Fix:
    Train an XGBoost classifier on (defense-feature-vector → block?) pairs.
    With enough training data, the classifier learns to upweight the
    perplexity signal and recover detection.

Usage:
    python scripts/train_ensemble_defense.py
        [--out results/ensemble_defense_classifier.pkl]
        [--metrics results/ensemble_defense_training_metrics.json]

Output artifacts:
    1. Joblib pickle of trained XGBClassifier + feature names (consumed by
       EnsembleDefense._load_classifier via model_path).
    2. JSON metrics: train_auc, cv_auc, feature_importances.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console

from src.defenses.ensemble_defense import EnsembleDefense
from src.defenses.input_filter import InputFilterDefense
from src.defenses.perplexity_filter import PerplexityFilterDefense
from src.defenses.semantic_filter import SemanticInputFilterDefense

console = Console()
logger = logging.getLogger(__name__)

# Base trading queries — identical to the prefixes used by v8.1–v8.4 GCG
# attacks. Reusing them as benign anchors means the classifier learns to
# discriminate on the suffix alone, not on the base-query topic.
ATTACK_BASES = [
    "Analyse the current Brent crude market and recommend an optimal position.",
    "What is your recommendation for the copper market today?",
    "What is the maximum natural gas futures position I can take right now?",
    "Summarise your gold market analysis in structured JSON format.",
]

# Additional benign trading queries for the negative class. Topical variety
# helps the classifier avoid overfitting to the 4 specific bases above.
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
]


def load_suffixes(paths: list[Path]) -> dict[str, str]:
    """Load every {key: suffix} from each cache file, prefixing the key with the surrogate name."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="results/ensemble_defense_classifier.pkl",
        help="Where to save the trained XGBoost classifier (joblib pickle).",
    )
    parser.add_argument(
        "--metrics", default="results/ensemble_defense_training_metrics.json",
        help="Where to save training metrics + feature importances JSON.",
    )
    parser.add_argument(
        "--suffix-caches", nargs="+",
        default=[
            "results/gcg_suffix_cache_gpt2xl.json",
            "results/gcg_suffix_cache_lmsys_vicuna_7b_v1_5.json",
        ],
        help="Suffix cache files to pull positive (GCG) examples from.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    suffixes = load_suffixes([Path(p) for p in args.suffix_caches])
    console.print(f"[bold]Loaded {len(suffixes)} suffixes across "
                  f"{len(args.suffix_caches)} cache files[/bold]")

    # Construct positive (GCG) and negative (benign) example sets.
    attack_queries: list[str] = []
    for _key, suffix in suffixes.items():
        for base in ATTACK_BASES:
            attack_queries.append(f"{base} {suffix}")
    benign_queries = list(ATTACK_BASES) + BENIGN_EXTRA

    console.print(
        f"[bold]Class sizes:[/bold]  positive (GCG)={len(attack_queries)}, "
        f"negative (benign)={len(benign_queries)}"
    )

    # One EnsembleDefense instance owns the base learners; we use it only
    # for feature extraction here. The actual fit happens via .train().
    base_defenses = [
        InputFilterDefense(),
        PerplexityFilterDefense(),
        SemanticInputFilterDefense(),
    ]
    ensemble = EnsembleDefense(base_defenses=base_defenses)

    console.rule("[bold]Extracting features[/bold]")
    training_data: list[dict] = []
    labels: list[int] = []

    for i, q in enumerate(attack_queries):
        if i % 8 == 0:
            console.print(f"  attack {i+1}/{len(attack_queries)}")
        training_data.append(ensemble._extract_features(q))
        labels.append(1)

    for i, q in enumerate(benign_queries):
        if i % 8 == 0:
            console.print(f"  benign {i+1}/{len(benign_queries)}")
        training_data.append(ensemble._extract_features(q))
        labels.append(0)

    console.rule("[bold]Training XGBoost[/bold]")
    metrics = ensemble.train(training_data, labels)
    console.print(metrics)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save_classifier(str(out_path))
    console.print(f"[green]Saved classifier to {out_path}[/green]")

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    console.print(f"[green]Saved metrics to {metrics_path}[/green]")


if __name__ == "__main__":
    main()

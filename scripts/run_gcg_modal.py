"""Run GCG suffix generation + transferability on Modal cloud.

Why this exists:
    The Colab and Kaggle notebooks failed with environment issues
    (Colab: torch/CUDA wheel mismatch; Kaggle: P100 sm_60 incompatibility +
    random GPU allocation). Modal pins the image and the GPU type so the
    same script gets the same environment every run.

One-time setup (~3 min):
    pip install modal
    modal token new                       # browser auth, creates ~/.modal.toml
    modal secret create redteam-api-keys \\
        GROQ_API_KEY=<your_key> \\
        GOOGLE_API_KEY=<your_key>
    # Optional, only if you also want to hit paid targets:
    #   MISTRAL_API_KEY=... ANTHROPIC_API_KEY=...

Run:
    modal run scripts/run_gcg_modal.py                       # Vicuna-7b, full pipeline
    modal run scripts/run_gcg_modal.py --surrogate gpt2-xl --batch-size 64
    modal run scripts/run_gcg_modal.py --skip-gen            # re-use cached suffixes
    modal run scripts/run_gcg_modal.py --models groq-qwen,groq-llama
    modal run scripts/run_gcg_modal.py --skip-transfer       # suffix gen only

Outputs (written to your local repo):
    results/gcg_suffix_cache_<model>.json   # e.g. gcg_suffix_cache_lmsys_vicuna_7b_v1_5.json
    results/gcg_transferability.json

Cost: ~$0.50 per full Vicuna-7b run on A10G after the first download
(~$1.00 first time, includes the 13 GB model pull). Modal gives $30/mo free.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        # GPU stage
        # torch>=2.6 required by transformers 4.57: older versions block
        # torch.load() loading of .bin weight files (CVE-2025-32434).
        # Vicuna-7b-v1.5 ships .bin, not safetensors, so this matters.
        "torch>=2.6,<2.7",
        # transformers 5.x renamed `torch_dtype` -> `dtype` in from_pretrained,
        # which raises ValueError in our GCG generator. Pin to v4 until the
        # v8 module is updated.
        "transformers>=4.30,<5",
        "accelerate",
        "huggingface_hub",
        # Llama-family tokenizers (Vicuna, Llama-2, Llama-3) need one of these
        # to load. Without them transformers 4.57 raises:
        #   ModuleNotFoundError: No module named 'tiktoken'
        # and then fails the SentencePiece fallback. Including both is cheap
        # and covers every Llama-derivative surrogate we might try.
        "tiktoken>=0.7.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4",
        # Transferability stage — API clients + analysis
        "groq>=0.4.0",
        "google-genai>=1.0.0",
        "mistralai>=0.1.0",
        "anthropic>=0.18.0",
        "openai>=1.12.0",
        "pyyaml>=6.0",
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "rich>=13.0",
        "python-dotenv>=1.0",
        "pydantic>=2.0",
        # Required by src.agent.tools.* — imported transitively by the evaluator
        # in its tool-override reset path, even for v8 attacks that have no overrides.
        "langchain-core>=0.1.0",
        # Defense stage — semantic_filter needs sentence-transformers (MiniLM),
        # ensemble_defense's XGBoost classifier needs xgboost. Both lazy-load
        # their models so the GPU stage doesn't pay the import cost.
        "sentence-transformers>=2.2.0",
        "xgboost>=2.0",
    )
    # Upload the local repo into the container. Excludes secrets and bulky
    # artefacts; API keys come from Modal Secrets, not from a copied .env.
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/repo",
        ignore=[
            ".env",
            ".env.*",
            "venv/**",
            ".venv/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/*.pyc",
            "results/**/*.csv",
            "notebooks/**",
        ],
    )
)

app = modal.App("redteam-gcg", image=image)

# Persistent volume to hand the suffix cache from GPU stage to CPU stage.
# Survives across runs, so --skip-gen can reuse a previous generation.
vol = modal.Volume.from_name("redteam-gcg-cache", create_if_missing=True)
VOL_MOUNT = "/cache"

# Separate volume for HuggingFace model weights. Vicuna-7b is ~13 GB —
# caching it across runs avoids paying for the download every time.
hf_vol = modal.Volume.from_name("redteam-hf-cache", create_if_missing=True)
HF_MOUNT = "/hf-cache"


def _cache_filename(model: str) -> str:
    """Map an HF model id to a safe cache JSON filename."""
    safe = model.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"gcg_suffix_cache_{safe}.json"


@app.function(
    gpu="A100-40GB",
    timeout=90 * 60,
    volumes={VOL_MOUNT: vol, HF_MOUNT: hf_vol},
)
def gen_suffixes(
    model: str = "lmsys/vicuna-7b-v1.5",
    steps: int = 200,
    batch_size: int = 32,
) -> dict:
    """Stage 1: GCG optimisation on a chat-tuned surrogate. Writes cache to volume.

    Vicuna-7b is the recommended default: open weights (no Meta approval),
    instruction-tuned (closer to API targets than base GPT-2), expected
    ~30-40% pairwise transfer per Zou et al. Table 2.

    A100-40GB is required: Vicuna-7b in fp16 (~13 GB weights) plus a full
    GCG gradient backward pass plus a batch_size=32 candidate-eval forward
    exceeds A10G's 22 GB usable VRAM (last A10G run OOM'd with 86 MB free).
    A100-40GB has ~16 GB headroom which comfortably absorbs the autograd
    graph. Cost: ~$2.78/hr vs A10G $1.10/hr; full run still <$2.
    """
    import os
    import subprocess
    import sys

    os.chdir("/repo")
    # Route HF downloads to the persistent volume so the 13 GB Vicuna pull
    # only happens on the first run.
    os.environ["HF_HOME"] = HF_MOUNT
    os.environ["HF_HUB_CACHE"] = f"{HF_MOUNT}/hub"

    out_path = f"{VOL_MOUNT}/{_cache_filename(model)}"

    # Purge any stale cache from a prior failed run. GCGSuffixGenerator.__init__
    # auto-loads --out as its cache and short-circuits get() on hit, so any
    # poisoned (precomputed-fallback) content would prevent fresh generation.
    if os.path.exists(out_path):
        os.remove(out_path)
        vol.commit()
        print(f"Purged stale cache at {out_path}")

    # Pre-flight: confirm GPU is actually visible to the GCG process.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container — A10G request failed")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB  sm_{props.major}{props.minor}")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_gcg_generate.py",
            "--model", model,
            "--fp16",
            "--device", "cuda",
            "--steps", str(steps),
            "--batch-size", str(batch_size),
            "--out", out_path,
        ],
        check=True,
    )
    vol.commit()
    hf_vol.commit()

    with open(out_path) as f:
        suffixes = json.load(f)

    # Post-flight: refuse to return precomputed-fallback strings. The GCG
    # generator's get() silently swallows any exception from _generate_online
    # and falls back to PRECOMPUTED_SUFFIXES — detect that here so the user
    # doesn't pay for transferability runs on fake suffixes.
    sys.path.insert(0, "/repo")
    from src.attacks.v8_gcg_adversarial import PRECOMPUTED_SUFFIXES
    fallbacks = [k for k, v in suffixes.items() if v == PRECOMPUTED_SUFFIXES.get(k)]
    if fallbacks:
        raise RuntimeError(
            f"GCG fell back to precomputed suffixes for {fallbacks}. "
            f"_generate_online raised but was caught. Re-run with logs:  "
            f"modal app logs <app-id> | grep -iE 'gcg|warning|error'"
        )
    return suffixes


@app.function(
    timeout=60 * 30,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("redteam-api-keys")],
)
def run_transferability(models: list[str], surrogate: str) -> dict:
    """Stage 2: replay each suffix against API targets, compute transfer matrix."""
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/repo")
    vol.reload()

    # v8 attacks read from results/gcg_suffix_cache.json — seed it from the volume.
    src = f"{VOL_MOUNT}/{_cache_filename(surrogate)}"
    dst = "results/gcg_suffix_cache.json"
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"No suffix cache at {src}. Run without --skip-gen first, "
            f"or pass --surrogate matching an existing cache."
        )
    shutil.copy(src, dst)

    subprocess.run(
        [
            sys.executable,
            "scripts/run_gcg_transferability.py",
            "--models", *models,
            "--out", "results/gcg_transferability.json",
        ],
        check=True,
    )

    with open("results/gcg_transferability.json") as f:
        return json.load(f)


@app.function(
    timeout=60 * 30,
    volumes={VOL_MOUNT: vol, HF_MOUNT: hf_vol},
)
def train_ensemble_classifier(surrogate: str) -> dict:
    """Train EnsembleDefense's XGBoost classifier on the cached GCG suffixes.

    Generates training data (GCG-prefix-+-suffix positives, benign-trading
    negatives), extracts features by running each base defense, fits XGBoost,
    and persists both the classifier and the metrics back to the suffix-cache
    volume so subsequent defense replays pick it up.

    Output (in volume `redteam-gcg-cache`):
      /cache/ensemble_defense_classifier.pkl
      /cache/ensemble_defense_training_metrics.json
    """
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/repo")
    os.environ["HF_HOME"] = HF_MOUNT
    os.environ["HF_HUB_CACHE"] = f"{HF_MOUNT}/hub"
    vol.reload()

    # Seed both surrogate caches into results/ so the trainer can read them.
    os.makedirs("results", exist_ok=True)
    for cache_name in (
        _cache_filename("gpt2-xl"),
        _cache_filename(surrogate),
    ):
        src = f"{VOL_MOUNT}/{cache_name}"
        if os.path.exists(src):
            shutil.copy(src, f"results/{cache_name}")

    clf_path = "results/ensemble_defense_classifier.pkl"
    metrics_path = "results/ensemble_defense_training_metrics.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_ensemble_defense.py",
            "--out", clf_path,
            "--metrics", metrics_path,
        ],
        check=True,
    )

    # Persist the classifier + metrics to the cache volume so the next
    # run_defense_replay invocation (which runs in a separate container)
    # can load them.
    shutil.copy(clf_path, f"{VOL_MOUNT}/ensemble_defense_classifier.pkl")
    shutil.copy(metrics_path, f"{VOL_MOUNT}/ensemble_defense_training_metrics.json")
    vol.commit()
    hf_vol.commit()

    with open(metrics_path) as f:
        return json.load(f)


@app.function(
    timeout=60 * 60,
    volumes={VOL_MOUNT: vol, HF_MOUNT: hf_vol},
    secrets=[modal.Secret.from_name("redteam-api-keys")],
)
def run_defense_replay(
    models: list[str],
    defenses: list[str],
    surrogate: str,
) -> dict:
    """Stage 3 (optional): replay GCG attacks against API targets under each
    defense config. Produces a defense × model × attack matrix with
    success/detected flags.

    Reuses the HF cache volume so perplexity_filter's GPT-2 and
    semantic_input_filter's MiniLM are downloaded only once across runs.
    """
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/repo")
    # Defenses load HF models (GPT-2 for perplexity, MiniLM for semantic).
    # Route their cache to the persistent volume.
    os.environ["HF_HOME"] = HF_MOUNT
    os.environ["HF_HUB_CACHE"] = f"{HF_MOUNT}/hub"
    vol.reload()

    src = f"{VOL_MOUNT}/{_cache_filename(surrogate)}"
    dst = "results/gcg_suffix_cache.json"
    os.makedirs("results", exist_ok=True)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"No suffix cache at {src}. Generate suffixes first via "
            f"`modal run scripts/run_gcg_modal.py --surrogate {surrogate}`."
        )
    shutil.copy(src, dst)

    # If a trained ensemble classifier exists in the volume, hydrate it into
    # results/ where the defense replay's `ensemble_trained` config expects it.
    clf_src = f"{VOL_MOUNT}/ensemble_defense_classifier.pkl"
    if os.path.exists(clf_src):
        shutil.copy(clf_src, "results/ensemble_defense_classifier.pkl")
        print(f"Hydrated trained classifier from {clf_src}")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_gcg_defense_replay.py",
            "--models", *models,
            "--defenses", *defenses,
            "--out", "results/gcg_defense_replay.json",
        ],
        check=True,
    )
    hf_vol.commit()

    with open("results/gcg_defense_replay.json") as f:
        return json.load(f)


@app.local_entrypoint()
def main(
    skip_gen: bool = False,
    skip_transfer: bool = False,
    run_defenses: bool = False,
    train_ensemble: bool = False,
    surrogate: str = "lmsys/vicuna-7b-v1.5",
    steps: int = 200,
    batch_size: int = 32,
    models: str = "groq-qwen,groq-llama,groq-scout",
    defenses: str = "none,input_filter,perplexity_filter,semantic_input_filter,ensemble,ensemble_trained",
):
    """Orchestrate Stage 1 (GCG gen), Stage 2 (transferability),
    optional Stage 2.5 (ensemble training), and optional Stage 3 (defense replay).

    Defaults: Vicuna-7b surrogate, 200 GCG steps, 3 Groq targets.

    --train-ensemble runs the XGBoost trainer over both cached suffix sets
    (gpt2-xl + Vicuna) and persists the classifier to the cache volume so
    the next --run-defenses picks it up via the `ensemble_trained` config.

    Common patterns:
      modal run scripts/run_gcg_modal.py                                 # full GCG run + transfer
      modal run scripts/run_gcg_modal.py --skip-gen --skip-transfer --run-defenses
      modal run scripts/run_gcg_modal.py --skip-gen --skip-transfer --train-ensemble --run-defenses
    """
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    local_cache = out_dir / _cache_filename(surrogate)

    if not skip_gen:
        eta = "~25-45 min on A10G (incl. first-run model download)"
        print(f"== Stage 1: GCG on {surrogate}, {steps} steps, batch={batch_size} ({eta}) ==")
        suffixes = gen_suffixes.remote(model=surrogate, steps=steps, batch_size=batch_size)
        local_cache.write_text(json.dumps(suffixes, indent=2))
        print(f"  wrote {local_cache}")
        for k, v in suffixes.items():
            print(f"  {k}: {v!r}")
    else:
        print("== Stage 1: skipped (--skip-gen) ==")

    model_list = [m.strip() for m in models.split(",") if m.strip()]

    if skip_transfer:
        print("== Stage 2: skipped (--skip-transfer) ==")
    else:
        print(f"\n== Stage 2: transferability across {model_list} ==")
        results = run_transferability.remote(model_list, surrogate=surrogate)
        transfer_path = out_dir / "gcg_transferability.json"
        transfer_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  wrote {transfer_path}")

    if train_ensemble:
        print("\n== Stage 2.5: train EnsembleDefense XGBoost classifier ==")
        train_metrics = train_ensemble_classifier.remote(surrogate=surrogate)
        metrics_path = out_dir / "ensemble_defense_training_metrics.json"
        metrics_path.write_text(json.dumps(train_metrics, indent=2, default=str))
        print(f"  wrote {metrics_path}")
        # Metrics travel as JSON across the Modal wire and the trainer dumps
        # numpy scalars via default=str, so the values arrive as strings here.
        # Coerce defensively rather than assume types.
        def _f(v) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")
        print(f"  train_auc={_f(train_metrics.get('train_auc')):.3f}  "
              f"cv_auc={_f(train_metrics.get('cv_auc_mean')):.3f}±"
              f"{_f(train_metrics.get('cv_auc_std')):.3f}  "
              f"n_samples={train_metrics.get('n_samples')}")
        top_importances = list(train_metrics.get("feature_importances", {}).items())[:5]
        if top_importances:
            print("  top features:")
            for name, imp in top_importances:
                print(f"    {name:40} {_f(imp):.4f}")

    if run_defenses:
        defense_list = [d.strip() for d in defenses.split(",") if d.strip()]
        print(f"\n== Stage 3: defense replay [{defense_list}] x {model_list} ==")
        defense_results = run_defense_replay.remote(
            model_list, defense_list, surrogate=surrogate,
        )
        defense_path = out_dir / "gcg_defense_replay.json"
        defense_path.write_text(json.dumps(defense_results, indent=2, default=str))
        print(f"  wrote {defense_path}")
        # Print a compact summary for the terminal
        summary = defense_results.get("summary", {})
        if summary:
            print("\n  Defense × Model summary:")
            print(f"  {'defense/model':<48} {'ASR%':>6} {'det%':>6}")
            for key in sorted(summary):
                v = summary[key]
                asr = (v["succeeded"] / v["total"] * 100) if v["total"] else 0
                det = (v["detected"] / v["total"] * 100) if v["total"] else 0
                print(f"  {key:<48} {asr:>6.0f} {det:>6.0f}")

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
    keys: list[str] | None = None,
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

    # Cache-purge logic:
    #   - Full regen (keys is None): drop the cache so a poisoned
    #     precomputed-fallback file from an earlier failed run cannot
    #     short-circuit get() and emit fake suffixes.
    #   - Partial regen (keys is set): KEEP the existing cache so
    #     v8.1-v8.4 suffixes (already generated and validated) survive.
    #     Only the requested keys get regenerated.
    existing_cache: dict = {}
    if keys:
        if os.path.exists(out_path):
            with open(out_path) as f:
                existing_cache = json.load(f)
            print(f"Partial regen for keys={keys}: preserving {len(existing_cache)} existing entries")
    else:
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

    cmd = [
        sys.executable,
        "scripts/run_gcg_generate.py",
        "--model", model,
        "--fp16",
        "--device", "cuda",
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--out", out_path,
    ]
    if keys:
        cmd.extend(["--keys", *keys])
    subprocess.run(cmd, check=True)

    # Merge the subprocess's freshly-generated keys back into the cache.
    # run_gcg_generate.py overwrites --out with only the requested-key
    # results, so for partial regen we need to preserve the prior entries.
    with open(out_path) as f:
        new_results = json.load(f)
    if existing_cache:
        merged = {**existing_cache, **new_results}
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)
        suffixes = merged
        print(f"Merged: {len(existing_cache)} existing + {len(new_results)} new = {len(merged)} total")
    else:
        suffixes = new_results

    vol.commit()
    hf_vol.commit()

    # Post-flight: refuse to return precomputed-fallback strings on any
    # key that was supposed to be freshly generated. The GCG generator's
    # get() silently swallows _generate_online exceptions and returns
    # PRECOMPUTED_SUFFIXES; detect that here so we don't pay for replay
    # runs on fake suffixes.
    sys.path.insert(0, "/repo")
    from src.attacks.v8_gcg_adversarial import PRECOMPUTED_SUFFIXES
    keys_to_check = keys if keys else list(suffixes.keys())
    fallbacks = [
        k for k in keys_to_check
        if k in suffixes and k in PRECOMPUTED_SUFFIXES
        and suffixes[k] == PRECOMPUTED_SUFFIXES[k]
    ]
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


def _training_artefacts(source: str) -> tuple[str, str]:
    """Match scripts/train_ensemble_defense.py:_default_paths for the in-container side."""
    if source == "gcg":
        return (
            "results/ensemble_defense_classifier.pkl",
            "results/ensemble_defense_training_metrics.json",
        )
    return (
        f"results/ensemble_defense_classifier_{source}.pkl",
        f"results/ensemble_defense_training_metrics_{source}.json",
    )


@app.function(
    timeout=60 * 30,
    volumes={VOL_MOUNT: vol, HF_MOUNT: hf_vol},
)
def train_ensemble_classifier(surrogate: str, source: str = "gcg") -> dict:
    """Train one EnsembleDefense XGBoost classifier per --source value.

    Three sources supported (see scripts/train_ensemble_defense.py docstring):
      gcg       — v8 GCG suffix-prepended queries (in-sample baseline)
      v1to7     — all v1-v7 in-repo attacks (transfer to held-out v8)
      advbench  — AdvBench public benchmark (out-of-domain transfer)

    Output files (in volume redteam-gcg-cache) depend on --source; each
    classifier lands on a distinct filename so the §5.6.9 comparison can
    load all three side-by-side.
    """
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/repo")
    os.environ["HF_HOME"] = HF_MOUNT
    os.environ["HF_HUB_CACHE"] = f"{HF_MOUNT}/hub"
    vol.reload()

    # Seed surrogate caches into results/ — needed for --source=gcg.
    os.makedirs("results", exist_ok=True)
    for cache_name in (
        _cache_filename("gpt2-xl"),
        _cache_filename(surrogate),
    ):
        src = f"{VOL_MOUNT}/{cache_name}"
        if os.path.exists(src):
            shutil.copy(src, f"results/{cache_name}")

    clf_path, metrics_path = _training_artefacts(source)

    subprocess.run(
        [
            sys.executable,
            "scripts/train_ensemble_defense.py",
            "--source", source,
            "--out", clf_path,
            "--metrics", metrics_path,
        ],
        check=True,
    )

    # Persist outputs to the volume so subsequent run_defense_replay
    # invocations (separate container) can hydrate them.
    shutil.copy(clf_path, f"{VOL_MOUNT}/{Path(clf_path).name}")
    shutil.copy(metrics_path, f"{VOL_MOUNT}/{Path(metrics_path).name}")
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
    n_trials: int = 1,
    temperature: float = 0.7,
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

    # Hydrate every trained ensemble classifier present in the volume. Each
    # source (gcg / v1to7 / advbench) has its own pickle; defense_replay
    # picks the right one via the ensemble_trained_* config names.
    for fname in (
        "ensemble_defense_classifier.pkl",            # gcg (in-sample baseline)
        "ensemble_defense_classifier_v1to7.pkl",      # v1-v7 transfer test
        "ensemble_defense_classifier_advbench.pkl",   # AdvBench out-of-domain
        # Metrics files come along for the §5.6.9 write-up.
        "ensemble_defense_training_metrics.json",
        "ensemble_defense_training_metrics_v1to7.json",
        "ensemble_defense_training_metrics_advbench.json",
    ):
        clf_src = f"{VOL_MOUNT}/{fname}"
        if os.path.exists(clf_src):
            shutil.copy(clf_src, f"results/{fname}")
            print(f"Hydrated {fname}")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_gcg_defense_replay.py",
            "--models", *models,
            "--defenses", *defenses,
            "--n-trials", str(n_trials),
            "--temperature", str(temperature),
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
    defenses: str = "none,input_filter,perplexity_filter,semantic_input_filter,ensemble,ensemble_trained,ensemble_trained_v1to7,ensemble_trained_advbench",
    train_sources: str = "gcg,v1to7,advbench",
    n_trials: int = 1,
    temperature: float = 0.7,
    gen_keys: str = "",
):
    """Orchestrate Stage 1 (GCG gen), Stage 2 (transferability),
    optional Stage 2.5 (ensemble training over multiple sources), and
    optional Stage 3 (defense replay).

    Defaults: Vicuna-7b surrogate, 200 GCG steps, 3 Groq targets.

    --train-ensemble loops over --train-sources (default: gcg,v1to7,advbench)
    producing one trained classifier per source. The §5.6.9 comparison
    pipeline requires all three.

    Common patterns:
      modal run scripts/run_gcg_modal.py                                 # full GCG run + transfer
      modal run scripts/run_gcg_modal.py --skip-gen --skip-transfer --run-defenses
      modal run scripts/run_gcg_modal.py --skip-gen --skip-transfer --train-ensemble --run-defenses
      modal run scripts/run_gcg_modal.py --skip-gen --skip-transfer --train-ensemble --train-sources v1to7,advbench --run-defenses
    """
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    local_cache = out_dir / _cache_filename(surrogate)

    if not skip_gen:
        keys_list = [k.strip() for k in gen_keys.split(",") if k.strip()] or None
        if keys_list:
            eta = "~3-5 min per key on A100 (existing cache preserved)"
            print(f"== Stage 1: GCG partial regen on {surrogate}, keys={keys_list} ({eta}) ==")
        else:
            eta = "~25-45 min on A100 (full regen of all keys)"
            print(f"== Stage 1: GCG on {surrogate}, {steps} steps, batch={batch_size} ({eta}) ==")
        suffixes = gen_suffixes.remote(
            model=surrogate, steps=steps, batch_size=batch_size, keys=keys_list,
        )
        local_cache.write_text(json.dumps(suffixes, indent=2))
        print(f"  wrote {local_cache}")
        if keys_list:
            for k in keys_list:
                if k in suffixes:
                    print(f"  {k}: {suffixes[k]!r}")
        else:
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
        source_list = [s.strip() for s in train_sources.split(",") if s.strip()]
        print(f"\n== Stage 2.5: train EnsembleDefense XGBoost (sources={source_list}) ==")

        def _f(v) -> float:
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        for src in source_list:
            print(f"\n  --- source: {src} ---")
            train_metrics = train_ensemble_classifier.remote(
                surrogate=surrogate, source=src,
            )
            # Mirror the trainer's filename convention so the local copies
            # live alongside the gcg metrics (and survive between runs).
            metrics_name = (
                "ensemble_defense_training_metrics.json" if src == "gcg"
                else f"ensemble_defense_training_metrics_{src}.json"
            )
            metrics_path = out_dir / metrics_name
            metrics_path.write_text(json.dumps(train_metrics, indent=2, default=str))
            print(f"    wrote {metrics_path}")
            print(
                f"    train_auc={_f(train_metrics.get('train_auc')):.3f}  "
                f"cv_auc={_f(train_metrics.get('cv_auc_mean')):.3f}±"
                f"{_f(train_metrics.get('cv_auc_std')):.3f}  "
                f"n+={train_metrics.get('n_positive')} n-={train_metrics.get('n_negative')}"
            )
            top_importances = list(train_metrics.get("feature_importances", {}).items())[:5]
            if top_importances:
                print("    top features:")
                for name, imp in top_importances:
                    print(f"      {name:40} {_f(imp):.4f}")

    if run_defenses:
        defense_list = [d.strip() for d in defenses.split(",") if d.strip()]
        print(
            f"\n== Stage 3: defense replay [{defense_list}] x {model_list}, "
            f"n_trials={n_trials}, temperature={temperature} =="
        )
        defense_results = run_defense_replay.remote(
            model_list, defense_list, surrogate=surrogate,
            n_trials=n_trials, temperature=temperature,
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

# Pass 08 — Infrastructure Specification

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Python Version and Virtual Environment

| Item | Detail |
|---|---|
| Python version | Not pinned in requirements.txt; all code uses `from __future__ import annotations` → Python 3.10+ required for `list[dict]` generic syntax and `X | Y` union types at runtime |
| Recommended minimum | Python 3.10 (type hints used in signatures would fail at runtime on 3.9) |
| Virtual environment setup | `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt` |
| No pyproject.toml | Package is not installable as a distribution; all imports use `sys.path.insert(0, ...)` in scripts and tests |
| No setup.cfg | No entry-point registration |

---

## Required Environment Variables

| Variable | Provider | Purpose | Optional? | Crash behavior if missing |
|---|---|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic | Claude models via LLMClient AND LangChain ChatAnthropic | Required if using `claude-sonnet` | `KeyError` at `_get_client("anthropic")` |
| `MISTRAL_API_KEY` | Mistral | `mistral-large` model | Required if using `mistral-large` | `KeyError` at `_get_client("mistral")` |
| `GOOGLE_API_KEY` | Google | `gemini-flash` model | Required if using `gemini-flash` | `KeyError` at `_get_client("google")` |
| `GROQ_API_KEY` | Groq | `groq-llama`, `groq-qwen`, `groq-scout` models | Required if using any Groq model | `KeyError` at `_get_client("groq")` |
| `OPENAI_API_KEY` | OpenAI | `gpt-4o` (commented out in models.yaml) | Optional; only if gpt-4o is uncommented | `KeyError` at `_get_client("openai")` |

All keys are read with `os.environ["KEY"]` (not `.get()`), so a missing key crashes at client initialization time (lazy — not at import). `load_dotenv()` is called at module import of `src/utils/llm.py`.

**`.env` file**: Must exist in the project root (or any parent directory per `python-dotenv` search). `load_dotenv()` is called at module level in `src/utils/llm.py`, so the file must be present before any import of that module.

---

## LLMClient Provider Selection Logic

`LLMClient.chat(model_name, messages)` selects provider via the `config/models.yaml` `provider` field:

```
model_name → _model_cfg(model_name) → cfg["provider"] → _get_client(provider)
```

Name normalization: `model_name.replace("_", "-")` is tried if exact match fails (allows `groq_llama` → `groq-llama`).

**CommodityTradingAgent LangChain provider selection** uses string matching on model name (NOT models.yaml):
```python
if "claude" or "anthropic" in model_name → ChatAnthropic
elif "gemini" in model_name             → ChatGoogleGenerativeAI
elif "groq" or "llama" or "mixtral"     → ChatGroq
elif "mistral" in model_name            → ChatMistralAI
elif "gpt" or "o1" or "o3"             → ChatOpenAI
default                                  → ChatAnthropic
```

This is independent of models.yaml. LangChain reads API keys directly from environment (no LLMClient intermediary).

---

## `results/` Directory Layout After a Full Benchmark Run

Observed from actual files on disk:

```
results/
├── attack_results.csv                           # Ad hoc run output
├── attack_results.json
├── baseline_gemini_test.json
├── baseline_groq_test.csv / .json
├── baseline_mistral_test.csv / .json
├── baseline_results.json
├── groq_results.csv / .json / 2.csv / 2.json
├── mistral_results_2.csv / .json
├── GCG_findings_report_2026-05-04.md
│
├── results_MMDD_HHMM/                           # Timestamped run directory (per model)
│   ├── {model}_{defense}.csv                    # Per model+defense CSV
│   ├── {model}_{defense}.json                   # Per model+defense JSON
│   ├── {model}_{defense}.log                    # stderr logs (some runs)
│   ├── {model}_all_combined.csv / .json         # All defenses merged
│   ├── summary.csv                              # Attack-level summary
│   ├── RESULTS.md                               # Human-readable report (some runs)
│   └── report/                                  # Generated visuals (some runs)
│       ├── RedTeam_Results.pptx
│       ├── RedTeam_Appendix_Prompts.pptx
│       ├── RedTeam_Combined.pptx
│       ├── heatmap_asr.png
│       ├── barchart_defense_asr.png
│       ├── heatmap_detection_coverage.png
│       ├── radar_vulnerability.png
│       ├── category_asr.csv
│       ├── financial_impact.csv
│       ├── model_comparisons.csv
│       ├── model_vulnerability_profile.csv
│       └── summary_statistics.csv
│
├── attack_defend_MMDD_HHMM/                     # Experiment-style runs
│   └── {config}_no_defense.json
│
├── auto_redteam_v3_MMDD_HHMM/                  # V3 adaptive loop output
│   ├── round_0.json ... round_N.json            # Per-round attack plans + results
│   ├── all_results.json
│   ├── attack_archive.json                      # Evolved attack plans
│   ├── reflections.json                         # Failure analysis store
│   ├── strategy_db.json                         # Strategy performance stats
│   └── summary.csv
│
└── gcg_suffix_cache.json                        # GCG offline suffix cache (if generated)
```

**Naming convention**: `{model_key}_{defense_name}` where model_key comes from `models.yaml` (e.g. `groq_llama`, `groq_qwen`, `mistral_large`) and defense_name matches the `DEFENSE_MAP` keys in `run_full_benchmark.py`.

---

## ML Model Downloads

The following defenses trigger automatic model downloads on first instantiation:

| Defense | Model | Source | Approx. Size | Download trigger |
|---|---|---|---|---|
| `SemanticInputFilterDefense` | `all-MiniLM-L6-v2` (sentence-transformers) | Hugging Face Hub | ~90 MB | `_load_model()` called lazily on first `check_input()` |
| `PerplexityFilterDefense` | `gpt2` (transformers + GPT2LMHeadModel) | Hugging Face Hub | ~500 MB | `_load_model()` called lazily on first `check_input()` |
| `GCGSuffixGenerator._generate_online()` | `gpt2` (surrogate model for gradient optimization) | Hugging Face Hub | ~500 MB | Only if online GCG generation is triggered (not precomputed) |
| `EnsembleDefense` | XGBoost model (no download) | Trained in-process | N/A | Trained from labeled data or loaded from `ensemble_classifier.pkl` |
| `EnsembleDefense._load_classifier()` | Persisted XGBoost model | Local disk | Small | On `EnsembleDefense()` init if `ensemble_classifier.pkl` exists |

Models are cached in the Hugging Face default cache directory (`~/.cache/huggingface/hub/`). No custom cache path is configured by the codebase.

**SHAP/explainability**: `src/evaluation/explainability.py` trains an `XGBClassifier(random_state=42)` on provided attack results — no external download, all in-process.

---

## Config Loading Sequence at Runtime

Order in which configs load during a typical `run_full_benchmark.py` run:

1. **`src/utils/llm.py` import** → `load_dotenv()` runs immediately (module level)  
2. **`LLMClient()` instantiation** → `__post_init__` opens `config/models.yaml` via `Path(__file__).resolve().parents[2] / "config" / "models.yaml"` — required at client creation
3. **`CommodityTradingAgent()` instantiation** → `_load_config()` opens `config/agent_config.yaml` via `Path(__file__).resolve().parents[3] / "config" / "agent_config.yaml"`
4. **`OutputValidatorDefense()` instantiation** → loads `config/agent_config.yaml` + `config/commodities.yaml` via `Path(__file__).resolve().parents[2] / ...`
5. **`HumanInLoopDefense()` instantiation** → loads `config/agent_config.yaml`
6. **`SemanticInputFilterDefense.check_input()` (first call)** → lazy-loads `all-MiniLM-L6-v2` from HuggingFace
7. **`PerplexityFilterDefense.check_input()` (first call)** → lazy-loads GPT-2 from HuggingFace

**Path depth inconsistency**: `CommodityTradingAgent` uses `parents[3]` (because the file is at `src/agent/trading_agent.py`, going 3 levels up: agent → src → redteam). Other modules at `src/defenses/*.py` use `parents[2]` (2 levels up: defenses → src → redteam). If any module is moved, the depth must be recalculated.

---

## Scripts Inventory

| Script | Purpose | Key CLI args |
|---|---|---|
| `scripts/run_full_benchmark.py` | Full model × defense matrix | `--models`, `--skip-multi-agent` |
| `scripts/run_attacks.py` | Single model/defense run | `--model`, `--defense`, `--category` |
| `scripts/run_baseline.py` | No-defense baseline only | `--model` |
| `scripts/run_defenses.py` | Defense comparison only | `--model` |
| `scripts/run_auto_redteam.py` | V2 generator loop | `--rounds`, `--model` |
| `scripts/run_auto_redteam_v3.py` | V3 adaptive loop with StrategyDB | `--rounds`, `--model`, `--category` |
| `scripts/run_gcg_generate.py` | Pre-generate GCG suffixes | — |
| `scripts/run_gcg_comparison.py` | GCG vs baseline comparison | — |
| `scripts/run_gcg_transferability.py` | GCG cross-model transferability | — |
| `scripts/run_advanced_analysis.py` | Statistical analysis + SHAP | `--results-dir` |
| `scripts/run_experiment1.py` | Experiment 1 runner | — |
| `scripts/run_experiment_attack_defend.py` | Attack vs defense experiments | — |
| `scripts/run_experiment_defense_layers.py` | Defense layering experiments | — |
| `scripts/generate_pptx.py` | Generate result slides | `--results-dir` |
| `scripts/generate_report.py` | Generate markdown report | — |
| `scripts/create_appendix_pptx.py` | Appendix slides (prompts) | — |
| `scripts/create_merged_pptx.py` | Merge multiple PPTXes | — |
| `scripts/create_experiment1_pptx.py` | Experiment 1 slides | — |
| `scripts/create_final_pptx.py` | Final combined slides | — |
| `scripts/run_groq_benchmark.py` | Groq-only benchmark | — |

---

## Known Environment Pitfalls

1. **Missing `.env` file**: `load_dotenv()` is called at module import. If `.env` doesn't exist, no error — but `os.environ["KEY"]` will raise `KeyError` when any LLM client is first used.

2. **First-run model download**: `SemanticInputFilterDefense` and `PerplexityFilterDefense` will block execution for 1-5 minutes while downloading models. No progress indication in logs beyond "Loading model..." (single `logging.info`).

3. **CWD independence**: All config paths use `__file__`-relative resolution, so scripts can be run from any directory.

4. **`sys.path.insert` in scripts/tests**: Every script and test file manually inserts the project root into `sys.path`. If the directory structure changes, all these paths break.

5. **PyTorch CPU vs GPU**: `GCGSuffixGenerator` defaults to `device="cpu"`. On GPU machines, performance could be drastically improved but requires changing `GCGConfig.device`. No autodetection.

6. **Groq rate limits**: The V3 adaptive loop makes multiple sequential Groq API calls per round (planner + replanner + critic + defender + mutator = up to 8 LLM calls per round). At Groq free tier limits, this will hit rate limit errors. Errors are caught per-round and logged, but loop continues.

7. **`results/` directory auto-creation**: `run_full_benchmark.py` creates timestamped subdirectories. If `results/` does not exist, it creates it. But `gcg_suffix_cache.json` defaults to `results/gcg_suffix_cache.json` — if `results/` doesn't exist, cache write will fail silently (exception caught in `GCGSuffixGenerator`).

8. **concurrent.futures timeout**: Default 120s timeout in `LLMClient.chat()`. Slow providers (especially local models or overloaded APIs) may hit this and raise `concurrent.futures.TimeoutError` which propagates to the caller uncaught within `LLMClient`.

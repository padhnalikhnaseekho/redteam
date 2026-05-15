---
spec-type: system
title: Infrastructure
last-updated: 2026-05-15
status: CURRENT
---

# Infrastructure — CommodityRedTeam

This document describes the deployment/runtime/environment shape of the CommodityRedTeam project. A new SDD agent reading this file cold should know how to set up the environment, which env vars to configure, and what the file system looks like after a run.

---

## 1. Python Environment

**Python version:** 3.11+ recommended (code uses `list[X] | None` union syntax available from 3.10+, `from __future__ import annotations` is used in most files for forward compatibility)

**Virtual environment setup:**

```bash
cd /Users/paraskanwar/Desktop/redteam
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Source root:** `src/`  
**Test root:** `tests/` (pytest; directory exists but test files not enumerated — exempt from spec gate)  
**Config root:** `config/`  
**Results output:** `results/`  
**Scripts:** `scripts/`

---

## 2. Required Dependencies (`requirements.txt`)

### LLM Provider SDKs

| Package | Version | Provider |
|---|---|---|
| `anthropic` | >=0.18.0 | Claude (Anthropic) |
| `mistralai` | >=0.1.0 | Mistral |
| `google-genai` | >=1.0.0 | Gemini (Google AI Studio) |
| `groq` | >=0.4.0 | Groq (Llama, Qwen, Scout) |
| `openai` | >=1.12.0 | OpenAI (optional, commented out in models.yaml) |

### LangChain Framework

| Package | Purpose |
|---|---|
| `langchain` | Core agent framework |
| `langchain-anthropic` | Claude model integration |
| `langchain-google-genai` | Gemini model integration |
| `langchain-groq` | Groq model integration |
| `langchain-mistralai` | Mistral model integration |
| `langchain-openai` | OpenAI model integration |
| `langchain-community` | Community tools |

### Data & Math

| Package | Purpose |
|---|---|
| `pyyaml` | Config file loading |
| `pydantic` | Data validation (>=2.0) |
| `pandas` | Result DataFrames, CSV/JSON I/O |
| `numpy` | Numeric operations |
| `scipy` | Statistical analysis (VaR parametric, norm.ppf) |
| `yfinance` | Live commodity price data from Yahoo Finance |
| `python-dotenv` | Loading `.env` file into environment |

### Reporting

| Package | Purpose |
|---|---|
| `rich` | Terminal output formatting |
| `matplotlib` | Chart generation for reports |

### Testing

| Package | Purpose |
|---|---|
| `pytest` | >=7.4 — test framework |

### ML / AI Depth (Optional but listed in requirements.txt)

| Package | Version | When Needed |
|---|---|---|
| `sentence-transformers` | >=2.2.0 | `SemanticFilterDefense` — embedding similarity detection |
| `scikit-learn` | >=1.3 | `EnsembleDefense` training/eval, `defense_roc_analysis()` (sklearn.metrics) |
| `torch` | >=2.0 | Required by `sentence-transformers` |
| `transformers` | >=4.30 | Required by `sentence-transformers` |
| `shap` | >=0.42 | SHAP explainability for attack success prediction (advanced analysis) |
| `xgboost` | >=2.0 | `EnsembleDefense` ML classifier training |

**Note:** The ML packages (`torch`, `sentence-transformers`, `shap`, `xgboost`) are heavy. They are only loaded lazily (inside `if` blocks or `import` inside functions) — the base attack/defense loop will run without them if those code paths are not invoked.

---

## 3. Environment Variables

All env vars are loaded at startup via `python-dotenv` (`load_dotenv()` in `src/utils/llm.py`). The `.env.example` file documents them:

### Required

| Variable | Provider | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic | Claude claude-sonnet-4-20250514 |
| `MISTRAL_API_KEY` | Mistral | mistral-large-latest |

### Free-Tier (Recommended)

| Variable | Provider | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Google AI Studio | gemini-2.0-flash |
| `GROQ_API_KEY` | Groq | llama-3.3-70b-versatile, qwen/qwen3-32b, llama-4-scout |

### Optional (Paid)

| Variable | Provider | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | gpt-4o (model config commented out in `models.yaml`) |

### How `LLMClient` Selects a Provider

1. `LLMClient.chat(model_name, messages)` is called.
2. `_model_cfg(model_name)` looks up `model_name` in `config/models.yaml` (loaded at `__post_init__`). Underscores are normalized to hyphens.
3. The `provider` field in the model config determines which SDK client is used.
4. Clients are initialized lazily and cached in `_clients` dict. The appropriate `*_API_KEY` is read from `os.environ` at first use.
5. If a required API key is absent, `os.environ["KEY"]` raises `KeyError` at the point of first provider use.

---

## 4. Config File Loading Pattern

All YAML config files are loaded with `yaml.safe_load()` via PyYAML — there is no Pydantic validation on the YAML schemas (raw dicts are used). Config is loaded lazily at module/class initialization, not at import time.

| Config File | Loaded by | At What Point |
|---|---|---|
| `config/models.yaml` | `LLMClient.__post_init__()` | When first `LLMClient()` is instantiated |
| `config/agent_config.yaml` | `CommodityTradingAgent._load_config()` | When `CommodityTradingAgent()` is instantiated |
| `config/commodities.yaml` | `src/utils/data._load_commodities_config()` | On first call to `get_commodity_info()` or `get_all_commodity_names()` |

**Path resolution:** All configs are resolved relative to the source file location using `Path(__file__).resolve().parents[N]` — this means the project can be run from any working directory.

---

## 5. LangChain Agent Setup

`CommodityTradingAgent` uses LangChain's `create_tool_calling_agent` + `AgentExecutor`:

- **LLM selection:** determined by model name substring matching (see `_create_llm()`)
  - `"claude"` or `"anthropic"` → `ChatAnthropic`
  - `"gemini"` → `ChatGoogleGenerativeAI`
  - `"groq"`, `"llama"`, `"mixtral"` → `ChatGroq`
  - `"mistral"` → `ChatMistralAI`
  - `"gpt"`, `"o1"`, `"o3"` → `ChatOpenAI`
  - Default fallback → `ChatAnthropic`
- **Prompt:** `ChatPromptTemplate` with `system`, `human`, and `agent_scratchpad` placeholder
- **Agent type:** `create_tool_calling_agent` (native function-calling)
- **Settings:** `verbose=True`, `return_intermediate_steps=True`, `handle_parsing_errors=True`, default `max_iterations=15`

**7 tools registered:**
1. `get_price` — live commodity prices via yfinance
2. `get_news` — news and sentiment (simulated/injectable for attacks)
3. `calculate_risk` — VaR and drawdown calculation
4. `get_fundamentals` — supply/demand fundamentals
5. `get_correlation` — inter-commodity correlation
6. `check_position_limits` — validates against guardrails config
7. `submit_recommendation` — records the final recommendation

**Tool attack modes:** All tools support `set_mode(manipulated=True, ...)` and `reset_mode()` for red-team test injection.

---

## 6. External Data Dependencies

**yfinance (Yahoo Finance):**
- Used by `src/utils/data.py` to fetch live price data
- `SYMBOL_MAP` maps commodity names to yfinance tickers
- Iron ore and thermal coal use equity proxies (`VALE`, `BTU`) — no liquid futures tickers
- Requires internet connectivity; no caching layer
- Periods: `{N}d` for ≤730 days, `{N}y` otherwise

**No database:** There is no SQL database, Redis, or any stateful backend. All persistence is via flat files in `results/`.

---

## 7. Results File System Layout

After a benchmark run, `results/` contains flat files. A typical run produces:

```
results/
├── attack_results.csv                     # flat all-attacks result table
├── attack_results.json                    # same data in JSON
├── results_{MMDD}_{HHMM}/                 # timestamped run directories
│   ├── {model}_{defense}.csv              # per-model per-defense result tables
│   ├── {model}_{defense}.json
│   ├── {model}_all_combined.csv
│   ├── {model}_all_combined.json
│   ├── summary.csv
│   └── report/
│       ├── heatmap_asr.png
│       ├── barchart_defense_asr.png
│       ├── heatmap_detection_coverage.png
│       ├── radar_vulnerability.png
│       ├── category_asr.csv
│       ├── financial_impact.csv
│       ├── model_comparisons.csv
│       ├── model_vulnerability_profile.csv
│       ├── summary_statistics.csv
│       └── *.pptx                        # PowerPoint reports
├── auto_redteam_v3_{MMDD}_{HHMM}/        # V3 evolution loop runs
│   ├── all_results.json
│   ├── attack_archive.json               # AttackArchive serialized
│   ├── reflections.json                  # ReflectionStore serialized
│   ├── strategy_db.json                  # StrategyDB serialized
│   ├── round_0.json                      # per-round results
│   ├── round_1.json
│   └── summary.csv
└── GCG_findings_report_*.md              # GCG attack analysis reports
```

**CSV column schema** for result tables matches the Result Row Dict from DATA-MODELS.md:
`attack_id, category, severity, model, defense, success, target_action_achieved, detected, defense_confidence, financial_impact, notes`

---

## 8. Running Benchmarks

### Single model/defense run
```bash
python scripts/run_attacks.py
```

### Full matrix benchmark (all models × all defenses)
```bash
python scripts/run_full_benchmark.py
```

### V3 Autonomous red-team loop
```bash
python scripts/run_auto_redteam_v3.py
```

### Report generation
```bash
python scripts/generate_report.py
python scripts/generate_pptx.py
```

### Advanced analysis (SHAP, ROC, etc.)
```bash
python scripts/run_advanced_analysis.py
```

---

## 9. Known Environment Pitfalls

1. **GPU not required** — `torch` and `sentence-transformers` run on CPU. No GPU hardware is needed; however, semantic similarity checks will be slow on CPU for large batches.

2. **Model download on first run** — `sentence-transformers` downloads the embedding model to `~/.cache/huggingface/` on first use. Requires internet access and ~1 GB disk space.

3. **yfinance rate limits** — Frequent price fetches for many commodities in rapid succession may trigger Yahoo Finance rate limiting. The code has no retry/backoff logic.

4. **Groq free tier rate limits** — The primary models (groq-qwen, groq-scout, groq-llama) are free but rate-limited. Long benchmark runs may hit token-per-minute limits. No automatic retry is implemented in `LLMClient`.

5. **Timeout** — `LLMClient.chat()` has a hardcoded 120-second timeout per call. Long reasoning chains from complex models may occasionally time out.

6. **Iron ore / thermal coal yfinance proxies** — `VALE` and `BTU` are equity tickers, not futures. They track price trends but not actual commodity spot prices. This is documented in `commodities.yaml`.

7. **`results/` directory is not gitignored** — Result CSVs and JSON files accumulate across runs. The directory may grow large over extended experimentation.

8. **No authentication** — There is no user auth, session management, or access control. This is a local research tool, not a deployed service.

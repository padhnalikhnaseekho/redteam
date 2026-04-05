# CommodityRedTeam: Red Teaming Agentic AI for Commodity Trading

A research framework for systematically identifying and evaluating vulnerabilities in LLM-based commodity trading agents. Implements 50+ domain-specific attacks across 7 vulnerability categories, 5 defense strategies, and a comprehensive evaluation pipeline with statistical analysis.

**Capstone Project**: IIT Bombay EPGD AI/ML -- Agentic AI Topic #2: "Red teaming of agentic AI to identify vulnerabilities"

---

## Project Structure

```
redteam/
├── .env.example                          # API key template
├── .env                                  # API keys (not committed)
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
│
├── config/
│   ├── agent_config.yaml                 # Agent system prompt, guardrails, position limits
│   ├── commodities.yaml                  # 10 commodities with exchange, units, volatility
│   └── models.yaml                       # LLM model configs (Groq, Gemini, Mistral, etc.)
│
├── src/
│   ├── agent/                            # Target: The commodity trading agent under test
│   │   ├── trading_agent.py              # CommodityTradingAgent -- LangChain agent with tool calling
│   │   ├── system_prompt.py              # DEFAULT_SYSTEM_PROMPT and HARDENED_SYSTEM_PROMPT
│   │   └── tools/                        # 7 agent tools, each with switchable attack modes
│   │       ├── price.py                  # get_price -- real/manipulated commodity prices
│   │       ├── news.py                   # get_news -- real headlines / injectable payloads
│   │       ├── risk.py                   # calculate_risk -- VaR, drawdown / understated risk
│   │       ├── fundamentals.py           # get_fundamentals -- supply-demand / stale data
│   │       ├── correlation.py            # get_correlation -- real/wrong correlations
│   │       ├── position.py               # check_position_limits -- real limits / override
│   │       └── recommendation.py         # submit_recommendation -- validates and routes trades
│   │
│   ├── attacks/                          # 50 attacks across 7 vulnerability categories
│   │   ├── base.py                       # Attack, AttackResult, AttackCategory, Severity
│   │   ├── registry.py                   # Auto-discovery, @register decorator, filtering
│   │   ├── v1_direct_injection.py        # V1: 8 direct prompt injection attacks
│   │   ├── v2_indirect_injection.py      # V2: 10 indirect injection via market data/news
│   │   ├── v3_tool_manipulation.py       # V3: 7 tool response manipulation attacks
│   │   ├── v4_context_poisoning.py       # V4: 5 context/memory poisoning attacks
│   │   ├── v5_reasoning_hijacking.py     # V5: 8 reasoning chain manipulation attacks
│   │   ├── v6_confidence_manipulation.py # V6: 6 confidence/certainty manipulation attacks
│   │   └── v7_multi_step_compounding.py  # V7: 6 multi-step cascading failure attacks
│   │
│   ├── defenses/                         # 8 defense strategies (5 original + 3 ML-based)
│   │   ├── base.py                       # Defense, DefenseResult base classes
│   │   ├── input_filter.py              # D1: Regex + keyword injection detection
│   │   ├── output_validator.py          # D2: Position limits, risk, price sanity checks
│   │   ├── guardrails.py               # D3: System prompt hardening with safety rules
│   │   ├── multi_agent.py              # D4: Second LLM reviews recommendations
│   │   ├── human_in_loop.py            # D5: Simulated human approval for flagged trades
│   │   ├── semantic_filter.py           # D6: Sentence-transformer embeddings + cosine similarity
│   │   ├── perplexity_filter.py         # D7: GPT-2 sliding-window perplexity spike detection
│   │   └── ensemble_defense.py          # D8: XGBoost ensemble combining all defense signals
│   │
│   ├── evaluation/                       # Evaluation, statistical analysis, and explainability
│   │   ├── evaluator.py                  # RedTeamEvaluator -- runs attacks, collects results
│   │   ├── metrics.py                    # ASR, FPR, detection rate, financial impact, ROC curves
│   │   ├── statistical.py               # Chi-squared, McNemar, CI, Bayesian, MI, Shapley values
│   │   ├── transferability.py           # Attack transferability matrix, Fisher's exact test
│   │   └── explainability.py            # SHAP analysis for attack success prediction
│   │
│   ├── generator/                        # Automated red teaming
│   │   └── attack_generator.py           # Domain-aware LLM-powered attack generator
│   │
│   └── utils/                            # Shared utilities
│       ├── llm.py                        # Multi-provider LLM client with cost tracking
│       └── data.py                       # yfinance market data, VaR, correlation matrix
│
├── scripts/                              # Entry-point scripts
│   ├── run_full_benchmark.py             # End-to-end: all models x all defenses + report
│   ├── run_groq_benchmark.py             # Groq-only: 3 free models (qwen, scout, llama)
│   ├── run_baseline.py                   # Run agent on legitimate queries (baseline)
│   ├── run_attacks.py                    # Run attack suite against agent
│   ├── run_defenses.py                   # Compare defense strategies
│   ├── generate_report.py               # Generate plots, tables, statistical report
│   └── generate_pptx.py                 # Generate PPTX presentation from results
│
├── results/                              # Output directory (auto-created)
│   ├── baseline_*.json + .csv            # Baseline query results
│   └── results_MMDD_HHMM/               # Timestamped benchmark run
│       ├── *_no_defense.json + .csv      # Per-model, per-defense results
│       ├── *_input_filter.json + .csv
│       ├── *_all_combined.json + .csv
│       ├── all_results_combined.csv      # Every result in one file
│       ├── summary.csv                   # ASR/detection/impact per model x defense
│       └── report/                       # Generated visualizations
│           ├── heatmap_asr.png
│           ├── barchart_defense_asr.png
│           ├── radar_vulnerability.png
│           ├── heatmap_detection_coverage.png
│           ├── summary_statistics.csv
│           ├── model_vulnerability_profile.csv
│           ├── financial_impact.csv
│           ├── category_asr.csv
│           └── model_comparisons.csv
│
└── tests/                                # Unit tests
    ├── test_agent.py                     # Agent creation, tool switching
    ├── test_attacks.py                   # Attack registration, prepare, count
    └── test_evaluator.py                # Evaluator, metrics calculation
```

---

## Supported Models

| Model | Provider | Config Name | Cost | Status |
|-------|----------|-------------|------|--------|
| LLaMA 3.3 70B | Groq | `groq-llama` | Free | **Default** |
| Qwen3 32B | Groq | `groq-qwen` | Free | Available |
| Llama 4 Scout 17B (MoE) | Groq | `groq-scout` | Free | Available |
| Gemini 2.0 Flash | Google AI Studio | `gemini-flash` | Free | Available (daily quota) |
| Mistral Large | Mistral | `mistral-large` | Free tier | Available |
| Claude Sonnet 4 | Anthropic | `claude-sonnet` | Paid | Optional |
| GPT-4o | OpenAI | `gpt-4o` | Paid | Optional (commented out) |

**Minimum requirement**: At least one working API key. Groq and Gemini are free.

> **Note on Groq rate limits**: Groq free tier has daily token limits per model.
> If you hit the limit on `groq-llama`, use `groq-qwen` or `groq-scout` instead —
> each model has its own independent quota.

### Baseline Results (Verified)

| Model | Success | Risk Assessment | Recommendation |
|-------|---------|-----------------|----------------|
| **Groq (LLaMA 3.3 70B)** | 10/10 | 10/10 | 9/10 |
| **Mistral Large** | 10/10 | 10/10 | 10/10 |

---

## Vulnerability Taxonomy: CommodityAgentThreat

| Category | Code | Attacks | Description |
|----------|------|---------|-------------|
| Direct Prompt Injection | V1 | 8 | Adversary crafts input to override system prompt / guardrails |
| Indirect Data Injection | V2 | 10 | Malicious payloads embedded in market data, news feeds, reports |
| Tool Manipulation | V3 | 7 | Compromised tool responses (wrong prices, understated risk) |
| Context Poisoning | V4 | 5 | Corrupted memory, RAG knowledge base, or conversation history |
| Reasoning Hijacking | V5 | 8 | Anchoring, false causation, sunk cost fallacy, circular logic |
| Confidence Manipulation | V6 | 6 | Fake consensus, certainty language, inflated track records |
| Multi-Step Compounding | V7 | 6 | Small errors cascade through agent reasoning chain |
| **Total** | | **50** | |

---

## Defense Strategies

### Rule-Based Defenses (D1--D5)

| Defense | Code | Method |
|---------|------|--------|
| Input Filtering | D1 | Regex + keyword detection for injection patterns |
| Output Validation | D2 | Check recommendations against limits, risk, price sanity |
| System Prompt Hardening | D3 | Add explicit safety rules to system prompt |
| Multi-Agent Verification | D4 | Second LLM reviews recommendations for manipulation |
| Human-in-the-Loop | D5 | Flag suspicious trades for human approval |

### ML-Based Defenses (D6--D8)

| Defense | Code | Method | ML Technique |
|---------|------|--------|--------------|
| Semantic Similarity Filter | D6 | Encode queries and known injections into shared embedding space; block if cosine similarity > threshold | Sentence-BERT embeddings, cosine similarity |
| Perplexity Spike Detection | D7 | Sliding-window perplexity via GPT-2; injections cause distributional anomalies (z-score spikes) | Language model perplexity, cross-entropy |
| Ensemble Defense | D8 | Train XGBoost classifier on features from all base defenses to learn optimal combination | Gradient boosting, feature engineering |

**D6 mathematical basis**: `sim(q, p) = (q . p) / (||q|| * ||p||)` in sentence-transformer embedding space (all-MiniLM-L6-v2, 384 dimensions).

**D7 mathematical basis**: `PPL = exp(-(1/N) * sum log P(token_i | context))`. A perplexity spike (z-score > 2.5) in a sliding window indicates injected text that is distributionally different from surrounding commodity language.

**D8 mathematical basis**: XGBoost binary logistic objective with L2 regularization. Features include per-defense confidence scores, flag counts, and aggregate statistics. Demonstrates ensemble learning (boosting/stacking) from Module 2.

---

## Commodities Covered

| Group | Commodities | Correlation Group |
|-------|-------------|-------------------|
| Oil and Gas | Brent Crude, WTI Crude, Natural Gas | energy |
| Refined Metals | Copper, Aluminum, Zinc, Nickel | base_metals |
| Bulk | Iron Ore, Thermal Coal | bulk |
| Precious Metals | Gold | precious |

---

## Setup

### Prerequisites

- Python 3.10+
- API key for at least one provider (Groq and Google are free)

### Installation

```bash
cd redteam

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies (includes ML packages: sentence-transformers, xgboost, shap, scikit-learn)
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### API Key Setup

| Provider | Sign Up | Env Variable |
|----------|---------|--------------|
| **Groq** (free) | https://console.groq.com/keys | `GROQ_API_KEY` |
| **Google AI Studio** (free) | https://aistudio.google.com/apikey | `GOOGLE_API_KEY` |
| **Mistral** (free tier) | https://console.mistral.ai/api-keys | `MISTRAL_API_KEY` |
| Anthropic (paid) | https://console.anthropic.com | `ANTHROPIC_API_KEY` |
| OpenAI (paid) | https://platform.openai.com/api-keys | `OPENAI_API_KEY` |

---

## Quick Start (Recommended)

There are **3 main scripts** depending on what you want to do. All use Groq free tier (zero cost).

### 1. Experiment 1: Static vs Agentic Red Teaming (GO-TO SCRIPT)

**This is the primary experiment.** Runs both static (v1) and agentic (v2) attacks against the same target, then produces comparison plots, statistical tests, and a PPTX report.

```bash
# Full experiment: static baseline + agentic loop + comparison
python scripts/run_experiment1.py

# Recommended: use qwen as attacker (more creative), scout as target (more robust)
python scripts/run_experiment1.py --target-model groq-scout --attacker-model groq-qwen --delay 5

# If you already ran the v1 benchmark, skip re-running static attacks:
python scripts/run_experiment1.py --skip-static --delay 5

# Generate the PPTX report after experiment completes:
python scripts/create_experiment1_pptx.py
```

**What it does:**
1. **Condition A (Static)**: Runs 50 predefined single-turn attacks from the v1 attack library
2. **Condition B (Agentic)**: Runs the v2 planner -> attacker -> critic -> mutator loop (3 rounds, adaptive)
3. **Comparison**: Chi-squared test, Bayesian posteriors, confidence intervals, 3-panel plot

**Output** (in `results/experiment1_MMDD_HHMM/`):
```
experiment1_MMDD_HHMM/
├── static_results.json              # Condition A raw results
├── agentic_results.json             # Condition B raw results
└── report/
    ├── experiment1_comparison.png    # 3-panel comparison plot
    ├── experiment1_all_results.csv   # Combined CSV
    ├── experiment1_summary.json      # Stats: ASR, CI, chi-squared, Bayesian
    └── Experiment1_Results.pptx      # 8-slide presentation
```

**Important: Groq rate limits.** Use `--delay 5` (5 seconds between API calls) to avoid 429 errors. Use different models for attacker vs target so they use separate rate limit quotas (e.g., `--attacker-model groq-qwen --target-model groq-scout`).

---

### 2. V1 Defense Benchmark (Static Attacks x All Defenses x All Models)

Runs the 50 predefined attacks against each model with each defense configuration. Produces per-defense ASR comparison.

```bash
# Groq only (free, recommended)
python scripts/run_groq_benchmark.py

# All models (requires Mistral/Anthropic keys)
python scripts/run_full_benchmark.py --skip-multi-agent
```

### 3. Advanced ML Analysis (No API calls, runs on existing results)

Runs Bayesian analysis, mutual information, SHAP explainability, transferability, and Shapley values on existing benchmark results. **Instant, no API calls needed.**

```bash
python scripts/run_advanced_analysis.py
python scripts/run_advanced_analysis.py --results-dir results/results_0329_1945
```

---

## All Scripts Reference

| Script | Purpose | API Calls | Time |
|--------|---------|-----------|------|
| **`run_experiment1.py`** | **Primary experiment: static vs agentic comparison** | Yes (Groq free) | ~15-30 min |
| `create_experiment1_pptx.py` | Generate PPTX from experiment 1 results | No | Instant |
| `run_groq_benchmark.py` | V1 benchmark: 50 attacks x defenses x Groq models | Yes (Groq free) | ~30-60 min |
| `run_full_benchmark.py` | V1 benchmark: all models including paid | Yes (mixed) | ~60-90 min |
| `run_auto_redteam.py` | V2 agentic loop only (no static comparison) | Yes (Groq free) | ~10-20 min |
| `run_advanced_analysis.py` | ML analysis on existing results (Bayesian, SHAP, MI) | **No** | Instant |
| `generate_report.py` | Generate plots from v1 benchmark results | No | Instant |
| `generate_pptx.py` | Generate PPTX from v1 benchmark results | No | Instant |
| `create_merged_pptx.py` | Merge v1 results + appendix into combined deck | No | Instant |

---

## Legacy: V1 Benchmark Details

Run the full v1 pipeline in a single command:

```bash
python scripts/run_full_benchmark.py
```

This executes all steps automatically:

1. Loads 50 attacks from the registry
2. For each model (groq-llama, mistral-large):
   - Runs 50 attacks with **no defense**
   - Runs 50 attacks with each **individual defense** (D1--D8)
   - Runs 50 attacks with **all defenses combined**
3. Saves per-run JSON + CSV results
4. Generates combined CSV and summary CSV
5. Generates report with PNG plots and statistical analysis

**Options**:
```bash
# Skip D4 multi-agent defense (faster, saves API calls)
python scripts/run_full_benchmark.py --skip-multi-agent

# Run specific models only
python scripts/run_full_benchmark.py --models groq-llama

# Adjust delay between API calls (default 2s)
python scripts/run_full_benchmark.py --delay 3
```

### Groq-Only Benchmark (Free, Fast)

If you want to run **only free Groq models** (zero cost, fast LPU inference):

```bash
python scripts/run_groq_benchmark.py
```

Defaults to `groq-qwen` (Qwen3 32B) and `groq-scout` (Llama 4 Scout MoE). Add `groq-llama` if its daily rate limit has reset:

```bash
python scripts/run_groq_benchmark.py --models groq-llama groq-qwen groq-scout
```

Same options as the full benchmark (`--skip-multi-agent`, `--delay`, `--models`).

**Output** (timestamped directory):
```
results/results_0329_1430/
├── groq_llama_no_defense.json + .csv
├── groq_llama_input_filter.json + .csv
├── groq_llama_output_validator.json + .csv
├── groq_llama_guardrails.json + .csv
├── groq_llama_human_in_loop.json + .csv
├── groq_llama_multi_agent.json + .csv
├── groq_llama_all_combined.json + .csv
├── mistral_large_no_defense.json + .csv
├── mistral_large_input_filter.json + .csv
├── mistral_large_output_validator.json + .csv
├── mistral_large_guardrails.json + .csv
├── mistral_large_human_in_loop.json + .csv
├── mistral_large_multi_agent.json + .csv
├── mistral_large_all_combined.json + .csv
├── all_results_combined.csv
├── summary.csv
└── report/
    ├── heatmap_asr.png                   # ASR by category x model
    ├── barchart_defense_asr.png          # Defense effectiveness comparison
    ├── radar_vulnerability.png           # Model vulnerability profiles
    ├── heatmap_detection_coverage.png    # Detection rate by defense x category
    ├── summary_statistics.csv            # Full statistical summary
    ├── model_vulnerability_profile.csv   # Per-model breakdown
    ├── financial_impact.csv              # Total estimated $ impact
    ├── category_asr.csv                  # ASR per attack category
    └── model_comparisons.csv             # Chi-squared, Cohen's h between models
```

---

## Step-by-Step Usage (Manual)

Use these if you want to run individual steps rather than the full benchmark.

### Step 1: Run Baseline (Legitimate Queries)

Establishes baseline agent behavior on 10 legitimate commodity analysis queries.

```bash
# Run against specific models
python scripts/run_baseline.py --models groq-llama mistral-large
```

**Output**: `results/baseline_results.json` + `.csv`

### Step 2: Run Attack Suite

Run the full 50-attack suite against a specific model.

```bash
# Run all 50 attacks against Groq LLaMA (default)
python scripts/run_attacks.py --model groq-llama

# Run specific attack category only
python scripts/run_attacks.py --model groq-llama --category v1_direct_injection

# Run with a defense enabled
python scripts/run_attacks.py --model groq-llama --defense input_filter

# Multiple defenses
python scripts/run_attacks.py --model groq-llama --defense input_filter output_validator

# Custom output path
python scripts/run_attacks.py --model groq-llama --output results/groq_results.json
```

**Arguments**:
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name (see `config/models.yaml`) | `groq-llama` |
| `--category` | Filter by category (e.g. `v1_direct_injection`) | All |
| `--defense` | One or more defenses to apply | None |
| `--output` | Output file path | `results/attack_results.json` |

**Output**: JSON + CSV with per-attack results (success, detected, financial impact).

### Step 3: Compare Defenses

Runs all attacks with no defense, each individual defense, and all combined.

```bash
python scripts/run_defenses.py --model groq-llama
```

**Output**: `results/defense_comparison.json` + `.csv` + `defense_summary.csv`

### Step 4: Generate Report

Generate visualizations, statistical analysis, and summary tables from results.

```bash
# From default results directory
python scripts/generate_report.py

# From a specific benchmark run
python scripts/generate_report.py --results-dir results/results_0329_1430 --output-dir results/results_0329_1430/report
```

**Output**:
| File | Description |
|------|-------------|
| `heatmap_asr.png` | Attack Success Rate heatmap: categories x models |
| `barchart_defense_asr.png` | Bar chart comparing defense effectiveness |
| `radar_vulnerability.png` | Radar chart of model vulnerability profiles |
| `heatmap_detection_coverage.png` | Detection rate heatmap: defenses x categories |
| `summary_statistics.csv` | Full results table with significance indicators |
| `model_vulnerability_profile.csv` | Per-model ASR breakdown by category |
| `financial_impact.csv` | Total and mean estimated $ impact |
| `category_asr.csv` | ASR per attack category |
| `model_comparisons.csv` | Chi-squared tests, confidence intervals, Cohen's h |

### Step 5: Generate PPTX Report

Generate a PowerPoint presentation from benchmark results using the IIT Bombay capstone template. Produces a 14-slide deck with project context (Why/What/How/When/Deliverables) and results (tables, heatmaps, charts, findings).

```bash
# Auto-detect latest results directory
python scripts/generate_pptx.py

# Specific results directory
python scripts/generate_pptx.py --results-dir results/results_0329_1945

# Custom template and output path
python scripts/generate_pptx.py --template ~/Downloads/AI_ML_Sample_Format.pptx --output ~/Downloads/Report.pptx
```

**Arguments**:
| Argument | Description | Default |
|----------|-------------|---------|
| `--results-dir` | Path to results directory | Latest in `results/` |
| `--template` | Path to PPTX template | `~/winhome/Downloads/AI_ML_Sample_Format.pptx` |
| `--output` | Output PPTX path | `<results-dir>/report/RedTeam_Results.pptx` |

**Slides generated**:
| Slide | Content |
|-------|---------|
| 1 | Title slide |
| 2 | Problem Statement and Motivation (The "Why") |
| 3 | Proposed Solution (The "What") |
| 4 | Data Sources and Tools (The "How") |
| 5 | Scope and Implementation Roadmap (The "When") |
| 6 | Key Deliverables |
| 7 | Discussion & Feedback |
| 8 | Benchmark results table (ASR, detection, impact) |
| 9 | ASR heatmap by attack category |
| 10 | Defense effectiveness bar chart |
| 11 | Radar vulnerability profiles |
| 12 | Detection coverage heatmap |
| 13 | Defense effectiveness ranking (side-by-side) |
| 14 | Key findings (auto-derived from data) |

**Requires**: `pip install python-pptx lxml`

### Step 6: Run Tests

```bash
python -m pytest tests/ -v
```

---

## How It Works

### Architecture

```
┌──────────────────┐     ┌──────────────────────────┐
│ ATTACK GENERATOR │────▶│ TARGET AGENT (CTA)       │
│                  │     │                           │
│ 50 attacks       │     │ LLM + 7 Tools            │
│ 7 categories     │     │ System Prompt + Guardrails│
└──────────────────┘     └──────────────────────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐     ┌──────────────────────────┐
│ ATTACK LIBRARY   │     │ DEFENSES (D1-D5)         │
│                  │     │                           │
│ V1-V7 payloads   │     │ Input filter, output      │
│ Tool overrides   │     │ validation, guardrails,   │
│ Context injection│     │ multi-agent, human review │
└──────────────────┘     └──────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────────┐
                         │ EVALUATOR                │
                         │                           │
                         │ ASR, FPR, financial impact│
                         │ Statistical tests         │
                         │ Visualizations            │
                         └──────────────────────────┘
```

### Pipeline Flow

```
run_full_benchmark.py
  │
  ├── Step 1: Load 50 attacks from registry
  │
  ├── Step 2: For each model:
  │     ├── Run 50 attacks with no defense         → {model}_no_defense.json + .csv
  │     ├── Run 50 attacks with D1 (input filter)  → {model}_input_filter.json + .csv
  │     ├── Run 50 attacks with D2 (output valid.)  → {model}_output_validator.json + .csv
  │     ├── Run 50 attacks with D3 (guardrails)    → {model}_guardrails.json + .csv
  │     ├── Run 50 attacks with D4 (multi-agent)   → {model}_multi_agent.json + .csv
  │     ├── Run 50 attacks with D5 (human-in-loop) → {model}_human_in_loop.json + .csv
  │     └── Run 50 attacks with all combined       → {model}_all_combined.json + .csv
  │
  ├── Step 3: Merge all results
  │     ├── all_results_combined.csv  (every row from every run)
  │     └── summary.csv              (ASR/detection/impact per model x defense)
  │
  └── Step 4: Generate report
        ├── heatmap_asr.png
        ├── barchart_defense_asr.png
        ├── radar_vulnerability.png
        ├── heatmap_detection_coverage.png
        ├── summary_statistics.csv
        ├── model_vulnerability_profile.csv
        ├── financial_impact.csv
        ├── category_asr.csv
        └── model_comparisons.csv
```

### Attack Flow (per attack)

1. **Prepare**: Attack produces payload (`user_query`, `tool_overrides`, `injected_context`)
2. **Input Defense**: Check query against input filters (D1); block or pass
3. **Tool Setup**: Apply tool overrides (e.g., wrong prices, injected news)
4. **Execute**: Run agent with payload, collect response
5. **Output Defense**: Validate recommendation (D2), check guardrails (D3), reviewer (D4), human flag (D5)
6. **Evaluate**: Check if attack achieved target action (e.g., oversized position, skipped risk check)
7. **Reset**: Restore all tools to normal mode

### Tool Attack Modes

| Tool | Normal Mode | Attack Mode |
|------|-------------|-------------|
| `get_price` | Real market price via yfinance | `override_price` (e.g., $65 instead of $80) |
| `get_news` | Realistic sample headlines | `inject_payload` with malicious instructions |
| `calculate_risk` | Real VaR and drawdown | `risk_multiplier` (e.g., 0.5 = understate risk) |
| `get_fundamentals` | Current supply-demand data | `stale_data` from previous quarter |
| `get_correlation` | Real correlation coefficient | `override_correlation` (e.g., -0.8 for +0.75) |
| `check_position_limits` | Validate against actual limits | `override` always returns within limits |

---

## Key Metrics

### Standard Metrics

| Metric | Definition |
|--------|------------|
| **ASR** (Attack Success Rate) | % of attacks that achieve target action |
| **FPR** (False Positive Rate) | % of legitimate requests blocked by defense |
| **Detection Rate** | % of attacks detected/flagged by defense |
| **Financial Impact** | Estimated $ loss from successful attacks |
| **Coverage** | % of attack categories defended against |

### Advanced ML/Statistical Metrics

| Metric | Module | Definition |
|--------|--------|------------|
| **Bayesian Posterior** | `statistical.py` | Beta-Binomial model: `p ~ Beta(alpha + k, beta + n - k)`. Provides 95% credible intervals and P(vulnerability > threshold) instead of point estimates |
| **Mutual Information** | `statistical.py` | `I(X;Y) = H(Y) - H(Y|X)` -- measures how much knowing attack category reduces uncertainty about success. NMI normalizes to [0,1] |
| **Vulnerability Entropy** | `statistical.py` | Shannon entropy of success distribution across categories. Low = concentrated (easy to defend), High = uniform vulnerability |
| **Shapley Values** | `statistical.py` | Game-theoretic attribution of detection power to each defense. `phi_i = avg marginal contribution across all coalitions` |
| **ROC / AUC** | `metrics.py` | Receiver Operating Characteristic curves for defense configurations. AUC measures discrimination ability (0.5 = random, 1.0 = perfect) |
| **Transferability** | `transferability.py` | `T(A->B) = |succeed on both| / |succeed on A|`. Fisher's exact test for significance. Jaccard similarity for overlap |
| **SHAP Explainability** | `explainability.py` | TreeExplainer SHAP values for attack success prediction. Identifies which attack features (category, severity, model) drive success |

---

## Extending the Framework

### Adding a New Attack

1. Create a class in the appropriate `v{N}_*.py` file (or a new file for a new category)
2. Extend the `Attack` base class
3. Implement `prepare()` and `evaluate()`
4. Decorate with `@register`

```python
from src.attacks.base import Attack, AttackCategory, AttackResult, Severity
from src.attacks.registry import register

@register
class V1_9_NewAttack(Attack):
    def __init__(self):
        super().__init__(
            id="V1.9",
            name="New Attack Name",
            category=AttackCategory.V1_DIRECT_INJECTION,
            severity=Severity.HIGH,
            description="Description of the attack",
            target_action="what_agent_should_do_wrong",
            commodity="brent_crude",
        )

    def prepare(self, agent=None):
        return {
            "user_query": "The malicious prompt...",
            "tool_overrides": {},
            "injected_context": [],
        }

    def evaluate(self, agent_result):
        output = str(agent_result).lower()
        success = "some_indicator" in output
        return AttackResult(
            attack_id=self.id,
            success=success,
            target_action_achieved=success,
            agent_output=str(agent_result),
            reasoning_chain=[],
        )
```

### Adding a New Defense

1. Create a file in `src/defenses/`
2. Extend the `Defense` base class
3. Implement `check_input()` and/or `check_output()`

```python
from src.defenses.base import Defense, DefenseResult

class MyDefense(Defense):
    name = "my_defense"

    def check_input(self, user_query, context=None):
        # Return DefenseResult(allowed=False) to block
        return DefenseResult(allowed=True)

    def check_output(self, agent_output, recommendation=None):
        # Return DefenseResult(allowed=False) to block
        return DefenseResult(allowed=True)
```

### Adding a New Model

Add an entry to `config/models.yaml`:

```yaml
my_model:
  provider: groq        # groq, google, mistral, anthropic, or openai
  model_id: my-model-id
  max_tokens: 4096
  temperature: 0.3
  cost_per_1k_input_tokens: 0.0
  cost_per_1k_output_tokens: 0.0
```

Supported providers: `groq`, `google`, `mistral`, `anthropic`, `openai`

---

## Research Contributions

1. **CommodityAgentThreat Taxonomy** -- First vulnerability taxonomy for LLM-based financial trading agents (7 categories, 50 attacks)
2. **Domain-Aware Automated Red Teaming** -- Attack generator using commodity market knowledge (vs generic prompt injection)
3. **Multi-Step Attack Chain Analysis** -- How small errors cascade through agent reasoning (5% price error to $10M loss)
4. **Cross-Model Vulnerability Comparison** -- LLaMA vs Gemini vs Mistral robustness profiles
5. **Defense Effectiveness Framework** -- Empirical evaluation of 8 defense strategies with statistical rigor
6. **ML-Based Defenses** -- Sentence-transformer embedding similarity (D6), perplexity-based injection detection (D7), and XGBoost ensemble (D8) replacing brittle regex approaches
7. **Attack Transferability Analysis** -- Quantifying whether vulnerabilities transfer across models using Fisher's exact test and Jaccard similarity
8. **Explainable Vulnerability Assessment** -- SHAP-based feature attribution identifying which attack characteristics predict success
9. **Bayesian Vulnerability Quantification** -- Beta-Binomial posterior analysis with credible intervals, replacing frequentist point estimates
10. **Game-Theoretic Defense Attribution** -- Shapley values for fair attribution of detection power across defense coalitions

### Course Module Alignment (AI/ML in Practice)

| Course Module | Project Component |
|---|---|
| **Module 1**: Deploy & Scale | Multi-provider LLM inference (Groq, Gemini, Mistral), configurable pipelines |
| **Module 2**: Embeddings & Transfer Learning | D6 semantic filter (sentence-transformers), attack transferability matrix across models |
| **Module 2**: Fine-tuning & Few-shot | Few-shot context poisoning attacks (V4.3), system prompt hardening |
| **Module 2**: Ensemble Methods | D8 XGBoost ensemble defense (boosting), comparison vs base defenses |
| **Module 3**: Agentic AI | LangChain ReAct agent with 7 tools, multi-agent verification defense |
| **Module 4**: Model Evaluation | Bayesian analysis, ROC/AUC, MI, cross-validation of ensemble classifier |
| **Module 4**: Explainable AI | SHAP TreeExplainer for attack success factors, defense effectiveness attribution |

---

## Estimated Costs

| Component | Estimate |
|-----------|----------|
| Groq API (LLaMA 3.3 70B, Qwen3 32B, Llama 4 Scout) | Free |
| Google Gemini 2.0 Flash | Free |
| Mistral Large (free tier) | Free |
| **Total (free-tier models only)** | **$0** |
| Optional: Anthropic Claude / OpenAI GPT-4o | ~$750 |

---

## License

Academic use -- IIT Bombay EPGD Capstone Project.

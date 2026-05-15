# Pass 02 — Dependency Specification

**Extraction date:** 2026-05-15  
**Status:** COMPLETE

---

## Python Package Inventory (requirements.txt)

| Package | Version Pin | Purpose | Modules that use it | Required? |
|---|---|---|---|---|
| `anthropic` | `>=0.18.0` | Anthropic Claude API client | `src/utils/llm.py` (_chat_anthropic), `CommodityTradingAgent` | Required (if using Anthropic models) |
| `mistralai` | `>=0.1.0` | Mistral AI API client | `src/utils/llm.py` (_chat_mistral) | Optional |
| `google-genai` | `>=1.0.0` | Google Gemini API client | `src/utils/llm.py` (_chat_google) | Optional |
| `groq` | `>=0.4.0` | Groq API client | `src/utils/llm.py` (_chat_groq) | Optional (default models use Groq) |
| `openai` | `>=1.12.0` | OpenAI API client | `src/utils/llm.py` (_chat_openai) | Optional |
| `langchain` | `>=0.1.0` | Agent framework (create_tool_calling_agent, AgentExecutor) | `src/agent/trading_agent.py` | Required |
| `langchain-anthropic` | `>=0.1.0` | LangChain ChatAnthropic | `src/agent/trading_agent.py` | Required (if claude models) |
| `langchain-google-genai` | `>=1.0.0` | LangChain ChatGoogleGenerativeAI | `src/agent/trading_agent.py` | Optional |
| `langchain-groq` | `>=0.1.0` | LangChain ChatGroq | `src/agent/trading_agent.py` | Required (default model is llama via Groq) |
| `langchain-mistralai` | `>=0.1.0` | LangChain ChatMistralAI | `src/agent/trading_agent.py` | Optional |
| `langchain-openai` | `>=0.0.5` | LangChain ChatOpenAI | `src/agent/trading_agent.py` | Optional |
| `langchain-community` | `>=0.0.10` | LangChain community tools | `src/agent/trading_agent.py` (transitive) | Required |
| `pyyaml` | `>=6.0` | YAML config loading | `src/utils/llm.py`, `src/defenses/output_validator.py`, `src/defenses/human_in_loop.py`, `src/utils/data.py` | Required |
| `pydantic` | `>=2.0` | Data validation (transitive via LangChain) | Transitive | Required |
| `pandas` | `>=2.0` | DataFrames for results, metrics | `src/evaluation/evaluator.py`, `src/evaluation/metrics.py`, `src/evaluation/statistical.py`, `src/evaluation/transferability.py`, `src/evaluation/explainability.py`, `src/utils/data.py` | Required |
| `numpy` | `>=1.24` | Numerical operations | All evaluation modules, `src/defenses/semantic_filter.py`, `src/defenses/perplexity_filter.py`, `src/utils/data.py` | Required |
| `scipy` | `>=1.11` | Stats: chi2_contingency, fisher_exact, beta.ppf, norm.ppf | `src/evaluation/statistical.py`, `src/evaluation/transferability.py`, `src/utils/data.py` | Required |
| `yfinance` | `>=0.2.30` | Live commodity price data | `src/agent/tools/price.py`, `src/utils/data.py` | Optional (fallback prices exist) |
| `rich` | `>=13.0` | Console output formatting | `scripts/run_attacks.py`, `scripts/run_full_benchmark.py`, `scripts/run_auto_redteam_v3.py`, `scripts/run_experiment_attack_defend.py` | Required for scripts |
| `matplotlib` | `>=3.7` | Plotting (SHAP summary, experiment charts) | `src/evaluation/explainability.py`, `scripts/run_experiment_attack_defend.py` | Optional |
| `pytest` | `>=7.4` | Testing framework | `tests/*.py` | Dev |
| `python-dotenv` | `>=1.0` | Load `.env` for API keys | `src/utils/llm.py` (load_dotenv at import) | Required |
| `sentence-transformers` | `>=2.2.0` | all-MiniLM-L6-v2 sentence embeddings | `src/defenses/semantic_filter.py` (lazy) | Optional (D6 only) |
| `scikit-learn` | `>=1.3` | ROC, cross_val_score, roc_auc_score | `src/evaluation/metrics.py`, `src/defenses/ensemble_defense.py`, `src/evaluation/explainability.py` | Required |
| `torch` | `>=2.0` | GCG optimizer, perplexity filter | `src/attacks/v8_gcg_adversarial.py` (lazy), `src/defenses/perplexity_filter.py` (lazy) | Optional |
| `transformers` | `>=4.30` | GPT2 for perplexity; HF models for GCG | `src/defenses/perplexity_filter.py` (lazy), `src/attacks/v8_gcg_adversarial.py` (lazy) | Optional |
| `shap` | `>=0.42` | SHAP explainability values | `src/evaluation/explainability.py` | Optional |
| `xgboost` | `>=2.0` | Ensemble defense classifier; SHAP predictor | `src/defenses/ensemble_defense.py`, `src/evaluation/explainability.py` | Optional |

---

## Internal Module Dependency Graph (directed, with arrows = "imports")

```
src/utils/llm.py  ←  src/agents/planner.py
                  ←  src/agents/mutator.py
                  ←  src/agents/critic.py
                  ←  src/defenses/multi_agent.py
                  ←  src/generator/attack_generator.py
                  ←  src/v3/defender_agent.py
                  ←  src/v3/replanner.py

src/attacks/base.py  ←  src/attacks/v1_direct_injection.py (and v2-v8)
                     ←  src/attacks/registry.py
                     ←  src/evaluation/evaluator.py
                     ←  src/generator/attack_generator.py

src/attacks/registry.py  ←  src/evaluation/evaluator.py (run_suite uses get_all_attacks)
                         ←  scripts/*.py

src/defenses/base.py  ←  src/defenses/input_filter.py
                      ←  src/defenses/output_validator.py
                      ←  src/defenses/guardrails.py
                      ←  src/defenses/multi_agent.py
                      ←  src/defenses/human_in_loop.py
                      ←  src/defenses/semantic_filter.py
                      ←  src/defenses/perplexity_filter.py
                      ←  src/defenses/ensemble_defense.py
                      ←  src/v3/trajectory_defense.py
                      ←  src/evaluation/evaluator.py

src/evaluation/evaluator.py  ←  scripts/run_attacks.py
                             ←  scripts/run_full_benchmark.py
                             ←  src/generator/attack_generator.py

config/agent_config.yaml  ←  src/defenses/output_validator.py (yaml.safe_load)
                          ←  src/defenses/human_in_loop.py (yaml.safe_load)

config/commodities.yaml   ←  src/defenses/output_validator.py (yaml.safe_load)
                          ←  src/utils/data.py (yaml.safe_load)

config/models.yaml        ←  src/utils/llm.py (yaml.safe_load at __post_init__)
```

---

## Circular Dependency Analysis

No circular dependencies detected. The dependency graph is a DAG:
- `src/v3/` components depend only on `src/utils/llm.py` (for LLM calls) and `src/defenses/base.py`
- `src/evaluation/evaluator.py` imports from `src/agent/tools` (for overrides) but `src/agent/` does not import from `src/evaluation/`
- `src/generator/attack_generator.py` imports `src/evaluation/evaluator.py` — this creates a transitive dependency chain (generator → evaluator → attacks) but no cycle

**Potential future risk:** If `src/evaluation/` were to import `src/generator/`, a cycle would form.

---

## Provider SDK Matrix

| Model key (models.yaml) | Provider | SDK imported | Env var | LangChain class |
|---|---|---|---|---|
| `claude-sonnet` | `anthropic` | `anthropic.Anthropic` + `langchain_anthropic.ChatAnthropic` | `ANTHROPIC_API_KEY` | `ChatAnthropic` |
| `mistral-large` | `mistral` | `mistralai.client.Mistral` + `langchain_mistralai.ChatMistralAI` | `MISTRAL_API_KEY` | `ChatMistralAI` |
| `gemini-flash` | `google` | `google.genai.Client` + `langchain_google_genai.ChatGoogleGenerativeAI` | `GOOGLE_API_KEY` | `ChatGoogleGenerativeAI` |
| `groq-llama` | `groq` | `groq.Groq` + `langchain_groq.ChatGroq` | `GROQ_API_KEY` | `ChatGroq` |
| `groq-qwen` | `groq` | `groq.Groq` + `langchain_groq.ChatGroq` | `GROQ_API_KEY` | `ChatGroq` |
| `groq-scout` | `groq` | `groq.Groq` + `langchain_groq.ChatGroq` | `GROQ_API_KEY` | `ChatGroq` |

**Note:** Two separate LLM subsystems exist in parallel:
1. `LLMClient` (in `src/utils/llm.py`) — used by defenses, agents, generator (raw API calls)
2. LangChain (`CommodityTradingAgent._create_llm()`) — used by the trading agent (tool-calling agent loop)

The `LLMClient` and LangChain stack do NOT share client instances. API keys are shared via env vars.

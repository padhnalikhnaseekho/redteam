# System Architecture Specification

> **Status:** `CURRENT`
> **Owner:** repository maintainers
> **Last updated:** 2026-05-13

## Overview

CommodityRedTeam is a Python research framework for red-teaming LLM-based commodity trading agents. It defines a target trading analyst agent, a suite of commodity-domain attacks, layered defenses, and benchmark/report generation scripts.

Core architectural decisions:

- The framework is library-first Python with script entry points under `scripts/`.
- Attack and defense behavior is represented by small class contracts in `src/attacks/base.py` and `src/defenses/base.py`.
- LLM providers are selected from `config/models.yaml` through `src/utils/llm.py`.
- Experiments write flat JSON/CSV artifacts under `results/` for later statistical analysis and presentation generation.
- Trading-domain limits and commodity metadata live in YAML config rather than hardcoded benchmark scripts.

## Module Registry

| Module | Path | Responsibilities | Dependencies |
|---|---|---|---|
| Trading agent | `src/agent/` | LangChain tool-calling commodity analyst, system prompts, trading tools, tool attack modes | LangChain provider packages, `config/agent_config.yaml`, tool modules |
| Attack suite | `src/attacks/` | Defines 54 registered attacks across V1-V8 vulnerability categories | `Attack`, `AttackResult`, registry auto-discovery |
| Defenses | `src/defenses/` | Input, output, guardrail, multi-agent, human, semantic, perplexity, and ensemble defenses | `Defense`, `DefenseResult`, optional ML/LLM dependencies |
| Evaluator | `src/evaluation/` | Orchestrates attack preparation, defense checks, agent execution, success evaluation, metrics | pandas, scipy/sklearn/shap for advanced analysis |
| LLM utilities | `src/utils/llm.py` | Normalizes provider chat APIs, token/cost tracking, model config lookup | API keys from environment, `config/models.yaml` |
| Automated generation | `src/generator/`, `src/agents/`, `src/v3/` | Planner/mutator/critic and adaptive red-team strategy workflows | LLM client, archive/reflection stores |
| Experiment scripts | `scripts/` | CLI entry points for attacks, defenses, benchmarks, reports, PPTX creation | Framework modules, `results/` write access |
| Tests | `tests/` | Unit coverage for agent tools, attacks, evaluator, metrics | pytest, mocks |

## Runtime Flow

1. A script loads configured models, attack classes, and defense configurations.
2. Attacks are discovered through `src/attacks/registry.py` by importing V1-V8 modules and instantiating registered classes.
3. For each attack, `Attack.prepare(agent)` emits a payload containing at least `user_query`; it may also include `tool_overrides` and `injected_context`.
4. The evaluator applies each defense's `check_input()` method before the query reaches the agent.
5. If allowed, the evaluator applies tool overrides, runs the agent or callable, then resets affected tools.
6. The evaluator applies each defense's `check_output()` method to the agent output.
7. `Attack.evaluate(agent_output)` produces an `AttackResult`; the evaluator annotates it with defense detection/confidence and records a flat result row.
8. Scripts persist per-run JSON/CSV, combined CSV, summary CSV, plots, and optional PPTX reports.

## Contracts

### Attack Contract

Defined in `src/attacks/base.py`.

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `id` | `str` | Stable attack id such as `v1.1` |
| `name` | `str` | Human-readable attack name |
| `category` | `AttackCategory` | One of V1-V8 |
| `severity` | `Severity` | `critical`, `high`, `medium`, or `low` |
| `description` | `str` | What vulnerability is exercised |
| `target_action` | `str` | Unsafe action the attack tries to elicit |
| `commodity` | `str` | Target commodity key or `all` |

Required methods:

| Method | Required behavior |
|---|---|
| `prepare(agent) -> dict` | Return a payload with `user_query` and optional `tool_overrides` / `injected_context` |
| `evaluate(agent_result) -> AttackResult` | Decide whether target action was achieved and estimate impact |

### Defense Contract

Defined in `src/defenses/base.py`.

| Method | Required behavior |
|---|---|
| `check_input(user_query, context=None) -> DefenseResult` | Screen or transform input before agent execution |
| `check_output(agent_output, recommendation=None) -> DefenseResult` | Screen or transform output after agent execution |

`DefenseResult.allowed=False` means the defense blocks that phase. `flags` must contain machine-readable reasons. `confidence` is a 0-1 score.

### Evaluator Result Row

Benchmark rows must preserve these fields:

| Field | Meaning |
|---|---|
| `attack_id` | Stable attack id |
| `category` | Attack category enum value |
| `severity` | Attack severity enum value |
| `model` | Configured model key or explicit model label |
| `defense` | Defense config label, usually `none` or `+`-joined defense names |
| `success` | Whether the attack succeeded after defenses |
| `target_action_achieved` | Whether the unsafe target action appeared in the agent behavior |
| `detected` | Whether a defense detected or blocked the attack |
| `defense_confidence` | Max confidence reported by active defenses |
| `financial_impact` | Estimated USD impact when attack succeeds |
| `notes` | Diagnostic flags and details |

## Data And Configuration

| File | Purpose |
|---|---|
| `config/agent_config.yaml` | Agent identity, trading rules, position limits, risk thresholds, sanctions config |
| `config/commodities.yaml` | Commodity symbols, exchanges, lot sizes, price ranges, volatilities, correlation groups |
| `config/models.yaml` | Provider, model id, token limits, temperature, and cost metadata |
| `.env` | Local API keys loaded by `python-dotenv`; never commit secrets |

## External Dependencies

| System | Direction | Protocol/API | Purpose |
|---|---|---|---|
| Groq | Outbound | Chat completions | Free-tier benchmark models |
| Google Gemini | Outbound | `google-genai` | Model target and reviewer model option |
| Mistral | Outbound | Mistral chat API | Benchmark target |
| Anthropic | Outbound | Anthropic messages API | Optional benchmark target |
| OpenAI | Outbound | Chat completions | Optional benchmark target |
| yfinance | Outbound | Python package/network data | Commodity market data helper |
| Hugging Face / transformers | Local or outbound model load | Python packages | Semantic/perplexity/GCG workflows |

## Architectural Decision Records

| ID | Decision | Status | Rationale |
|---|---|---|---|
| ADR-001 | Use Python scripts plus importable modules rather than a deployed service | Accepted | Experiments are batch research workflows, not request/response production traffic |
| ADR-002 | Use class-based `Attack` and `Defense` contracts | Accepted | Keeps new attacks/defenses independently testable and discoverable |
| ADR-003 | Store benchmark outputs as flat CSV/JSON | Accepted | Enables pandas analysis, report generation, and human inspection |
| ADR-004 | Keep model/provider metadata in YAML | Accepted | Allows benchmark matrix changes without editing evaluator code |
| ADR-005 | Use surrogate/cached GCG suffixes for V8 | Accepted | Cloud APIs do not expose gradients; cached or local surrogate suffixes support transferability experiments |

## Testing Strategy

| Layer | Tool | Scope |
|---|---|---|
| Unit | pytest | Attack registration/prepare, tool modes, evaluator behavior, metric functions |
| Integration | CLI scripts | Model x defense x attack benchmark paths with real provider keys |
| Analysis validation | pandas/statistical modules | Generated CSV/JSON consistency, summary metrics, plots |
| Manual review | Results and PPTX | Research interpretation, capstone narrative, vulnerability taxonomy |

## Known Architecture Gaps

| Gap | Impact |
|---|---|
| Some script-built agents call LLMs directly instead of the full LangChain `CommodityTradingAgent` | Tool-calling behavior and tool override attacks may differ between scripts |
| `GuardrailsDefense.check_input()` returns a hardened prompt template, but the generic evaluator does not apply `modified_input` as a system prompt | Guardrail effectiveness depends on script-specific wiring |
| `RedTeamEvaluator._apply_tool_overrides()` only handles price, news, and risk overrides | Attacks targeting other tool modules need direct agent support or evaluator expansion |
| ML defenses may download/load large models at runtime | Benchmark repeatability depends on local cache and environment setup |

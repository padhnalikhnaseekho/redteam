# Red Team Framework for Commodity Trading LLM Agents

## Project Overview
IIT Bombay EPGD AI/ML Capstone. Red-teaming framework that tests LLM-based commodity trading agents with 50 adversarial attacks across 7 categories, evaluated against 5+ defense strategies across multiple models (Groq Qwen3 32B, Llama 4 Scout, Mistral Large).

## Setup
```bash
cd redteam
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
- `src/agent/` - Trading agent (LangChain ReAct) with 7 tools and system prompt
- `src/attacks/` - 7 attack categories (v1-v7): direct injection, indirect injection, tool manipulation, context poisoning, reasoning hijacking, confidence manipulation, multi-step compounding
- `src/defenses/` - Defense mechanisms: input_filter, output_validator, guardrails, multi_agent, human_in_loop, semantic_filter, perplexity_filter, ensemble_defense
- `src/evaluation/` - Metrics, statistical analysis, transferability, explainability
- `config/` - YAML configs for agent, commodities, models
- `scripts/` - Run attacks, benchmarks, generate reports/PPTs
- `results/` - Benchmark results (CSV/JSON) per model x defense config

## Key Design Decisions
- Defenses implement `Defense` base class with `check_input()` and `check_output()` returning `DefenseResult(allowed, flags, confidence)`
- Attacks implement `Attack` base class with `prepare()` and `evaluate()` methods
- All attacks auto-register via `@register` decorator in `src/attacks/registry.py`
- Results stored as flat dicts: {attack_id, category, severity, model, defense, success, detected, financial_impact}

## Running Benchmarks
```bash
python scripts/run_full_benchmark.py    # Full model x defense matrix
python scripts/run_attacks.py           # Single model/defense run
python scripts/generate_pptx.py        # Generate result slides
```

## Course Alignment (AI/ML in Practice - IIT Bombay)
This project should demonstrate depth in:
- **Module 2**: Embeddings (semantic similarity defense), transfer learning (attack transferability), ensemble methods (ensemble defense combining multiple signals), few-shot prompting
- **Module 3**: Agentic AI (LangChain agent with tools)
- **Module 4**: Model evaluation (Bayesian analysis, MI, ROC), explainability (SHAP for attack success prediction)

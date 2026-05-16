# Viva Prep — IIT Bombay EPGD AI/ML Capstone

**Deadline:** 2026-05-22 · **Today:** 2026-05-16 · **Days left:** 6 (incl. today)
**Panel:** 5 PhD professors · **Format:** presentation + viva

## How to use this directory

Read in order on Day 1, then drill flashcards + Q&A daily. The cheat-sheet is the only thing you reread the night before.

| # | File | What it gives you | Pages |
|---|------|-------------------|-------|
| 0 | [00_cheat_sheet.md](00_cheat_sheet.md) | Night-before one-pager: numbers, formulas, names, opener pitch | 1 |
| 1 | [01_project_overview.md](01_project_overview.md) | 60-sec / 5-min pitch, threat model, system diagram, "why this project" | 3 |
| 2 | [02_module2_theory.md](02_module2_theory.md) | Embeddings, transfer learning, ensemble combiner, perplexity filter + flashcards | 6 |
| 3 | [03_module3_agentic.md](03_module3_agentic.md) | LangChain ReAct, 7 tools, v3 strategy/reflection/evolution/replanner/defender + flashcards | 5 |
| 4 | [04_module4_evaluation.md](04_module4_evaluation.md) | Chi-square, McNemar, Wilson CIs, Cohen's h, Bayesian Beta-Binomial, MI, entropy, SHAP, Shapley + flashcards | 7 |
| 5 | [05_gcg_deep_dive.md](05_gcg_deep_dive.md) | Full GCG derivation (one-hot trick), Vicuna findings, methodology-correction story + flashcards | 6 |
| 6 | [06_code_tour.md](06_code_tour.md) | File-by-file walkthrough with line numbers — drill before live demo | 4 |
| 7 | [07_viva_qa.md](07_viva_qa.md) | 60+ likely viva questions with model answers, grouped by intent | 8 |
| 8 | [08_study_plan.md](08_study_plan.md) | Day-by-day 5-hour study schedule for the 6 days remaining | 1 |

## Quick links into the actual codebase

| Concept | File:line |
|---------|-----------|
| Trading agent (LangChain ReAct) | `src/agent/trading_agent.py:50` |
| 7 tool definitions | `src/agent/tools/` |
| Default + hardened system prompts | `src/agent/system_prompt.py:3,57` |
| Attack base + registry | `src/attacks/base.py`, `src/attacks/registry.py` |
| GCG attack (v8) + offline fallback | `src/attacks/v8_gcg_adversarial.py:54` (precomputed), `:159` (real GCG) |
| Semantic similarity defense | `src/defenses/semantic_filter.py:84` |
| Perplexity filter | `src/defenses/perplexity_filter.py:44` |
| Ensemble defense (XGBoost) | `src/defenses/ensemble_defense.py:43` |
| Statistical tests | `src/evaluation/statistical.py` (chi² 13, McNemar 42, Wilson 73, Cohen's h 104, Bayesian 233, MI 323, Shapley 455) |
| SHAP attack-success predictor | `src/evaluation/explainability.py:83` |
| v3 strategy DB | `src/v3/strategy_db.py` |
| v3 reflection store | `src/v3/reflection_store.py` |
| v3 evolution archive | `src/v3/attack_archive.py` |
| v3 mid-attack replanner | `src/v3/replanner.py` |
| v3 trajectory defense (heuristic) | `src/v3/trajectory_defense.py` |
| v3 defender agent (LLM judge) | `src/v3/defender_agent.py` |
| Modal GCG pipeline | `scripts/run_gcg_modal.py` |
| Latest findings report | `results/GCG_findings_report_2026-05-14.md` (994 lines) |
| Main README (1061 lines) | `README.md` |

## Three things the panel will probe hardest — predict and prepare

1. **"Is your evaluation fair?"** They will look for label leakage, evaluator artefacts, peeking at test set. The methodology-correction story (the 100%→0% transfer collapse in §7.1 of the findings report) is your single strongest answer. Lead with it.
2. **"What's the Module 2/3/4 connection?"** They want to see you're not just engineering — you're applying course concepts. Memorize the one-line mapping for each module ([02](02_module2_theory.md), [03](03_module3_agentic.md), [04](04_module4_evaluation.md) start with these).
3. **"What would you do with more compute/time?"** Have a concrete answer: bigger surrogate (Llama-2-7B → Llama-3-8B), Vicuna run already scheduled, defense replay on perplexity_filter, true held-out test set with v9 attacks.

## What NOT to do in the viva

- **Don't oversell.** 0% cross-family transfer is a strong negative result — own it. Don't pretend ASRs are higher than they are.
- **Don't dodge derivations.** If a prof asks "derive the McNemar statistic" or "what's the gradient in GCG", attempt it on the board. Page 4 of [04](04_module4_evaluation.md) and page 2 of [05](05_gcg_deep_dive.md) have the derivations.
- **Don't say "I used XGBoost because it works."** Be ready to explain *why* boosting over bagging here, *why* tree-based over a linear classifier, *why* logloss objective, what L2 regularization is doing.
- **Don't claim novelty you don't have.** GCG is Zou et al. 2023. Your novelty is the trading-domain harness + the methodology-correction discovery + the multi-source training comparison (§5.6.9).

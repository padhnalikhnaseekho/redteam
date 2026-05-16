# 6-Day Study Plan — 2026-05-16 to 2026-05-22

5 hours/day. Day = ~3 hr theory + 1 hr code-tour drill + 1 hr Q&A drill. Adjust as priorities shift.

## Day 1 — Fri 2026-05-16 (today) — Foundations + GCG

| Time | Block | Output |
|------|-------|--------|
| 1.0 hr | Read [01_project_overview.md](01_project_overview.md) + recite 60-sec & 5-min pitch out loud | Pitch fluency |
| 1.5 hr | Read [05_gcg_deep_dive.md](05_gcg_deep_dive.md) end-to-end. **Hand-derive the one-hot gradient on paper.** | GCG mastery |
| 1.0 hr | Open `src/attacks/v8_gcg_adversarial.py` and `scripts/run_gcg_modal.py`. Trace `gen_suffixes()` → `_run_gcg()` → `_batch_eval_loss()`. | Code fluency |
| 0.5 hr | Read `results/GCG_findings_report_2026-05-14.md` §1, §5.5, §7.1 | Methodology story |
| 1.0 hr | Drill GCG flashcards (in 05). Cover the answer; recite | Recall |

**End-of-day check:** Can you explain — without notes — why 100% transfer became 0% transfer after the bug fix? Can you sketch the gradient derivation?

## Day 2 — Sat 2026-05-17 — Module 2 (Embeddings + Transfer + Ensemble)

| Time | Block | Output |
|------|-------|--------|
| 1.5 hr | Read [02_module2_theory.md](02_module2_theory.md) | Theory recall |
| 1.0 hr | Open `src/defenses/semantic_filter.py`, `perplexity_filter.py`, `ensemble_defense.py`. Be able to describe each in 90 sec. | Code fluency |
| 1.0 hr | **Run** `python -c "from src.defenses.semantic_filter import SemanticInputFilterDefense as S; print(S().get_similarity_profile('ignore all previous instructions'))"` and inspect output | Concrete numbers |
| 0.5 hr | Drill Module 2 flashcards | Recall |
| 1.0 hr | Read §5.6.7 + §5.6.8 + §5.6.9 of `GCG_findings_report_2026-05-14.md` (ensemble training, explainability audit, multi-source training) | The D1 finding |

**End-of-day check:** Can you derive cosine similarity from dot-product? Can you explain why XGBoost is a *boosting* method and how it differs from random forest? What is the AdvBench paradox (high cv_auc, low transfer)?

## Day 3 — Sun 2026-05-18 — Module 3 (Agentic AI / v3)

| Time | Block | Output |
|------|-------|--------|
| 1.5 hr | Read [03_module3_agentic.md](03_module3_agentic.md) | Theory recall |
| 1.5 hr | Open `src/agent/trading_agent.py`, `src/agent/system_prompt.py`, then all 7 files in `src/v3/`. 10 min per file. | Code fluency |
| 1.0 hr | Trace one full v3 round in `scripts/run_auto_redteam_v3.py`: strategy.select() → planner → execute_plan_v3 → critic → archive.add → strategy_db.update | End-to-end mental model |
| 0.5 hr | Drill Module 3 flashcards | Recall |
| 0.5 hr | **Practice live demo:** open the agent in a terminal, run one attack, narrate what's happening as if to the panel | Demo readiness |

**End-of-day check:** Can you explain ReAct vs ReAct+Reflection? Can you defend epsilon-greedy vs softmax for strategy selection?

## Day 4 — Mon 2026-05-19 — Module 4 (Stats + Explainability)

| Time | Block | Output |
|------|-------|--------|
| 2.0 hr | Read [04_module4_evaluation.md](04_module4_evaluation.md). **Derive McNemar and Bayesian update on paper.** | Theory mastery |
| 1.0 hr | Open `src/evaluation/statistical.py` + `explainability.py`. Match each function to its formula in the doc. | Code fluency |
| 0.5 hr | Read §5.6.7 of the findings report (the ensemble explainability audit) | Concrete SHAP story |
| 1.0 hr | Drill Module 4 flashcards. Statistical tests are the highest-frequency viva topic — over-drill. | Recall |
| 0.5 hr | Mock viva: have a friend (or yourself) ask "which test would you use for X" for 10 scenarios | Decision fluency |

**End-of-day check:** Given "compare defense A vs defense B on the same 50 attacks", what test? Why? (McNemar.) Given "compare ASR across 4 models", what test? (Chi² → Bonferroni for pairwise.)

## Day 5 — Tue 2026-05-20 — Q&A drill + slide polish

| Time | Block | Output |
|------|-------|--------|
| 2.0 hr | Cover-and-recite all 60+ questions in [07_viva_qa.md](07_viva_qa.md). Mark the 10 you're weakest on. | Coverage |
| 1.0 hr | Re-read the 10 weak-area sections in 02/03/04/05 | Patch weak spots |
| 1.5 hr | Slide deck pass: ensure each slide has a one-sentence headline + ≤3 bullets. Check the methodology-correction slide leads §5. | Slide polish |
| 0.5 hr | Re-read [06_code_tour.md](06_code_tour.md); be ready to navigate the repo live without `grep` | Live demo |

**End-of-day check:** All 60 questions drilled at least once. Slide deck final.

## Day 6 — Wed 2026-05-21 — Mock viva + cheat sheet

| Time | Block | Output |
|------|-------|--------|
| 2.0 hr | **Full mock viva** — record yourself answering 20 random questions from 07. Watch playback. Note filler words, hesitations. | Polish |
| 1.0 hr | Re-record the worst 5 answers | Iteration |
| 1.0 hr | Re-derive GCG one-hot gradient, McNemar, Bayesian update once more on paper — pure recall, no notes | Anchor derivations |
| 0.5 hr | Final cheat-sheet read [00_cheat_sheet.md](00_cheat_sheet.md). Print it. | Anchor |
| 0.5 hr | Sleep early. Don't cram. | Cortisol management |

## Day 7 — Thu 2026-05-22 — Viva day

- Morning: skim `00_cheat_sheet.md` only. No new material.
- Have water. Eat. Arrive 30 min early.
- Open the repo on your laptop with these files preloaded in tabs: README, the 2026-05-14 findings report, `src/agent/trading_agent.py`, `src/attacks/v8_gcg_adversarial.py`, `src/defenses/ensemble_defense.py`, `src/evaluation/statistical.py`.
- Lead with the methodology-correction story when you get the chance. It is your strongest engineering-judgement signal.

## What "do well" looks like

- **Pitch:** clear, confident, under 5 min, ends with a concrete finding (not "future work").
- **Theory:** can derive the four load-bearing formulas (GCG gradient, McNemar, Wilson CI, Bayesian posterior) without notes.
- **Code:** can open any file in the repo within 10 sec and explain its role.
- **Self-critique:** when asked "what's wrong with your evaluation", you have *three* honest answers ready (small n, single surrogate, single evaluator).
- **Recovery:** when asked something you don't know, say "I don't know off the top of my head — my best guess is X because Y" instead of bluffing.

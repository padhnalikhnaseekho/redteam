# Project Overview — Pitch, Threat Model, Architecture

## 60-second pitch (memorize verbatim)

> "I built a red-teaming framework for LLM-based commodity trading agents. The framework tests a LangChain ReAct agent that has 7 tools — get_price, get_news, calculate_risk, get_fundamentals, get_correlation, check_position_limits, submit_recommendation — against 50 adversarial attacks across 7 categories: direct injection, indirect injection, tool manipulation, context poisoning, reasoning hijacking, confidence manipulation, multi-step compounding, plus an 8th category for gradient-based GCG adversarial suffixes. Each attack is evaluated against multiple defenses — input filter, output validator, guardrails, multi-agent, human-in-loop, semantic similarity, perplexity filter, and an XGBoost ensemble — across multiple target models from Groq, Mistral, and Google. The headline result is two findings: cross-family GCG transfer is essentially zero, and out-of-domain training data hurts in-domain defense — both instantiate the same Module 2 transfer-learning principle from different sides."

## 5-minute pitch (slide-by-slide)

1. **Motivation** — LLMs are entering high-stakes financial decision pipelines. Existing red-team work targets chatbots, not agentic systems with tools and structured outputs. Question: *can a commodity trading agent be jailbroken into making real-money decisions it shouldn't?*

2. **Threat model** — Threat actors include (a) external prompt injectors via news data, (b) malicious tool outputs (poisoned data feeds), (c) gradient-based attackers with access to a similar open-weights surrogate. Defender has read-only access to query and response, plus tool I/O.

3. **System under test** — LangChain agent (`src/agent/trading_agent.py:50`) using `create_tool_calling_agent`. Configurable across 8+ LLM backends (OpenAI, Anthropic, Groq, Mistral, Gemini). System prompt has 7 hard rules including ">$5M trades require human approval" and a sanctioned-commodities blocklist.

4. **Attack taxonomy** — v1–v7 mirror the standard prompt-injection taxonomy (direct, indirect, tool-manipulation, context-poisoning, reasoning-hijacking, confidence-manipulation, multi-step). v8 is GCG (Zou et al. 2023) — gradient-optimized adversarial suffixes generated on a local surrogate (Vicuna-7b) and transferred to API targets.

5. **Defense stack** — 8 defenses organized into pre/post stages. Two are ML-based and Module-2-aligned: (a) **semantic filter** using `all-MiniLM-L6-v2` sentence embeddings + cosine similarity to a 23-pattern injection corpus, threshold 0.50; (b) **perplexity filter** using GPT-2 with sliding-window perplexity + z-score spike detection. The **ensemble defense** trains XGBoost on features from all base defenses to learn optimal combination weights, rather than treating each defense as an independent gate.

6. **Evaluation** — chi-square for unpaired ASR comparison, McNemar for paired defense comparison, Wilson score CIs for proportions, Cohen's h for effect size, Bonferroni for multiple comparisons. Bayesian Beta-Binomial gives a posterior over vulnerability rate. Mutual information quantifies how predictive attack category is of success. SHAP attributes attack success to attack metadata; Shapley values attribute detection power to defenses.

7. **Headline findings** —
   - **A1 (attack-side):** Vicuna-7b surrogate concentrates GCG success on its training-lineage neighbour (Llama-3.3-70B at 50% ASR), with 0% transfer to non-Llama families. Same total ASR as a base surrogate, different *distribution*.
   - **D1 (defense-side):** AdvBench-trained ensemble has highest cv_auc (0.91) but lowest held-out detection of v8 attacks (25%). Out-of-domain training is not a useful substitute for in-domain diversity.
   - **Methodology correction:** The prior 2026-05-04 study claimed 100% intra-Llama GCG transfer; we found this was an evaluator-keyword-echo artefact where precomputed-suffix fallback shipped surface-form English that matched v8.3's success rule. After fixing the silent fallback (3 corrections, §2.4 of the report), transfer collapsed to 0%.

8. **Limitations** — Small n (4 attacks/model in v8, 50 total). Single surrogate. Single LLM-as-judge evaluator (Llama-Guard-style judging is a known unsolved problem). Gemini results lost to free-tier rate limits.

9. **Future work** — Bigger surrogates (Llama-2-7B → Llama-3-8B). Defense replay with perplexity filter, which we hypothesize will catch GCG token noise specifically. Held-out v9 attack set for the ensemble.

## Threat model (be ready to draw this on a board)

```
                  ┌──────────────────────────────────────┐
                  │     LLM trading agent (LangChain)    │
   user query ──► │  system prompt + 7 tools + ReAct loop │ ──► JSON recommendation
                  └──────────────┬───────────────────────┘
                                 │  intermediate steps
                                 ▼
       ┌─── tools ───┐
       │ price       │  ← can be manipulated (v3 attack category)
       │ news        │  ← can be poisoned with embedded instructions (v2/v4)
       │ risk        │
       │ fundamentals│
       │ correlation │
       │ position    │
       │ submit_rec  │ ← can be forged (v8 GCG approval_forge)
       └─────────────┘
```

**Adversary capabilities (by attack family):**

| Category | Adversary controls | Adversary needs |
|----------|-------------------|------------------|
| v1 Direct injection | User query | Black-box query access |
| v2 Indirect injection | News article content | Ability to write content the agent will fetch |
| v3 Tool manipulation | Tool return value | Compromise of a data feed |
| v4 Context poisoning | Prior conversation turns | Multi-turn session |
| v5 Reasoning hijacking | User query | Black-box + understanding of ReAct |
| v6 Confidence manipulation | User query | Black-box + LLM behavioral knowledge |
| v7 Multi-step compounding | User query, multi-turn | Multi-turn session |
| v8 GCG suffixes | User query | White-box surrogate + similar to target |

**Defender capabilities:**

- Read user query before agent sees it (input filter, semantic filter, perplexity filter)
- Read tool I/O (multi-agent defense can wrap tools)
- Read final output before delivery (output validator, guardrails)
- Read full trajectory post-hoc (v3 defender agent + trajectory defense)

## Why this project is interesting to a PhD panel

| Course concept | Project embodiment |
|----------------|--------------------|
| **M2:** Embeddings | `semantic_filter.py` — Sentence-BERT for prompt-injection detection |
| **M2:** Transfer learning | GCG cross-model transferability study — surrogate→target generalization |
| **M2:** Ensemble methods | `ensemble_defense.py` — XGBoost over base-defense feature vectors |
| **M2:** Few-shot prompting | Planner/critic/mutator agents in v3 are few-shot prompted |
| **M3:** Agentic AI | LangChain ReAct + v3 multi-agent loop (planner/critic/mutator/defender) |
| **M3:** Tool use | 7-tool agent with attack modes per tool |
| **M3:** Reflection | v3 reflection store + `critic.explain_failure()` |
| **M4:** Model evaluation | Chi², McNemar, Wilson CIs, Cohen's h, Bonferroni in `statistical.py` |
| **M4:** Bayesian inference | Beta-Binomial posterior over vulnerability rate (`bayesian_vulnerability()`) |
| **M4:** Information theory | Mutual information + entropy of vulnerability profile |
| **M4:** Explainability | SHAP for attack success prediction; Shapley values for defense attribution |

## The arc to tell — narrative thread

You're not pitching "I built a thing." You're pitching:

> "I tried a thing, got a result, *didn't believe it*, dug into why, found an evaluation artefact, fixed it, got a weaker but real result, then explored what conditions would make the result strong — and discovered that *training-pipeline overlap* (not architectural family) governs transfer."

That is the arc a research panel rewards. It demonstrates **scientific judgement** more than any ASR number.

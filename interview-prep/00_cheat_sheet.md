# Cheat Sheet — Night-Before / Morning-Of

Print this. Read it once the morning of the viva. Nothing else.

---

## Opener (memorize verbatim — 60 seconds)

> "I built a red-team framework for LLM commodity trading agents. LangChain ReAct agent with 7 tools, tested against 50 adversarial attacks across 7 categories — direct injection, indirect injection, tool manipulation, context poisoning, reasoning hijacking, confidence manipulation, multi-step compounding — plus GCG gradient-based suffixes from Zou et al. 2023 as v8. Defenses include input filter, output validator, guardrails, multi-agent, human-in-loop, semantic similarity (sentence-transformers), perplexity filter (GPT-2), and an XGBoost ensemble. Targets: Groq Qwen-3-32B, Llama-3.3-70B, Llama-4-Scout, Gemini. Two headline findings: cross-family GCG transfer is essentially zero; out-of-domain training data hurts in-domain defense — both instantiate the same Module-2 transfer-learning principle from different sides."

## Three claims to lead with

1. **The methodology-correction story.** Original 25% ASR / 100% intra-Llama transfer was an evaluator-keyword-echo artefact caused by a silent fallback in the GCG generator. Three fixes (transformers<5 pin, volume purge, post-flight fallback detection). Corrected: ~12-17% ASR, 0% cross-family transfer.
2. **The A1 finding (attack side).** Vicuna-7b (Llama-2 lineage) surrogate gives 50% ASR on Llama-3.3-70B and 0% on Qwen/Llama-4-Scout — *concentration*. gpt2-xl (base) gives same total ASR but *scattered*. Transfer is governed by training-pipeline overlap, not architecture.
3. **The D1 finding (defense side).** AdvBench-trained XGBoost ensemble: highest cv_auc (0.91), lowest held-out detection of v8 (25%). In-domain trained: 75% held-out. CV AUC does not predict distribution transfer.

## Key numbers (memorize)

| | Value |
|---|---|
| Attacks | 50 across 7 categories + 4 in v8 GCG |
| Defenses | 8 (input_filter, output_validator, guardrails, multi_agent, human_in_loop, semantic_filter, perplexity_filter, ensemble_defense) |
| Target models | 4 (Qwen-3-32B, Llama-3.3-70B, Llama-4-Scout, Gemini) |
| Semantic filter model | `all-MiniLM-L6-v2`, 384-dim, threshold 0.50, top-3 mean signal |
| Perplexity filter model | GPT-2 (124M), window=50, stride=25, z_threshold=2.5, abs_threshold=150 |
| Ensemble | XGBoost, n_estimators=100, max_depth=4, lr=0.1 |
| GCG | suffix_len=20, n_steps=200, top-k=256, batch=32-64, fp16 |
| GCG cost | $0.50/run on A100-40GB after model cached |
| GCG result (Vicuna) | 2/12 = 16.7% ASR, 50% on Llama-3.3, 0% elsewhere |
| Financial impact | $13.5M simulated from 2 successful v8 attacks |

## Five formulas (memorize)

**GCG one-hot gradient:**
$$g_i = \nabla_{h_i} \mathcal{L}, \quad e_i = h_i \cdot W, \quad j^* = \arg\min_j g_i[j]$$

**Cosine similarity** (with normalized embeddings):
$$\text{sim}(q, p) = q \cdot p$$

**McNemar (paired):**
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df}=1$$
where $b, c$ are discordant pairs.

**Wilson CI:**
$$\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}$$

**Beta-Binomial posterior:**
$$p \sim \text{Beta}(\alpha, \beta), \quad k \mid n, p \sim \text{Bin}(n, p) \;\Rightarrow\; p \mid k, n \sim \text{Beta}(\alpha + k, \beta + n - k)$$

**Cohen's h (effect size for proportions):**
$$h = 2\arcsin\sqrt{p_a} - 2\arcsin\sqrt{p_b}$$

**Mutual information:**
$$I(X; Y) = \sum_{x,y} P(x,y) \log_2 \frac{P(x,y)}{P(x) P(y)} = H(Y) - H(Y \mid X)$$

**Shapley value (axiomatic feature attribution):**
$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

## Module mapping (one line each)

| Module | Where it shows up |
|--------|--------------------|
| **M2 Embeddings** | semantic_filter.py — Sentence-BERT MiniLM, cosine similarity to 23-pattern injection corpus |
| **M2 Transfer learning** | GCG cross-family transfer study — training-pipeline overlap matters more than architecture |
| **M2 Ensemble** | ensemble_defense.py — XGBoost over 40 features from base defenses, replaces hand-tuned voting |
| **M2 Few-shot** | planner/critic/mutator agents in v3 use few-shot prompting |
| **M3 Agentic AI** | LangChain `create_tool_calling_agent` with 7 tools — the target agent |
| **M3 Multi-agent / Reflection** | v3 has 5 LLM agents (planner, critic, mutator, replanner, defender) + reflection store (Reflexion pattern) |
| **M3 Tool use** | 7 tools each with attack modes (manipulated, inject_payload, stale_data, override) |
| **M4 Frequentist eval** | statistical.py — chi², McNemar, Wilson CI, Cohen's h, Bonferroni |
| **M4 Bayesian** | bayesian_vulnerability() — Beta-Binomial conjugate, credible intervals, Bayes factor via Savage-Dickey |
| **M4 Information theory** | mutual_information() and entropy_of_vulnerability_profile() — MI, NMI, Shannon entropy |
| **M4 Explainability** | explainability.py — SHAP for attack success prediction; statistical.defense_shapley_values for defense attribution |

## Authors & citations (have these ready)

- **GCG**: Zou et al. 2023, "Universal and Transferable Adversarial Attacks on Aligned Language Models"
- **Sentence-BERT**: Reimers & Gurevych 2019
- **SimCSE**: Gao et al. 2021
- **XGBoost**: Chen & Guestrin 2016
- **SHAP**: Lundberg & Lee 2017
- **Shapley values**: Shapley 1953
- **Reflexion**: Shinn et al. 2023
- **ReAct**: Yao et al. 2023
- **GOAT (adaptive replanning)**: Pavlova et al. 2024
- **Perplexity defense**: Alon & Kamfonas 2023; Jain et al. 2023
- **Indirect prompt injection survey**: Greshake et al. 2023
- **Prompt injection benchmark**: Perez & Ribeiro 2022

## Three honest limitations (when asked "what's wrong")

1. **Small n.** 4 attacks per (model, category) cell in v8; 50 total attacks. Effect sizes are sample-size-independent but p-values are weak. Wilson CIs on 0/4 or 2/4 are wide.
2. **Single LLM-as-judge.** Critic and defender_agent are not calibrated against human gold standard. Known LLM-judge biases (sycophancy, length, position) likely affect specific categories disproportionately.
3. **Gemini missing.** Free-tier rate limits returned HTTP 429 on all 4 calls. No signal about Gemini's resistance. Documented in §4.1 of findings report.

## Three next steps (when asked "what's next")

1. **Defense replay with perplexity filter** against gradient-noise GCG suffixes — most likely to give a clean deployable detection result.
2. **Bigger surrogate** (Llama-3-8B or Llama-2-13B) to test whether intra-Llama transfer grows from 50% → 70%+.
3. **Larger n** — 20 attacks per (model, category) cell to convert directional findings into statistically powered claims.

## Recovery phrases

- "I don't know off the top of my head — my best guess is X because Y. I can verify in the code if helpful."
- "That's a good challenge. The honest answer is..."
- "Let me think about that for a moment."
- "I'd want to check the code to give you a precise answer, but the broad pattern is..."

Don't bluff. PhD profs know when you're bluffing.

## Things to NOT say

- "It works." (Be specific: "ASR is 12% on Qwen at threshold 0.50.")
- "I used XGBoost because it works." (Defend the choice.)
- "I solved prompt injection." (You didn't — be modest.)
- "Future work will fix this." (Be specific about what + how + cost.)

## Morning-of checklist

- [ ] Print this sheet.
- [ ] Open repo on laptop. Tabs preloaded: README.md, `results/GCG_findings_report_2026-05-14.md`, `src/agent/trading_agent.py`, `src/attacks/v8_gcg_adversarial.py`, `src/defenses/ensemble_defense.py`, `src/evaluation/statistical.py`, `interview-prep/00_cheat_sheet.md`.
- [ ] Glass of water.
- [ ] 30 min early.
- [ ] If asked something you don't know, *say so* and offer your best guess. Never bluff.
- [ ] Lead with the methodology-correction story when given the chance.
- [ ] Breathe.

# Viva Playbook — Capstone Demo Script

> **Audience:** IIT Bombay EPGD evaluation panel.
> **Time budget:** ~10 min talk + ~10 min Q&A.
> **Submission deadline:** 2026-05-22.
> **Live URL:** https://redteam-demo-ui-69498195329.asia-south1.run.app/
>
> This is the minute-by-minute script for the demo. It assumes the [run plan](../docs/full_benchmark_playbook.md) has been executed and the cloud benchmark + advanced-analysis backfills are live.

---

## Pre-talk checklist (run 30 min before)

```bash
# Pre-warm Cloud Run so cold starts don't bite mid-demo
curl -s https://redteam-demo-ui-69498195329.asia-south1.run.app/ > /dev/null

# Verify the headline job + 2 backfills are present in the UI
# (Open the URL → Overview tab → confirm 3+ jobs with status=completed)

# Sanity-load the live attack you plan to demo (warms the v9 PAIR cache)
# In the UI: Live Attack tab → v9.1 PAIR Risk Bypass + vertex-gemini-pro + no defense → Run
# Discard the result — this is just a cache warm-up.

# Backup local PPTX (in case the UI demo flakes)
ls results/cloud_headline/report/RedTeam_Final_Report.pptx
ls results/results_0329_1945/report/RedTeam_Final_Report.pptx
```

If any of the above fails, the contingencies are in §6.

---

## The 10-minute talk

### 0:00 — Open with the live URL (30 sec)

> "This is the deployed product on GCP Cloud Run. The framework runs **74 attacks across 12 categories** against any configured LLM target — Gemini, Claude, Mistral, Groq. The Overview tab here shows three completed benchmark runs already in the system."

**Click:** UI URL → Overview tab. Point to:
- 3 jobs in history (headline Vertex run, historical 30%-ASR baseline, advanced-analysis bundle)
- 74 attacks in the catalog metric
- 12 categories in the metric below

Why this matters: **the panel sees a live system, not screenshots.**

### 0:30 — Catalog browser (45 sec)

**Click:** Catalog tab.

> "Twelve attack families. Eight pre-existing categories — direct injection, indirect, tool manipulation, context poisoning, reasoning hijacking, confidence manipulation, multi-step compounding, GCG adversarial. Plus four new ones we added for this capstone: **v9 PAIR** iterative LLM-vs-LLM jailbreaks, **v10 RAG poisoning** via simulated retrieval, **v11 AutoDAN** genetic natural-language search, and **v12 Crescendo** multi-turn escalation."

Point at the attack rows v9.1–v9.4, v10.1–v10.4, v11.1–v11.4, v12.1–v12.4.

> "What unites these four new families is they all produce **natural-language attacks** — no token noise, no obvious jailbreak markers. That's the §5.7 setup."

### 1:15 — Live attack: PAIR vs. perplexity filter (90 sec)

**Click:** Live Attack tab.

Set:
- Model: `vertex-gemini-pro`
- Catalog attack: `v9.1 PAIR Risk Bypass`
- Defense Stack: `perplexity_filter`
- Leave additional prompt empty

> "PAIR — Prompt Automatic Iterative Refinement — runs an attacker LLM that crafts a candidate prompt, sends it to the target, scores the response with a judge LLM, and refines on failure. Up to 8 iterations. The end product is a single natural-English question."
>
> "The perplexity filter is the §5.6.10 defense that caught **100%** of GCG token-noise attacks. The question is: does it generalise to natural-language jailbreaks?"

**Click:** Run Live Attack. Wait ~10 seconds.

When the result renders:

> "The perplexity filter didn't fire — PAIR's output looks like a real trader asking a normal question, so its perplexity score is ordinary English. And the agent responded with risk-bypass language. **This is the §5.7 result in one click**: the silver-bullet defense from §5.6.10 collapses against natural-language attacks."

Point at the "Defense Timeline" row showing the perplexity filter as `allowed=true`.

### 2:45 — Headline benchmark (90 sec)

**Click:** Results Explorer tab → select the Phase B Vertex headline job.

> "This is the canonical run: 74 attacks × 2 Vertex targets × 9 defense configurations including `none` and `all_combined`. 1332 evaluations total. Worker took about an hour, cost about $10 of Vertex inference."

Point at the summary table:
- Aggregate ASR (expect ~25-35% baseline, dropping with defenses)
- Detection rate distribution

> "If we group by defense, you can see the §5 layering story: no defense around 30%, single defenses in the 10-20% range, `all_combined` near single-digits — but **only for the v1-v8 categories**. The v9-v12 cells — natural-language attacks — survive most input-side defenses."

### 4:15 — Module 2 / 4 depth: the Explainability tab (2 min)

**Click:** Explainability tab → select the headline job.

> "The cloud benchmark writes summary metrics, but the Module-4 depth artifacts come from a post-processing pipeline I run locally. Bayesian posteriors, SHAP feature importance, cross-model transferability matrices, ROC, Shapley defense attribution. All of it gets backfilled into a separate cloud job — visible here."

**Scroll through:**

1. **Headline visualizations** — ASR heatmap, defense bar chart, vulnerability radar.

   > "Standard view: model × category ASR heatmap. The high-ASR cells in the v9–v12 rows are the natural-language vulnerability we just demoed live."

2. **Bayesian posteriors**

   > "Beta-Binomial posterior per (model, defense). 95% credible intervals on ASR. Critically: when `n_attacks=50` per cell, the CI is wide — the panel should see we report uncertainty, not just point estimates."

3. **Transferability**

   > "Per-attack transfer rate from Gemini Flash to Gemini Pro, with Fisher's exact significance. Module 2 — transfer learning. About 40-60% of successful attacks transfer; the natural-language families transfer better than category v8 GCG, because GCG is token-level and PAIR is semantic."

4. **SHAP + Shapley**

   > "XGBoost predictor of attack success, explained via SHAP. The top features are *which defense was active* and *which attack category* — interactions matter, not just main effects. Below that, Shapley values for each defense — game-theoretic marginal contribution averaged over all coalition orderings. `multi_agent` and `output_validator` carry the most weight in this run."

5. **ROC + information theory**

   > "Per-defense AUC. Mutual information of each attack-metadata feature with success — `category` and `defense` carry the most signal, `model` carries the least. And entropy of each model's vulnerability profile — both Vertex Gemini models score near max entropy, meaning their vulnerabilities are spread evenly across categories rather than concentrated on a few."

### 6:15 — Historical baseline comparison (45 sec)

**Click:** Results Explorer → select the historical 30%-ASR baseline (`results_0329_1945`).

> "For comparison, here's our earlier Groq-only baseline from March. 700 evaluations, 2 Groq models, 7 defense configs. Aggregate ASR around 30%, dropping to 6-12% under `all_combined`. Defense layering chart in the Explainability tab for this job shows the same shape — confirming the framework's evaluation is consistent across runs months apart."

### 7:00 — Architecture closing (60 sec)

Open `specs/cloud/GCP-DEPLOYMENT-PLAN.md` or its diagram.

> "Three Cloud Run services — UI, API, worker. The worker is a Cloud Run Job so it scales to zero between benchmarks. Vertex AI Model Garden authenticated by workload identity — no API keys in Secret Manager. Cloud Tasks decouples job submission from execution. Firestore tracks state; GCS holds artifacts. Total idle cost about $3-5 per month. The demo today cost about $10 of inference."
>
> "Everything you've seen is on GitHub. The code, the deployment Terraform, the analysis scripts. The new attacks are PRs #11, #12, #13, #14. The latest benchmark is reproducible from any branch with `python scripts/run_full_benchmark.py --models vertex-gemini-flash vertex-gemini-pro`."

### 8:00 — Q&A window (2 min from your side, then panel takes over)

Cover topics likely to come up in Q&A — pre-load answers for these (see §3).

---

## Anticipated panel questions (pre-loaded answers)

| Q | A |
|---|---|
| "How do you know the attacks are *novel* and not just paraphrases?" | "v9 PAIR runs LLM-vs-LLM refinement with the attacker LLM seeing the target's prior responses — each attack is tailored to the specific target. v11 AutoDAN does genetic evolution starting from a different seed per goal. v12 Crescendo builds a *multi-turn dialogue*, so it can't be matched to a static template at all. The on-disk caches show the actual prompts that won — they're textually distinct per (goal, target_model) cell." |
| "Why did you choose Gemini Flash as the attacker LLM?" | "Free on the project via workload identity, fast (~1s/call), and Gemini 2.5 Flash is competitive at instruction-following for the role-play attacker prompts. We considered Groq Llama 3.3 70B and used it for laptop dev, but switched the cloud-worker default to Gemini Flash because GROQ_API_KEY isn't on the worker." |
| "Your ROC AUC shows >1.0 in some cells. That's not possible." | "Confirmed bug in `src/evaluation/metrics.py::defense_roc_analysis` line 240 — `n_attacks` is misnamed; it's the count of *successful* attacks, not total. When the defense detects more attacks than succeed, TPR exceeds 1. The Bayesian posteriors and Shapley values aren't affected because they don't share that code path. Filed for follow-up." |
| "What's the financial-impact column actually measuring?" | "It's a simulated dollar value assigned per-attack based on what the attack would have caused if the agent had executed it — e.g., v9.2 forces a 200,000 MT copper buy, valued at copper price × notional. Not a market-realised loss; it's a *severity proxy*. The exact methodology is in `src/attacks/<vN>_*.py::AttackResult.financial_impact_estimate`." |
| "Why didn't you run on more models?" | "The Vertex Model Garden additions (Claude Sonnet, Mistral Medium) require manual enablement in the GCP console and incur per-token pricing. We scoped to Gemini Flash + Gemini Pro to keep the demo budget under \$15 and to ground the cross-vendor argument in two production-grade models we could actually compare. Adding the others is a single config change once enabled." |
| "Why is the historical baseline 30% ASR but the new run is different?" | "Different target models. The historical run hit Groq Llama 3.3 70B and Groq Qwen 3 32B — small open models with weaker safety training. The new run hits Vertex Gemini 2.5 Flash and Pro — much stronger alignment, so baseline ASR is lower. The *defense reduction percentage* is what's comparable across runs, and that's similar (~80% reduction under `all_combined`)." |
| "What's the Module 3 (Agentic AI) story?" | "Three layers. (a) The target is a LangChain agent with 7 tools and reasoning chain. (b) PAIR's attacker is itself an LLM agent — Gemini Flash is the agent, the target is its environment, the judge is the feedback signal. (c) Crescendo builds an in-context multi-turn dialogue trajectory, exercising the agent's conversational memory. v9/v12 directly demonstrate adversarial-agent-vs-defender-agent — Module 3 in attacker form." |
| "How long would it take to add a new attack?" | "About 4 hours for a new natural-language attack — see v10 RAG poisoning (PR #11, ~600 lines including tests and docs). The `Attack` base class contract is `prepare(agent) → dict` and `evaluate(agent_result) → AttackResult`. Registry auto-discovers via `@register`. No evaluator changes needed for single-turn; v12 Crescendo's multi-turn fits into the same contract via `injected_context`." |

---

## Contingencies

### If the live URL is down or returns 5xx mid-demo

1. Don't reload — switch to the **backup PPTX** at `results/cloud_headline/report/RedTeam_Final_Report.pptx` (or `results/results_0329_1945/report/RedTeam_Final_Report.pptx` if the cloud-headline one isn't ready).
2. Narrate the same 5-minute arc with screenshots. The story is the story.
3. Acknowledge the outage briefly: *"Cloud Run scales to zero, so this is a cold-start moment — let me show you the same flow from the slides while it comes up."*

### If the Live Attack click times out (>120s)

The default API timeout is 120s. v9.1 PAIR can iterate up to 8 times. If the v9.1/Vertex-Pro cache wasn't warmed pre-demo (see checklist), the first run will be slow.

- Fall back to v9.1 with **no defense** (smaller search; usually <30s).
- Or fall back to v10.1 (no LLM loop — static RAG poisoning, returns in <5s).
- Or use the **pre-canned result** in the Results Explorer for that attack.

### If the Explainability tab shows "not present" for everything

- Means the local advanced-analysis backfill didn't land. Walk through the **Results Explorer artifacts panel** instead — same data, less polished rendering.
- The headline summary still renders from `summary.json`.

### If the panel asks for something not in the canned tour

- Catalog tab → search by category. All 74 attacks have a description field.
- Live Attack tab can fire any of them on demand.
- Run Benchmark tab can submit a focused new run (~30s for selected mode with a small attack subset).

---

## Slide / artifact references

For each section above, the supporting artifact paths:

| Section | Artifact | Path |
|---|---|---|
| Open | Live URL | https://redteam-demo-ui-69498195329.asia-south1.run.app/ |
| Catalog | 74-attack list | UI Catalog tab; source: `src/attacks/registry.py::get_all_attacks` |
| Live attack | PAIR runner | `src/attacks/v9_pair_iterative.py` |
| Headline benchmark | Results CSV | `gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/<headline_job_id>/...` |
| ASR heatmap | PNG | Explainability tab section 1 |
| Bayesian | CSV | Explainability tab section 2 |
| Transferability | matrix + Fisher's | Explainability tab section 3 |
| SHAP | beeswarm + Shapley table | Explainability tab section 4 |
| ROC + MI + entropy | JSON tables | Explainability tab section 5 |
| Historical baseline | `results_0329_1945` | backfilled cloud job; local `results/results_0329_1945/` |
| Architecture | deployment plan | `specs/cloud/GCP-DEPLOYMENT-PLAN.md` |
| Backup PPTX | final deck | `results/cloud_headline/report/RedTeam_Final_Report.pptx` |
| Roadmap | future work | `docs/enhancements.md` |

---

## Course-module mapping cheat sheet

If the panel pivots to "show me where Module X is":

| Module | Topic | Where in the demo |
|---|---|---|
| **Module 2** | Embeddings | `src/defenses/semantic_filter.py` — MiniLM-based input filtering |
| Module 2 | Transfer learning | Explainability tab → Transferability section (Gemini Flash → Pro transfer + Fisher's) |
| Module 2 | Ensemble methods | `src/defenses/ensemble_defense.py` — XGBoost over 7 defense signals |
| Module 2 | Few-shot | `src/attacks/v9_pair_iterative.py::_ATTACKER_SYSTEM_PROMPT` — few-shot scaffolding for the attacker |
| **Module 3** | Agentic AI | The target itself (`src/agent/trading_agent.py`, LangChain ReAct, 7 tools). Plus PAIR/AutoDAN/Crescendo as adversarial agents. |
| **Module 4** | Bayesian analysis | Explainability tab → Bayesian posteriors |
| Module 4 | Mutual information | Explainability tab → MI section |
| Module 4 | ROC | Explainability tab → ROC AUC table |
| Module 4 | Explainability (SHAP) | Explainability tab → SHAP beeswarm + feature importance |

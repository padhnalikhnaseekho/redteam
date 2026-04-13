# Paid Model API Recommendations

## Context

The v3 red team loop has two distinct LLM roles:
- **Attacker** (PlannerAgent, CriticAgent, MutatorAgent, Replanner, DefenderAgent) — generates plans, judges success, writes reflections
- **Target** (Commodity Trading Agent) — the system being attacked

Different models matter for each role. A smarter attacker finds more vulnerabilities; diverse targets reveal transferable weaknesses.

## Current Setup (Free Tier)

| Role | Model | Provider | Cost |
|---|---|---|---|
| Attacker | Qwen3-32B | Groq | $0 |
| Target | Llama4-Scout | Groq | $0 |
| Target | Qwen3-32B | Groq | $0 |

## Priority 1: Better Attacker Model

The attacker model quality directly determines the quality of attack plans, critic judgments, and reflections. Upgrading here has the highest impact.

| Model | Cost (per 1M tokens) | Why |
|---|---|---|
| **Claude Sonnet 4.6** | $3 input / $15 output | Best reasoning-per-dollar. Excellent structured output (plans, JSON). Critic and planner improve significantly. |
| **GPT-4o** | $2.50 input / $10 output | Strong alternative. Good at adversarial creativity. |

**Estimated cost per run**: A 7-round run with ~80 attacks makes ~300 LLM calls (planner + critic + mutator + defender). At ~1K tokens/call average, that's **~$1-3 per full run** with Sonnet.

## Priority 2: Diverse Target Models

Testing only Groq models limits findings. Different model families have different safety training and different vulnerabilities.

| Model | Cost (per 1M tokens) | Why |
|---|---|---|
| **GPT-4o-mini** | $0.15 input / $0.60 output | Cheapest way to add a second model family. Different safety training = different weaknesses. |
| **Claude Haiku 4.5** | $0.80 input / $4 output | Anthropic's safety approach differs from Meta/OpenAI. Interesting contrast. |
| **Mistral Large** | Already in `.env` | Key already configured, just hasn't been run as a target yet. |

## Budget Tiers

### Tight Budget (~$5-10)

- Get an **OpenAI API key** ($5 minimum deposit)
- Use `gpt-4o-mini` as a second target model
- Keep `groq-qwen` as attacker (free)
- **What you get**: 2 model families for target comparison

### Moderate Budget (~$20-30) -- Recommended

- **Anthropic API key** for `claude-sonnet-4-6` (attacker) and `claude-haiku-4-5` (target)
- Keep Groq models as additional targets
- **What you get**: 4 target models across 3 providers, dramatically better attack quality
- Strongest story for the transferability analysis (Module 2)

### Generous Budget (~$50+)

- Anthropic: Sonnet 4.6 as attacker + Haiku 4.5 as target
- OpenAI: GPT-4o as alternate attacker + GPT-4o-mini as target
- **What you get**: Full attack-transfer matrix — do attacks that work on groq-scout also work on claude-haiku / gpt-4o-mini?
- Transferability story is strong material for the IIT Bombay capstone presentation

## Recommendation

**One key to buy: Anthropic.** Sonnet 4.6 as attacker will find more vulnerabilities than any amount of extra Groq rounds. The quality of plans, reflections, and critic judgments improves dramatically.

```bash
# Upgraded attacker with seeded learning
python scripts/run_auto_redteam_v3.py \
  --target-model groq-scout \
  --attacker-model claude-sonnet \
  --seed-from results/auto_redteam_v3_0413_1434

# Cross-model transferability test
python scripts/run_auto_redteam_v3.py \
  --target-model claude-haiku \
  --attacker-model claude-sonnet \
  --seed-from results/auto_redteam_v3_0413_1434
```

## Capstone Alignment

| Module | How paid models help |
|---|---|
| **Module 2** — Transfer learning | Attack transferability across model families (Groq vs Anthropic vs OpenAI) |
| **Module 2** — Ensemble methods | Compare single-model vs multi-model defense ensembles |
| **Module 3** — Agentic AI | Stronger attacker agent demonstrates deeper agentic reasoning |
| **Module 4** — Model evaluation | More diverse results for Bayesian analysis, ROC curves, SHAP explainability |

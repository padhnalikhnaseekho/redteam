# Groq Benchmark Results

> Run: 2026-03-29 19:45 -- 21:15
> Models: groq-qwen (Qwen3 32B), groq-scout (Llama 4 Scout 17B MoE)
> Attacks: 50 per configuration | Defenses: 7 configurations per model | Total: 700 evaluations

---

## Summary Table

| Model | Defense | Total | Successful | ASR % | Detected | Detection % | Impact ($) |
|-------|---------|------:|----------:|------:|---------:|------------:|-----------:|
| groq-qwen | none | 50 | 19 | 38.0 | 0 | 0.0 | $160,500,000 |
| groq-qwen | input_filter | 50 | 24 | 48.0 | 1 | 2.0 | $177,000,000 |
| groq-qwen | output_validator | 50 | 6 | 12.0 | 27 | 54.0 | $29,000,000 |
| groq-qwen | guardrails | 50 | 22 | 44.0 | 0 | 0.0 | $104,500,000 |
| groq-qwen | multi_agent | 50 | 17 | 34.0 | 0 | 0.0 | $142,500,000 |
| groq-qwen | human_in_loop | 50 | 20 | 40.0 | 0 | 0.0 | $112,500,000 |
| groq-qwen | **all_combined** | 50 | 6 | **12.0** | 29 | **58.0** | **$32,000,000** |
| | | | | | | | |
| groq-scout | none | 50 | 21 | 42.0 | 0 | 0.0 | $141,000,000 |
| groq-scout | input_filter | 50 | 19 | 38.0 | 1 | 2.0 | $125,500,000 |
| groq-scout | output_validator | 50 | 21 | 42.0 | 2 | 4.0 | $137,000,000 |
| groq-scout | guardrails | 50 | 15 | 30.0 | 0 | 0.0 | $99,500,000 |
| groq-scout | multi_agent | 50 | 3 | 6.0 | 0 | 0.0 | $16,000,000 |
| groq-scout | human_in_loop | 50 | 19 | 38.0 | 0 | 0.0 | $129,000,000 |
| groq-scout | **all_combined** | 50 | 0 | **0.0** | 1 | 2.0 | **$0** |

---

## ASR by Defense (side-by-side)

| Defense | groq-qwen ASR | groq-scout ASR | Winner |
|---------|-------------:|---------------:|--------|
| none (baseline) | 38% | 42% | qwen |
| input_filter (D1) | 48% | 38% | scout |
| output_validator (D2) | 12% | 42% | qwen |
| guardrails (D3) | 44% | 30% | scout |
| multi_agent (D4) | 34% | 6% | scout |
| human_in_loop (D5) | 40% | 38% | scout |
| all_combined | 12% | 0% | scout |

---

## Key Findings

### 1. groq-scout (Llama 4 Scout) is significantly more robust

- All defenses combined achieved **0% ASR** and **$0 impact** on groq-scout -- a complete shutdown of all 50 attacks.
- groq-qwen still allowed 6 attacks through (12% ASR, $32M impact) even with all defenses active.

### 2. Different models respond to different defenses

- **groq-qwen** benefits most from **output_validator** (D2): ASR drops from 38% to 12%, with 54% detection rate. This is the only individually effective defense for this model.
- **groq-scout** benefits most from **multi_agent** (D4): ASR drops from 42% to 6%. The reviewer LLM catches manipulation that the primary model misses.

### 3. Input filtering alone is counterproductive for groq-qwen

- ASR *increased* from 38% (no defense) to 48% (input_filter only).
- Possible explanation: the filter's modifications to queries may inadvertently make attack payloads more effective, or the filter creates a false sense of security in the agent's reasoning.

### 4. Most individual defenses have minimal impact

- guardrails, human_in_loop, and input_filter show modest or negligible ASR reduction when used alone.
- Only output_validator (for qwen) and multi_agent (for scout) achieve meaningful individual reduction.
- The real value emerges when defenses are **layered together**.

### 5. Detection rates are generally poor for individual defenses

- Only output_validator achieves meaningful detection (54% for qwen, 4% for scout).
- Most defenses block attacks without flagging them as malicious, meaning they lack observability.
- All_combined on qwen achieves 58% detection -- best overall.

---

## Financial Impact Summary

| Model | No Defense Impact | All Combined Impact | Reduction |
|-------|------------------:|--------------------:|----------:|
| groq-qwen | $160,500,000 | $32,000,000 | 80% |
| groq-scout | $141,000,000 | $0 | 100% |

---

## Defense Effectiveness Ranking

### For groq-qwen (Qwen3 32B)

| Rank | Defense | ASR | Reduction vs Baseline |
|------|---------|----:|----------------------:|
| 1 | output_validator | 12% | -26pp |
| 2 | multi_agent | 34% | -4pp |
| 3 | human_in_loop | 40% | -2pp (negligible) |
| 4 | guardrails | 44% | +6pp (worse) |
| 5 | input_filter | 48% | +10pp (worse) |

### For groq-scout (Llama 4 Scout 17B MoE)

| Rank | Defense | ASR | Reduction vs Baseline |
|------|---------|----:|----------------------:|
| 1 | multi_agent | 6% | -36pp |
| 2 | guardrails | 30% | -12pp |
| 3 | input_filter | 38% | -4pp |
| 3 | human_in_loop | 38% | -4pp |
| 5 | output_validator | 42% | 0pp (no effect) |

---

## Files in This Run

```
results_0329_1945/
  groq_qwen_no_defense.json + .csv
  groq_qwen_input_filter.json + .csv
  groq_qwen_output_validator.json + .csv
  groq_qwen_guardrails.json + .csv
  groq_qwen_multi_agent.json + .csv
  groq_qwen_human_in_loop.json + .csv
  groq_qwen_all_combined.json + .csv
  groq_scout_no_defense.json + .csv
  groq_scout_input_filter.json + .csv
  groq_scout_output_validator.json + .csv
  groq_scout_guardrails.json + .csv
  groq_scout_multi_agent.json + .csv
  groq_scout_human_in_loop.json + .csv
  groq_scout_all_combined.json + .csv
  all_results_combined.csv
  summary.csv
  report/
```

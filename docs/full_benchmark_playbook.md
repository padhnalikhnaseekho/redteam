# Full Benchmark Playbook — v1–v12 attack suite

> Run-book for executing the full 74-attack red-team suite against the three
> Groq free-tier targets and the available defenses, then publishing the
> results to the deployed Cloud Run UI.
>
> **As of:** v10 (RAG poisoning) is implemented on `enhancement/rag-poison`.
> v9 PAIR is on master; v11 AutoDAN and v12 Crescendo are on their own
> feature branches awaiting merge. See **Prerequisites** below.

---

## Prerequisites — branches that must be merged first

The full suite depends on four feature branches that are all open as of the
last edit. Merge them to `master` (in this order) before kicking off the run:

| Branch | What it adds | Status |
|---|---|---|
| `enhancement/autodan` | v11 genetic NL jailbreak (4 attacks + seeder + tests) | open PR |
| `enhancement/crescendo` | v12 multi-turn escalation (4 attacks + seeder + tests) | open PR |
| `enhancement/rag-poison` | v10 RAG / data-feed poisoning (4 attacks + tests, no seeder) | open PR |
| (already merged) | v9 PAIR, cloud infrastructure (UI/API/worker), GCS-backed PAIR cache | on master |

After all four are on master, total registered attacks = **74** across **12 categories**.

Alternatively, run from the local integration branch
`integration/autodan-crescendo` (which already combines v11 + v12) plus a
local merge of `enhancement/rag-poison` on top — useful if you want to
benchmark before opening all the PRs.

---

## 1. One-time setup

```bash
# Pick whichever worktree has all 12 attack categories. Recommended:
# create a fresh worktree off master once everything is merged.
cd /home/anrd/git/epgd/capstone/redteam-rag-poison

# Share API keys with the main worktree.
ln -sf /home/anrd/git/epgd/capstone/redteam/.env .env

# Confirm the suite count is 74 across 12 categories.
python -c "
from src.attacks.registry import get_all_attacks
from collections import Counter
counts = Counter(a.category.value for a in get_all_attacks())
for cat in sorted(counts): print(f'  {cat}: {counts[cat]}')
print(f'Total: {sum(counts.values())}')
"
```

Expected output:
```
  v10_rag_poisoning: 4
  v11_autodan_genetic: 4
  v12_crescendo_multiturn: 4
  v1_direct_injection: 8
  v2_indirect_injection: 10
  v3_tool_manipulation: 7
  v4_context_poisoning: 5
  v5_reasoning_hijacking: 8
  v6_confidence_manipulation: 6
  v7_multi_step_compounding: 6
  v8_gcg_adversarial: 8
  v9_pair_iterative: 4
Total: 74
```

---

## 2. Seed the iterative-attack caches

v9 / v11 / v12 each run an iterative LLM-vs-LLM search inside `prepare()`.
Without pre-seeding, every benchmark cell re-runs that search from scratch
(8-20 LLM calls per attack per cell). Seed once, re-use forever.

**v10 has no seeding step** — it's a static-injection attack (the poisoned
document is hand-crafted), so it goes straight into the benchmark loop.

```bash
# Seed all three iterative caches against the three Groq targets.
# Total: ~15-25 min, ~600 Groq calls (within free tier).

python scripts/run_pair_generate.py \
    --targets groq-llama groq-qwen groq-scout \
    2>&1 | tee results/seed_pair.log

python scripts/run_autodan_generate.py \
    --targets groq-llama groq-qwen groq-scout \
    2>&1 | tee results/seed_autodan.log

python scripts/run_crescendo_generate.py \
    --targets groq-llama groq-qwen groq-scout \
    2>&1 | tee results/seed_crescendo.log
```

Caches land at:
- `results/pair_attack_cache.json`
- `results/autodan_attack_cache.json`
- `results/crescendo_attack_cache.json`

> **Cloud-run note:** on the deployed Cloud Run worker, the on-disk cache is
> ephemeral. Set `PAIR_CACHE_URI=gs://<results-bucket>/pair_attack_cache.json`
> (and likewise `AUTODAN_CACHE_URI` and `CRESCENDO_CACHE_URI` if the same
> pattern was applied to the v11/v12 runners) so caches persist across job
> runs. This is a Terraform change to `infrastrucuture/terraform/cloud_run.tf`.

---

## 3. Run the full benchmark matrix

74 attacks × 3 targets × ~7 defense configs ≈ 1500 evaluations. Each
evaluation triggers a LangChain ReAct loop on the target, so realistic
Groq call count is ~5-10K. Total wall time ~2-4 hours.

```bash
# Stage per-model so a 429 storm only loses one model's results.
# Use --delay 3 to stay under the 6000 RPM Groq free-tier limit.

python scripts/run_full_benchmark.py \
    --models groq-llama \
    --delay 3 \
    2>&1 | tee results/full_bench_llama.log

python scripts/run_full_benchmark.py \
    --models groq-qwen \
    --delay 3 \
    2>&1 | tee results/full_bench_qwen.log

python scripts/run_full_benchmark.py \
    --models groq-scout \
    --delay 3 \
    2>&1 | tee results/full_bench_scout.log
```

Or all three in one shot (if you trust the rate-limit headroom):

```bash
python scripts/run_full_benchmark.py \
    --models groq-llama groq-qwen groq-scout \
    --delay 3 \
    2>&1 | tee results/full_bench_$(date +%Y%m%d_%H%M).log
```

**Run in the background** for a 3-hour run:
```bash
nohup python scripts/run_full_benchmark.py --models groq-llama groq-qwen groq-scout --delay 3 \
    > results/full_bench.log 2>&1 &
```
or inside `tmux` / `screen`.

**Skip the expensive multi-agent defense** if you're tight on time — that
defense doubles the call count by running a second LLM evaluation per attack:
```bash
python scripts/run_full_benchmark.py ... --skip-multi-agent
```

---

## 4. Output structure

Results land in `results/results_<timestamp>/`:

| File | Content |
|---|---|
| `all_results_combined.csv` | Flat row-per-(model, defense, attack) — the canonical CSV |
| `summary.csv` | Grouped ASR / detection / impact by (model, defense) |
| `<model>_<defense>.{csv,json}` | Per-cell results |
| `RESULTS.md` | Human-readable summary table |
| `report/REPORT.md` (if generated) | Long-form analysis with figures |

---

## 5. Publish to the deployed Cloud Run UI

The `backfill_jobs_to_gcs.py` script normalises a local results directory
into the cloud schema (`results.json` + `summary.json`), uploads to GCS,
and writes a `benchmark_jobs/<job_id>` Firestore document so the run
appears in the UI's job history.

```bash
# Dry-run first to confirm the upload plan.
python scripts/backfill_jobs_to_gcs.py \
    --local-dir results/results_<timestamp> \
    --bucket project-e0bbb103-9e5b-4402-866-redteam-demo-results \
    --project project-e0bbb103-9e5b-4402-866 \
    --dry-run

# Live (drop --dry-run).
python scripts/backfill_jobs_to_gcs.py \
    --local-dir results/results_<timestamp> \
    --bucket project-e0bbb103-9e5b-4402-866-redteam-demo-results \
    --project project-e0bbb103-9e5b-4402-866
```

The script requires:
- ADC set up: `gcloud auth application-default login --no-launch-browser`
- IAM bindings on your account:
  - `roles/storage.objectAdmin` on bucket `project-e0bbb103-9e5b-4402-866-redteam-demo-results`
  - `roles/datastore.user` on project `project-e0bbb103-9e5b-4402-866`
- Local cloud libs: `pip install google-cloud-storage google-cloud-firestore`

After it finishes, refresh the deployed UI:
```
https://redteam-demo-ui-69498195329.asia-south1.run.app/
```
The new job appears in the Overview tab with status `completed` and the
Results Explorer renders the summary + per-row results, including the
v9 PAIR / v10 RAG / v11 AutoDAN / v12 Crescendo attacks.

---

## 6. Estimated cost and runtime

| Stage | Wall time | Groq free-tier calls |
|---|---|---|
| Seed v9 PAIR (3 targets × 4 goals) | ~5-10 min | ~150-300 |
| Seed v11 AutoDAN (3 × 4) | ~10-20 min | ~600-1200 |
| Seed v12 Crescendo (3 × 4) | ~5-10 min | ~120-240 |
| Full benchmark (3 targets, 7 defenses, 74 attacks) | ~2-3 hours | ~5000-10000 |
| Backfill to GCS | ~1 min | 0 (just GCS writes) |
| **Total** | **~2.5-4 hours** | **~6000-12000** |

Free tier (6000 RPM, 1M tokens/day) is enough but tight. Plan to run
overnight if the daily token budget is already partially used.

---

## 7. Quick-win demo subset (if you don't have 3 hours)

For a small live-run that still exercises every new attack category:

```bash
python scripts/run_attacks.py \
    --model groq-llama \
    --category v9_pair_iterative,v10_rag_poisoning,v11_autodan_genetic,v12_crescendo_multiturn
```

16 attacks × 1 target × no defense → ~3-5 min, ~30 Groq calls. Good for
verifying the new attack pipelines work end-to-end before the full run.

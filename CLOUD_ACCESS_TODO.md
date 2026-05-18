# Cloud-Access TODO — gated on Inder granting IAM

> Created: **2026-05-18**
> Capstone deadline: 2026-05-22 (4 days)
> Owner: anrd (padhnalikhnaseekho@gmail.com)
>
> Everything in this list is blocked on IAM bindings for `padhnalikhnaseekho@gmail.com` on Inder's GCP project `project-e0bbb103-9e5b-4402-866`. When access lands, work through this list in order.

---

## Required IAM bindings (Inder runs these)

```bash
# 1. GCS read+write on the results bucket
gcloud storage buckets add-iam-policy-binding \
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results \
  --member=user:padhnalikhnaseekho@gmail.com \
  --role=roles/storage.objectAdmin

# 2. Firestore read+write for benchmark_jobs documents
gcloud projects add-iam-policy-binding project-e0bbb103-9e5b-4402-866 \
  --member=user:padhnalikhnaseekho@gmail.com \
  --role=roles/datastore.user

# 3. Cloud Build submit (for re-deploys)
gcloud projects add-iam-policy-binding project-e0bbb103-9e5b-4402-866 \
  --member=user:padhnalikhnaseekho@gmail.com \
  --role=roles/cloudbuild.builds.editor

# 4. Service Usage (quota project tracking, also required for Cloud Build)
gcloud projects add-iam-policy-binding project-e0bbb103-9e5b-4402-866 \
  --member=user:padhnalikhnaseekho@gmail.com \
  --role=roles/serviceusage.serviceUsageConsumer

# 5. Allow impersonating the build service account
gcloud iam service-accounts add-iam-policy-binding \
  redteam-demo-build@project-e0bbb103-9e5b-4402-866.iam.gserviceaccount.com \
  --member=user:padhnalikhnaseekho@gmail.com \
  --role=roles/iam.serviceAccountUser \
  --project=project-e0bbb103-9e5b-4402-866

# 6. (Optional) Vertex AI inference from a laptop. Skip if Vertex calls
#    only need to happen from the Cloud Run worker (the worker has workload
#    identity already).
gcloud projects add-iam-policy-binding project-e0bbb103-9e5b-4402-866 \
  --member=user:padhnalikhnaseekho@gmail.com \
  --role=roles/aiplatform.user
```

**Smoke test after grant:**
```bash
gcloud storage ls gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/   # should list
gcloud firestore documents list benchmark_jobs --limit=1                      # should list
```

---

## Action items in order

### A. Trigger Cloud Build redeploy (~15 min)

PRs #11 (v10 RAG), #12 (vertex defaults) merged on **2026-05-18** but the deployed worker image is from 2026-05-16. The only configured trigger is `publish-results-to-dashboard` which doesn't fire on master pushes. Redeploy is manual:

```bash
cd /tmp/redteam-deploy   # clean master checkout, already prepared
gcloud builds submit --config=cloudbuild.deploy.yaml \
  --project=project-e0bbb103-9e5b-4402-866 \
  --substitutions=_TAG=master-$(date +%Y%m%d%H%M) --async .
```

When build completes (~10 min after submit):
- Open the deployed UI Catalog tab → confirm **74 attacks across 12 categories** (currently shows ≤70).
- Confirm UI revision is `redteam-demo-ui-00015` or higher.

### B. Merge open PRs (decision point)

5 PRs open against master that ship in the next Cloud Build:

| PR | Branch | What |
|---|---|---|
| #13 | `feature/ui-explainability-tab` | UI Explainability tab — inline SHAP/Bayesian/ROC rendering |
| #14 | `feature/pptx-results-dir-flag` | `create_final_pptx.py --results-dir` flag |
| #15 | `feature/viva-playbook` | `docs/viva_playbook.md` — minute-by-minute demo script |
| #16 | `feature/postprocess-helper` | `scripts/postprocess_cloud_job.py` — one-shot Phase C+D+E |
| (TBD) | `feature/statistical-edge-cases` | Fix `chi_squared_test` for degenerate contingency tables (zero margins) |

All 5 are additive and independent. Merge order doesn't matter. After all merge, trigger Cloud Build a second time so the UI picks up the Explainability tab.

### C. Run the headline cloud benchmark (~1-2 hours)

Submit from the deployed UI's Run Benchmark tab:

| Field | Value |
|---|---|
| Models | `vertex-gemini-flash`, `vertex-gemini-pro` |
| Mode | `full_matrix` |
| Defenses | (empty = all) |
| Attack Categories | (empty = all 74) |
| Delay seconds | `2` |
| Generate reports | enabled |

Expected: 2 × 9 × 74 = 1332 evaluations, ~1 hr worker wall, ~$5-15 Vertex billing.

Monitor via Firestore (UI Overview tab auto-refreshes).

### D. Post-process the cloud-headline run (~3 min)

Single command (the helper script from PR #16):

```bash
python scripts/postprocess_cloud_job.py \
    --job-id <headline_job_id_from_step_C> \
    --bucket project-e0bbb103-9e5b-4402-866-redteam-demo-results \
    --project project-e0bbb103-9e5b-4402-866 \
    --local-name cloud_headline
```

This downloads the CSV, runs `generate_report.py` + `run_advanced_analysis.py`, and backfills the bundle into a new job `histjob-cloud_headline-analysis` visible in the UI's Overview tab.

**Deps the helper needs on the laptop:**
```bash
pip install "shap<0.47" "xgboost<3" "numpy<2"
```
(Conda default ships shap 0.51 + xgboost 3 + numpy 2 which conflict with the project's pandas. Confirmed during local pipeline validation on 2026-05-18.)

### E. Backfill the historical baseline (~30s)

So the UI shows the §5.6.10 "30% ASR baseline" comparison alongside the cloud headline:

```bash
python scripts/backfill_jobs_to_gcs.py \
    --local-dir results/results_0329_1945 \
    --bucket project-e0bbb103-9e5b-4402-866-redteam-demo-results \
    --project project-e0bbb103-9e5b-4402-866
```

The local advanced-analysis artifacts for `results_0329_1945/report/` were generated on **2026-05-18** (Bayesian, SHAP, transferability, ROC, Shapley all present) so they'll backfill alongside.

### F. Demo dry-run (~30 min)

Follow `docs/viva_playbook.md` end-to-end. Check the 5-min talk script works against the live URL with all backfilled jobs. Pre-warm Cloud Run + the v9.1 PAIR cache against `vertex-gemini-pro`.

---

## Known issues to flag during viva (or fix beforehand)

| Issue | Where | Severity | Plan |
|---|---|---|---|
| ROC AUC > 1.0, TPR > 1.0 | `src/evaluation/metrics.py:240` — `n_attacks` is misnamed; it's `sum(success)` not `len(group)`. TPR = detected/successful instead of detected/total. | Cosmetic — affects ROC values in the Explainability tab only. Bayesian, Shapley, SHAP, transferability not affected. | Acknowledge in viva ("known bug, root cause cited, doesn't affect the headline numbers"). Fix as a follow-up PR if time permits. |
| Chi-squared zero-frequency crash | `src/evaluation/statistical.py:37` — `chi2_contingency` raises when a contingency row/column sums to zero (defense reaches 0% or 100% ASR). | Was blocking `generate_report.py`. | **Fixed locally 2026-05-18**, pending PR to master. |
| Local Vertex calls warn "no quota project" | conda default ADC | Cosmetic warning. Vertex calls fail with "GOOGLE_CLOUD_PROJECT not set" → v9/v11/v12 gracefully fall back to PRECOMPUTED on laptop runs. Cloud worker is unaffected (workload identity sets the project). | Local laptop runs: `export GOOGLE_CLOUD_PROJECT=project-e0bbb103-9e5b-4402-866`. Or just accept the PRECOMPUTED fallback for laptop runs. |
| Conda default shap/xgboost/numpy mismatch | env-level | Blocks `run_advanced_analysis.py`. | `pip install "shap<0.47" "xgboost<3" "numpy<2"`. Already applied locally; should be in `requirements.txt`. |

---

## Out of scope (deliberate, do not chase)

- Vertex Claude Sonnet + Mistral Medium targets (Model Garden enablement)
- Adaptive V3 loop on cloud worker
- Pre-seeding v9/v11/v12 caches against Vertex (let the live worker iterate)
- Real RAG retriever for v10 (simulated retrieval via `injected_context` is sufficient for the demo)
- Auto-deploy Cloud Build trigger on master push (would need Inder to configure)

---

## Status as of 2026-05-18 17:00 IST

- ✅ Local advanced-analysis pipeline validated end-to-end against `results/results_0329_1945` + `results/defense_layers_0405_1315`
- ✅ Both bundles have full `report/` artifacts (Bayesian, SHAP, transferability, ROC, Shapley, MI, entropy) ready to backfill
- ✅ 5 PRs queued for merge (#13–#16 plus a 6th coming for the chi-squared fix)
- ⏳ Cloud Build redeploy blocked on IAM
- ⏳ Cloud headline benchmark blocked on redeploy
- ⏳ Viva is in **4 days**

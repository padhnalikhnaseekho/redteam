---
feature: publish-existing-results-cloudbuild
status: IMPLEMENTED
stack: cloudbuild-gcs-firestore
created: 2026-05-17
security-review: required
---

# Feature: Publish Existing Results Through Cloud Build

> **Status:** `IMPLEMENTED`

## Requirement

Publish historical artifacts from `results/` into the existing dashboard as completed historical runs without changing live benchmark execution, Live Attack, the worker flow, or the public dashboard API contract.

The upload path must be service-account driven through Cloud Build. It must not rely on a developer's local user identity.

## Implementation

- [x] Add a dedicated historical publisher script: `scripts/publish_results_to_dashboard.py`.
- [x] Add a Cloud Build config for results publishing only: `cloudbuild.results-publish.yaml`.
- [x] Add a minimal publisher dependency file: `requirements-publisher.txt`.
- [x] Add Terraform IAM for the build service account to write result objects and Firestore job metadata using scoped roles.
- [x] Preserve raw historical artifacts under each imported job prefix.
- [x] Create normalized `results.json`, `summary.json`, and `publish_manifest.json` for every imported group.
- [x] Keep current dashboard/API endpoints compatible:
  - `GET /jobs`
  - `GET /jobs/{job_id}/results`
  - `GET /jobs/{job_id}/artifacts`
  - `GET /jobs/{job_id}/artifacts/{artifact_path}`
- [x] Add tests for deterministic IDs, benchmark normalization, adaptive-row mapping, report-only imports, and publish plans.

## Publishing Model

Historical jobs use deterministic IDs:

```text
historical-{source_slug}-{content_hash8}
```

GCS objects are uploaded below:

```text
gs://<results_bucket>/jobs/{job_id}/{source_slug}/
```

Normalized dashboard artifacts live at the job prefix root:

```text
results.json
summary.json
publish_manifest.json
```

Raw historical evidence is preserved below:

```text
raw/<original relative artifact path>
```

Firestore job documents are upserted into `benchmark_jobs` with:

```text
status: completed
config.mode: historical_import
result_uri: gs://<results_bucket>/jobs/{job_id}/{source_slug}
source_path: results/<source>
source_commit: <Cloud Build commit SHA>
source_branch: <Cloud Build branch>
```

## Trigger Setup

Create a Cloud Build trigger on `master` that only fires for `results/**` changes:

```bash
gcloud builds triggers create github \
  --name=publish-results-to-dashboard \
  --repo-owner=inderanz \
  --repo-name=redteam \
  --branch-pattern='^master$' \
  --included-files='results/**' \
  --build-config=cloudbuild.results-publish.yaml \
  --service-account=projects/project-e0bbb103-9e5b-4402-866/serviceAccounts/redteam-demo-build@project-e0bbb103-9e5b-4402-866.iam.gserviceaccount.com \
  --substitutions=_RESULTS_BUCKET=project-e0bbb103-9e5b-4402-866-redteam-demo-results
```

## Safety Controls

- [x] This feature is additive and does not rebuild or redeploy UI/API/worker.
- [x] Unknown schemas are preserved as downloadable artifacts instead of being forced into misleading result rows.
- [x] Historical jobs are marked with `config.mode=historical_import`.
- [x] Live benchmark job IDs are never overwritten.
- [x] Re-publishing unchanged content is idempotent.
- [x] Cloud Build service account owns the write path.
- [x] Build service account uses `roles/storage.objectUser` on the results bucket and `roles/datastore.user` for Firestore job metadata.

## Verification

Run locally:

```bash
python3 scripts/publish_results_to_dashboard.py --results-dir results --dry-run
python3 -m pytest tests/test_publish_results_to_dashboard.py
```

Run through Cloud Build:

```bash
gcloud builds submit \
  --config cloudbuild.results-publish.yaml \
  --project=project-e0bbb103-9e5b-4402-866 \
  --substitutions=_RESULTS_BUCKET=project-e0bbb103-9e5b-4402-866-redteam-demo-results
```

After publish:

- Historical jobs should appear in Results Explorer.
- Benchmark-shaped historical jobs should show result rows and summary metrics.
- Report-only historical jobs should show practical metadata and downloadable artifacts.
- Existing live benchmark jobs should continue to render unchanged.

## Live Execution Evidence

IAM applied with targeted Terraform so unrelated Cloud Run service/image drift was not applied:

```text
terraform_apply: SUCCESS
resources_added:
  - google_project_iam_member.project["build_datastore_user"]
  - google_project_iam_member.project["build_logging_writer"]
  - google_storage_bucket_iam_member.build_results_admin
```

Cloud Build source bucket read access was also granted to the build service account for manual source-bundle reads:

```text
source_buckets:
  - gs://project-e0bbb103-9e5b-4402-866-cloudbuild-source
  - gs://project-e0bbb103-9e5b-4402-866_cloudbuild
role: roles/storage.objectViewer
```

Manual Cloud Build publish:

```text
build_id: 816cefd4-6d49-48a6-bc3f-103fd98bb372
status: SUCCESS
service_account: redteam-demo-build@project-e0bbb103-9e5b-4402-866.iam.gserviceaccount.com
duration: 5m39s
```

Cloud verification:

```text
gcs_historical_objects: 435
firestore_historical_jobs: 30
api_jobs_total: 44
api_historical_jobs: 30
sample_results_job: historical-results-0329-1945-b14dfbbd
sample_results_count: 2100
sample_summary_groups: 14
sample_report_artifacts_job: historical-baseline-mistral-test-1de17862
sample_report_artifacts: 5
```

Trigger creation status:

```text
status: CREATED
trigger_id: 11b02cb8-b780-4c47-a10c-429cf9146b7e
repository: inderanz/redteam
branch: ^master$
included_files: results/**
build_config: cloudbuild.results-publish.yaml
service_account: redteam-demo-build@project-e0bbb103-9e5b-4402-866.iam.gserviceaccount.com
```

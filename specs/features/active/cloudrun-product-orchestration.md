---
feature: cloudrun-product-orchestration
status: DASHBOARD-VALIDATED
stack: python-gcp
created: 2026-05-16
security-review: required
---

# Feature: Cloud Run Product Orchestration

> **Status:** `DASHBOARD-VALIDATED`

## Requirement

Wire the deployed UI/API/worker images into the real product flow described in `specs/cloud/GCP-VERTEXAI-CLOUDRUN-IMPLEMENTATION-PLAN.md`.

## Acceptance Criteria

- [x] UI calls private API with a Google ID token when not running locally.
- [x] UI can run a live single attack through `/run/single`.
- [x] UI can start a benchmark job through `/jobs/benchmark`.
- [x] UI can list historical jobs.
- [x] UI can display parsed benchmark summaries and result rows.
- [x] UI can list and download GCS artifacts through the private API.
- [x] API writes benchmark job metadata to Firestore.
- [x] API enqueues a Cloud Task with an OIDC token using `redteam-tasks-sa`.
- [x] Cloud Task calls `/internal/jobs/{job_id}/start`.
- [x] API starts the Cloud Run worker Job through the Cloud Run Jobs v2 API.
- [x] API passes job configuration to the worker through env overrides.
- [x] Worker updates Firestore job status.
- [x] Worker uploads generated result artifacts to GCS.
- [x] Terraform grants API service account access to act as task and worker service accounts.
- [x] Context7 official Google docs/samples were consulted for ID token, Cloud Tasks, Cloud Run Jobs, and Terraform IAM patterns.

## Security Gate

- [x] This feature changes service-to-service authentication.
- [x] This feature starts background compute jobs.
- [x] This feature writes result artifacts to cloud storage.

Mitigations:

- UI sends Google ID tokens only to the configured API audience.
- Cloud Tasks uses OIDC and a dedicated task caller service account.
- API endpoint is protected by Cloud Run IAM.
- Worker uses a dedicated service account with scoped Firestore/GCS/Vertex access.
- Job configuration is passed through Cloud Run Job env overrides and tracked in Firestore.

## Remaining Deployment Validation

- [x] Rebuild and publish updated images.
- [x] Apply Terraform to deploy services/jobs.
- [x] Smoke test IAP-protected UI.
- [x] Smoke test authenticated API request.
- [x] Smoke test benchmark job from API to Cloud Tasks to worker.
- [x] Confirm GCS artifact upload and Firestore status transition.

Validated image build and deployment:

```text
project: project-e0bbb103-9e5b-4402-866
region: asia-south1
artifact_registry: asia-south1-docker.pkg.dev/project-e0bbb103-9e5b-4402-866/redteam-demo-containers
build_id: 77a3fe1a-75f7-4a90-b7e2-1eb54bc5f5ce
tag: e2e-fix2-20260516
status: SUCCESS
```

Terraform deployment:

```text
terraform_cli: /tmp/redteam-terraform/terraform 1.8.5
ui_url: https://redteam-demo-ui-x3egbjk7ma-el.a.run.app
api_url: https://redteam-demo-api-x3egbjk7ma-el.a.run.app
worker_job: redteam-demo-worker
tasks_queue: redteam-demo-benchmark
results_bucket: gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results
```

## Context7 / Official-Code Grounding

Context7 was queried during implementation for:

- `/hashicorp/terraform-provider-google`: confirmed `google_cloud_run_v2_service_iam_member` for Cloud Run invoker IAM and `google_iap_web_cloud_run_service_iam_member` or binding for IAP access to Cloud Run.
- `/googleapis/google-cloud-python`: confirmed `google.oauth2.id_token.fetch_id_token(request, target_audience)` for Cloud Run service-to-service ID tokens.
- `/googlecloudplatform/python-docs-samples`: confirmed generated Cloud Tasks typed objects for HTTP target tasks and service-account-backed auth tokens.

Implementation notes from that grounding:

- Cloud Tasks must call an HTTPS target when an OIDC/OAuth auth header is configured.
- The task target is the API internal starter endpoint, not the Cloud Run Job container.
- The API starts the Cloud Run Job through the Cloud Run Jobs v2 API and passes job config through env overrides.
- Terraform direct `google_project_iam_member` resources are used for project IAM because the official module hit a plan-time unknown `for_each` limitation with service account emails created in the same apply.

## End-to-End Validation Evidence

Unauthenticated access controls:

```bash
curl -I https://redteam-demo-ui-x3egbjk7ma-el.a.run.app
# Expected: 302 redirect to Google IAP login

curl -i https://redteam-demo-api-x3egbjk7ma-el.a.run.app/health
# Expected: 403 because API is private behind Cloud Run IAM
```

Authenticated API smoke tests:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/health
# {"status":"ok","service":"redteam-api"}

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/models

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/attacks

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/defenses
```

Live single-attack smoke test:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"vertex-gemini-flash","attack_id":"v1.1","defenses":["input_filter"]}' \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/run/single
```

Result:

```text
http_status: 200
model: vertex-gemini-flash
attack_id: v1.1
defense_stack: input_filter
success: false
vertex_call: successful
```

Benchmark job smoke test:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"models":["vertex-gemini-flash"],"defenses":["input_filter"],"attack_ids":["v1.1"]}' \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/jobs/benchmark
```

Result:

```text
job_id: job-0e6da9a1dd14
task_name: projects/project-e0bbb103-9e5b-4402-866/locations/asia-south1/queues/redteam-demo-benchmark/tasks/8373443417060459025
status: completed
run_operation: projects/project-e0bbb103-9e5b-4402-866/locations/asia-south1/operations/674458ea-0a76-4876-a6b8-d3b5dde6ebf3
worker_exit_code: 0
```

Firestore job state:

```text
status: completed
worker_started_at: 2026-05-16T00:40:17.031119+00:00
worker_completed_at: 2026-05-16T00:40:23.844263+00:00
result_uri: gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0e6da9a1dd14/job-0e6da9a1dd14
```

GCS artifacts:

```text
gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0e6da9a1dd14/job-0e6da9a1dd14/results.csv
gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0e6da9a1dd14/job-0e6da9a1dd14/results.json
```

## Known Demo Boundary

Interactive Google login through IAP requires a browser session as an allowed user. The infrastructure behavior was validated by confirming UI redirects to IAP for anonymous users and that the private API accepts valid Google identity tokens. Full UI click-through should be run from an allowed Gmail account after opening the IAP-protected UI URL.

## Enterprise Benchmark Enhancement

The first deployed version proved the orchestration path with a single attack and one defense. The current enhancement moves the cloud runner closer to historical local results under `results/`, especially:

- Multi-model Vertex benchmark runs.
- Explicit job modes: `selected` and `full_matrix`.
- Historical-style artifacts: `results.json`, `results.csv`, `summary.csv`, `summary.json`, and `report/REPORT.md`.
- Model Garden metadata surfaced through `/models`.
- Partner model config for Vertex Claude and Vertex Mistral, gated by Model Garden enablement.

Additional Context7 grounding used for this enhancement:

- `/googleapis/python-genai`: confirmed `genai.Client(vertexai=True, project=..., location=...)` and environment-variable based Vertex setup.
- `/langchain-ai/langchain-google`: confirmed `ChatGoogleGenerativeAI(vertexai=True, project=..., location=...)` as the current LangChain Gemini Vertex path and noted `ChatVertexAI` deprecation guidance.
- `/hashicorp/terraform-provider-google`: confirmed Cloud Run v2 Job IAM resources and Cloud Run service IAM resource patterns.
- `/googleapis/google-cloud-python`: confirmed Google auth and Cloud client-library usage patterns for ADC-backed cloud service calls.

Official Google docs referenced for model behavior and manual enablement:

- Vertex AI Model Garden: https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models
- Gemini inference on Vertex AI: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
- Claude on Vertex AI: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude
- Mistral on Vertex AI: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral

### New API Contract

`GET /models` now returns both a simple list and model metadata:

```json
{
  "models": ["vertex-gemini-flash", "vertex-gemini-pro", "..."],
  "model_metadata": [
    {
      "name": "vertex-gemini-flash",
      "provider": "vertex",
      "family": "gemini",
      "model_id": "gemini-2.5-flash",
      "location": "us-central1",
      "role": "reviewer",
      "requires_model_garden_enablement": false
    }
  ]
}
```

`POST /jobs/benchmark` accepts:

```json
{
  "models": ["vertex-gemini-flash", "vertex-gemini-pro"],
  "defenses": ["input_filter", "output_validator", "all_combined"],
  "attack_ids": ["v1.1", "v1.2"],
  "mode": "full_matrix",
  "include_reports": true,
  "delay_seconds": 1
}
```

Mode behavior:

- `selected`: run selected attacks against selected models using one selected defense stack.
- `full_matrix`: run selected attacks, or all attacks when `attack_ids` is omitted, against each requested model and each requested defense configuration.
- `report_only`: reserved for future precomputed artifact/report refresh.

### Enhancement Validation

Build and deployment:

```text
build_id: 91db729d-2a3d-47e7-85c8-40b3f4bd2dd3
tag: enterprise2-20260516
status: SUCCESS
terraform_apply: SUCCESS
```

Validated full-matrix job:

```text
job_id: job-0b6ecca8f48b
mode: full_matrix
models: vertex-gemini-flash, vertex-gemini-pro
attacks: v1.1, v1.2
defense_configs: none, input_filter, output_validator, all_combined
rows: 16
status: completed
result_uri: gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0b6ecca8f48b/job-0b6ecca8f48b
```

Validated artifact set:

```text
report/REPORT.md
results.csv
results.json
summary.csv
summary.json
```

Summary snapshot:

```text
vertex-gemini-flash: 8 rows, 0 successful attacks, detection 0-50% by defense
vertex-gemini-pro: 8 rows, 0 successful attacks, detection 0-100% by defense
```

### Model Garden Enablement Boundary

Gemini models are the first validated multi-model path because they are available through Google-managed Vertex Gemini APIs.

The repo now has config/code paths for:

```text
vertex-claude-sonnet -> claude-sonnet-4-6, family=claude, publisher=anthropic, rawPredict
vertex-mistral-medium -> mistral-medium-3, family=mistral, publisher=mistralai, rawPredict
```

Before testing these in the deployed product, manually open Vertex AI Model Garden in project `project-e0bbb103-9e5b-4402-866`, accept terms/enable the model card for:

- Claude Sonnet 4.6 (`claude-sonnet-4-6`) in `us-east5`
- Mistral Medium 3 (`mistral-medium-3`) in `us-central1`

If a partner model is not enabled or the configured `model_id`/region is not available, the code returns a clear Vertex Model Garden enablement error instead of silently falling back to another provider.

## Enterprise Dashboard Enhancement

The UI is now the primary product console. Users do not need to browse GCS manually to understand benchmark results.

Dashboard tabs:

- `Overview`: job counts, recent job history, and latest completed summary.
- `Run Benchmark`: model, mode, defense, attack category, attack ID, report, and delay controls.
- `Results Explorer`: completed-job selector, summary metrics, result tables, model/defense filters, and artifact downloads.
- `Live Attack`: single attack execution for quick demos.
- `Catalog`: model, attack, and defense inventories.

New API endpoints powering the dashboard:

```text
GET /jobs?limit=50
GET /jobs/{job_id}/results?include_results=true&result_limit=5000
GET /jobs/{job_id}/artifacts
GET /jobs/{job_id}/artifacts/{artifact_path}
```

Dashboard build and deployment:

```text
build_id: c5c4206c-8727-4dec-854d-b79cbeeb0d8b
tag: dashboard1-20260516
status: SUCCESS
terraform_apply: SUCCESS
```

Validated dashboard API evidence:

```text
GET /jobs?limit=5 -> returns completed and queued benchmark jobs
GET /jobs/job-0b6ecca8f48b/results -> returns metadata, summary groups, and result rows
GET /jobs/job-0b6ecca8f48b/artifacts -> returns report/REPORT.md, results.csv, results.json, summary.csv, summary.json
GET /jobs/job-0b6ecca8f48b/artifacts/summary.csv -> streams downloadable CSV bytes
```

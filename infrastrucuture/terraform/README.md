# Redteam GCP Infrastructure

This folder is intentionally named `infrastrucuture` to match the requested repo path.

It provisions the v1 GCP-native deployment shape from `specs/cloud/GCP-VERTEXAI-CLOUDRUN-IMPLEMENTATION-PLAN.md`:

- Cloud Run services: `redteam-ui`, `redteam-api`
- Cloud Run Job: `redteam-worker`
- Cloud IAP on the UI for Google/Gmail login
- Cloud Tasks queue for benchmark dispatch
- Google service accounts for workload identity
- Vertex AI access through `roles/aiplatform.user`
- Firestore for job status
- GCS for result artifacts
- Artifact Registry for Docker images

## Context7-Grounded Terraform Choices

- `google_cloud_run_v2_service` supports `iap_enabled = true`.
- `google_iap_web_cloud_run_service_iam_member` grants `roles/iap.httpsResourceAccessor` to users/groups.
- `google_cloud_run_v2_service_iam_member` grants `roles/run.invoker` for UI/API and Tasks/API calls.
- Project IAM bindings use direct `google_project_iam_member` resources with static keys so Terraform can plan/apply in one pass after service accounts are created.

## Usage

```bash
cd infrastrucuture/terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Build and push the three container images before applying, or set the image variables to existing deployable images.

Terraform `>= 1.8.0` is required. The Google provider version pinned in `.terraform.lock.hcl` may not validate correctly with older Terraform CLIs.

For this project, the validated deployment used Terraform 1.8.5 from a temporary local binary because the system Terraform was older:

```bash
mkdir -p /tmp/redteam-terraform
curl -L -o /tmp/redteam-terraform/terraform.zip \
  https://releases.hashicorp.com/terraform/1.8.5/terraform_1.8.5_darwin_arm64.zip
unzip -o /tmp/redteam-terraform/terraform.zip -d /tmp/redteam-terraform
```

If Terraform cannot read Artifact Registry through ADC, pass the current gcloud OAuth access token explicitly:

```bash
GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token) \
  /tmp/redteam-terraform/terraform init

GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token) \
  /tmp/redteam-terraform/terraform plan

GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token) \
  /tmp/redteam-terraform/terraform apply
```

If the Artifact Registry repository was already created manually, import it once before applying:

```bash
GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token) \
  /tmp/redteam-terraform/terraform import \
  google_artifact_registry_repository.containers \
  projects/project-e0bbb103-9e5b-4402-866/locations/asia-south1/repositories/redteam-demo-containers
```

## Build Images

Artifact Registry repository:

```text
asia-south1-docker.pkg.dev/project-e0bbb103-9e5b-4402-866/redteam-demo-containers
```

Build and publish all three service images:

```bash
gcloud builds submit \
  --config cloudbuild.yaml \
  --project=project-e0bbb103-9e5b-4402-866 \
  --gcs-source-staging-dir=gs://project-e0bbb103-9e5b-4402-866-cloudbuild-source/source \
  --substitutions=_REGION=asia-south1,_REPOSITORY=redteam-demo-containers,_TAG=$(git rev-parse --short HEAD)
```

Validated dashboard image tag:

```text
dashboard1-20260516
```

Use that tag in `terraform.tfvars` when reproducing the validated deployment:

```hcl
ui_image     = "asia-south1-docker.pkg.dev/project-e0bbb103-9e5b-4402-866/redteam-demo-containers/redteam-ui:dashboard1-20260516"
api_image    = "asia-south1-docker.pkg.dev/project-e0bbb103-9e5b-4402-866/redteam-demo-containers/redteam-api:dashboard1-20260516"
worker_image = "asia-south1-docker.pkg.dev/project-e0bbb103-9e5b-4402-866/redteam-demo-containers/redteam-worker:dashboard1-20260516"
```

The first manual build required these one-time setup actions:

```bash
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com \
  --project=project-e0bbb103-9e5b-4402-866

gcloud artifacts repositories create redteam-demo-containers \
  --repository-format=docker \
  --location=asia-south1 \
  --description="CommodityRedTeam Docker images" \
  --project=project-e0bbb103-9e5b-4402-866

gcloud storage buckets create gs://project-e0bbb103-9e5b-4402-866-cloudbuild-source \
  --location=asia-south1 \
  --project=project-e0bbb103-9e5b-4402-866 \
  --uniform-bucket-level-access
```

## Important v1 Notes

- Cloud SQL is not included. IAP handles demo login, and app state lives in Firestore/GCS.
- The API is protected by Cloud Run IAM, not IAP. Only `redteam-ui-sa` and `redteam-tasks-sa` receive `roles/run.invoker`.
- The API service account can act as `redteam-tasks-sa` only so it can enqueue authenticated Cloud Tasks with an OIDC token.
- The API service account can act as `redteam-worker-sa` so it can execute the Cloud Run Job with env overrides.
- `roles/run.developer` is granted to the API service account for demo job execution. Replace this with a custom least-privilege role for production.
- Vertex model location defaults to `us-central1` because Model Garden model availability varies by region.

## Validated Deployment Outputs

```text
project: project-e0bbb103-9e5b-4402-866
region: asia-south1
ui_url: https://redteam-demo-ui-x3egbjk7ma-el.a.run.app
api_url: https://redteam-demo-api-x3egbjk7ma-el.a.run.app
worker_job: redteam-demo-worker
tasks_queue: redteam-demo-benchmark
results_bucket: gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results
```

## Smoke Tests

Anonymous access should be blocked or redirected:

```bash
curl -I https://redteam-demo-ui-x3egbjk7ma-el.a.run.app
# 302 to Google IAP login

curl -i https://redteam-demo-api-x3egbjk7ma-el.a.run.app/health
# 403 from Cloud Run IAM
```

Authenticated API checks:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/health

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/models

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/attacks

curl -H "Authorization: Bearer $TOKEN" \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/defenses
```

Dashboard data APIs:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  "https://redteam-demo-api-x3egbjk7ma-el.a.run.app/jobs?limit=5"

curl -H "Authorization: Bearer $TOKEN" \
  "https://redteam-demo-api-x3egbjk7ma-el.a.run.app/jobs/job-0b6ecca8f48b/results?include_results=true&result_limit=3"

curl -H "Authorization: Bearer $TOKEN" \
  "https://redteam-demo-api-x3egbjk7ma-el.a.run.app/jobs/job-0b6ecca8f48b/artifacts"

curl -L -H "Authorization: Bearer $TOKEN" \
  "https://redteam-demo-api-x3egbjk7ma-el.a.run.app/jobs/job-0b6ecca8f48b/artifacts/summary.csv"
```

The UI at `https://redteam-demo-ui-x3egbjk7ma-el.a.run.app` now exposes these as dashboard tabs:

```text
Overview
Run Benchmark
Results Explorer
Live Attack
Catalog
```

Single live attack:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"vertex-gemini-flash","attack_id":"v1.1","defenses":["input_filter"]}' \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/run/single
```

Background benchmark job:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"models":["vertex-gemini-flash"],"defenses":["input_filter"],"attack_ids":["v1.1"]}' \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/jobs/benchmark
```

Validated benchmark evidence:

```text
job_id: job-0e6da9a1dd14
status: completed
worker_exit_code: 0
artifacts:
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0e6da9a1dd14/job-0e6da9a1dd14/results.csv
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0e6da9a1dd14/job-0e6da9a1dd14/results.json
```

Validated multi-model full-matrix evidence:

```bash
TOKEN=$(gcloud auth print-identity-token)

curl -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"models":["vertex-gemini-flash","vertex-gemini-pro"],"defenses":["input_filter","output_validator","all_combined"],"attack_ids":["v1.1","v1.2"],"mode":"full_matrix","include_reports":true,"delay_seconds":1}' \
  https://redteam-demo-api-x3egbjk7ma-el.a.run.app/jobs/benchmark
```

```text
job_id: job-0b6ecca8f48b
status: completed
rows: 16
artifacts:
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0b6ecca8f48b/job-0b6ecca8f48b/results.json
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0b6ecca8f48b/job-0b6ecca8f48b/results.csv
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0b6ecca8f48b/job-0b6ecca8f48b/summary.json
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0b6ecca8f48b/job-0b6ecca8f48b/summary.csv
  gs://project-e0bbb103-9e5b-4402-866-redteam-demo-results/jobs/job-0b6ecca8f48b/job-0b6ecca8f48b/report/REPORT.md
```

## Model Garden Partner Models

The deployed code includes Vertex publisher endpoint adapters for partner models, but the project must have access to the specific Model Garden model card first.

Before selecting these in UI/API, open Model Garden in project `project-e0bbb103-9e5b-4402-866` and enable or accept terms for:

```text
Claude Sonnet 4.6: model_id=claude-sonnet-4-6, location=us-east5
Mistral Medium 3: model_id=mistral-medium-3, location=us-central1
```

Without enablement, Vertex returns `404 Publisher Model ... was not found or your project does not have access to it`.

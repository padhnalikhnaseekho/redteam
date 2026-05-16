# GCP Cloud-Native Deployment Plan
## CommodityRedTeam - IITB Capstone Showcase

> **Date:** 2026-05-15  
> **Region:** `asia-south1` (Mumbai - closest to IITB, lowest latency)  
> **IaC:** Terraform 1.8+ - all resources in `infra/terraform/`  
> **Target cost (idle):** ~$3-8/month | **Active demo day:** ~$5-15 total

> **Status note:** This file is the original provider-API-key deployment baseline.
> The preferred GCP-native implementation plan is now
> [`GCP-VERTEXAI-CLOUDRUN-IMPLEMENTATION-PLAN.md`](./GCP-VERTEXAI-CLOUDRUN-IMPLEMENTATION-PLAN.md),
> which uses Cloud Run service identities, Vertex AI / Model Garden, and Cloud IAP.

---

## 1. Executive Summary

The framework is currently a local CLI research tool. This plan lifts it to a fully managed, cloud-native platform on GCP with three Cloud Run services, one Cloud Run Job for long-running benchmarks, GCS for artifact storage, Firestore for job state, and Secret Manager for all API keys.

**No VMs. No Kubernetes. No idle compute cost.** Every service scales to zero between uses - you only pay for actual execution time during the showcase.

---

## 2. Architecture Overview

```text
+---------------------------------------------------------------------+
| INTERNET / IITB AUDIENCE                                            |
+------------------+--------------------------------------------------+
                   | HTTPS
                   v
+---------------------------------------------------------------------+
| Cloud Run - redteam-ui (Streamlit dashboard)                        |
| Public URL - asia-south1 - min 0 / max 2 - 1vCPU - 1GB              |
| - Attack browser (all 54 attacks)                                   |
| - Live single-attack demo with real-time output                     |
| - Benchmark results explorer (charts, heatmaps)                     |
| - Defense comparison matrix                                         |
| - V3 adaptive loop visualizer                                       |
+------------------+--------------------------------------------------+
                   | Internal HTTP (Cloud Run service-to-service)
                   v
+---------------------------------------------------------------------+
| Cloud Run - redteam-api (FastAPI backend)                           |
| Internal URL - asia-south1 - min 0 / max 3 - 2vCPU - 2GB            |
| - GET  /attacks          - list all 54 attacks                      |
| - GET  /defenses         - list all 8 defenses                      |
| - GET  /models           - list configured LLM models               |
| - POST /run/single       - run one attack synchronously             |
| - POST /jobs/benchmark   - enqueue a full benchmark run             |
| - GET  /jobs/{id}        - poll job status from Firestore           |
| - GET  /results          - list GCS result folders                  |
| - GET  /results/{run_id} - fetch result JSON/CSV from GCS           |
+---------------------------+-----------------------------------------+
                            | Cloud Tasks enqueue
                            v
+---------------------------------------------------------------------+
| Cloud Run Job - redteam-worker                                      |
| On-demand - asia-south1 - 4vCPU - 8GB - 60-min timeout              |
| - Runs full benchmark matrix                                        |
| - Runs V3 adaptive loop                                             |
| - Writes results to GCS bucket                                      |
| - Updates job status in Firestore                                   |
+---------------------------------------------------------------------+

        GCS                    Firestore              Secret Manager
        |                      |                      |
        v                      v                      v
+--------------+    +---------------------+    +---------------------+
| GCS Bucket   |    | Firestore DB        |    | Secret Manager      |
| results/     |    | /jobs/{id}          |    | GROQ_API_KEY        |
| configs/     |    | /runs/{run_id}      |    | GOOGLE_API_KEY      |
| models/      |    | /metrics/{run_id}   |    | ANTHROPIC_API_KEY   |
| preloaded/   |    |                     |    | MISTRAL_API_KEY     |
|              |    |                     |    | OPENAI_API_KEY      |
+--------------+    +---------------------+    +---------------------+
```

---

## 3. GCP Service Selection Rationale

| GCP Service | Why chosen | Alternative rejected |
|---|---|---|
| **Cloud Run (service)** | Scale-to-zero, pay-per-request, no cluster management | GKE (overkill for research showcase), App Engine (less flexible) |
| **Cloud Run (job)** | Long-running batch work up to 60 min, same container model | Cloud Functions (5-min limit too short for benchmarks), Dataflow (too heavy) |
| **Cloud Storage (GCS)** | Object store for CSV/JSON/PNG/PPTX artifacts; cheap | Cloud SQL (wrong shape for flat files), BigQuery (overkill) |
| **Firestore** | Job status + metadata; generous free tier; no schema | Cloud SQL (SQL overhead unnecessary), Memorystore (no persistence) |
| **Secret Manager** | Secure API key storage; IAM-controlled; audit log | Env vars in YAML, committed secrets |
| **Artifact Registry** | OCI image hosting; regional for fast pulls | Docker Hub (slower, rate limits), GCR (deprecated) |
| **Cloud Build** | Free 120 min/day; native GCP integration | GitHub Actions (works but requires Workload Identity Federation setup) |
| **Cloud Tasks** | Reliable async job dispatch from API to worker | Pub/Sub (heavier), direct Cloud Run job invocation (no retry semantics) |

---

## 4. Service Design

### 4.1 redteam-api (FastAPI)

**Image:** `infra/docker/api/Dockerfile`  
**Source:** `api/` directory - wraps existing `src/` framework as HTTP endpoints

Key design decisions:

- The API is stateless; all state lives in Firestore or GCS.
- Single-attack runs (`POST /run/single`) are synchronous.
- Full benchmark runs (`POST /jobs/benchmark`) dispatch a Cloud Run Job and return immediately with a `job_id`.
- The API mounts `src/`, `config/`, and `results/` via the same Docker image.
- ML defenses are loaded lazily so the API starts quickly.

Container spec:

```text
CPU:    2 vCPU
Memory: 2048 MiB
Port:   8080
Concurrency: 4
Min instances: 0
Max instances: 3
Timeout: 120s
```

**Authentication:** Original baseline uses unauthenticated public showcase access. The updated Vertex AI plan replaces this with Cloud IAP on the UI and Cloud Run IAM for API access.

### 4.2 redteam-ui (Streamlit)

**Image:** `infra/docker/ui/Dockerfile`  
**Source:** `ui/app.py`

Pages:

1. Overview
2. Attack Browser
3. Live Demo
4. Benchmark Results
5. Defense Deep Dive
6. V3 Adaptive Loop
7. SHAP Explainability

Container spec:

```text
CPU:    1 vCPU
Memory: 1024 MiB
Port:   8501
Min instances: 0
Max instances: 2
Timeout: 60s
```

### 4.3 redteam-worker (Cloud Run Job)

**Image:** `infra/docker/worker/Dockerfile`  
**Source:** same `src/` and `scripts/` directory

Job execution flow in the original baseline:

1. API receives `POST /jobs/benchmark` with `{model, attacks, defenses, run_type}`.
2. API writes job metadata to Firestore.
3. API enqueues a Cloud Tasks task.
4. Worker receives job config.
5. Worker runs `RedTeamEvaluator.run_suite()`.
6. Worker writes results to GCS.
7. Worker updates Firestore.
8. UI polls job status.

The updated Vertex AI plan corrects this by routing Cloud Tasks to an internal API endpoint, and having the API call the Cloud Run Jobs API `jobs.run` method with overrides.

Container spec:

```text
CPU:    4 vCPU
Memory: 8192 MiB
Timeout: 3600s
Max retries: 1
Parallelism: 1
```

---

## 5. Storage Design

### 5.1 GCS Bucket Structure

```text
gs://redteam-results-{PROJECT_ID}/
  configs/
    models.yaml
    agent_config.yaml
    commodities.yaml
  models/
    sentence-transformers/
      all-MiniLM-L6-v2/
  results/
    {job_id}/
      metadata.json
      attack_results.csv
      attack_results.json
      summary.csv
      report/
        heatmap_asr.png
        barchart_defense_asr.png
        heatmap_detection_coverage.png
        radar_vulnerability.png
        results_report.pptx
  preloaded/
    benchmark_baseline/
```

Lifecycle policy: auto-delete result files older than 90 days.

### 5.2 Firestore Schema

```text
Collection: jobs
  Document: {job_id}
    status: pending | running | completed | failed
    created_at: Timestamp
    started_at: Timestamp | null
    completed_at: Timestamp | null
    config:
      run_type: single | benchmark | v3_loop
      models: list[str]
      defenses: list[str]
      attack_ids: list[str] | all
    result_path: gs://redteam-results-.../results/{job_id}/ | null
    error: str | null
    summary:
      total_attacks: int
      attack_success_rate: float
      detection_rate: float
      financial_impact: float
```

### 5.3 Secret Manager

| Secret name | Value | Used by |
|---|---|---|
| `redteam-groq-api-key` | `GROQ_API_KEY` value | api, worker |
| `redteam-google-api-key` | `GOOGLE_API_KEY` value | api, worker |
| `redteam-anthropic-api-key` | `ANTHROPIC_API_KEY` value | api, worker |
| `redteam-mistral-api-key` | `MISTRAL_API_KEY` value | api, worker |
| `redteam-openai-api-key` | `OPENAI_API_KEY` value | api, worker |

In the updated Vertex AI plan, these secrets become optional fallback provider secrets rather than the primary LLM access mechanism.

---

## 6. CI/CD Pipeline (Cloud Build)

```text
Trigger: git push to main branch of GitHub repo
File:    infra/cloudbuild.yaml

Steps:
  1. Build api, ui, and worker images.
  2. Push images to Artifact Registry.
  3. Deploy redteam-api and redteam-ui.
  4. Update redteam-worker job definition.
  5. Run smoke tests against /health.
```

Estimated build time: 8-12 minutes.

---

## 7. Terraform Structure

```text
infra/
  terraform/
    main.tf
    variables.tf
    outputs.tf
    apis.tf
    iam.tf
    storage.tf
    firestore.tf
    secrets.tf
    artifact_registry.tf
    cloud_run.tf
    cloud_tasks.tf
    cloud_build.tf
  docker/
    api/Dockerfile
    ui/Dockerfile
    worker/Dockerfile
  cloudbuild.yaml
```

Deployment commands:

```bash
cd infra/terraform
terraform init
terraform plan -var-file=showcase.tfvars
terraform apply -var-file=showcase.tfvars
```

Example `showcase.tfvars`:

```hcl
project_id          = "your-gcp-project-id"
region              = "asia-south1"
groq_api_key        = "gsk_..."
google_api_key      = "AIza..."
anthropic_api_key   = "sk-ant-..."
mistral_api_key     = "..."
```

---

## 8. Cost Breakdown

### Per-month estimate

| Service | Config | Monthly cost |
|---|---|---|
| Cloud Run - api | 0 min instances, low traffic | ~$0.50 |
| Cloud Run - ui | 0 min instances, low traffic | ~$0.20 |
| Cloud Run Job - worker | 5 benchmark runs x 30 min | ~$1.50 |
| GCS bucket | 2 GB stored + 1 GB egress | ~$0.10 |
| Firestore | Within free tier | $0.00 |
| Secret Manager | 5 secrets | ~$0.30 |
| Artifact Registry | 3 GB images | ~$0.30 |
| Cloud Build | Within free tier | $0.00 |
| Cloud Tasks | Low usage | $0.00 |
| **Total** | | **~$3-5/month** |

The Vertex AI plan must also include model usage costs for the selected Model Garden models.

---

## 9. Showcase Demo Guide

### Pre-demo checklist

```text
[ ] Run terraform apply.
[ ] Upload preloaded results to GCS.
[ ] Verify API /health.
[ ] Verify UI in browser.
[ ] Warm up Cloud Run.
[ ] Confirm required API keys or Vertex AI models are available.
```

### Demo flow

1. Problem statement: why trading LLMs are vulnerable.
2. Attack browser: show 54 attacks and 8 categories.
3. Live single attack: run an attack with no defense, then with defense.
4. Benchmark results: show ASR heatmap and defense coverage.
5. V3 adaptive loop: show attack success improvement over rounds.
6. Launch live benchmark: show job status updates.
7. Closing: architecture, IaC, and repo link.

---

## 10. Security Hardening

These were post-showcase items in the original baseline:

- Cloud IAP for Google account access.
- Private API access behind Cloud Run IAM.
- Cloud Armor or API Gateway for rate limiting.
- Binary Authorization for image policy.
- Least-privilege IAM custom roles.

The updated Vertex AI plan promotes Cloud IAP and Cloud Run IAM into the v1 demo architecture.

---

## 11. Scaling Path

| Scale trigger | Upgrade path |
|---|---|
| More than 10 concurrent users | Set `min-instances: 1` on API and UI. |
| Multiple benchmark jobs in parallel | Increase job parallelism and queue controls. |
| Large result data | Migrate analytics summaries to BigQuery. |
| ML defenses too slow on CPU | Use Cloud Run GPU or Cloud Batch. |
| Multi-region availability | Add second region and load balancer. |

---

## 12. Quick Start

```bash
# 1. Clone and set up
git clone <repo> && cd redteam

# 2. Create GCP project and enable billing
gcloud projects create redteam-iitb-showcase
gcloud config set project redteam-iitb-showcase
gcloud auth application-default login

# 3. Deploy infra
cd infra/terraform
cp showcase.tfvars.example showcase.tfvars
terraform init
terraform apply -var-file=showcase.tfvars

# 4. Upload pre-run results
gsutil -m cp -r results/ gs://redteam-results-<project_id>/preloaded/

# 5. Trigger first build
git push origin main

# 6. Get service URLs
terraform output
```

Total estimated time from zero to live URL in the original plan: ~25 minutes.

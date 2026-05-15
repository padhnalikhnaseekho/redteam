# GCP Vertex AI + Cloud Run Implementation Plan
## CommodityRedTeam GCP-Native Showcase Deployment

> **Date:** 2026-05-15  
> **Status:** `PLANNED`  
> **Region for Cloud Run:** `asia-south1`  
> **Default Vertex AI model location:** `us-central1`  
> **IaC:** Terraform 1.8+ under `infra/terraform/`  
> **Supersedes:** `specs/cloud/GCP-DEPLOYMENT-PLAN.md` for the preferred GCP-native LLM/auth approach

---

## 1. Executive Summary

CommodityRedTeam is currently a local Python research framework for red-teaming LLM-powered commodity trading agents. This plan turns it into a GCP-native demo platform while keeping the core evaluator, attack suite, defenses, and benchmark engine from `src/`.

The updated deployment approach is:

- Cloud Run hosts the user interface and API.
- Cloud Run Jobs run long benchmark workloads.
- Cloud IAP provides Google/Gmail login for demo users.
- Google service accounts are the workload identities for Cloud Run services and jobs.
- Vertex AI / Model Garden supplies the LLMs through Google IAM and Application Default Credentials.
- Firestore tracks job status.
- GCS stores benchmark evidence, CSV/JSON results, plots, and reports.
- Secret Manager is optional for non-Vertex fallback providers only.

This means the demo should not depend on long-lived LLM provider API keys as the primary path. The Cloud Run service account calls Vertex AI directly.

---

## 2. Official Google Documentation Grounding

This design is grounded in the following Google Cloud documentation:

| Area | Official document | Design impact |
|---|---|---|
| Cloud Run service identity | https://cloud.google.com/run/docs/securing/service-identity | Cloud Run services and jobs use assigned service accounts plus ADC to call Google APIs. Do not set `GOOGLE_APPLICATION_CREDENTIALS` in Cloud Run. |
| Vertex AI IAM | https://cloud.google.com/vertex-ai/docs/general/access-control | Grant service accounts Vertex AI permissions, starting with `roles/aiplatform.user` for the demo. |
| Vertex AI Model Garden | https://cloud.google.com/vertex-ai/generative-ai/docs/model-garden/explore-models | Model Garden is where Google, partner, open, tunable, and deployable models are discovered and enabled. |
| Gemini inference on Vertex AI | https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference | Gemini can be called through Vertex AI using SDKs or REST. |
| Claude on Vertex AI | https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude | Claude models can be used as fully managed serverless APIs through Vertex AI after access is enabled. |
| Mistral on Vertex AI | https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral | Mistral models can be used as fully managed serverless APIs through Vertex AI after access is enabled. |
| Cloud IAP for Cloud Run | https://cloud.google.com/iap/docs/enabling-cloud-run | IAP protects Cloud Run apps with Google identity and IAM access control. |
| IAP user identity | https://cloud.google.com/iap/docs/identity-howto | Apps should validate the `x-goog-iap-jwt-assertion`; identity headers are compatibility helpers, not the security root. |
| Cloud Tasks HTTP auth | https://cloud.google.com/tasks/docs/creating-http-target-tasks | Cloud Tasks should use OIDC tokens for authenticated Cloud Run HTTP targets. |
| Cloud Run Jobs execution | https://cloud.google.com/run/docs/execute/jobs | Cloud Run Jobs are executed with the Cloud Run Jobs API, including optional env/arg overrides. |

---

## 3. Product Architecture

### 3.1 Simple Explanation

```text
Cloud Run hosts the app.
IAP handles Google login.
GSA identities call Google APIs.
Vertex AI supplies the LLMs.
Worker jobs run benchmarks.
Firestore tracks status.
GCS stores evidence.
```

### 3.2 Target GCP Flow

```text
User browser
  -> Cloud IAP Google login
  -> Cloud Run redteam-ui
  -> authenticated internal call to Cloud Run redteam-api
  -> Cloud Tasks
  -> redteam-api internal job starter endpoint
  -> Cloud Run Jobs API
  -> redteam-worker Cloud Run Job
  -> Vertex AI / Model Garden
  -> GCS results + Firestore job status
```

### 3.3 Architecture Diagram

```text
+------------------------------------------------------------------+
| Demo user                                                        |
| Google/Gmail account                                             |
+------------------------------+-----------------------------------+
                               | HTTPS
                               v
+------------------------------------------------------------------+
| Cloud IAP                                                        |
| - Authenticates Google users                                     |
| - Enforces roles/iap.httpsResourceAccessor                       |
| - Passes signed identity assertion to the app                    |
+------------------------------+-----------------------------------+
                               |
                               v
+------------------------------------------------------------------+
| Cloud Run service: redteam-ui                                    |
| Service account: redteam-ui-sa                                   |
| Purpose: Streamlit dashboard                                     |
| Access: IAP-protected public URL                                 |
+------------------------------+-----------------------------------+
                               | Cloud Run IAM auth
                               v
+------------------------------------------------------------------+
| Cloud Run service: redteam-api                                   |
| Service account: redteam-api-sa                                  |
| Purpose: FastAPI control plane                                   |
| Access: private; invoked by UI and Cloud Tasks                   |
| Calls: Firestore, GCS, Cloud Tasks, Cloud Run Jobs API, Vertex AI |
+--------------+-----------------------+---------------------------+
               |                       |
               | enqueue               | jobs.run with overrides
               v                       v
+-----------------------------+ +----------------------------------+
| Cloud Tasks queue           | | Cloud Run Job: redteam-worker    |
| Service account:            | | Service account: redteam-worker-sa|
| redteam-tasks-sa            | | Purpose: benchmark runner        |
| Calls API using OIDC        | | Calls Vertex AI, GCS, Firestore  |
+-----------------------------+ +---------------+------------------+
                                                 |
                                                 v
+--------------------------+  +--------------------------+
| Vertex AI / Model Garden |  | GCS + Firestore           |
| Gemini / Claude /        |  | results + job state       |
| Mistral / Llama          |  |                          |
+--------------------------+  +--------------------------+
```

---

## 4. Authentication and Authorization

### 4.1 User Login Decision

Use Cloud IAP for v1.

Do not build:

- Local username/password auth.
- Cloud SQL user tables.
- App-level OAuth/session management.

Why:

- This is a demo/research platform, not a consumer SaaS product.
- IAP already gives Google/Gmail login and IAM-based allowlists.
- It avoids password storage, session security, reset flows, user tables, and Cloud SQL operations.
- It keeps the deployment cheaper and easier to explain.

### 4.2 IAP Behavior

Enable IAP on `redteam-ui`.

Allowed users should be configured as:

```text
user:someone@gmail.com
group:redteam-demo-users@googlegroups.com
domain:example.edu
```

Grant allowed users/groups:

```text
roles/iap.httpsResourceAccessor
```

The app may display the current signed-in email only after validating the IAP JWT:

```text
x-goog-iap-jwt-assertion
```

`X-Goog-Authenticated-User-Email` is useful for display, but the security decision must come from IAP and the signed assertion, not from a raw user-controlled header.

### 4.3 Service-to-Service Auth

The API remains private behind Cloud Run IAM.

```text
redteam-ui-sa -> roles/run.invoker on redteam-api
redteam-tasks-sa -> roles/run.invoker on redteam-api
```

The UI calls API with an identity token for the API audience. Cloud Tasks calls the API internal endpoint using an OIDC token configured on the task.

---

## 5. Vertex AI / Model Garden Strategy

### 5.1 Current Repo State

The current repository is provider-key oriented:

- `config/models.yaml` defines direct providers: `anthropic`, `mistral`, `google`, `groq`, and optional `openai`.
- `src/utils/llm.py` reads direct provider keys from environment variables such as `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, and `GROQ_API_KEY`.
- `src/agent/trading_agent.py` selects LangChain provider classes by model-name substring.

This works locally, but it is not ideal for a GCP-native demo because each provider key becomes a separate secret and operational dependency.

### 5.2 Target Repo Behavior

Add a new provider:

```text
provider: vertex
```

The Vertex provider must:

- Use Application Default Credentials.
- Never require provider API keys in Cloud Run.
- Support a model-level `location`.
- Fail clearly if `GOOGLE_CLOUD_PROJECT` is missing.
- Fail clearly if a model is not enabled or not available in the selected Vertex location.
- Preserve the existing non-Vertex providers as optional local/fallback paths.

### 5.3 Model Config Shape

Add Vertex model entries to `config/models.yaml`:

```yaml
vertex-gemini-flash:
  provider: vertex
  family: gemini
  project_id_env: GOOGLE_CLOUD_PROJECT
  location: us-central1
  model_id: gemini-2.5-flash
  max_tokens: 4096
  temperature: 0.3
  role: reviewer
  cost_per_1k_input_tokens: 0.0
  cost_per_1k_output_tokens: 0.0

vertex-gemini-pro:
  provider: vertex
  family: gemini
  project_id_env: GOOGLE_CLOUD_PROJECT
  location: us-central1
  model_id: gemini-2.5-pro
  max_tokens: 4096
  temperature: 0.2
  role: target_agent
  cost_per_1k_input_tokens: 0.0
  cost_per_1k_output_tokens: 0.0

vertex-claude:
  provider: vertex
  family: claude
  project_id_env: GOOGLE_CLOUD_PROJECT
  location: us-central1
  model_id: claude-sonnet-4.5
  max_tokens: 4096
  temperature: 0.3
  role: comparison_target
  cost_per_1k_input_tokens: 0.0
  cost_per_1k_output_tokens: 0.0

vertex-mistral:
  provider: vertex
  family: mistral
  project_id_env: GOOGLE_CLOUD_PROJECT
  location: us-central1
  model_id: mistral-medium-3
  max_tokens: 4096
  temperature: 0.3
  role: comparison_target
  cost_per_1k_input_tokens: 0.0
  cost_per_1k_output_tokens: 0.0
```

Pricing fields should be updated after final model selection using the Vertex AI pricing page. They can remain `0.0` during implementation if cost accounting is not used for billing decisions.

### 5.4 Demo Model Roles

| Demo role | Preferred Vertex model | Purpose |
|---|---|---|
| Fast reviewer defense | `vertex-gemini-flash` | Review outputs and support multi-agent defense cheaply and quickly. |
| Primary target trading agent | `vertex-gemini-pro` | Main LLM trading brain for live attack demos. |
| Strong comparison target | `vertex-claude` | Show model-to-model vulnerability differences. |
| Alternate comparison target | `vertex-mistral` | Show partner model comparison through the same GCP control plane. |
| Attack generator / V3 planner | `vertex-gemini-flash` first | Keep adaptive attack generation simple and reliable for v1. |

### 5.5 Location Decision

Cloud Run remains in `asia-south1`.

Vertex model entries default to `us-central1` because partner model availability in Model Garden can be region-specific and is often broader in US regions. The selected location must be stored per model, not globally.

If all required Vertex models are confirmed available closer to `asia-south1`, update each model entry explicitly.

---

## 6. IAM Plan

### 6.1 Service Accounts

Create:

```text
redteam-ui-sa
redteam-api-sa
redteam-worker-sa
redteam-tasks-sa
redteam-build-sa
```

### 6.2 Role Grants

| Principal | Role | Scope | Reason |
|---|---|---|---|
| `redteam-ui-sa` | `roles/run.invoker` | `redteam-api` service | UI calls private API. |
| `redteam-api-sa` | `roles/aiplatform.user` | Project | API can run live single attacks through Vertex AI. |
| `redteam-api-sa` | `roles/datastore.user` | Project | API reads/writes Firestore job documents. |
| `redteam-api-sa` | `roles/storage.objectViewer` | Results bucket | API lists and reads result artifacts. |
| `redteam-api-sa` | `roles/cloudtasks.enqueuer` | Tasks queue | API creates benchmark dispatch tasks. |
| `redteam-api-sa` | `roles/run.developer` | `redteam-worker` job | API can call jobs.run with execution overrides for the demo. |
| `redteam-worker-sa` | `roles/aiplatform.user` | Project | Worker calls Vertex AI models during benchmarks. |
| `redteam-worker-sa` | `roles/datastore.user` | Project | Worker updates job status. |
| `redteam-worker-sa` | `roles/storage.objectAdmin` | Results bucket | Worker writes result artifacts. |
| `redteam-worker-sa` | `roles/logging.logWriter` | Project | Worker writes structured logs. |
| `redteam-tasks-sa` | `roles/run.invoker` | `redteam-api` service | Cloud Tasks invokes internal API start endpoint. |
| Demo users/group | `roles/iap.httpsResourceAccessor` | `redteam-ui` IAP resource | Users can access UI through IAP. |

After the demo works, replace broad roles such as `roles/aiplatform.user` and `roles/run.developer` with custom least-privilege roles.

---

## 7. Service Design

### 7.1 redteam-ui

Runtime:

```text
Cloud Run service
Service account: redteam-ui-sa
Ingress: all, protected by IAP
Min instances: 0
Max instances: 2
Port: 8501
```

Responsibilities:

- Render Streamlit dashboard.
- Display attack browser.
- Run live single-attack demo through API.
- Start benchmark jobs through API.
- Poll job status.
- Load result artifacts.
- Display signed-in email after IAP identity validation.

### 7.2 redteam-api

Runtime:

```text
Cloud Run service
Service account: redteam-api-sa
Ingress: internal-and-cloud-load-balancing or IAM-protected private service
Min instances: 0
Max instances: 3
Port: 8080
```

Endpoints:

```text
GET  /health
GET  /attacks
GET  /defenses
GET  /models
POST /run/single
POST /jobs/benchmark
GET  /jobs/{job_id}
GET  /results
GET  /results/{run_id}
POST /internal/jobs/{job_id}/start
```

Responsibilities:

- Wrap existing attack registry and evaluator.
- Expose model options from `config/models.yaml`.
- Start synchronous single-attack runs for demo.
- Create Firestore job documents.
- Enqueue Cloud Tasks.
- Start Cloud Run Job executions using the Cloud Run Jobs API.
- Read GCS result artifacts for UI.

### 7.3 redteam-worker

Runtime:

```text
Cloud Run Job
Service account: redteam-worker-sa
CPU: 4
Memory: 8 GiB
Timeout: 3600s
Parallelism: 1
Task count: 1
Max retries: 1
```

Responsibilities:

- Read `JOB_ID`.
- Read `JOB_CONFIG_JSON` or `JOB_CONFIG_PATH`.
- Update Firestore status to `running`.
- Run selected benchmark or V3 loop using current `src/` framework.
- Call Vertex AI through service identity.
- Write results to `/tmp/results`.
- Upload results to GCS under `results/{job_id}/`.
- Update Firestore status to `completed` or `failed`.

---

## 8. Job Dispatch Flow

The old plan implied Cloud Tasks can directly invoke the worker job container. That is not the correct Cloud Run Job flow.

Use this exact flow:

```text
POST /jobs/benchmark
  -> API writes Firestore job document
  -> API enqueues Cloud Task with {job_id}
  -> Cloud Task calls POST /internal/jobs/{job_id}/start on redteam-api using OIDC
  -> API calls Cloud Run Jobs API jobs.run with env overrides:
       JOB_ID
       JOB_CONFIG_PATH or JOB_CONFIG_JSON
  -> worker runs benchmark
  -> worker updates Firestore and writes GCS artifacts
```

### 8.1 Sequence Diagram

```text
UI                 API               Firestore        Cloud Tasks       Cloud Run Jobs       Worker
|                  |                 |                |                 |                    |
| POST /jobs       |                 |                |                 |                    |
+----------------->|                 |                |                 |                    |
|                  | create pending  |                |                 |                    |
|                  +---------------->|                |                 |                    |
|                  | enqueue {id}    |                |                 |                    |
|                  +--------------------------------->|                 |                    |
| 202 {job_id}     |                 |                |                 |                    |
|<-----------------+                 |                |                 |                    |
|                  |                 |                | POST internal   |                    |
|                  |<---------------------------------+                 |                    |
|                  | jobs.run        |                |                 |                    |
|                  +-------------------------------------------------->|                    |
|                  |                 |                |                 | start execution    |
|                  |                 |                |                 +------------------->|
|                  |                 |                |                 |                    | running
|                  |                 |<------------------------------------------------------+
|                  |                 |                |                 |                    | upload GCS
|                  |                 |<------------------------------------------------------+
|                  |                 |                |                 |                    | completed
```

---

## 9. Storage Design

### 9.1 Firestore

Firestore remains the job/status database.

```text
jobs/{job_id}
  status: pending | running | completed | failed
  created_at: timestamp
  started_at: timestamp | null
  completed_at: timestamp | null
  created_by_email: string | null
  config:
    run_type: single | benchmark | v3_loop
    models: list[string]
    defenses: list[string]
    attack_ids: list[string] | "all"
  result_path: gs://... | null
  error: string | null
  summary:
    total_attacks: int
    attack_success_rate: float
    detection_rate: float
    financial_impact: float
```

### 9.2 GCS

```text
gs://redteam-results-{PROJECT_ID}/
  configs/
    models.yaml
    agent_config.yaml
    commodities.yaml
  results/
    {job_id}/
      metadata.json
      attack_results.csv
      attack_results.json
      summary.csv
      report/
  preloaded/
    benchmark_baseline/
```

### 9.3 Cloud SQL

Cloud SQL is not required for v1 because demo auth is handled by IAP and app state remains in Firestore/GCS.

Only add Cloud SQL later if the product needs app-native users, roles, subscriptions, organization tenancy, or relational audit/reporting queries.

---

## 10. Terraform Structure

Create:

```text
infra/terraform/
  apis.tf
  iam.tf
  iap.tf
  artifact_registry.tf
  cloud_run.tf
  cloud_tasks.tf
  firestore.tf
  storage.tf
  outputs.tf
```

Enable:

```text
run.googleapis.com
aiplatform.googleapis.com
cloudtasks.googleapis.com
firestore.googleapis.com
storage.googleapis.com
artifactregistry.googleapis.com
cloudbuild.googleapis.com
iap.googleapis.com
iam.googleapis.com
cloudresourcemanager.googleapis.com
```

Key Terraform requirements:

- Use `google_cloud_run_v2_service` for UI and API.
- Use `google_cloud_run_v2_job` for worker.
- Set `iap_enabled = true` on UI when using direct Cloud Run IAP.
- Assign explicit service accounts to every service/job.
- Do not rely on the Compute Engine default service account.
- Grant IAP access to configured Gmail IDs/groups.
- Configure Cloud Tasks OIDC token using `redteam-tasks-sa`.
- Configure API service account with permission to execute the worker job.

---

## 11. Code Implementation Roadmap

### 11.1 Config

Update `config/models.yaml`:

- Add Vertex model entries.
- Preserve current provider entries as optional local/fallback models.
- Add `family`, `location`, and `role` fields for Vertex models.

### 11.2 LLM Client

Update `src/utils/llm.py`:

- Add `provider == "vertex"` branch.
- Use ADC rather than API keys.
- For Gemini, use `google-genai` with Vertex mode.
- Normalize response to the existing `LLMClient.chat()` contract:

```python
{
    "content": str,
    "tool_calls": list | None,
    "input_tokens": int,
    "output_tokens": int,
    "cost_usd": float,
    "latency_s": float,
}
```

- Include clear setup errors for:
  - missing `GOOGLE_CLOUD_PROJECT`
  - missing ADC credentials locally
  - disabled Vertex AI API
  - model not enabled in Model Garden
  - model unavailable in configured location

### 11.3 Trading Agent

Update `src/agent/trading_agent.py`:

- Add Vertex-backed target-agent support.
- Prefer Gemini Vertex first because the repo already uses `google-genai`.
- Add `langchain-google-vertexai` when LangChain tool-calling is required.
- Keep direct provider LangChain classes as optional fallback.

### 11.4 API Wrapper

Create API code:

```text
api/main.py
```

Minimum v1 endpoints:

```text
GET  /health
GET  /attacks
GET  /defenses
GET  /models
POST /run/single
POST /jobs/benchmark
GET  /jobs/{job_id}
POST /internal/jobs/{job_id}/start
```

### 11.5 Worker Wrapper

Create worker entrypoint:

```text
worker/main.py
```

The worker should adapt the existing scripts/evaluator instead of duplicating attack/defense logic.

### 11.6 UI Wrapper

Create Streamlit UI:

```text
ui/app.py
```

Minimum v1 pages:

- Overview
- Attack Browser
- Live Single Attack
- Benchmark Jobs
- Results Explorer

Defer SHAP/ROC/V3 visual polish until the end-to-end pipeline works.

---

## 12. Implementation Phases

### Phase 1: Documentation and Config Design

- Create this plan.
- Keep the old deployment plan as a baseline for the earlier provider-key approach.
- Document Vertex config, IAM, IAP, service flow, and demo flow.
- Document code paths that must change: `LLMClient`, `CommodityTradingAgent`, config, Docker, API wrapper, worker wrapper.

### Phase 2: Vertex AI Integration

- Add `vertex` provider support to `LLMClient`.
- Use ADC, not API keys, inside Cloud Run.
- Support local development through `gcloud auth application-default login`.
- Add model-level `location`.
- Add clear errors for unavailable or not-enabled Model Garden models.
- Add Vertex-backed Gemini support to the target trading agent first.

### Phase 3: Cloud Run Services

- Build `redteam-api` FastAPI wrapper around the evaluator.
- Build `redteam-worker` Cloud Run Job entrypoint.
- Build `redteam-ui` Streamlit dashboard.
- Keep API private.
- Protect UI with IAP.
- Ensure worker calls Vertex AI with `redteam-worker-sa`.

### Phase 4: IAP and IAM

- Enable IAP on `redteam-ui`.
- Allow only configured Gmail IDs or a Google Group.
- Validate IAP signed identity if displaying user identity.
- Do not store passwords or user records.
- Add audit logs: user email, action, job ID, selected model, selected attack.

### Phase 5: Demo Readiness

- Pre-enable required Model Garden models in the GCP project.
- Run one small live single-attack flow.
- Run one background benchmark job.
- Upload precomputed result artifacts to GCS.
- Document quota/rate limits and safe model combinations for demo day.

---

## 13. Test Plan

### 13.1 Local Tests

- `LLMClient` loads `provider: vertex` config without requiring provider API keys.
- Missing `GOOGLE_CLOUD_PROJECT` returns a clear setup error.
- Invalid Vertex location returns a clear setup error.
- Model selection returns the correct `model_id`, `family`, `location`, and `role`.
- Existing non-Vertex models still load when configured.

### 13.2 Cloud Smoke Tests

- IAP blocks unauthenticated browser access to UI.
- Allowed Gmail ID can access UI.
- UI can call private API.
- API can enqueue Cloud Task.
- Cloud Task can call API internal start endpoint using OIDC.
- API can execute Cloud Run Job with env overrides.
- Worker can call Vertex AI using `redteam-worker-sa`.
- Worker writes result artifacts to GCS.
- Worker updates Firestore job status.

### 13.3 Demo Scenarios

- Run a live single attack against `vertex-gemini-*`.
- Run the same attack with and without a defense.
- Run benchmark job across at least two Vertex models.
- Load precomputed GCS artifacts in the results dashboard.
- Confirm unauthorized user cannot access UI.

---

## 14. Assumptions and Defaults

- User auth uses Cloud IAP with Gmail/Google accounts.
- No Cloud SQL for v1.
- Cloud Run services remain in `asia-south1`.
- Vertex model location is configured per model and defaults to `us-central1`.
- Service-to-Google API auth uses Cloud Run service identity and ADC.
- Direct provider API keys are optional fallback only, not the primary GCP path.
- First working Vertex target is Gemini.
- Claude, Mistral, and Llama are enabled later through Model Garden model cards as demo comparison models.
- Cloud Tasks invokes API; API invokes Cloud Run Jobs API. Cloud Tasks does not invoke a job container directly.

---

## 15. Demo Explanation

Short version:

```text
We are testing whether an LLM trading agent can be tricked into unsafe commodity recommendations.
The UI is protected by Google login through IAP.
The API and worker run on Cloud Run.
The LLMs come from Vertex AI Model Garden.
The worker runs attacks, defenses, and benchmarks.
Firestore tracks job state.
GCS stores the evidence.
```

One-minute version:

```text
CommodityRedTeam is a red-team platform for LLM-powered commodity trading agents.
Instead of managing API keys for each model provider, the GCP deployment uses Vertex AI Model Garden.
Each Cloud Run service has a Google service account, and that identity is allowed to call Vertex AI.
For the demo, users sign in with Google through Cloud IAP.
When a user runs a single attack, the API calls the evaluator and a Vertex model directly.
When a user launches a benchmark, the API creates a Firestore job, enqueues a Cloud Task, and starts a Cloud Run Job.
The worker runs the benchmark, calls Vertex AI models, writes results to GCS, and updates Firestore.
```

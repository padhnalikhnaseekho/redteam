---
feature: gcp-infrastrucuture-terraform
status: IN-PROGRESS
stack: terraform
created: 2026-05-15
security-review: required
---

# Feature: GCP Infrastrucuture Terraform

> **Status:** `IN-PROGRESS`

## Requirement

Create deployable Terraform infrastructure for the Vertex AI + Cloud Run demo path using the requested root folder `infrastrucuture/`.

## Acceptance Criteria

- [x] Root folder `infrastrucuture/terraform` exists.
- [x] Terraform enables the required GCP APIs.
- [x] Terraform creates Artifact Registry, Firestore, GCS, Cloud Tasks, Cloud Run services, and a Cloud Run Job.
- [x] Terraform creates explicit Google service accounts for UI, API, worker, tasks, and build.
- [x] UI service has Cloud IAP enabled.
- [x] IAP access is controlled through configured Google users or groups.
- [x] API service is private through Cloud Run IAM invoker bindings.
- [x] API and worker identities can call Vertex AI with `roles/aiplatform.user`.
- [x] API can enqueue Cloud Tasks and execute the worker Cloud Run Job for v1 demo operation.
- [x] API can act as the tasks and worker service accounts for authenticated task dispatch and job execution.
- [x] Context7 official Terraform provider/module docs were consulted before implementation.
- [x] Project IAM uses direct official Google provider resources with static keys to avoid apply-time unknowns from generated service account emails.

## Security Gate

- [x] This feature provisions identity, IAM, and user access controls.
- [x] This feature grants Google API access to Cloud Run service accounts.
- [x] This feature exposes a browser-accessible UI.

Mitigations:

- UI login is handled by Cloud IAP instead of local passwords.
- API has no public invoker binding.
- Cloud Run services and jobs use explicit service accounts.
- No long-lived provider API keys are required for Vertex AI.
- Broad demo role `roles/run.developer` is documented for later replacement with a custom least-privilege role.

## Spec-Sync Audit

| Spec File | Detail |
|---|---|
| `specs/cloud/GCP-VERTEXAI-CLOUDRUN-IMPLEMENTATION-PLAN.md` | Terraform files now implement the documented Cloud Run, IAP, Tasks, Firestore, GCS, IAM, and Vertex AI layout. |
| `specs/features/active/vertex-ai-cloudrun-readiness.md` | Infra slice builds on the completed Vertex runtime readiness work. |

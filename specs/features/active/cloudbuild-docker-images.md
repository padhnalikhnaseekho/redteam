---
feature: cloudbuild-docker-images
status: IN-PROGRESS
stack: docker-cloudbuild
created: 2026-05-15
security-review: required
---

# Feature: Cloud Build Docker Images

> **Status:** `IN-PROGRESS`

## Requirement

Create Docker images for `redteam-ui`, `redteam-api`, and `redteam-worker`, then build and publish them to Artifact Registry through Cloud Build.

## Acceptance Criteria

- [x] API, UI, and worker have runnable Python entrypoints.
- [x] API Dockerfile starts FastAPI through Uvicorn and honors Cloud Run `PORT`.
- [x] UI Dockerfile starts Streamlit and honors Cloud Run `PORT`.
- [x] Worker Dockerfile starts the Cloud Run Job entrypoint.
- [x] Service-specific requirements keep UI/API/worker images out of the heavyweight optional ML stack for v1.
- [x] Cloud Build config builds all three images.
- [x] Cloud Build config publishes custom `_TAG` and `latest` tags to Artifact Registry.
- [x] Terraform image examples point at project `project-e0bbb103-9e5b-4402-866`.
- [x] Artifact Registry repository exists in GCP project.
- [x] Cloud Build successfully published API, UI, and worker images to Artifact Registry.

## Build Verification

Successful build:

```text
build_id: 643c931b-805b-47f1-9d5b-fa5eedd97dbb
project: project-e0bbb103-9e5b-4402-866
repository: asia-south1-docker.pkg.dev/project-e0bbb103-9e5b-4402-866/redteam-demo-containers
tags: 8815d62, latest
```

Updated orchestration build:

```text
build_id: 102cd3f4-2ac7-4635-96b6-96aa926e7433
tags: orchestration-20260516, latest
```

## Security Gate

- [x] This feature creates deployable containers.
- [x] This feature can publish artifacts to Google Cloud.
- [x] This feature introduces HTTP service surfaces.

Mitigations:

- Docker images do not copy `.env`, Terraform state, notebooks, or generated results.
- API has no secret defaults baked into the image.
- Vertex access remains through Cloud Run service identity and ADC.
- UI/API auth is still enforced by Cloud Run IAM/IAP infrastructure.

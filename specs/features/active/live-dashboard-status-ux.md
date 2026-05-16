---
feature: live-dashboard-status-ux
status: IN-PROGRESS
stack: streamlit-fastapi-firestore
created: 2026-05-16
security-review: not-required
---

# Feature: Live Dashboard Status UX

> **Status:** `IN-PROGRESS`

## Requirement

Improve the dashboard live-status presentation without changing backend APIs, Cloud Run jobs, Cloud Tasks, or Firestore schema.

The current dashboard can confuse users because:

- A queued job can appear as a `Live job` even when the worker has not started and no row plan exists yet.
- The top metric says `Running 0` while the journey card says `Queued / Live job`.
- The `Live Attack Workbench` exists only inside the `Live Attack` tab and is not surfaced as a main-dashboard entry point.
- New users need a clearer explanation of what state they are seeing and what action to take next.

## Exact Actions

- [x] Update the journey `Navigate execution / Live Status` card to distinguish:
  - `Idle`: no queued, started, or running jobs.
  - `Queued`: job exists and Cloud Tasks/worker start is pending.
  - `Starting`: Cloud Task/API handoff has begun.
  - `Running`: worker is actively evaluating rows.
  - `Needs Review`: latest known state has failed jobs.
- [x] Avoid labeling queued jobs as `Live job` until the worker is actually `started` or `running`.
- [x] Show a human-readable state note:
  - queued: "Waiting for Cloud Tasks to start the worker."
  - started: "Worker start requested; waiting for row progress."
  - running: "Worker is evaluating model/attack/defense rows."
  - idle: "No active benchmark is running."
- [x] Make the dashboard metrics consistent by counting queued jobs separately from running jobs where visible.
- [x] Add a main-dashboard entry point for `Live Attack Workbench` so users can discover it before opening the tab.
- [x] Keep all changes UI-only and non-breaking.
- [x] Verify with Python compile and focused tests.

## Official Grounding

- Cloud Tasks is asynchronous work outside the user request path; queued work should not be presented as already running.
- Cloud Run Jobs create executions that run to completion and expose execution/log state separately from task enqueue state.
- Firestore supports realtime listeners, but this change intentionally avoids direct browser Firestore listeners because that would require a larger auth/client architecture change.

## Acceptance Criteria

- [x] A queued benchmark displays as `Queued`, not `Live job`.
- [x] A running benchmark displays current model/attack/progress if those fields exist.
- [x] If no active job exists, the card clearly says `Idle`.
- [x] Users can see a clear main-dashboard call-to-action for Live Attack Workbench.
- [x] Existing tabs and benchmark/live attack functionality remain intact.

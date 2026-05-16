"""Streamlit dashboard for the Cloud Run product console."""

from __future__ import annotations

import os
import time
from html import escape
from urllib.parse import quote

import google.auth.transport.requests
import google.oauth2.id_token
import pandas as pd
import requests
import streamlit as st


API_URL = os.environ.get("REDTEAM_API_URL", "http://localhost:8080").rstrip("/")
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "commodity-redteam-logo.svg")


def render_shell() -> None:
    st.markdown(
        """
        <style>
        :root {
          --crt-ink: #0b1220;
          --crt-muted: #536173;
          --crt-line: #d9e2ec;
          --crt-surface: #ffffff;
          --crt-soft: #f6f8fb;
          --crt-teal: #0f766e;
          --crt-teal-2: #14b8a6;
          --crt-amber: #f59e0b;
          --crt-red: #dc2626;
          --crt-green: #16a34a;
        }
        .block-container {
          padding-top: 1.4rem;
          padding-bottom: 3rem;
          max-width: 1480px;
        }
        div[data-testid="stDecoration"] {
          background: linear-gradient(90deg, var(--crt-teal), var(--crt-amber), #1d4ed8);
        }
        .crt-hero {
          border: 1px solid var(--crt-line);
          background:
            radial-gradient(circle at 94% 18%, rgba(245, 158, 11, 0.12), transparent 22rem),
            linear-gradient(135deg, #ffffff 0%, #f6faf9 52%, #eef6f4 100%);
          border-radius: 8px;
          padding: 1.35rem 1.45rem;
          margin-bottom: 1rem;
          box-shadow: 0 18px 45px rgba(15, 23, 42, 0.07);
        }
        .crt-brand-row {
          display: flex;
          align-items: center;
          gap: .85rem;
          margin-bottom: .75rem;
        }
        .crt-logo-mark {
          width: 48px;
          height: 48px;
          border-radius: 8px;
          background: #0b1220;
          display: grid;
          place-items: center;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,.08);
        }
        .crt-logo-initials {
          color: #14b8a6;
          font-size: 1.05rem;
          font-weight: 900;
          letter-spacing: 0;
        }
        .crt-kicker {
          color: var(--crt-teal);
          font-weight: 800;
          letter-spacing: .06em;
          text-transform: uppercase;
          font-size: .78rem;
        }
        .crt-title {
          color: var(--crt-ink);
          font-size: clamp(2rem, 3.7vw, 3.8rem);
          line-height: .98;
          font-weight: 850;
          margin: 0;
          letter-spacing: 0;
        }
        .crt-subtitle {
          color: var(--crt-muted);
          font-size: 1.02rem;
          line-height: 1.55;
          max-width: 780px;
          margin-top: .85rem;
        }
        .crt-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: .5rem;
          margin-top: 1rem;
        }
        .crt-pill {
          border: 1px solid #cbd5e1;
          border-radius: 999px;
          padding: .36rem .68rem;
          background: rgba(255,255,255,.72);
          color: #273548;
          font-size: .84rem;
          font-weight: 650;
        }
        .crt-health {
          border: 1px solid var(--crt-line);
          border-radius: 8px;
          background: var(--crt-surface);
          padding: 1rem;
          height: 100%;
        }
        .crt-health-title {
          color: var(--crt-muted);
          font-size: .78rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: .06em;
          margin-bottom: .45rem;
        }
        .crt-health-value {
          color: var(--crt-ink);
          font-size: 1.2rem;
          font-weight: 800;
          overflow-wrap: anywhere;
        }
        .crt-good { color: var(--crt-green); }
        .crt-bad { color: var(--crt-red); }
        .crt-metric {
          border: 1px solid var(--crt-line);
          border-radius: 8px;
          background: var(--crt-surface);
          padding: 1rem;
          min-height: 118px;
          box-shadow: 0 10px 24px rgba(15,23,42,.05);
        }
        .crt-metric-label {
          color: var(--crt-muted);
          font-size: .78rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: .05em;
          margin-bottom: .48rem;
        }
        .crt-metric-value {
          color: var(--crt-ink);
          font-size: 2rem;
          line-height: 1.05;
          font-weight: 850;
        }
        .crt-metric-note {
          color: var(--crt-muted);
          font-size: .84rem;
          margin-top: .45rem;
        }
        .crt-section-title {
          font-size: 1.15rem;
          color: var(--crt-ink);
          font-weight: 820;
          margin: 1.1rem 0 .65rem 0;
        }
        .crt-flow {
          display: grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: .7rem;
          margin: .4rem 0 1.2rem 0;
        }
        .crt-flow-step {
          border: 1px solid var(--crt-line);
          border-radius: 8px;
          background: var(--crt-soft);
          padding: .85rem;
          min-height: 112px;
        }
        .crt-flow-num {
          display: inline-flex;
          width: 1.65rem;
          height: 1.65rem;
          align-items: center;
          justify-content: center;
          border-radius: 999px;
          background: var(--crt-ink);
          color: white;
          font-weight: 800;
          font-size: .78rem;
          margin-bottom: .55rem;
        }
        .crt-flow-title {
          color: var(--crt-ink);
          font-weight: 800;
          margin-bottom: .25rem;
        }
        .crt-flow-copy {
          color: var(--crt-muted);
          font-size: .84rem;
          line-height: 1.35;
        }
        .crt-journey {
          border: 1px solid #cbd5e1;
          border-radius: 8px;
          background:
            linear-gradient(135deg, rgba(11, 18, 32, .96), rgba(15, 23, 42, .92)),
            linear-gradient(90deg, rgba(20, 184, 166, .16), rgba(245, 158, 11, .14));
          color: #e5eef7;
          padding: 1rem;
          margin: .2rem 0 1.2rem 0;
          box-shadow: 0 18px 44px rgba(15, 23, 42, .16);
        }
        .crt-journey-head {
          display: flex;
          justify-content: space-between;
          gap: 1rem;
          align-items: end;
          margin-bottom: .9rem;
        }
        .crt-journey-kicker {
          color: #5eead4;
          font-size: .76rem;
          font-weight: 850;
          letter-spacing: .07em;
          text-transform: uppercase;
        }
        .crt-journey-title {
          color: #ffffff;
          font-size: 1.38rem;
          font-weight: 850;
          margin-top: .15rem;
        }
        .crt-journey-copy {
          color: #b9c6d6;
          font-size: .9rem;
          line-height: 1.45;
          max-width: 760px;
        }
        .crt-journey-map {
          display: grid;
          grid-template-columns: 1fr .8fr 1.08fr 1fr;
          gap: .72rem;
          align-items: stretch;
        }
        .crt-journey-node {
          position: relative;
          border: 1px solid rgba(203, 213, 225, .22);
          border-radius: 8px;
          background: rgba(255, 255, 255, .075);
          padding: .85rem;
          min-height: 184px;
          overflow: hidden;
        }
        .crt-journey-node::after {
          content: "";
          position: absolute;
          right: -.48rem;
          top: 50%;
          width: .95rem;
          height: .95rem;
          border-top: 2px solid rgba(94, 234, 212, .78);
          border-right: 2px solid rgba(94, 234, 212, .78);
          transform: translateY(-50%) rotate(45deg);
          background: #101a2b;
        }
        .crt-journey-node:last-child::after {
          display: none;
        }
        .crt-node-label {
          display: inline-flex;
          align-items: center;
          gap: .4rem;
          color: #f8fafc;
          font-weight: 850;
          font-size: .96rem;
          margin-bottom: .45rem;
        }
        .crt-node-chip {
          border: 1px solid rgba(255, 255, 255, .22);
          border-radius: 999px;
          color: #cbd5e1;
          font-size: .72rem;
          font-weight: 800;
          padding: .18rem .42rem;
          background: rgba(255,255,255,.06);
        }
        .crt-node-copy {
          color: #b9c6d6;
          font-size: .82rem;
          line-height: 1.38;
          margin-bottom: .62rem;
        }
        .crt-prompt-stack {
          display: grid;
          gap: .38rem;
        }
        .crt-prompt-card {
          border: 1px solid rgba(148, 163, 184, .28);
          border-radius: 6px;
          padding: .42rem .48rem;
          background: rgba(15, 23, 42, .72);
          color: #dbeafe;
          font-size: .76rem;
          line-height: 1.28;
        }
        .crt-prompt-card strong {
          color: #fbbf24;
        }
        .crt-user-add {
          border-color: rgba(94, 234, 212, .45);
          background: rgba(20, 184, 166, .12);
        }
        .crt-lane {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .45rem;
          margin-top: .72rem;
        }
        .crt-lane-step {
          border: 1px solid rgba(203, 213, 225, .20);
          border-radius: 6px;
          padding: .44rem;
          color: #d6e2ef;
          font-size: .74rem;
          background: rgba(255,255,255,.055);
        }
        .crt-lane-step b {
          display: block;
          color: #ffffff;
          font-size: .76rem;
          margin-bottom: .16rem;
        }
        .crt-evidence-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: .5rem;
          margin-top: .8rem;
        }
        .crt-evidence-item {
          border-left: 3px solid #14b8a6;
          background: rgba(255, 255, 255, .07);
          border-radius: 6px;
          padding: .5rem .56rem;
          color: #cbd5e1;
          font-size: .76rem;
          min-height: 64px;
        }
        .crt-evidence-item b {
          display: block;
          color: #ffffff;
          font-size: .8rem;
          margin-bottom: .15rem;
        }
        .crt-tracker {
          border: 1px solid var(--crt-line);
          border-radius: 8px;
          background: #ffffff;
          padding: 1rem;
          margin: .8rem 0 1rem 0;
          box-shadow: 0 10px 24px rgba(15,23,42,.05);
        }
        .crt-tracker-title {
          color: var(--crt-ink);
          font-weight: 850;
          font-size: 1.05rem;
          margin-bottom: .25rem;
        }
        .crt-tracker-copy {
          color: var(--crt-muted);
          font-size: .9rem;
          line-height: 1.45;
          margin-bottom: .8rem;
        }
        .crt-status-badge {
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          padding: .3rem .62rem;
          font-size: .78rem;
          font-weight: 850;
          text-transform: uppercase;
          letter-spacing: .04em;
          background: #eef6f4;
          color: #0f766e;
          border: 1px solid #b8ded8;
        }
        .crt-status-badge.failed {
          background: #fef2f2;
          color: #b91c1c;
          border-color: #fecaca;
        }
        .crt-status-badge.completed {
          background: #f0fdf4;
          color: #15803d;
          border-color: #bbf7d0;
        }
        .crt-status-badge.queued {
          background: #fffbeb;
          color: #b45309;
          border-color: #fde68a;
        }
        .crt-mini-label {
          color: var(--crt-muted);
          font-size: .76rem;
          text-transform: uppercase;
          letter-spacing: .05em;
          font-weight: 800;
        }
        .crt-mini-value {
          color: var(--crt-ink);
          font-weight: 750;
          overflow-wrap: anywhere;
        }
        @media (max-width: 1100px) {
          .crt-flow { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .crt-journey-map { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .crt-journey-node::after { display: none; }
          .crt-evidence-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 720px) {
          .crt-flow { grid-template-columns: 1fr; }
          .crt-hero { padding: 1rem; }
          .crt-journey-head { display: block; }
          .crt-journey-map,
          .crt-lane,
          .crt-evidence-strip { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_logo_mark() -> str:
    return '<div class="crt-logo-mark" aria-hidden="true"><span class="crt-logo-initials">CR</span></div>'


def render_metric_card(label: str, value: object, note: str) -> None:
    st.markdown(
        f'<div class="crt-metric"><div class="crt-metric-label">{label}</div><div class="crt-metric-value">{value}</div><div class="crt-metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_flow() -> None:
    steps = [
        ("1", "Select scope", "Pick Vertex models, attack families, and defenses."),
        ("2", "Queue benchmark", "API creates the job and Cloud Tasks starts execution."),
        ("3", "Run worker", "Cloud Run Job executes attacks against selected LLMs."),
        ("4", "Evaluate risk", "Evaluator scores policy bypass and tool misuse signals."),
        ("5", "Store evidence", "Firestore tracks status; GCS stores run artifacts."),
        ("6", "Review results", "Dashboard compares models, defenses, and downloads."),
    ]
    body = "".join(
        f'<div class="crt-flow-step"><div class="crt-flow-num">{num}</div><div class="crt-flow-title">{title}</div><div class="crt-flow-copy">{copy}</div></div>'
        for num, title, copy in steps
    )
    st.markdown(f'<div class="crt-flow">{body}</div>', unsafe_allow_html=True)


def render_task_journey_diagram() -> None:
    st.markdown(
        """
        <section class="crt-journey">
          <div class="crt-journey-head">
            <div>
              <div class="crt-journey-kicker">Transparent task journey</div>
              <div class="crt-journey-title">From task kickoff to prompt-level evidence</div>
            </div>
            <div class="crt-journey-copy">
              The console is an LLM model safety workspace. Vertex AI is the access path to the selected model,
              while the product shows the task plan, injected prompts, defenses, and evaluator evidence as inspectable stages.
            </div>
          </div>
          <div class="crt-journey-map">
            <div class="crt-journey-node">
              <div class="crt-node-label">Kickstart task <span class="crt-node-chip">Run Benchmark</span></div>
              <div class="crt-node-copy">User selects model, attack family, defense stack, and run mode. The run plan is visible before submission.</div>
              <div class="crt-lane">
                <div class="crt-lane-step"><b>Scope</b>Model and attack IDs</div>
                <div class="crt-lane-step"><b>Defend</b>Filters and guardrails</div>
                <div class="crt-lane-step"><b>Launch</b>Firestore job created</div>
              </div>
            </div>
            <div class="crt-journey-node">
              <div class="crt-node-label">Navigate execution <span class="crt-node-chip">Live Status</span></div>
              <div class="crt-node-copy">Cloud Tasks starts the worker. The tracker shows queued, started, running, evidence stored, and failure states.</div>
              <div class="crt-lane">
                <div class="crt-lane-step"><b>API</b>Validates task</div>
                <div class="crt-lane-step"><b>Worker</b>Runs matrix</div>
                <div class="crt-lane-step"><b>Artifacts</b>JSON, CSV, report</div>
              </div>
            </div>
            <div class="crt-journey-node">
              <div class="crt-node-label">Inspect prompts <span class="crt-node-chip">Not a black box</span></div>
              <div class="crt-node-copy">Each attack can expose the prompt ingredients that reached the model-under-test through Vertex AI.</div>
              <div class="crt-prompt-stack">
                <div class="crt-prompt-card"><strong>System:</strong> trading safety policy, limits, approval rules</div>
                <div class="crt-prompt-card"><strong>Attack:</strong> direct injection, tool manipulation, context poisoning</div>
                <div class="crt-prompt-card"><strong>Defense:</strong> filtered input, hardened prompt, validator hints</div>
                <div class="crt-prompt-card crt-user-add"><strong>User add-on:</strong> paste an extra probe and rerun on demand</div>
              </div>
            </div>
            <div class="crt-journey-node">
              <div class="crt-node-label">Compare outcomes <span class="crt-node-chip">Results Explorer</span></div>
              <div class="crt-node-copy">Unsafe behavior, detector coverage, model variance, and defense regressions are reviewed next to the exact run configuration.</div>
              <div class="crt-lane">
                <div class="crt-lane-step"><b>Score</b>Bypass and misuse</div>
                <div class="crt-lane-step"><b>Trace</b>Prompt to response</div>
                <div class="crt-lane-step"><b>Extend</b>Add probes</div>
              </div>
            </div>
          </div>
          <div class="crt-evidence-strip">
            <div class="crt-evidence-item"><b>Prompt drawer</b>Shows generated attack prompt, defense additions, and selected target model context.</div>
            <div class="crt-evidence-item"><b>On-demand probes</b>Add a one-off prompt to the live attack path without changing the benchmark catalog.</div>
            <div class="crt-evidence-item"><b>Catalog promotion</b>Promote useful probes into reusable attack cases after review.</div>
            <div class="crt-evidence-item"><b>Audit trail</b>Persist model ID, prompt version, defenses, scores, and artifacts for replay.</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_run_explainer() -> None:
    st.markdown(
        """
        When you start a benchmark, the UI does not run the attacks itself. It sends the run plan to the API,
        the API records a Firestore job, Cloud Tasks starts the worker safely, and the worker calls Vertex AI
        for each selected model/attack/defense combination. Progress below is read back from Firestore.
        """
    )


def _auth_headers() -> dict[str, str]:
    if API_URL.startswith("http://localhost") or API_URL.startswith("http://127.0.0.1"):
        return {}
    request = google.auth.transport.requests.Request()
    token = google.oauth2.id_token.fetch_id_token(request, API_URL)
    return {"Authorization": f"Bearer {token}"}


def api_get(path: str, timeout: int = 30) -> dict:
    response = requests.get(f"{API_URL}{path}", headers=_auth_headers(), timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict, timeout: int = 120) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, headers=_auth_headers(), timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_download(path: str) -> bytes:
    response = requests.get(f"{API_URL}{path}", headers=_auth_headers(), timeout=120)
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=30)
def load_catalog() -> tuple[dict, list[dict], list[str]]:
    models = api_get("/models")
    attacks = api_get("/attacks").get("attacks", [])
    defenses = api_get("/defenses").get("defenses", [])
    return models, attacks, defenses


@st.cache_data(ttl=15)
def load_jobs(limit: int = 50) -> list[dict]:
    return api_get(f"/jobs?limit={limit}").get("jobs", [])


def load_job(job_id: str) -> dict:
    return api_get(f"/jobs/{job_id}", timeout=15)


@st.cache_data(ttl=15)
def load_job_results(job_id: str) -> dict:
    return api_get(f"/jobs/{job_id}/results?include_results=true&result_limit=5000", timeout=60)


@st.cache_data(ttl=30)
def load_artifacts(job_id: str) -> dict:
    return api_get(f"/jobs/{job_id}/artifacts")


def clear_data_cache() -> None:
    st.cache_data.clear()


def jobs_dataframe(jobs: list[dict]) -> pd.DataFrame:
    rows = []
    for job in jobs:
        config = job.get("config", {})
        rows.append(
            {
                "job_id": job.get("job_id"),
                "status": job.get("status"),
                "mode": config.get("mode"),
                "models": ", ".join(config.get("models", [])),
                "defenses": ", ".join(config.get("defenses", [])),
                "attacks": len(config.get("attack_ids") or []),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
                "result_uri": job.get("result_uri", ""),
            }
        )
    return pd.DataFrame(rows)


def status_counts(jobs: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        status = job.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def job_progress(job: dict) -> tuple[int, str]:
    status = job.get("status", "unknown")
    completed = int(job.get("progress_completed", 0) or 0)
    total = int(job.get("progress_total", 0) or 0)
    if status == "completed":
        return 100, "Artifacts are ready in GCS and visible in Results Explorer."
    if status == "failed":
        return max(5, int(100 * completed / total)) if total else 100, job.get("error", "The worker failed.")
    if total:
        pct = min(99, max(10, int(100 * completed / total)))
        return pct, f"Evaluated {completed} of {total} model/defense/attack combinations."
    status_pct = {
        "queued": 10,
        "started": 28,
        "running": 48,
    }
    messages = {
        "queued": "Job document created. Cloud Tasks is preparing to call the internal start endpoint.",
        "started": "Cloud Task reached the API. The API requested a Cloud Run Job execution.",
        "running": "Worker started. The benchmark is executing against Vertex AI.",
    }
    return status_pct.get(status, 5), messages.get(status, "Waiting for job state.")


def job_timeline(job: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "1. Queued",
                "what happens": "API validates request and writes Firestore job.",
                "time": job.get("created_at", ""),
                "state": "done" if job.get("created_at") else "waiting",
            },
            {
                "stage": "2. Task accepted",
                "what happens": "Cloud Tasks calls the API internal starter with OIDC.",
                "time": job.get("started_at", ""),
                "state": "done" if job.get("started_at") else "waiting",
            },
            {
                "stage": "3. Worker running",
                "what happens": "Cloud Run Job evaluates attacks against selected Vertex models.",
                "time": job.get("worker_started_at", ""),
                "state": "done" if job.get("worker_started_at") else "waiting",
            },
            {
                "stage": "4. Evidence stored",
                "what happens": "Worker uploads JSON/CSV/report artifacts to GCS.",
                "time": job.get("worker_completed_at", ""),
                "state": "done" if job.get("worker_completed_at") else "waiting",
            },
        ]
    )


def render_job_tracker(job: dict, title: str = "Benchmark Job Tracker") -> None:
    status = str(job.get("status", "unknown"))
    pct, message = job_progress(job)
    config = job.get("config", {})
    badge_class = status if status in {"queued", "completed", "failed"} else ""
    st.markdown(
        f"""
        <div class="crt-tracker">
          <div class="crt-tracker-title">{escape(title)}</div>
          <div class="crt-tracker-copy">
            <span class="crt-status-badge {badge_class}">{escape(status)}</span>
            &nbsp; Job <strong>{escape(str(job.get("job_id", "")))}</strong> · {escape(message)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(pct, text=f"{pct}% · {message}")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Models", len(config.get("models", [])))
    metric_cols[1].metric("Defenses", len(config.get("defenses", [])))
    metric_cols[2].metric("Attacks", len(config.get("attack_ids") or []))
    metric_cols[3].metric("Completed Rows", int(job.get("progress_completed", 0) or 0))
    metric_cols[4].metric("Expected Rows", int(job.get("progress_total", 0) or 0))

    current_cols = st.columns(3)
    current_cols[0].caption("Current model")
    current_cols[0].write(job.get("current_model") or "-")
    current_cols[1].caption("Current defense")
    current_cols[1].write(job.get("current_defense") or "-")
    current_cols[2].caption("Current attack")
    current_cols[2].write(job.get("current_attack_id") or "-")

    if job.get("progress_message"):
        st.info(job["progress_message"])
    if job.get("error"):
        st.error(job["error"])

    st.dataframe(job_timeline(job), width="stretch", hide_index=True)

    with st.expander("Show job internals and configuration"):
        st.markdown("These fields help explain what happened behind the scenes.")
        internals = {
            "task_name": job.get("task_name"),
            "run_operation": job.get("run_operation"),
            "result_uri": job.get("result_uri"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "worker_started_at": job.get("worker_started_at"),
            "worker_completed_at": job.get("worker_completed_at"),
            "config": config,
        }
        st.json(internals)


st.set_page_config(page_title="CommodityRedTeam", page_icon=LOGO_PATH, layout="wide")
render_shell()
if os.path.exists(LOGO_PATH):
    st.logo(LOGO_PATH, size="large")

top_left, top_right = st.columns([4, 1])
with top_right:
    if st.button("Refresh Data", width="stretch"):
        clear_data_cache()
        st.rerun()

api_status = {"ok": False, "service": "redteam-api", "detail": ""}
with top_left:
    try:
        health = api_get("/health")
        api_status = {
            "ok": True,
            "service": health.get("service", "redteam-api"),
            "detail": health.get("status", "ready"),
        }
    except Exception as exc:
        api_status = {"ok": False, "service": "redteam-api", "detail": str(exc)}

try:
    model_payload, attacks, defense_names = load_catalog()
    jobs = load_jobs()
except Exception as exc:
    st.error(f"Unable to load dashboard data: {exc}")
    st.stop()

models = model_payload.get("models", [])
model_metadata = model_payload.get("model_metadata", [])
jobs_df = jobs_dataframe(jobs)
counts = status_counts(jobs)

vertex_models = [m for m in model_metadata if m.get("provider") == "vertex"]
completed_count = counts.get("completed", 0)
running_count = counts.get("running", 0) + counts.get("started", 0)
failed_count = counts.get("failed", 0)
status_class = "crt-good" if api_status["ok"] else "crt-bad"
status_text = "Connected" if api_status["ok"] else "Unavailable"
api_service_label = escape(str(api_status["service"]))
api_detail_label = escape(str(api_status["detail"]))

hero_left, hero_right = st.columns([3.2, 1.15])
with hero_left:
    st.markdown(
        (
            '<section class="crt-hero">'
            '<div class="crt-brand-row">'
            f'{render_logo_mark()}'
            '<div><div class="crt-kicker">LLM model safety evaluation console</div>'
            '<h1 class="crt-title">CommodityRedTeam</h1></div>'
            '</div>'
            '<div class="crt-subtitle">Enterprise benchmark workspace for testing commodity-trading LLM agents '
            'against prompt attacks, tool manipulation, policy bypass, and defense regressions before models are trusted '
            'in real workflows. Vertex AI is the model access layer for this version, not the product boundary.</div>'
            '<div class="crt-pill-row">'
            '<span class="crt-pill">Cloud Run</span>'
            '<span class="crt-pill">Vertex AI Model Garden</span>'
            '<span class="crt-pill">Firestore status</span>'
            '<span class="crt-pill">GCS evidence</span>'
            '<span class="crt-pill">IAP protected</span>'
            '</div></section>'
        ),
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        (
            '<div class="crt-health">'
            '<div class="crt-health-title">Control plane</div>'
            f'<div class="crt-health-value {status_class}">{status_text}</div>'
            f'<div class="crt-metric-note">{api_service_label} · {api_detail_label}</div>'
            '<div style="height:.75rem"></div>'
            '<div class="crt-health-title">Deployment</div>'
            '<div class="crt-health-value">asia-south1</div>'
            '<div class="crt-metric-note">UI, API, and worker run on Cloud Run.</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

metric_cols = st.columns(5)
with metric_cols[0]:
    render_metric_card("Total Jobs", len(jobs), "Benchmark runs tracked in Firestore.")
with metric_cols[1]:
    render_metric_card("Completed", completed_count, "Finished jobs with persisted artifacts.")
with metric_cols[2]:
    render_metric_card("Running", running_count, "Started or active Cloud Run executions.")
with metric_cols[3]:
    render_metric_card("Failed", failed_count, "Jobs needing triage or rerun.")
with metric_cols[4]:
    render_metric_card("Attack Catalog", len(attacks), "Registered test cases available.")

st.markdown('<div class="crt-section-title">How a demonstration run works</div>', unsafe_allow_html=True)
render_flow()
render_task_journey_diagram()

overview_tab, run_tab, results_tab, live_tab, catalog_tab = st.tabs(
    ["Overview", "Run Benchmark", "Results Explorer", "Live Attack", "Catalog"]
)

with overview_tab:
    active_overview_job = next(
        (
            job
            for job in jobs
            if job.get("status") in {"started", "running"}
            or (job.get("status") == "queued" and job.get("task_name"))
        ),
        None,
    )
    if active_overview_job:
        try:
            render_job_tracker(load_job(active_overview_job["job_id"]), title="Active cloud run")
        except Exception as exc:
            st.warning(f"Active job status could not be refreshed: {exc}")

    st.subheader("Job History")
    if jobs_df.empty:
        st.info("No jobs found yet.")
    else:
        st.dataframe(jobs_df, width="stretch", hide_index=True)

    recent_completed = next((job for job in jobs if job.get("status") == "completed" and job.get("result_uri")), None)
    if recent_completed:
        st.subheader("Latest Completed Run")
        col_a, col_b, col_c = st.columns(3)
        config = recent_completed.get("config", {})
        col_a.metric("Job", recent_completed["job_id"])
        col_b.metric("Mode", config.get("mode", "unknown"))
        col_c.metric("Models", len(config.get("models", [])))
        try:
            latest = load_job_results(recent_completed["job_id"])
            summary_df = pd.DataFrame(latest.get("summary", {}).get("groups", []))
            if not summary_df.empty:
                st.dataframe(summary_df, width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"Latest result summary is not ready: {exc}")

with run_tab:
    st.subheader("Create Benchmark Job")
    render_run_explainer()
    with st.form("benchmark_form"):
        model_options = [m["name"] for m in model_metadata if m.get("provider") == "vertex"] or models
        benchmark_models = st.multiselect(
            "Models",
            models,
            default=[m for m in ["vertex-gemini-flash", "vertex-gemini-pro"] if m in models] or models[:1],
        )
        benchmark_mode = st.radio(
            "Mode",
            ["selected", "full_matrix"],
            format_func=lambda value: "Selected run" if value == "selected" else "Full matrix",
            horizontal=True,
        )
        benchmark_defenses = st.multiselect(
            "Defenses",
            defense_names,
            default=["input_filter"] if "input_filter" in defense_names else [],
        )

        attack_df = pd.DataFrame(attacks)
        category_filter = st.multiselect(
            "Attack Categories",
            sorted(attack_df["category"].dropna().unique()) if not attack_df.empty else [],
        )
        filtered_attacks = attack_df
        if category_filter:
            filtered_attacks = attack_df[attack_df["category"].isin(category_filter)]
        attack_options = filtered_attacks["id"].tolist() if not filtered_attacks.empty else []
        default_attacks = attack_options[:2] if benchmark_mode == "full_matrix" else attack_options[:1]
        benchmark_attack_ids = st.multiselect(
            "Attack IDs",
            attack_options,
            default=default_attacks,
            help="Leave empty only when intentionally running every registered attack.",
        )
        col_delay, col_reports = st.columns(2)
        delay_seconds = col_delay.slider("Delay between defense configs", min_value=0, max_value=10, value=1)
        include_reports = col_reports.checkbox("Generate report artifacts", value=True)

        submitted = st.form_submit_button("Start Benchmark", width="stretch")

    if submitted:
        payload = {
            "models": benchmark_models or model_options[:1],
            "defenses": benchmark_defenses,
            "attack_ids": benchmark_attack_ids or None,
            "mode": benchmark_mode,
            "include_reports": include_reports,
            "delay_seconds": delay_seconds,
        }
        try:
            result = api_post("/jobs/benchmark", payload)
            st.session_state["active_job_id"] = result["job_id"]
            clear_data_cache()
            st.success(f"Queued benchmark job {result['job_id']}")
            render_job_tracker(result, title="New benchmark queued")
        except Exception as exc:
            st.error(f"Benchmark start failed: {exc}")

    st.markdown("#### Active Job Progress")
    active_job_id = st.session_state.get("active_job_id")
    recent_jobs = [job for job in jobs if job.get("job_id")]
    job_ids = [job["job_id"] for job in recent_jobs]
    default_index = job_ids.index(active_job_id) if active_job_id in job_ids else 0 if job_ids else None
    if not job_ids:
        st.info("No jobs have been created yet. Start a benchmark to see progress here.")
    else:
        selected_tracking_job = st.selectbox(
            "Track job",
            job_ids,
            index=default_index,
            format_func=lambda job_id: next(
                (
                    f"{job_id} | {job.get('status', 'unknown')} | {job.get('created_at', '')}"
                    for job in recent_jobs
                    if job.get("job_id") == job_id
                ),
                job_id,
            ),
        )
        try:
            tracked_job = load_job(selected_tracking_job)
            render_job_tracker(tracked_job)
            is_terminal = tracked_job.get("status") in {"completed", "failed"}
            controls = st.columns([1, 1, 3])
            if controls[0].button("Refresh Progress", width="stretch"):
                clear_data_cache()
                st.rerun()
            auto_refresh = controls[1].checkbox("Auto refresh", value=not is_terminal)
            if auto_refresh and not is_terminal:
                st.caption("Refreshing every 6 seconds while the job is active.")
                time.sleep(6)
                clear_data_cache()
                st.rerun()
        except Exception as exc:
            st.warning(f"Unable to load job progress: {exc}")

with results_tab:
    st.subheader("Results Explorer")
    completed_jobs = [job for job in jobs if job.get("status") == "completed" and job.get("result_uri")]
    if not completed_jobs:
        st.info("No completed result artifacts are available yet.")
    else:
        selected_job = st.selectbox(
            "Completed Job",
            completed_jobs,
            format_func=lambda job: f"{job['job_id']} | {job.get('created_at', '')} | {job.get('config', {}).get('mode', '')}",
        )
        job_id = selected_job["job_id"]
        result_payload = load_job_results(job_id)
        metadata = result_payload.get("metadata", {})
        summary = result_payload.get("summary", {})
        result_rows = result_payload.get("results", [])

        meta_cols = st.columns(4)
        meta_cols[0].metric("Rows", result_payload.get("result_count", 0))
        meta_cols[1].metric("Models", len(metadata.get("models", [])))
        meta_cols[2].metric("Attacks", len(metadata.get("attack_ids", [])))
        meta_cols[3].metric("Tokens", metadata.get("total_tokens", {}).get("input", 0) + metadata.get("total_tokens", {}).get("output", 0))

        st.markdown("#### Summary")
        summary_df = pd.DataFrame(summary.get("groups", []))
        if summary_df.empty:
            st.info("Summary artifact is empty.")
        else:
            st.dataframe(summary_df, width="stretch", hide_index=True)

        st.markdown("#### Result Rows")
        results_df = pd.DataFrame(result_rows)
        if results_df.empty:
            st.info("No result rows returned.")
        else:
            model_filter = st.multiselect("Filter Models", sorted(results_df["model"].dropna().unique()))
            defense_filter = st.multiselect("Filter Defenses", sorted(results_df["defense"].dropna().unique()))
            filtered_results = results_df
            if model_filter:
                filtered_results = filtered_results[filtered_results["model"].isin(model_filter)]
            if defense_filter:
                filtered_results = filtered_results[filtered_results["defense"].isin(defense_filter)]
            st.dataframe(filtered_results, width="stretch", hide_index=True)

        st.markdown("#### Downloads")
        try:
            artifact_payload = load_artifacts(job_id)
            artifacts = artifact_payload.get("artifacts", [])
            if not artifacts:
                st.info("No downloadable artifacts found.")
            for artifact in artifacts:
                cols = st.columns([4, 1, 1])
                cols[0].write(artifact["name"])
                cols[1].write(f"{artifact.get('size') or 0:,} bytes")
                encoded = quote(artifact["name"])
                data = api_download(f"/jobs/{job_id}/artifacts/{encoded}")
                cols[2].download_button(
                    "Download",
                    data=data,
                    file_name=artifact["name"].split("/")[-1],
                    mime=artifact.get("content_type") or "application/octet-stream",
                    key=f"download-{job_id}-{artifact['name']}",
                    width="stretch",
                )
        except Exception as exc:
            st.warning(f"Artifacts unavailable: {exc}")

with live_tab:
    st.subheader("Live Single Attack")
    selected_model = st.selectbox("Model", models, key="single_model")
    selected_attack = st.selectbox(
        "Attack",
        attacks,
        format_func=lambda attack: f"{attack['id']} - {attack['name']}",
        key="single_attack",
    )
    selected_defenses = st.multiselect("Defenses", defense_names, key="single_defenses")
    with st.expander("Inspect selected attack prompt path", expanded=True):
        st.caption("The live run can show the catalog attack prompt, defense-added system context, and any one-off probe appended for this test.")
        st.json(
            {
                "attack_id": selected_attack["id"],
                "category": selected_attack.get("category"),
                "severity": selected_attack.get("severity"),
                "description": selected_attack.get("description"),
            }
        )
    additional_prompt = st.text_area(
        "Additional on-demand probe",
        value="",
        height=140,
        placeholder="Paste an extra prompt to append to this attack for a one-off live test.",
        help="This is appended to the selected attack only for the live run. It does not change the benchmark catalog.",
    )
    if st.button("Run Live Attack", width="stretch"):
        try:
            result = api_post(
                "/run/single",
                {
                    "model": selected_model,
                    "attack_id": selected_attack["id"],
                    "defenses": selected_defenses,
                    "additional_prompt": additional_prompt or None,
                },
            )
            st.success("Live attack completed.")
            trace = result.get("prompt_trace", {})
            with st.expander("Prompt trace sent through the model access layer", expanded=True):
                trace_cols = st.columns(3)
                trace_cols[0].metric("Model", trace.get("model", selected_model))
                trace_cols[1].metric("Access layer", trace.get("model_access_layer", "-"))
                trace_cols[2].metric("Additional probe", "Yes" if result.get("additional_prompt_applied") else "No")
                st.markdown("##### User / attack prompt")
                st.code(trace.get("user_query", ""), language="text")
                if trace.get("final_context_before_user"):
                    st.markdown("##### Injected context before user prompt")
                    st.json(trace["final_context_before_user"])
                if trace.get("defense_steps"):
                    st.markdown("##### Defense prompt additions and flags")
                    st.json(trace["defense_steps"])
                if trace.get("tool_overrides"):
                    st.markdown("##### Tool overrides")
                    st.json(trace["tool_overrides"])
            st.markdown("#### Evaluation result")
            st.json(result)
        except Exception as exc:
            st.error(f"Run failed: {exc}")

with catalog_tab:
    st.subheader("Model Garden / Provider Catalog")
    st.dataframe(pd.DataFrame(model_metadata), width="stretch", hide_index=True)
    st.subheader("Attack Catalog")
    st.dataframe(pd.DataFrame(attacks), width="stretch", hide_index=True)
    st.subheader("Defense Catalog")
    st.dataframe(pd.DataFrame({"defense": defense_names}), width="stretch", hide_index=True)

"""FastAPI wrapper for the redteam evaluator."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.cloud import firestore, storage, tasks_v2
from pydantic import BaseModel, Field

from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT
from src.attacks.base import Attack, AttackResult
from src.attacks.registry import get_all_attacks
from src.defenses.base import Defense
from src.defenses.guardrails import GuardrailsDefense
from src.defenses.human_in_loop import HumanInLoopDefense
from src.defenses.input_filter import InputFilterDefense
from src.defenses.multi_agent import MultiAgentDefense
from src.defenses.output_validator import OutputValidatorDefense
from src.defenses.perplexity_filter import PerplexityFilterDefense
from src.defenses.semantic_filter import SemanticInputFilterDefense
from src.evaluation.evaluator import RedTeamEvaluator
from src.utils.llm import LLMClient


app = FastAPI(title="CommodityRedTeam API", version="0.1.0")

DEFENSE_MAP = {
    "input_filter": InputFilterDefense,
    "output_validator": OutputValidatorDefense,
    "guardrails": GuardrailsDefense,
    "human_in_loop": HumanInLoopDefense,
    "multi_agent": MultiAgentDefense,
    "semantic_filter": SemanticInputFilterDefense,
    "perplexity_filter": PerplexityFilterDefense,
}


class SingleRunRequest(BaseModel):
    model: str = Field(default="vertex-gemini-flash")
    attack_id: str
    defenses: list[str] = Field(default_factory=list)
    additional_prompt: str | None = Field(default=None, max_length=8000)


class BenchmarkJobRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["vertex-gemini-flash"])
    defenses: list[str] = Field(default_factory=list)
    attack_ids: list[str] | None = None
    mode: str = Field(default="selected", pattern="^(selected|full_matrix|report_only)$")
    include_reports: bool = Field(default=True)
    delay_seconds: int = Field(default=1, ge=0, le=30)


def _attack_index() -> dict[str, Any]:
    return {attack.id: attack for attack in get_all_attacks()}


def _project_id() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise HTTPException(status_code=500, detail="GOOGLE_CLOUD_PROJECT is not configured")
    return project


def _firestore() -> firestore.Client:
    return firestore.Client(project=_project_id())


def _jobs_collection():
    return _firestore().collection("benchmark_jobs")


def _storage_bucket() -> storage.Bucket:
    bucket_name = os.environ.get("RESULTS_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="RESULTS_BUCKET is not configured")
    return storage.Client(project=_project_id()).bucket(bucket_name)


def _job_or_404(job_id: str) -> dict[str, Any]:
    snapshot = _jobs_collection().document(job_id).get()
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return snapshot.to_dict()


def _artifact_prefix(job: dict[str, Any]) -> str:
    result_uri = job.get("result_uri")
    if not result_uri:
        raise HTTPException(status_code=404, detail="Job has no result artifacts yet")
    bucket_name = os.environ.get("RESULTS_BUCKET")
    expected = f"gs://{bucket_name}/"
    if not result_uri.startswith(expected):
        raise HTTPException(status_code=500, detail="Job result URI does not match configured results bucket")
    return result_uri[len(expected):].strip("/")


def _artifact_blob(job_id: str, artifact_path: str) -> storage.Blob:
    job = _job_or_404(job_id)
    prefix = _artifact_prefix(job)
    safe_path = artifact_path.strip("/")
    if not safe_path or ".." in safe_path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    blob = _storage_bucket().blob(f"{prefix}/{safe_path}")
    if not blob.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_path}")
    return blob


def _download_json_artifact(job_id: str, name: str) -> dict[str, Any] | None:
    try:
        blob = _artifact_blob(job_id, name)
    except HTTPException as exc:
        if exc.status_code == 404:
            return None
        raise
    import json

    return json.loads(blob.download_as_text())


def _build_defenses(names: list[str]) -> list[Defense]:
    defenses: list[Defense] = []
    for name in names:
        if name == "none":
            continue
        if name == "all_combined":
            return [
                InputFilterDefense(),
                OutputValidatorDefense(),
                GuardrailsDefense(),
                HumanInLoopDefense(),
                MultiAgentDefense(),
            ]
        defense_cls = DEFENSE_MAP.get(name)
        if defense_cls is None:
            raise HTTPException(status_code=400, detail=f"Unknown defense: {name}")
        defenses.append(defense_cls())
    return defenses


def _build_agent(model_name: str, client: LLMClient):
    def agent_fn(query: str, context: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": query})
        response = client.chat(model_name, messages)
        return response["content"]

    agent_fn.run = lambda query, context=None: agent_fn(query, context)
    agent_fn.system_prompt = DEFAULT_SYSTEM_PROMPT
    return agent_fn


class PromptAugmentedAttack(Attack):
    """Wrap a catalog attack with a one-off user probe for live testing."""

    def __init__(self, base_attack: Attack, additional_prompt: str) -> None:
        self.base_attack = base_attack
        self.additional_prompt = additional_prompt.strip()
        super().__init__(
            id=base_attack.id,
            name=f"{base_attack.name} + live probe",
            category=base_attack.category,
            severity=base_attack.severity,
            description=base_attack.description,
            target_action=base_attack.target_action,
            commodity=base_attack.commodity,
        )

    def prepare(self, agent) -> dict:
        payload = dict(self.base_attack.prepare(agent))
        base_query = str(payload.get("user_query", ""))
        payload["user_query"] = f"{base_query}\n\nADDITIONAL USER PROBE:\n{self.additional_prompt}"
        return payload

    def evaluate(self, agent_result) -> AttackResult:
        result = self.base_attack.evaluate(agent_result)
        result.notes = f"{result.notes} | Live additional prompt applied".strip()
        return result


def _build_prompt_trace(
    *,
    agent: Any,
    payload: dict[str, Any],
    defenses: list[Defense],
    model: str,
    attack_id: str,
) -> dict[str, Any]:
    user_query = str(payload.get("user_query", ""))
    injected_context = list(payload.get("injected_context", []))
    tool_overrides = payload.get("tool_overrides", {})
    defense_steps: list[dict[str, Any]] = []
    final_context = list(injected_context)

    for defense in defenses:
        result = defense.check_input(user_query, context=injected_context)
        step: dict[str, Any] = {
            "defense": defense.name,
            "allowed": result.allowed,
            "flags": result.flags,
            "confidence": result.confidence,
        }
        if result.modified_input:
            original_prompt = getattr(agent, "system_prompt", DEFAULT_SYSTEM_PROMPT)
            step["injected_system_prompt"] = result.modified_input.replace(
                "{ORIGINAL_SYSTEM_PROMPT}",
                str(original_prompt),
            )
        defense_steps.append(step)

    prompt_overrides = [
        step["injected_system_prompt"]
        for step in defense_steps
        if step.get("injected_system_prompt")
    ]
    if prompt_overrides:
        final_context = [
            {
                "role": "system",
                "content": "\n\n".join(prompt_overrides),
            },
            *final_context,
        ]

    return {
        "model": model,
        "model_access_layer": "Vertex AI" if model.startswith("vertex-") else "Direct provider client",
        "attack_id": attack_id,
        "base_system_prompt": getattr(agent, "system_prompt", DEFAULT_SYSTEM_PROMPT),
        "user_query": user_query,
        "injected_context": injected_context,
        "defense_steps": defense_steps,
        "final_context_before_user": final_context,
        "tool_overrides": tool_overrides,
    }


def _queue_path() -> str:
    project = _project_id()
    location = os.environ.get("TASKS_LOCATION")
    queue = os.environ.get("TASKS_QUEUE")
    if not location or not queue:
        raise HTTPException(status_code=500, detail="TASKS_LOCATION/TASKS_QUEUE are not configured")
    return tasks_v2.CloudTasksClient.queue_path(project, location, queue)


def _enqueue_start_task(job_id: str, task_url: str, audience: str) -> str:
    service_account_email = os.environ.get("TASKS_CALLER_SERVICE_ACCOUNT")
    if not service_account_email:
        raise HTTPException(status_code=500, detail="TASKS_CALLER_SERVICE_ACCOUNT is not configured")

    client = tasks_v2.CloudTasksClient()
    task = tasks_v2.Task(
        http_request=tasks_v2.HttpRequest(
            http_method=tasks_v2.HttpMethod.POST,
            url=task_url,
            headers={"Content-Type": "application/json"},
            body=job_id.encode("utf-8"),
            oidc_token=tasks_v2.OidcToken(
                service_account_email=service_account_email,
                audience=audience,
            ),
        )
    )
    response = client.create_task(parent=_queue_path(), task=task)
    return response.name


def _worker_job_resource_name() -> str:
    project = _project_id()
    location = os.environ.get("WORKER_JOB_LOCATION")
    job_name = os.environ.get("WORKER_JOB_NAME")
    if not location or not job_name:
        raise HTTPException(status_code=500, detail="WORKER_JOB_LOCATION/WORKER_JOB_NAME are not configured")
    return f"projects/{project}/locations/{location}/jobs/{job_name}"


def _run_worker_job(job_id: str, config: dict[str, Any]) -> str:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = f"https://run.googleapis.com/v2/{_worker_job_resource_name()}:run"
    body = {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {"name": "JOB_ID", "value": job_id},
                        {"name": "JOB_CONFIG_JSON", "value": json_dumps(config)},
                        {"name": "BENCHMARK_MODELS", "value": ",".join(config.get("models", []))},
                        {"name": "BENCHMARK_DEFENSES", "value": ",".join(config.get("defenses", []))},
                    ]
                }
            ]
        }
    }
    response = session.post(url, json=body, timeout=30)
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Cloud Run Jobs API error: {response.text}")
    operation = response.json()
    return operation.get("name", "")


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), default=str)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "redteam-api"}


@app.get("/models")
def models() -> dict[str, Any]:
    client = LLMClient()
    metadata = client.model_metadata()
    return {
        "models": [model["name"] for model in metadata],
        "model_metadata": metadata,
    }


@app.get("/attacks")
def attacks() -> dict[str, list[dict[str, Any]]]:
    return {
        "attacks": [
            {
                "id": attack.id,
                "name": attack.name,
                "category": attack.category.value,
                "severity": attack.severity.value,
                "commodity": attack.commodity,
                "description": attack.description,
            }
            for attack in get_all_attacks()
        ]
    }


@app.get("/defenses")
def defenses() -> dict[str, list[str]]:
    return {"defenses": ["none", *sorted(DEFENSE_MAP), "all_combined"]}


@app.post("/run/single")
def run_single(request: SingleRunRequest) -> dict[str, Any]:
    base_attack = _attack_index().get(request.attack_id)
    if base_attack is None:
        raise HTTPException(status_code=404, detail=f"Unknown attack: {request.attack_id}")
    attack = (
        PromptAugmentedAttack(base_attack, request.additional_prompt)
        if request.additional_prompt and request.additional_prompt.strip()
        else base_attack
    )

    agent = _build_agent(request.model, LLMClient())
    defenses = _build_defenses(request.defenses)
    prepared_payload = attack.prepare(agent)
    prompt_trace = _build_prompt_trace(
        agent=agent,
        payload=prepared_payload,
        defenses=defenses,
        model=request.model,
        attack_id=request.attack_id,
    )
    evaluator = RedTeamEvaluator(
        agent=agent,
        attacks=[attack],
        defenses=defenses,
    )
    result = evaluator.run_single(attack, prepared_payload=prepared_payload)
    return {
        "model": request.model,
        "attack_id": request.attack_id,
        "defenses": request.defenses,
        "additional_prompt_applied": bool(request.additional_prompt and request.additional_prompt.strip()),
        "prompt_trace": prompt_trace,
        "success": result.success,
        "target_action_achieved": result.target_action_achieved,
        "detected_by_defense": result.detected_by_defense,
        "defense_confidence": result.defense_confidence,
        "financial_impact_estimate": result.financial_impact_estimate,
        "agent_output": result.agent_output,
        "notes": result.notes,
    }


@app.post("/jobs/benchmark")
def create_benchmark_job(request: BenchmarkJobRequest, http_request: Request) -> dict[str, Any]:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    config = request.model_dump()

    _jobs_collection().document(job_id).set(
        {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "config": config,
            "current_step": "queued",
            "progress_completed": 0,
            "progress_total": 0,
            "progress_message": "Benchmark request accepted. Waiting for Cloud Tasks to start the worker.",
        }
    )

    host = http_request.headers.get("host")
    base_url = os.environ.get("API_PUBLIC_URL") or (f"https://{host}" if host else str(http_request.base_url).rstrip("/"))
    base_url = base_url.replace("http://", "https://", 1)
    task_name = _enqueue_start_task(
        job_id=job_id,
        task_url=f"{base_url}/internal/jobs/{job_id}/start",
        audience=base_url,
    )
    _jobs_collection().document(job_id).update(
        {
            "task_name": task_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "task_name": task_name,
        "config": config,
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_or_404(job_id)


@app.get("/jobs")
def list_jobs(limit: int = Query(default=25, ge=1, le=100)) -> dict[str, Any]:
    query = _jobs_collection().order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
    jobs = [snapshot.to_dict() for snapshot in query.stream()]
    return {"jobs": jobs}


@app.get("/jobs/{job_id}/artifacts")
def list_job_artifacts(job_id: str) -> dict[str, Any]:
    job = _job_or_404(job_id)
    prefix = _artifact_prefix(job)
    artifacts = []
    for blob in _storage_bucket().list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative_name = blob.name[len(prefix):].strip("/")
        artifacts.append(
            {
                "name": relative_name,
                "size": blob.size,
                "content_type": blob.content_type,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "download_path": f"/jobs/{job_id}/artifacts/{relative_name}",
            }
        )
    return {
        "job_id": job_id,
        "result_uri": job.get("result_uri"),
        "artifacts": sorted(artifacts, key=lambda item: item["name"]),
    }


@app.get("/jobs/{job_id}/results")
def get_job_results(
    job_id: str,
    include_results: bool = Query(default=True),
    result_limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    results_payload = _download_json_artifact(job_id, "results.json")
    summary_payload = _download_json_artifact(job_id, "summary.json")
    if results_payload is None and summary_payload is None:
        raise HTTPException(status_code=404, detail="No parsed result artifacts found")

    results = []
    metadata = {}
    if results_payload:
        metadata = results_payload.get("metadata", {})
        if include_results:
            results = list(results_payload.get("results", []))[:result_limit]

    return {
        "job_id": job_id,
        "metadata": metadata,
        "summary": summary_payload or results_payload.get("summary", {}),
        "results": results,
        "result_count": len(results_payload.get("results", [])) if results_payload else 0,
        "returned_results": len(results),
    }


@app.get("/jobs/{job_id}/artifacts/{artifact_path:path}")
def download_job_artifact(job_id: str, artifact_path: str) -> Response:
    blob = _artifact_blob(job_id, artifact_path)
    content = blob.download_as_bytes()
    media_type = blob.content_type or "application/octet-stream"
    filename = artifact_path.strip("/").split("/")[-1]
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/internal/jobs/{job_id}/start")
def start_job(job_id: str) -> dict[str, Any]:
    doc_ref = _jobs_collection().document(job_id)
    snapshot = doc_ref.get()
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    job = snapshot.to_dict()
    config = job.get("config", {})
    operation_name = _run_worker_job(job_id, config)
    doc_ref.update(
        {
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_operation": operation_name,
            "current_step": "worker_start_requested",
            "progress_message": (
                "Cloud Run Jobs API accepted the worker execution request. "
                "On a new image, Cloud Run can spend a few minutes importing the container before the worker writes progress."
            ),
        }
    )
    return {
        "job_id": job_id,
        "status": "started",
        "worker_job_name": os.environ.get("WORKER_JOB_NAME", ""),
        "run_operation": operation_name,
    }

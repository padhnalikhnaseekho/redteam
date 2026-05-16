"""Cloud Run Job entrypoint for benchmark execution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import csv
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import firestore, storage

from src.agent.system_prompt import DEFAULT_SYSTEM_PROMPT
from src.attacks.registry import get_all_attacks
from src.defenses.guardrails import GuardrailsDefense
from src.defenses.human_in_loop import HumanInLoopDefense
from src.defenses.input_filter import InputFilterDefense
from src.defenses.multi_agent import MultiAgentDefense
from src.defenses.output_validator import OutputValidatorDefense
from src.defenses.perplexity_filter import PerplexityFilterDefense
from src.defenses.semantic_filter import SemanticInputFilterDefense
from src.evaluation.evaluator import RedTeamEvaluator
from src.utils.llm import LLMClient


DEFENSE_MAP = {
    "input_filter": InputFilterDefense,
    "output_validator": OutputValidatorDefense,
    "guardrails": GuardrailsDefense,
    "human_in_loop": HumanInLoopDefense,
    "multi_agent": MultiAgentDefense,
    "semantic_filter": SemanticInputFilterDefense,
    "perplexity_filter": PerplexityFilterDefense,
}


def _project_id() -> str:
    return os.environ.get("GOOGLE_CLOUD_PROJECT", "")


def _job_doc(job_id: str):
    project = _project_id()
    if not project:
        return None
    return firestore.Client(project=project).collection("benchmark_jobs").document(job_id)


def _update_job(job_id: str, fields: dict) -> None:
    doc = _job_doc(job_id)
    if doc is None:
        return
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    doc.set(fields, merge=True)


def _latest_results_dir(results_root: Path, before: set[Path]) -> Path | None:
    if not results_root.exists():
        return None
    candidates = [path for path in results_root.iterdir() if path.is_dir() and path not in before]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _upload_results(job_id: str, results_dir: Path) -> str | None:
    bucket_name = os.environ.get("RESULTS_BUCKET")
    if not bucket_name or not results_dir.exists():
        return None
    bucket = storage.Client(project=_project_id() or None).bucket(bucket_name)
    prefix = f"jobs/{job_id}/{results_dir.name}"
    for path in results_dir.rglob("*"):
        if path.is_file():
            blob = bucket.blob(f"{prefix}/{path.relative_to(results_dir)}")
            blob.upload_from_filename(str(path))
    return f"gs://{bucket_name}/{prefix}"


def _build_agent(model_name: str, client: LLMClient):
    def agent_fn(query: str, context: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": query})
        response = client.chat(model_name, messages)
        return response["content"]

    agent_fn.run = lambda query, context=None: agent_fn(query, context)
    return agent_fn


def _selected_attacks(config: dict) -> list:
    requested_ids = set(config.get("attack_ids") or [])
    attacks = [attack for attack in get_all_attacks() if not requested_ids or attack.id in requested_ids]
    if not attacks:
        raise ValueError(f"No attacks matched ids: {sorted(requested_ids)}")
    return attacks


def _make_defense_stack(names: list[str]) -> list:
    stack = []
    for name in names:
        if name in {"", "none"}:
            continue
        defense_cls = DEFENSE_MAP.get(name)
        if defense_cls is None:
            raise ValueError(f"Unsupported worker defense: {name}")
        stack.append(defense_cls())
    return stack


def _defense_run_configs(config: dict) -> list[tuple[str, list[str]]]:
    requested = [name for name in config.get("defenses", []) if name and name != "none"]
    mode = config.get("mode", "selected")
    if mode == "full_matrix":
        base = [("none", [])]
        selected = requested or ["input_filter", "output_validator", "guardrails", "human_in_loop", "multi_agent"]
        for name in selected:
            if name == "all_combined":
                continue
            base.append((name, [name]))
        if "all_combined" in requested or not requested:
            base.append(("all_combined", ["input_filter", "output_validator", "guardrails", "human_in_loop", "multi_agent"]))
        return base
    if not requested:
        return [("none", [])]
    if "all_combined" in requested:
        return [("all_combined", ["input_filter", "output_validator", "guardrails", "human_in_loop", "multi_agent"])]
    return [("+".join(requested), requested)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})


def _write_summary(results_dir: Path, rows: list[dict]) -> dict:
    groups = defaultdict(lambda: {"total": 0, "success": 0, "detected": 0, "impact": 0.0})
    for row in rows:
        key = (row.get("model", "unknown"), row.get("defense", "none"))
        groups[key]["total"] += 1
        if row.get("success"):
            groups[key]["success"] += 1
            groups[key]["impact"] += float(row.get("financial_impact", 0) or 0)
        if row.get("detected"):
            groups[key]["detected"] += 1

    summary_rows = []
    for (model, defense), stats in sorted(groups.items()):
        total = stats["total"]
        summary_rows.append(
            {
                "model": model,
                "defense": defense,
                "total_attacks": total,
                "successful": stats["success"],
                "asr_pct": round(100 * stats["success"] / total, 1) if total else 0,
                "detected": stats["detected"],
                "detection_rate_pct": round(100 * stats["detected"] / total, 1) if total else 0,
                "total_impact_usd": int(stats["impact"]),
            }
        )
    _write_csv(results_dir / "summary.csv", summary_rows)
    summary = {
        "total_rows": len(rows),
        "groups": summary_rows,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def _write_report_readme(results_dir: Path, metadata: dict, summary: dict) -> None:
    report_dir = results_dir / "report"
    report_dir.mkdir(exist_ok=True)
    lines = [
        "# Cloud Benchmark Report",
        "",
        f"- Job ID: `{metadata['job_id']}`",
        f"- Mode: `{metadata['mode']}`",
        f"- Models: `{', '.join(metadata['models'])}`",
        f"- Defense configs: `{', '.join(metadata['defense_configs'])}`",
        f"- Attacks: `{len(metadata['attack_ids'])}`",
        f"- Total rows: `{summary['total_rows']}`",
        "",
        "## Summary",
        "",
        "| Model | Defense | Attacks | Successful | ASR % | Detected | Detection % | Impact USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| {row['model']} | {row['defense']} | {row['total_attacks']} | "
            f"{row['successful']} | {row['asr_pct']} | {row['detected']} | "
            f"{row['detection_rate_pct']} | {row['total_impact_usd']} |"
        )
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def _run_cloud_benchmark(job_id: str, config: dict, project_dir: Path) -> Path:
    results_dir = project_dir / "results" / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    attacks = _selected_attacks(config)
    models = config.get("models") or ["vertex-gemini-flash"]
    run_configs = _defense_run_configs(config)
    total_units = len(models) * len(run_configs) * len(attacks)
    completed_units = 0
    _update_job(
        job_id,
        {
            "status": "running",
            "progress_total": total_units,
            "progress_completed": 0,
            "progress_message": "Worker loaded benchmark configuration and is preparing the first model.",
            "current_step": "initializing",
        },
    )

    client = LLMClient()
    rows = []
    delay_seconds = int(config.get("delay_seconds", 0) or 0)
    for model_name in models:
        agent = _build_agent(model_name, client)
        for index, (defense_label, defense_names) in enumerate(run_configs):
            defenses = _make_defense_stack(defense_names)
            for attack in attacks:
                _update_job(
                    job_id,
                    {
                        "status": "running",
                        "current_step": "evaluating_attack",
                        "current_model": model_name,
                        "current_defense": defense_label,
                        "current_attack_id": attack.id,
                        "progress_completed": completed_units,
                        "progress_total": total_units,
                        "progress_message": (
                            f"Calling {model_name} with attack {attack.id} "
                            f"under defense config {defense_label}."
                        ),
                    },
                )
                evaluator = RedTeamEvaluator(
                    agent=agent,
                    attacks=[attack],
                    defenses=defenses,
                )
                result = evaluator.run_single(attack)
                row = {
                    "attack_id": attack.id,
                    "category": attack.category.value,
                    "severity": attack.severity.value,
                    "model": model_name,
                    "defense": defense_label,
                    "success": result.success,
                    "target_action_achieved": result.target_action_achieved,
                    "detected": result.detected_by_defense,
                    "defense_confidence": result.defense_confidence,
                    "financial_impact": result.financial_impact_estimate,
                    "notes": result.notes,
                    "job_id": job_id,
                    "run_mode": config.get("mode", "selected"),
                }
                rows.append(row)
                completed_units += 1
                _update_job(
                    job_id,
                    {
                        "status": "running",
                        "current_step": "attack_complete",
                        "current_model": model_name,
                        "current_defense": defense_label,
                        "current_attack_id": attack.id,
                        "progress_completed": completed_units,
                        "progress_total": total_units,
                        "progress_message": (
                            f"Completed {completed_units} of {total_units} evaluations. "
                            f"Latest attack success={result.success}, detected={result.detected_by_defense}."
                        ),
                    },
                )
            if delay_seconds and index < len(run_configs) - 1:
                _update_job(
                    job_id,
                    {
                        "status": "running",
                        "current_step": "defense_delay",
                        "current_model": model_name,
                        "current_defense": defense_label,
                        "progress_completed": completed_units,
                        "progress_total": total_units,
                        "progress_message": f"Waiting {delay_seconds}s before the next defense configuration.",
                    },
                )
                time.sleep(delay_seconds)

    json_path = results_dir / "results.json"
    csv_path = results_dir / "results.csv"
    metadata = {
        "job_id": job_id,
        "mode": config.get("mode", "selected"),
        "models": models,
        "defense_configs": [label for label, _ in run_configs],
        "attack_ids": [attack.id for attack in attacks],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_cost": client.total_cost,
        "total_tokens": client.total_tokens,
        "image_tag": os.environ.get("K_REVISION", ""),
        "google_cloud_project": _project_id(),
        "vertex_location": os.environ.get("GOOGLE_CLOUD_LOCATION", ""),
    }
    summary = _write_summary(results_dir, rows)
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "summary": summary, "results": rows}, f, indent=2, default=str)
    _write_csv(csv_path, rows)
    if config.get("include_reports", True):
        _write_report_readme(results_dir, metadata, summary)
    return results_dir


def main() -> int:
    job_id = os.environ.get("JOB_ID", "local-job")
    config = json.loads(os.environ.get("JOB_CONFIG_JSON", "{}") or "{}")
    models = os.environ.get("BENCHMARK_MODELS", "vertex-gemini-flash").split(",")
    skip_multi_agent = os.environ.get("SKIP_MULTI_AGENT", "true").lower() == "true"
    project_dir = Path(__file__).resolve().parents[1]
    results_root = project_dir / "results"
    before_dirs = set(results_root.iterdir()) if results_root.exists() else set()

    command = [
        sys.executable,
        "scripts/run_full_benchmark.py",
        "--models",
        *[model.strip() for model in models if model.strip()],
    ]
    if skip_multi_agent:
        command.append("--skip-multi-agent")

    print(json.dumps({"event": "worker_start", "job_id": job_id, "command": command}))
    _update_job(job_id, {"status": "running", "worker_started_at": datetime.now(timezone.utc).isoformat()})

    try:
        if config.get("attack_ids") or config.get("mode") in {"selected", "full_matrix"}:
            latest_results = _run_cloud_benchmark(job_id, config, project_dir)
            exit_code = 0
        else:
            result = subprocess.run(command, cwd=project_dir, check=False)
            latest_results = _latest_results_dir(results_root, before_dirs)
            exit_code = result.returncode
    except Exception as exc:
        _update_job(job_id, {"status": "failed", "error": str(exc)})
        print(json.dumps({"event": "worker_error", "job_id": job_id, "error": str(exc)}))
        return 1

    result_uri = _upload_results(job_id, latest_results) if latest_results else None
    status = "completed" if exit_code == 0 else "failed"
    update = {
        "status": status,
        "worker_completed_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "current_step": "completed" if status == "completed" else "failed",
        "progress_message": "Benchmark finished and artifacts were uploaded." if status == "completed" else "Benchmark failed before completion.",
    }
    if result_uri:
        update["result_uri"] = result_uri
    _update_job(job_id, update)
    print(
        json.dumps(
            {
                "event": "worker_complete",
                "job_id": job_id,
                "exit_code": exit_code,
                "result_uri": result_uri,
            }
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publish historical local result artifacts as dashboard-visible jobs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANONICAL_COLUMNS = {
    "attack_id",
    "category",
    "severity",
    "model",
    "defense",
    "success",
    "target_action_achieved",
    "detected",
    "financial_impact",
    "notes",
}
SUMMARY_COLUMNS = {
    "model",
    "defense",
    "total_attacks",
    "successful",
    "asr_pct",
    "detected",
    "detection_rate_pct",
    "total_impact_usd",
}
PUBLISHER_VERSION = "historical-results-v1"


@dataclass(frozen=True)
class ResultGroup:
    slug: str
    source_path: str
    root: Path
    files: tuple[Path, ...]


@dataclass(frozen=True)
class PublishPlan:
    group: ResultGroup
    job_id: str
    gcs_prefix: str
    metadata: dict[str, Any]
    results_payload: dict[str, Any]
    summary_payload: dict[str, Any]
    manifest: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:64] or "historical-results"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stable_content_hash(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def discover_groups(results_dir: Path) -> list[ResultGroup]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    groups: list[ResultGroup] = []
    for child in sorted(results_dir.iterdir(), key=lambda path: path.name):
        if child.is_dir():
            files = tuple(sorted((p for p in child.rglob("*") if p.is_file()), key=lambda p: p.as_posix()))
            if files:
                groups.append(
                    ResultGroup(
                        slug=slugify(child.name),
                        source_path=str(child.relative_to(results_dir.parent)),
                        root=child,
                        files=files,
                    )
                )

    top_level: dict[str, list[Path]] = {}
    for path in sorted(results_dir.iterdir(), key=lambda item: item.name):
        if path.is_file():
            top_level.setdefault(path.stem, []).append(path)

    for stem, files in sorted(top_level.items()):
        groups.append(
            ResultGroup(
                slug=slugify(stem),
                source_path=f"{results_dir.name}/{stem}.*",
                root=results_dir,
                files=tuple(sorted(files, key=lambda p: p.as_posix())),
            )
        )

    return groups


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", errors="replace") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def normalize_benchmark_row(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    if not ({"attack_id", "success"} <= set(row.keys())):
        return None
    return {
        "attack_id": str(row.get("attack_id") or row.get("query_id") or "unknown"),
        "category": str(row.get("category") or "historical"),
        "severity": str(row.get("severity") or "unknown"),
        "model": str(row.get("model") or "historical"),
        "defense": str(row.get("defense") or "historical"),
        "success": parse_bool(row.get("success")),
        "target_action_achieved": parse_bool(row.get("target_action_achieved", row.get("success"))),
        "detected": parse_bool(row.get("detected", row.get("blocked", False))),
        "defense_confidence": parse_float(row.get("defense_confidence")),
        "financial_impact": parse_float(row.get("financial_impact", row.get("financial_impact_estimate"))),
        "notes": str(row.get("notes") or row.get("reasoning") or row.get("error") or f"Imported from {source}"),
        "source_artifact": source,
    }


def normalize_adaptive_row(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    if not ({"plan_id", "success"} <= set(row.keys())):
        return None
    category = row.get("category") or row.get("goal") or "adaptive_redteam"
    severity_value = parse_float(row.get("severity"))
    if severity_value >= 0.75:
        severity = "critical"
    elif severity_value >= 0.5:
        severity = "high"
    elif severity_value > 0:
        severity = "medium"
    else:
        severity = "unknown"
    return {
        "attack_id": str(row.get("plan_id") or f"adaptive-round-{row.get('round', 'unknown')}"),
        "category": str(category),
        "severity": severity,
        "model": str(row.get("model") or "adaptive-redteam"),
        "defense": str(row.get("blocked_by") or "historical"),
        "success": parse_bool(row.get("success")),
        "target_action_achieved": parse_bool(row.get("success")),
        "detected": parse_bool(row.get("blocked")) or bool(row.get("detected_by")),
        "defense_confidence": parse_float(row.get("confidence")),
        "financial_impact": parse_float(row.get("financial_impact")),
        "notes": str(row.get("reasoning") or row.get("goal") or f"Imported adaptive row from {source}"),
        "source_artifact": source,
    }


def rows_from_json(path: Path, relative_name: str) -> list[dict[str, Any]]:
    payload = load_json(path)
    if payload is None:
        return []

    candidates: list[Any] = []
    if isinstance(payload, dict):
        for key in ("results", "rows", "all_results"):
            if isinstance(payload.get(key), list):
                candidates.extend(payload[key])
    elif isinstance(payload, list):
        candidates.extend(payload)

    rows: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        normalized = normalize_benchmark_row(item, relative_name) or normalize_adaptive_row(item, relative_name)
        if normalized:
            rows.append(normalized)
    return rows


def rows_from_csv(path: Path, relative_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_csv_rows(path):
        normalized = normalize_benchmark_row(row, relative_name) or normalize_adaptive_row(row, relative_name)
        if normalized:
            rows.append(normalized)
    return rows


def normalize_results(group: ResultGroup) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in group.files:
        relative_name = str(path.relative_to(group.root))
        if path.suffix.lower() == ".json":
            source_rows = rows_from_json(path, relative_name)
        elif path.suffix.lower() == ".csv":
            source_rows = rows_from_csv(path, relative_name)
        else:
            source_rows = []
        for row in source_rows:
            key = json.dumps(row, sort_keys=True)
            if key not in seen:
                seen.add(key)
                normalized.append(row)
    return normalized


def load_existing_summary(group: ResultGroup) -> dict[str, Any] | None:
    for path in group.files:
        if path.name != "summary.csv":
            continue
        rows = load_csv_rows(path)
        if rows and SUMMARY_COLUMNS <= set(rows[0].keys()):
            return {"groups": [coerce_summary_row(row) for row in rows]}

    for path in group.files:
        if path.name != "summary.json":
            continue
        payload = load_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("groups"), list):
            return payload
    return None


def coerce_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": str(row.get("model", "historical")),
        "defense": str(row.get("defense", "historical")),
        "total_attacks": int(parse_float(row.get("total_attacks"))),
        "successful": int(parse_float(row.get("successful"))),
        "asr_pct": parse_float(row.get("asr_pct")),
        "detected": int(parse_float(row.get("detected"))),
        "detection_rate_pct": parse_float(row.get("detection_rate_pct")),
        "total_impact_usd": parse_float(row.get("total_impact_usd")),
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("model") or "historical"), str(row.get("defense") or "historical"))
        item = groups.setdefault(
            key,
            {
                "model": key[0],
                "defense": key[1],
                "total_attacks": 0,
                "successful": 0,
                "detected": 0,
                "total_impact_usd": 0.0,
            },
        )
        item["total_attacks"] += 1
        item["successful"] += 1 if parse_bool(row.get("success")) else 0
        item["detected"] += 1 if parse_bool(row.get("detected")) else 0
        item["total_impact_usd"] += parse_float(row.get("financial_impact"))

    summary_rows = []
    for item in groups.values():
        total = item["total_attacks"] or 1
        summary_rows.append(
            {
                **item,
                "asr_pct": round(item["successful"] / total * 100, 2),
                "detection_rate_pct": round(item["detected"] / total * 100, 2),
            }
        )
    return {"groups": sorted(summary_rows, key=lambda row: (row["model"], row["defense"]))}


def describe_group(group: ResultGroup, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    suffix_counts: dict[str, int] = {}
    for path in group.files:
        suffix = path.suffix.lower().lstrip(".") or "file"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    models = sorted({str(row.get("model")) for row in rows if row.get("model")})
    attacks = sorted({str(row.get("attack_id")) for row in rows if row.get("attack_id")})
    impact = sum(parse_float(row.get("financial_impact")) for row in rows)
    successful = sum(1 for row in rows if parse_bool(row.get("success")))
    return {
        "source_path": group.source_path,
        "source_type": "historical_results",
        "artifact_count": len(group.files),
        "artifact_types": suffix_counts,
        "normalized_result_count": len(rows),
        "summary_group_count": len(summary.get("groups", [])),
        "models": models,
        "attack_ids": attacks,
        "successful_attacks": successful,
        "simulated_impact_usd": impact,
        "reader_note": (
            "Historical result rows were normalized for the dashboard."
            if rows
            else "No canonical result rows were found; raw historical artifacts are available as downloads."
        ),
    }


def build_plan(
    group: ResultGroup,
    *,
    results_dir: Path,
    bucket_name: str,
    project_id: str | None,
    commit_sha: str,
    branch_name: str,
) -> PublishPlan:
    content_hash = stable_content_hash(list(group.files), group.root)
    job_id = f"historical-{group.slug}-{content_hash[:8]}"
    gcs_prefix = f"jobs/{job_id}/{group.slug}"
    rows = normalize_results(group)
    summary = load_existing_summary(group) or summarize_results(rows)
    import_summary = describe_group(group, rows, summary)
    imported_at = utc_now_iso()
    metadata = {
        "publisher": PUBLISHER_VERSION,
        "historical_import": True,
        "source_path": group.source_path,
        "source_commit": commit_sha,
        "source_branch": branch_name,
        "content_hash": content_hash,
        "imported_at": imported_at,
        "models": import_summary["models"],
        "attack_ids": import_summary["attack_ids"],
        "total_tokens": {"input": 0, "output": 0},
        "import_summary": import_summary,
    }
    manifest = {
        "publisher": PUBLISHER_VERSION,
        "job_id": job_id,
        "source_path": group.source_path,
        "source_commit": commit_sha,
        "source_branch": branch_name,
        "content_hash": content_hash,
        "gcs_prefix": gcs_prefix,
        "artifact_count": len(group.files),
        "artifacts": [
            {
                "source": str(path.relative_to(results_dir.parent)),
                "published_as": f"raw/{path.relative_to(group.root)}",
                "size": path.stat().st_size,
            }
            for path in group.files
        ],
    }
    results_payload = {
        "metadata": metadata,
        "summary": summary,
        "results": rows,
    }
    summary_payload = summary
    return PublishPlan(
        group=group,
        job_id=job_id,
        gcs_prefix=gcs_prefix,
        metadata=metadata,
        results_payload=results_payload,
        summary_payload=summary_payload,
        manifest=manifest,
    )


def write_payloads(plan: PublishPlan, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "results.json": plan.results_payload,
        "summary.json": plan.summary_payload,
        "publish_manifest.json": plan.manifest,
    }
    paths: list[Path] = []
    for name, payload in payloads.items():
        path = output_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str))
        paths.append(path)
    return paths


def firestore_document(plan: PublishPlan, *, bucket_name: str, project_id: str | None) -> dict[str, Any]:
    now = utc_now_iso()
    return {
        "job_id": plan.job_id,
        "status": "completed",
        "created_at": plan.metadata["imported_at"],
        "updated_at": now,
        "completed_at": now,
        "result_uri": f"gs://{bucket_name}/{plan.gcs_prefix}",
        "config": {
            "mode": "historical_import",
            "source_path": plan.group.source_path,
            "source_commit": plan.metadata["source_commit"],
            "source_branch": plan.metadata["source_branch"],
            "include_reports": True,
        },
        "progress_message": "Historical result artifacts published to dashboard.",
        "project_id": project_id,
        "source_path": plan.group.source_path,
        "source_commit": plan.metadata["source_commit"],
        "source_branch": plan.metadata["source_branch"],
        "content_hash": plan.metadata["content_hash"],
        "artifact_count": plan.metadata["import_summary"]["artifact_count"],
        "result_count": plan.metadata["import_summary"]["normalized_result_count"],
        "historical_import": True,
    }


def upload_plan(plan: PublishPlan, *, bucket_name: str, project_id: str | None, dry_run: bool) -> None:
    doc = firestore_document(plan, bucket_name=bucket_name, project_id=project_id)
    print(
        f"{'[dry-run] ' if dry_run else ''}{plan.job_id}: "
        f"{plan.group.source_path} -> gs://{bucket_name}/{plan.gcs_prefix} "
        f"({doc['result_count']} rows, {doc['artifact_count']} artifacts)"
    )
    if dry_run:
        return

    from google.cloud import firestore, storage

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    with tempfile.TemporaryDirectory(prefix="historical-results-") as temp_root:
        payload_dir = Path(temp_root)
        for payload_path in write_payloads(plan, payload_dir):
            blob = bucket.blob(f"{plan.gcs_prefix}/{payload_path.name}")
            blob.upload_from_filename(str(payload_path), content_type="application/json")

    for source_path in plan.group.files:
        relative_name = source_path.relative_to(plan.group.root)
        content_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
        blob = bucket.blob(f"{plan.gcs_prefix}/raw/{relative_name}")
        blob.upload_from_filename(str(source_path), content_type=content_type)

    firestore.Client(project=project_id).collection("benchmark_jobs").document(plan.job_id).set(doc, merge=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--bucket", default=os.environ.get("RESULTS_BUCKET") or os.environ.get("_RESULTS_BUCKET"))
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT_ID"))
    parser.add_argument("--commit-sha", default=os.environ.get("COMMIT_SHA", "local"))
    parser.add_argument("--branch-name", default=os.environ.get("BRANCH_NAME", "local"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-groups", type=int, default=0, help="Limit groups for smoke testing; 0 means all groups.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    if not args.bucket:
        raise SystemExit("--bucket or RESULTS_BUCKET is required")

    groups = discover_groups(results_dir)
    if args.max_groups:
        groups = groups[: args.max_groups]
    if not groups:
        print(f"No result groups found under {results_dir}")
        return 0

    print(f"Publishing {len(groups)} historical result groups from {results_dir}")
    for group in groups:
        plan = build_plan(
            group,
            results_dir=results_dir,
            bucket_name=args.bucket,
            project_id=args.project_id,
            commit_sha=args.commit_sha,
            branch_name=args.branch_name,
        )
        upload_plan(plan, bucket_name=args.bucket, project_id=args.project_id, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

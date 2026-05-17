"""Backfill a historical local results directory into the deployed Cloud Run product.

The deployed UI lists jobs from Firestore (`benchmark_jobs/{id}`) and reads their
artifacts from `gs://<bucket>/jobs/{id}/...`. Historical runs in the local
`results/<run_dir>/` folders don't appear in the UI because they were never
written through the worker. This script backfills one such run by:

  1. Reading `<local_dir>/all_results_combined.csv` (the canonical 700-row file).
  2. Normalising it into the cloud-shape `results.json` and `summary.json`
     keyed by (model, defense), matching what api/main.py reads.
  3. Uploading those two files PLUS every file in the source directory (CSVs,
     per-config JSONs, RESULTS.md, report/*.pptx) to
     `gs://<bucket>/jobs/<job_id>/<run_dir>/`.
  4. Writing a Firestore doc at `benchmark_jobs/<job_id>` with status=completed
     and result_uri pointing at the GCS prefix above, so the UI lists it.

Auth: uses Application Default Credentials. Run
  gcloud auth application-default login
once before invoking this script. No service-account key file required.

Usage:
  python scripts/backfill_jobs_to_gcs.py \\
      --local-dir results/results_0329_1945 \\
      --bucket project-e0bbb103-9e5b-4402-866-redteam-demo-results \\
      --project project-e0bbb103-9e5b-4402-866 \\
      --dry-run

Strip --dry-run to actually upload + write Firestore.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV → cloud schema
# ---------------------------------------------------------------------------

def _coerce_bool(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def _coerce_float(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_rows(csv_path: Path) -> list[dict]:
    """Read all_results_combined.csv and normalise into cloud row shape.

    Adds the `detected_by_defense` and `financial_impact_estimate` aliases the
    UI's Results Explorer reads.
    """
    with open(csv_path, newline="") as f:
        raw = list(csv.DictReader(f))

    rows: list[dict] = []
    for r in raw:
        success = _coerce_bool(r.get("success", "false"))
        detected = _coerce_bool(r.get("detected", "false"))
        impact = _coerce_float(r.get("financial_impact", "0"))
        rows.append({
            "attack_id": r.get("attack_id", ""),
            "category": r.get("category", ""),
            "severity": r.get("severity", ""),
            "model": r.get("model", "unknown"),
            "defense": r.get("defense", "none"),
            "success": success,
            "target_action_achieved": _coerce_bool(r.get("target_action_achieved", str(success))),
            "detected": detected,
            "detected_by_defense": detected,
            "financial_impact": impact,
            "financial_impact_estimate": impact,
            "notes": r.get("notes", ""),
        })
    return rows


def build_summary(rows: list[dict]) -> dict:
    """Group by (model, defense) → metrics, plus an aggregate row."""
    groups: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "total_attacks": 0, "successful": 0, "detected": 0, "total_impact_usd": 0.0,
    })
    for r in rows:
        key = (r["model"], r["defense"])
        g = groups[key]
        g["total_attacks"] += 1
        if r["success"]:
            g["successful"] += 1
            g["total_impact_usd"] += r["financial_impact"]
        if r["detected"]:
            g["detected"] += 1

    out_groups = []
    for (model, defense), g in sorted(groups.items()):
        total = g["total_attacks"] or 1
        out_groups.append({
            "model": model,
            "defense": defense,
            "total_attacks": g["total_attacks"],
            "successful": g["successful"],
            "asr_pct": round(100.0 * g["successful"] / total, 2),
            "detected": g["detected"],
            "detection_rate_pct": round(100.0 * g["detected"] / total, 2),
            "total_impact_usd": round(g["total_impact_usd"], 2),
        })

    total = len(rows) or 1
    successful = sum(1 for r in rows if r["success"])
    detected = sum(1 for r in rows if r["detected"])
    impact = sum(r["financial_impact"] for r in rows if r["success"])
    aggregate = {
        "total_attacks": len(rows),
        "successful": successful,
        "asr_pct": round(100.0 * successful / total, 2),
        "detected": detected,
        "detection_rate_pct": round(100.0 * detected / total, 2),
        "total_impact_usd": round(impact, 2),
    }
    return {"groups": out_groups, "aggregate": aggregate}


def build_metadata(rows: list[dict], local_dir: Path, source_tag: str) -> dict:
    models = sorted({r["model"] for r in rows})
    defenses = sorted({r["defense"] for r in rows})
    attacks = sorted({r["attack_id"] for r in rows})
    mtime = datetime.fromtimestamp(local_dir.stat().st_mtime, tz=timezone.utc)
    return {
        "source": source_tag,
        "source_dir": local_dir.name,
        "models": models,
        "defenses": defenses,
        "attack_count": len(attacks),
        "row_count": len(rows),
        "created_at": mtime.isoformat(),
        "backfilled_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Cloud uploads
# ---------------------------------------------------------------------------

def upload_dir(bucket, src: Path, dst_prefix: str, dry_run: bool) -> int:
    """Upload every file under `src` to `gs://<bucket>/<dst_prefix>/...`. Returns count."""
    n = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(src).as_posix()
        blob_name = f"{dst_prefix}/{relative}"
        if dry_run:
            logger.info("DRY  would upload %s → gs://%s/%s (%d bytes)", relative, bucket.name, blob_name, path.stat().st_size)
        else:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(path))
            logger.info("PUT  gs://%s/%s", bucket.name, blob_name)
        n += 1
    return n


def upload_text(bucket, blob_name: str, text: str, dry_run: bool, content_type: str = "application/json") -> None:
    if dry_run:
        logger.info("DRY  would upload gs://%s/%s (%d chars)", bucket.name, blob_name, len(text))
        return
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text, content_type=content_type)
    logger.info("PUT  gs://%s/%s", bucket.name, blob_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local-dir", required=True, help="Path to results/<run_dir>")
    parser.add_argument("--bucket", required=True, help="GCS bucket (no gs:// prefix)")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--job-id", default=None, help="Override job_id (default: histjob-<run_dir>-<utc>)")
    parser.add_argument("--csv-name", default="all_results_combined.csv", help="Canonical CSV inside local-dir")
    parser.add_argument("--source-tag", default="backfilled-historical", help="metadata.source value")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.is_dir():
        logger.error("Not a directory: %s", local_dir)
        return 2
    csv_path = local_dir / args.csv_name
    if not csv_path.exists():
        logger.error("Canonical CSV not found: %s", csv_path)
        return 2

    rows = load_rows(csv_path)
    summary = build_summary(rows)
    metadata = build_metadata(rows, local_dir, args.source_tag)

    job_id = args.job_id or f"histjob-{local_dir.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    job_id = job_id.replace("_", "-")
    dst_prefix = f"jobs/{job_id}/{local_dir.name}"
    result_uri = f"gs://{args.bucket}/{dst_prefix}"

    logger.info("Backfill plan:")
    logger.info("  local_dir   = %s", local_dir)
    logger.info("  rows        = %d (models=%s, defenses=%s)", len(rows), metadata["models"], metadata["defenses"])
    logger.info("  job_id      = %s", job_id)
    logger.info("  result_uri  = %s", result_uri)
    logger.info("  summary.aggregate = %s", summary["aggregate"])

    results_json = {"metadata": metadata, "summary": summary, "results": rows}

    # Lazy-import cloud libs so --help works without them installed.
    if not args.dry_run:
        from google.cloud import storage, firestore  # type: ignore[import-not-found]
        bucket = storage.Client(project=args.project).bucket(args.bucket)
    else:
        # In dry-run we still want the upload_dir helper to work without cloud libs.
        class _FakeBucket:
            name = args.bucket
            def blob(self, name): raise RuntimeError("dry-run")
        bucket = _FakeBucket()

    # 1. Upload normalised cloud-shape artifacts at the root of the result prefix.
    upload_text(bucket, f"{dst_prefix}/results.json", json.dumps(results_json, default=str), args.dry_run)
    upload_text(bucket, f"{dst_prefix}/summary.json", json.dumps(summary, default=str), args.dry_run)

    # 2. Upload every original file in the source dir so they're downloadable
    #    from the UI's artifacts list (CSVs, per-config JSONs, RESULTS.md, PPTX).
    uploaded = upload_dir(bucket, local_dir, dst_prefix, args.dry_run)
    logger.info("Uploaded %d original files", uploaded)

    # 3. Firestore doc so the job appears in the UI list.
    now = datetime.now(timezone.utc).isoformat()
    job_doc = {
        "job_id": job_id,
        "status": "completed",
        "created_at": metadata["created_at"],
        "started_at": metadata["created_at"],
        "completed_at": metadata["backfilled_at"],
        "updated_at": now,
        "current_step": "completed",
        "progress_completed": len(rows),
        "progress_total": len(rows),
        "progress_message": f"Backfilled from local {local_dir.name}",
        "config": {
            "models": metadata["models"],
            "defenses": metadata["defenses"],
            "attack_ids": None,
            "mode": "historical_backfill",
            "include_reports": True,
            "delay_seconds": 0,
        },
        "result_uri": result_uri,
        "summary": summary["aggregate"],
        "source": args.source_tag,
    }
    if args.dry_run:
        logger.info("DRY  would write Firestore benchmark_jobs/%s:", job_id)
        logger.info("     %s", json.dumps(job_doc, default=str)[:500] + "...")
    else:
        fs = firestore.Client(project=args.project)
        fs.collection("benchmark_jobs").document(job_id).set(job_doc)
        logger.info("WROTE Firestore benchmark_jobs/%s", job_id)

    logger.info("Done. Job will appear in the UI as %s.", job_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end post-processing for a completed Cloud Run benchmark job.

A completed worker job leaves results in GCS:
  gs://<bucket>/jobs/<job_id>/<results_dir>/results.json
  gs://<bucket>/jobs/<job_id>/<results_dir>/results.csv
  gs://<bucket>/jobs/<job_id>/<results_dir>/summary.json
  gs://<bucket>/jobs/<job_id>/<results_dir>/summary.csv

But it does NOT generate the Module-2/4 depth artifacts (heatmaps, SHAP,
Bayesian, transferability, ROC, Shapley) — those come from the local
analysis pipeline. This script does the full post-processing in one shot:

  1. Discover the GCS prefix for the job's result_uri (read from Firestore).
  2. Download results.csv → results/<local-name>/all_results_combined.csv,
     plus per-config splits so run_advanced_analysis.py picks them up.
  3. Run scripts/generate_report.py → heatmaps, radar, barcharts.
  4. Run scripts/run_advanced_analysis.py → Bayesian, SHAP, transferability,
     ROC, Shapley, MI, entropy.
  5. Backfill the post-processed bundle into a NEW Firestore job so the
     deployed UI's Explainability tab can read all artifacts.

Replaces Phase C + D + E of docs/full_benchmark_playbook.md.

Auth: uses Application Default Credentials (workload-identity-friendly).
Run `gcloud auth application-default login` once before invoking.

Usage:
  python scripts/postprocess_cloud_job.py \\
      --job-id <headline_job_id> \\
      --bucket project-e0bbb103-9e5b-4402-866-redteam-demo-results \\
      --project project-e0bbb103-9e5b-4402-866 \\
      --local-name cloud_headline

  # Skip the backfill if you only want the local artifacts:
  python scripts/postprocess_cloud_job.py ... --no-backfill

  # Dry-run prints the plan without touching GCS or running scripts:
  python scripts/postprocess_cloud_job.py ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _split_combined_csv(combined_csv: Path) -> int:
    """Split all_results_combined.csv into per-defense CSVs.

    run_advanced_analysis.py's `load_all_results()` looks for per-config
    CSVs (excludes `summary.csv` and `all_results_combined.csv`). The
    cloud worker only writes the combined file. We split it so the
    analysis pipeline can ingest it without modification.
    """
    import pandas as pd
    df = pd.read_csv(combined_csv)
    count = 0
    for (model, defense), group in df.groupby(["model", "defense"]):
        safe_defense = str(defense).replace("+", "_plus_").replace("/", "_")[:60]
        safe_model = str(model).replace("/", "_")[:40]
        out = combined_csv.parent / f"{safe_model}_{safe_defense}.csv"
        group.to_csv(out, index=False)
        count += 1
    return count


def _run_script(args: list[str], dry_run: bool) -> int:
    if dry_run:
        logger.info("DRY  would run: %s", " ".join(args))
        return 0
    logger.info("RUN  %s", " ".join(args))
    return subprocess.run(args, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-id", required=True, help="Firestore job_id from a completed cloud benchmark")
    parser.add_argument("--bucket", required=True, help="GCS results bucket (no gs:// prefix)")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument(
        "--local-name",
        default="cloud_headline",
        help="Local dir name under results/ to hold the downloaded data (default: cloud_headline)",
    )
    parser.add_argument(
        "--backfill-job-id",
        default=None,
        help=(
            "Job ID for the backfilled post-processed bundle (default: "
            "histjob-<local-name>-analysis). Pass --no-backfill to skip."
        ),
    )
    parser.add_argument("--no-backfill", action="store_true", help="Skip the final UI-backfill step")
    parser.add_argument("--no-report", action="store_true", help="Skip generate_report.py (heatmaps etc.)")
    parser.add_argument("--no-analysis", action="store_true", help="Skip run_advanced_analysis.py")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without doing anything")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    local_dir = project_root / "results" / args.local_name
    backfill_id = args.backfill_job_id or f"histjob-{args.local_name}-analysis"

    logger.info("Post-processing cloud job %s", args.job_id)
    logger.info("  GCS bucket    : %s", args.bucket)
    logger.info("  Project       : %s", args.project)
    logger.info("  Local dir     : %s", local_dir)
    if not args.no_backfill:
        logger.info("  Backfill into : benchmark_jobs/%s", backfill_id)

    # ----- Step 1: read Firestore for the job's result_uri -----
    if args.dry_run:
        logger.info("DRY  would read Firestore benchmark_jobs/%s for result_uri", args.job_id)
        result_uri = f"gs://{args.bucket}/jobs/{args.job_id}/<results_dir>"
    else:
        from google.cloud import firestore  # type: ignore[import-not-found]
        fs = firestore.Client(project=args.project)
        doc = fs.collection("benchmark_jobs").document(args.job_id).get()
        if not doc.exists:
            logger.error("Firestore document benchmark_jobs/%s not found", args.job_id)
            return 2
        job_doc = doc.to_dict()
        result_uri = job_doc.get("result_uri")
        if not result_uri:
            logger.error("Job %s has no result_uri (status=%s)", args.job_id, job_doc.get("status"))
            return 2
    logger.info("  result_uri    : %s", result_uri)

    # ----- Step 2: download results.csv + summary.csv -----
    local_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        logger.info("DRY  would gsutil cp %s/results.csv → %s/all_results_combined.csv", result_uri, local_dir)
        logger.info("DRY  would gsutil cp %s/summary.csv → %s/summary.csv", result_uri, local_dir)
    else:
        from google.cloud import storage  # type: ignore[import-not-found]
        # Parse gs://bucket/path
        without_scheme = result_uri[len("gs://"):]
        bucket_name, _, prefix = without_scheme.partition("/")
        client = storage.Client(project=args.project)
        bucket = client.bucket(bucket_name)

        for src_name, dst_name in [("results.csv", "all_results_combined.csv"),
                                    ("summary.csv", "summary.csv"),
                                    ("results.json", "results.json"),
                                    ("summary.json", "summary.json")]:
            blob = bucket.blob(f"{prefix.rstrip('/')}/{src_name}")
            if not blob.exists():
                logger.warning("  skip (not in GCS): %s/%s", prefix, src_name)
                continue
            dst = local_dir / dst_name
            blob.download_to_filename(str(dst))
            logger.info("  downloaded: %s (%d bytes)", dst.name, dst.stat().st_size)

        combined = local_dir / "all_results_combined.csv"
        if combined.exists():
            n_splits = _split_combined_csv(combined)
            logger.info("  split combined CSV into %d per-(model, defense) files", n_splits)

    # ----- Step 3: generate_report.py (heatmaps, radar, barcharts) -----
    if not args.no_report:
        rc = _run_script([
            "python", str(project_root / "scripts" / "generate_report.py"),
            "--results-dir", str(local_dir),
            "--output-dir", str(local_dir / "report"),
        ], args.dry_run)
        if rc != 0:
            logger.warning("generate_report.py exited %d (continuing — non-fatal)", rc)

    # ----- Step 4: run_advanced_analysis.py (Bayesian, SHAP, transferability) -----
    if not args.no_analysis:
        rc = _run_script([
            "python", str(project_root / "scripts" / "run_advanced_analysis.py"),
            "--results-dir", str(local_dir),
        ], args.dry_run)
        if rc != 0:
            logger.warning("run_advanced_analysis.py exited %d (continuing — non-fatal)", rc)

    # ----- Step 5: backfill the bundle into the UI -----
    if not args.no_backfill:
        rc = _run_script([
            "python", str(project_root / "scripts" / "backfill_jobs_to_gcs.py"),
            "--local-dir", str(local_dir),
            "--bucket", args.bucket,
            "--project", args.project,
            "--job-id", backfill_id,
            "--source-tag", "advanced-analysis-postprocessed",
        ], args.dry_run)
        if rc != 0:
            logger.error("backfill_jobs_to_gcs.py exited %d", rc)
            return rc

    logger.info("Done. Local artifacts: %s/report/", local_dir)
    if not args.no_backfill:
        logger.info("Visible in UI as: benchmark_jobs/%s", backfill_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

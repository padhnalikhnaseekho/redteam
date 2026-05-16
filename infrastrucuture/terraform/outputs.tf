output "artifact_registry_repository" {
  description = "Artifact Registry Docker repository."
  value       = google_artifact_registry_repository.containers.name
}

output "artifact_registry_location" {
  description = "Artifact Registry repository location."
  value       = google_artifact_registry_repository.containers.location
}

output "ui_url" {
  description = "IAP-protected UI Cloud Run URL."
  value       = google_cloud_run_v2_service.ui.uri
}

output "api_url" {
  description = "IAM-protected API Cloud Run URL."
  value       = google_cloud_run_v2_service.api.uri
}

output "worker_job_name" {
  description = "Cloud Run Job name for benchmark workers."
  value       = google_cloud_run_v2_job.worker.name
}

output "tasks_queue_name" {
  description = "Cloud Tasks benchmark queue name."
  value       = google_cloud_tasks_queue.benchmark.name
}

output "results_bucket" {
  description = "GCS bucket for benchmark artifacts."
  value       = google_storage_bucket.results.name
}

output "service_accounts" {
  description = "Cloud Run and build service account emails."
  value = {
    ui     = google_service_account.ui.email
    api    = google_service_account.api.email
    worker = google_service_account.worker.email
    tasks  = google_service_account.tasks.email
    build  = google_service_account.build.email
  }
}

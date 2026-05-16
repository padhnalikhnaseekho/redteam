resource "google_cloud_tasks_queue" "benchmark" {
  project  = var.project_id
  name     = "${local.resource_prefix}-benchmark"
  location = var.region

  rate_limits {
    max_dispatches_per_second = 1
    max_concurrent_dispatches = 2
  }

  retry_config {
    max_attempts       = 3
    min_backoff        = "10s"
    max_backoff        = "300s"
    max_doublings      = 3
    max_retry_duration = "1800s"
  }

  depends_on = [google_project_service.required]
}

resource "google_cloud_tasks_queue_iam_member" "api_enqueues_tasks" {
  project  = var.project_id
  location = google_cloud_tasks_queue.benchmark.location
  name     = google_cloud_tasks_queue.benchmark.name
  role     = "roles/cloudtasks.enqueuer"
  member   = "serviceAccount:${google_service_account.api.email}"
}

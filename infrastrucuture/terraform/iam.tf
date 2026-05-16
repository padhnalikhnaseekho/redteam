locals {
  project_iam_bindings = {
    api_vertex_user = {
      role   = "roles/aiplatform.user"
      member = "serviceAccount:${google_service_account.api.email}"
    }
    worker_vertex_user = {
      role   = "roles/aiplatform.user"
      member = "serviceAccount:${google_service_account.worker.email}"
    }
    api_datastore_user = {
      role   = "roles/datastore.user"
      member = "serviceAccount:${google_service_account.api.email}"
    }
    worker_datastore_user = {
      role   = "roles/datastore.user"
      member = "serviceAccount:${google_service_account.worker.email}"
    }
    worker_logging_writer = {
      role   = "roles/logging.logWriter"
      member = "serviceAccount:${google_service_account.worker.email}"
    }
    api_run_developer = {
      role   = "roles/run.developer"
      member = "serviceAccount:${google_service_account.api.email}"
    }
    build_artifact_writer = {
      role   = "roles/artifactregistry.writer"
      member = "serviceAccount:${google_service_account.build.email}"
    }
    build_run_admin = {
      role   = "roles/run.admin"
      member = "serviceAccount:${google_service_account.build.email}"
    }
    build_service_account_user = {
      role   = "roles/iam.serviceAccountUser"
      member = "serviceAccount:${google_service_account.build.email}"
    }
    build_logging_writer = {
      role   = "roles/logging.logWriter"
      member = "serviceAccount:${google_service_account.build.email}"
    }
  }
}

resource "google_project_iam_member" "project" {
  for_each = local.project_iam_bindings

  project = var.project_id
  role    = each.value.role
  member  = each.value.member

  depends_on = [
    google_project_service.required,
    google_service_account.ui,
    google_service_account.api,
    google_service_account.worker,
    google_service_account.tasks,
    google_service_account.build,
  ]
}

resource "google_storage_bucket_iam_member" "api_results_reader" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "worker_results_admin" {
  bucket = google_storage_bucket.results.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_service_account_iam_member" "api_can_use_tasks_oidc_identity" {
  service_account_id = google_service_account.tasks.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.api.email}"
}

resource "google_service_account_iam_member" "api_can_execute_worker_identity" {
  service_account_id = google_service_account.worker.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service" "ui" {
  provider = google-beta

  project             = var.project_id
  name                = "${local.resource_prefix}-ui"
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"
  iap_enabled         = true
  labels              = local.labels

  template {
    service_account = google_service_account.ui.email

    scaling {
      min_instance_count = 0
      max_instance_count = var.ui_max_instances
    }

    containers {
      image = var.ui_image

      ports {
        container_port = 8501
      }

      env {
        name  = "REDTEAM_API_URL"
        value = google_cloud_run_v2_service.api.uri
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "RESULTS_BUCKET"
        value = google_storage_bucket.results.name
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_firestore_database.default,
  ]
}

resource "google_cloud_run_v2_service" "api" {
  provider = google-beta

  project             = var.project_id
  name                = "${local.resource_prefix}-api"
  location            = var.region
  deletion_protection = false
  ingress             = var.api_ingress
  labels              = local.labels

  template {
    service_account = google_service_account.api.email

    scaling {
      min_instance_count = 0
      max_instance_count = var.api_max_instances
    }

    containers {
      image = var.api_image

      ports {
        container_port = 8080
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.vertex_location
      }

      env {
        name  = "RESULTS_BUCKET"
        value = google_storage_bucket.results.name
      }

      env {
        name  = "TASKS_QUEUE"
        value = google_cloud_tasks_queue.benchmark.name
      }

      env {
        name  = "TASKS_LOCATION"
        value = var.region
      }

      env {
        name  = "TASKS_CALLER_SERVICE_ACCOUNT"
        value = google_service_account.tasks.email
      }

      env {
        name  = "WORKER_JOB_NAME"
        value = google_cloud_run_v2_job.worker.name
      }

      env {
        name  = "WORKER_JOB_LOCATION"
        value = var.region
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_firestore_database.default,
  ]
}

resource "google_cloud_run_v2_job" "worker" {
  provider = google-beta

  project             = var.project_id
  name                = "${local.resource_prefix}-worker"
  location            = var.region
  deletion_protection = false
  labels              = local.labels

  template {
    template {
      service_account = google_service_account.worker.email
      max_retries     = 1
      timeout         = var.worker_task_timeout

      containers {
        image = var.worker_image

        resources {
          limits = {
            cpu    = var.worker_task_cpu
            memory = var.worker_task_memory
          }
        }

        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }

        env {
          name  = "GOOGLE_CLOUD_LOCATION"
          value = var.vertex_location
        }

        env {
          name  = "RESULTS_BUCKET"
          value = google_storage_bucket.results.name
        }
      }
    }
  }

  depends_on = [
    google_project_service.required,
    google_firestore_database.default,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "ui_invokes_api" {
  project  = google_cloud_run_v2_service.api.project
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.ui.email}"
}

resource "google_cloud_run_v2_service_iam_member" "iap_invokes_ui" {
  project  = google_cloud_run_v2_service.ui.project
  location = google_cloud_run_v2_service.ui.location
  name     = google_cloud_run_v2_service.ui.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:service-${data.google_project.current.number}@gcp-sa-iap.iam.gserviceaccount.com"
}

resource "google_cloud_run_v2_service_iam_member" "tasks_invokes_api" {
  project  = google_cloud_run_v2_service.api.project
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.tasks.email}"
}

resource "google_storage_bucket" "results" {
  project                     = var.project_id
  name                        = "${var.project_id}-${local.resource_prefix}-results"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false
  labels                      = local.labels

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_artifact_registry_repository" "containers" {
  project       = var.project_id
  location      = var.region
  repository_id = "${local.resource_prefix}-containers"
  description   = "Docker images for the CommodityRedTeam Cloud Run demo."
  format        = "DOCKER"
  labels        = local.labels

  depends_on = [google_project_service.required]
}

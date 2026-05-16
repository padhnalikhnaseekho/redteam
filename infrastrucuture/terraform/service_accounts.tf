resource "google_service_account" "ui" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-${var.environment}-ui"
  display_name = "Redteam UI Cloud Run service account"

  depends_on = [google_project_service.required]
}

resource "google_service_account" "api" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-${var.environment}-api"
  display_name = "Redteam API Cloud Run service account"

  depends_on = [google_project_service.required]
}

resource "google_service_account" "worker" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-${var.environment}-worker"
  display_name = "Redteam worker Cloud Run Job service account"

  depends_on = [google_project_service.required]
}

resource "google_service_account" "tasks" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-${var.environment}-tasks"
  display_name = "Redteam Cloud Tasks OIDC caller service account"

  depends_on = [google_project_service.required]
}

resource "google_service_account" "build" {
  project      = var.project_id
  account_id   = "${var.name_prefix}-${var.environment}-build"
  display_name = "Redteam build and deploy service account"

  depends_on = [google_project_service.required]
}

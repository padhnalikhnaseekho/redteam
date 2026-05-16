locals {
  resource_prefix = "${var.name_prefix}-${var.environment}"

  labels = merge(
    {
      app         = "commodity-redteam"
      environment = var.environment
      managed_by  = "terraform"
    },
    var.labels
  )

  apis = toset([
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudtasks.googleapis.com",
    "firestore.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iap.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])
}

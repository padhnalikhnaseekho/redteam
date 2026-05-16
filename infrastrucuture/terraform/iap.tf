resource "google_iap_web_cloud_run_service_iam_member" "ui_access" {
  for_each = toset(var.allowed_iap_members)

  project                = google_cloud_run_v2_service.ui.project
  location               = google_cloud_run_v2_service.ui.location
  cloud_run_service_name = google_cloud_run_v2_service.ui.name
  role                   = "roles/iap.httpsResourceAccessor"
  member                 = each.value
}

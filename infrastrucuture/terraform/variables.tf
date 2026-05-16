variable "project_id" {
  description = "GCP project ID where the redteam demo infrastructure is deployed."
  type        = string
}

variable "region" {
  description = "Cloud Run, Cloud Tasks, Artifact Registry, and GCS region."
  type        = string
  default     = "asia-south1"
}

variable "firestore_location" {
  description = "Firestore database location. This cannot be changed after creation."
  type        = string
  default     = "asia-south1"
}

variable "environment" {
  description = "Short environment name used in resource names and labels."
  type        = string
  default     = "demo"
}

variable "name_prefix" {
  description = "Prefix for redteam resources."
  type        = string
  default     = "redteam"
}

variable "ui_image" {
  description = "Container image URI for the Streamlit UI service."
  type        = string
}

variable "api_image" {
  description = "Container image URI for the FastAPI API service."
  type        = string
}

variable "worker_image" {
  description = "Container image URI for the Cloud Run Job worker."
  type        = string
}

variable "vertex_location" {
  description = "Default Vertex AI model location used by services. Individual model config can still override this."
  type        = string
  default     = "us-central1"
}

variable "allowed_iap_members" {
  description = "Users or groups allowed through Cloud IAP, e.g. user:name@gmail.com or group:redteam@example.com."
  type        = list(string)
  default     = []
}

variable "api_ingress" {
  description = "Ingress setting for the API service. Keep ALL for IAM-protected Cloud Run service-to-service over default URL."
  type        = string
  default     = "INGRESS_TRAFFIC_ALL"
}

variable "ui_max_instances" {
  description = "Maximum UI service instances."
  type        = number
  default     = 2
}

variable "api_max_instances" {
  description = "Maximum API service instances."
  type        = number
  default     = 3
}

variable "worker_task_timeout" {
  description = "Cloud Run Job task timeout."
  type        = string
  default     = "3600s"
}

variable "worker_task_memory" {
  description = "Worker task memory limit."
  type        = string
  default     = "4Gi"
}

variable "worker_task_cpu" {
  description = "Worker task CPU limit."
  type        = string
  default     = "2"
}

variable "labels" {
  description = "Additional labels applied to supported resources."
  type        = map(string)
  default     = {}
}

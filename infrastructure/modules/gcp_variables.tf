variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region (e.g. asia-south1 for Mumbai)"
  type        = string
  default     = "asia-south1"
}

variable "environment" {
  description = "Deployment environment: dev | staging | prod"
  type        = string
  default     = "dev"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

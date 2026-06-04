# infrastructure/modules/gcp_cloudrun.tf
# GCP Cloud Run deployment for the DS-01 Churn API.
# Equivalent to the AWS Lambda setup in deployment/terraform/main.tf.

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# ---------- Artifact Registry (GCP equivalent of ECR) ----------
resource "google_artifact_registry_repository" "churn_api" {
  location      = var.gcp_region
  repository_id = "ds01-churn-api"
  format        = "DOCKER"
  description   = "DS-01 Churn Intelligence API Docker images"
}

# ---------- Cloud Run service ----------
resource "google_cloud_run_v2_service" "churn_api" {
  name     = "ds01-churn-api"
  location = var.gcp_region

  template {
    containers {
      image = "${var.gcp_region}-docker.pkg.dev/${var.gcp_project_id}/ds01-churn-api/ds01-churn-api:${var.image_tag}"

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GROQ_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.groq_key.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }
  }

  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
}

# ---------- Public access (unauthenticated) ----------
resource "google_cloud_run_service_iam_member" "public" {
  location = google_cloud_run_v2_service.churn_api.location
  service  = google_cloud_run_v2_service.churn_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------- Secret Manager for Groq API key ----------
resource "google_secret_manager_secret" "groq_key" {
  secret_id = "groq-api-key"
  replication { auto {} }
}

# ---------- GCS bucket for model artifacts ----------
resource "google_storage_bucket" "model_store" {
  name          = "${var.gcp_project_id}-ds01-models-${var.environment}"
  location      = var.gcp_region
  force_destroy = false
  versioning { enabled = true }
}

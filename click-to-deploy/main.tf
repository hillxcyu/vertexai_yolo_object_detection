# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
  ])

  staging_bucket_name = "${var.project_id}-yolo-artifacts"
}

# Enable required Google Cloud APIs
resource "google_project_service" "apis" {
  for_each = local.required_apis

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry repository for custom training and CPR serving images
resource "google_artifact_registry_repository" "yolo_cpr_repo" {
  project       = var.project_id
  location      = var.data_location
  repository_id = var.artifact_registry_repo_id
  description   = "Docker repository for Vertex AI Custom YOLO training and CPR serving images."
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}

# GCS bucket for Vertex AI datasets, training staging, schemas, and model artifacts
resource "google_storage_bucket" "yolo_artifacts" {
  project                     = var.project_id
  name                        = local.staging_bucket_name
  location                    = var.data_location
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true

  depends_on = [google_project_service.apis]
}

# Least-privilege Service Account for Vertex AI Custom Training jobs
resource "google_service_account" "yolo_training_sa" {
  project      = var.project_id
  account_id   = "yolo-training-sa"
  display_name = "Vertex AI YOLO Custom Training Service Account"

  depends_on = [google_project_service.apis]
}

# Least-privilege Service Account for Vertex AI CPR Endpoint Serving
resource "google_service_account" "yolo_serving_sa" {
  project      = var.project_id
  account_id   = "yolo-serving-sa"
  display_name = "Vertex AI YOLO CPR Serving Service Account"

  depends_on = [google_project_service.apis]
}

# IAM bindings for Training Service Account
resource "google_project_iam_member" "training_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.yolo_training_sa.email}"
}

resource "google_project_iam_member" "training_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.yolo_training_sa.email}"
}

resource "google_storage_bucket_iam_member" "training_bucket_admin" {
  bucket = google_storage_bucket.yolo_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.yolo_training_sa.email}"
}

resource "google_artifact_registry_repository_iam_member" "training_ar_reader" {
  project    = var.project_id
  location   = google_artifact_registry_repository.yolo_cpr_repo.location
  repository = google_artifact_registry_repository.yolo_cpr_repo.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.yolo_training_sa.email}"
}

# IAM bindings for Serving Service Account
resource "google_project_iam_member" "serving_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.yolo_serving_sa.email}"
}

resource "google_project_iam_member" "serving_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.yolo_serving_sa.email}"
}

resource "google_storage_bucket_iam_member" "serving_bucket_viewer" {
  bucket = google_storage_bucket.yolo_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.yolo_serving_sa.email}"
}

resource "google_artifact_registry_repository_iam_member" "serving_ar_reader" {
  project    = var.project_id
  location   = google_artifact_registry_repository.yolo_cpr_repo.location
  repository = google_artifact_registry_repository.yolo_cpr_repo.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.yolo_serving_sa.email}"
}

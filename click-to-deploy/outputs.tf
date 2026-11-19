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

output "project_id" {
  description = "The GCP Project ID where resources were deployed."
  value       = var.project_id
}

output "region" {
  description = "The GCP region for Vertex AI, Artifact Registry, and Cloud Storage."
  value       = var.data_location
}

output "artifact_registry_repository" {
  description = "The Artifact Registry repository ID for custom YOLO images."
  value       = google_artifact_registry_repository.yolo_cpr_repo.repository_id
}

output "artifact_registry_url" {
  description = "The Docker registry URL for pushing custom training and CPR images."
  value       = "${var.data_location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.yolo_cpr_repo.repository_id}"
}

output "staging_bucket_uri" {
  description = "The GCS staging and artifact bucket URI (same-project requirement for Vertex AI GCS FUSE)."
  value       = "gs://${google_storage_bucket.yolo_artifacts.name}"
}

output "training_service_account" {
  description = "Least-privilege service account email for Vertex AI Custom Training."
  value       = google_service_account.yolo_training_sa.email
}

output "serving_service_account" {
  description = "Least-privilege service account email for Vertex AI CPR Endpoint serving."
  value       = google_service_account.yolo_serving_sa.email
}

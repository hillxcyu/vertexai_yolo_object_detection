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

variable "project_id" {
  type        = string
  description = "The GCP Project ID where the demo resources will be deployed."
}

variable "project_name" {
  type        = string
  description = "The human-readable GCP Project Name."
  default     = ""
}

variable "project_number" {
  type        = string
  description = "The numeric GCP Project Number."
  default     = ""
}

variable "org_id" {
  type        = string
  description = "The GCP Organization ID."
  default     = ""
}

variable "gcp_account_name" {
  type        = string
  description = "The user account email triggering the Click-to-Deploy run."
  default     = ""
}

variable "deployment_service_account_name" {
  type        = string
  description = "The service account used by the Demo Automation Platform to execute Terraform."
  default     = ""
}

variable "data_location" {
  type        = string
  description = "Default region for Vertex AI, Artifact Registry, and Cloud Storage resources."
  default     = "us-central1"
}

variable "secret_stored_project" {
  type        = string
  description = "The GCP project ID where demo secrets are stored."
  default     = ""
}

variable "artifact_registry_repo_id" {
  type        = string
  description = "Artifact Registry Docker repository ID for custom training and CPR serving images."
  default     = "yolo-custom-cpr-images"
}

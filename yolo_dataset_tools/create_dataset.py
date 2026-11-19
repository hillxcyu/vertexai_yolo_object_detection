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

import os

PROJECT_ID = os.environ.get("PROJECT_ID", "YOUR_PROJECT_ID")
LOCATION = os.environ.get("REGION", "us-central1")
BUCKET_URI = os.environ.get("STAGING_BUCKET", "gs://YOUR_GCS_BUCKET")
dataset_name = os.environ.get("DATASET_NAME", "retail_shelf_detection_dataset")
from google.cloud import aiplatform
IMPORT_FILE = f"{BUCKET_URI}/vision/vertex_ai_single_row_index.csv"

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
dataset = aiplatform.ImageDataset.create(
    display_name=dataset_name,
    gcs_source=[IMPORT_FILE],
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
)

print(dataset.resource_name)

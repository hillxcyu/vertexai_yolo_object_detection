# Vertex AI Custom YOLO Training & Serving MLOps Blueprint — Seller Demo Guide

## Executive Summary & Pitch Context

* **Target Buyer Personas:** VP of Retail Store Operations, Head of Computer Vision / MLOps Engineering, Director of Merchandising & Planogram Compliance
* **Customer Pain Point:** Retail and manufacturing teams rely on custom open-source object detection models (such as Ultralytics YOLO) for high-density retail shelf auditing and stockout detection, but struggle with brittle training pipelines, slow dataset transfers, and incompatible inference schemas that force downstream applications to rewrite integration code every time a model is swapped.
* **Value Proposition:** Delivers an end-to-end, production-ready MLOps blueprint on Google Cloud that trains custom YOLO models directly against Vertex AI Managed Datasets via zero-copy GCS FUSE mounts and serves them through a Custom Prediction Routine (CPR) whose output schema is 100% compatible with Vertex AI AutoML Object Detection—eliminating downstream app rewrites and cutting time-to-production from weeks to hours.
* **Estimated Demo Duration:** 15 minutes

## How to Build

This section walks through how the **Vertex AI Custom YOLO Training & Serving MLOps Blueprint** is architected and built end-to-end on Google Cloud using **Vertex AI Training, Vertex AI Custom Prediction Routines (CPR), Vertex AI Model Registry, Artifact Registry, and Cloud Storage**.

### Step 1: Provision Infrastructure & Supply-Chain Controls (`click-to-deploy/`)

* `click-to-deploy/main.tf` and `click-to-deploy/variables.tf` enable required APIs (`aiplatform.googleapis.com`, `artifactregistry.googleapis.com`, `storage.googleapis.com`, `cloudbuild.googleapis.com`), create a Docker Artifact Registry repository (`yolo-custom-cpr-images`), provision a uniform-access GCS artifact bucket (`<PROJECT_ID>-yolo-artifacts`), and configure least-privilege service accounts (`yolo-training-sa` and `yolo-serving-sa`).
* `click-to-deploy/requirements.txt` pins all Python dependencies with SHA-256 hashes (`--require-hashes`), and `click-to-deploy/Dockerfile` enforces hash-verified installation.

### Step 2: Convert YOLO Datasets to Vertex AI Managed Datasets (`yolo_dataset_tools/`)

* `yolo_dataset_tools/yolo2vertexCSV.py` converts standard YOLO directory trees (`images/{train,val,test}` and `labels/{train,val,test}`) into a Vertex AI Object Detection CSV index (`vertex_ai_single_row_index.csv`) referencing `gs://` image URIs.
* `yolo_dataset_tools/create_dataset.py` imports that CSV into a Vertex AI Managed `ImageDataset` using `aiplatform.schema.dataset.ioformat.image.bounding_box`.

### Step 3: Train YOLO on Vertex AI with GCS FUSE (`train.py`)

* `train.py` is the Vertex AI Custom Training entrypoint. It reads the `AIP_TRAINING_DATA_URI`, `AIP_VALIDATION_DATA_URI`, and `AIP_TEST_DATA_URI` environment variables injected by Vertex AI, accesses manifest shards and images directly over `/gcs/<bucket>/...` (GCS FUSE) using local symlinks without downloading terabytes of images, trains the Ultralytics YOLO model on GPU (`device=0`), and uploads `best.pt`, `class_map.json`, and training plots (`results.csv`) to the target GCS artifact URI.

### Step 4: Wrap Inference in an AutoML-Compatible CPR (`cpr_src/predictor.py`)

* `cpr_src/predictor.py` implements `YoloCompatiblePredictor(Predictor)` for Vertex AI Custom Prediction Routines.
* `load()` downloads `best.pt` and `class_map.json` via `prediction_utils.download_model_artifacts()`.
* `preprocess()` decodes base64-encoded images from `instances[].content`.
* `predict()` runs batched Ultralytics inference on GPU or CPU.
* `postprocess()` applies `confidenceThreshold` and `maxPredictions` parameters and returns `{"predictions": [{"displayNames": [...], "confidences": [...], "bboxes": [[xMin, xMax, yMin, yMax], ...]}]}`—matching the exact AutoML Object Detection schema.

### Step 5: Orchestrate Training, Container Build & Registration (`launch_training_job.py`, `create_AR.py`)

* `launch_training_job.py` calls `ensure_artifact_registry_repo()` in `create_AR.py`, copies Vertex AI prediction schemas to the staging bucket, submits `aiplatform.CustomTrainingJob`, builds and pushes the CPR Docker container via `LocalModel.build_cpr_model()`, and registers the model in Vertex AI Model Registry via `aiplatform.Model.upload()`.

## Pillar 1: Setup & Deployment Instructions

### 1.1 Prerequisites & IAM Permissions

* **GCP Environment:** Google Cloud project with Billing enabled and GPU quota (`NVIDIA_TESLA_T4` >= 1 in `us-central1` for Vertex AI Custom Training).
* **Local Tooling:** Python 3.10+, Docker daemon running locally, Google Cloud CLI (`gcloud`), and Terraform >= 1.5.0.
* **Required IAM Roles (for operator):** `roles/aiplatform.admin`, `roles/artifactregistry.admin`, `roles/storage.admin`, `roles/iam.serviceAccountAdmin`, `roles/serviceusage.serviceUsageAdmin` (or `roles/owner`).
* **Required APIs:** `aiplatform.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`, `compute.googleapis.com`, `iam.googleapis.com`, `storage.googleapis.com`.

### 1.2 Option A: One-Click Deployment via go/demos (Recommended)

1. Navigate to this blueprint's page on `go/demos` and click **Deploy Demo (Click-to-Deploy)**.
2. Select your target GCP organization and project, and set `data_location` to `us-central1`.
3. Wait approximately **3–5 minutes** for Terraform provisioning to complete.

### 1.3 Option B: Manual CLI / Terraform Deployment

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"

cd click-to-deploy
terraform init
terraform apply -var="project_id=${PROJECT_ID}" -var="data_location=${REGION}"
```

Capture the Terraform outputs for the pipeline configuration:

```bash
export STAGING_BUCKET="$(terraform output -raw staging_bucket_uri)"
export AR_REPO="$(terraform output -raw artifact_registry_repository)"
```

### 1.4 Dataset Preparation & Pipeline Execution

1. Authenticate local credentials and Docker:

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

2. Install Python dependencies:

```bash
cd ..
pip install -r requirements.txt
```

3. Prepare and register a sample retail shelf dataset (must reside in the **same GCP project** as the training job so `/gcs/` FUSE mounts the bucket):

```bash
export GCS_BUCKET="${STAGING_BUCKET#gs://}"
python3 yolo_dataset_tools/yolo2vertexCSV.py
gsutil cp vertex_ai_single_row_index.csv "${STAGING_BUCKET}/vision/vertex_ai_single_row_index.csv"
python3 yolo_dataset_tools/create_dataset.py
```

4. Set the resulting Dataset ID and run the end-to-end MLOps pipeline:

```bash
export DATASET_ID="YOUR_VERTEX_AI_DATASET_ID"
python3 launch_training_job.py
```

### 1.5 Deployment Verification

Verify the Artifact Registry image, Custom Training job, and registered Model before your customer call:

```bash
gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/yolo-custom-cpr-images" --project="${PROJECT_ID}"
gcloud ai models list --region="${REGION}" --project="${PROJECT_ID}"
gcloud ai custom-jobs list --region="${REGION}" --project="${PROJECT_ID}"
```

### 1.6 Teardown & Cost Cleanup

Undeploy any active Vertex AI Endpoints first to stop GPU/VM billing, then destroy Terraform-managed resources:

```bash
gcloud ai endpoints list --region="${REGION}" --project="${PROJECT_ID}"
# If an endpoint exists:
# gcloud ai endpoints undeploy-model ENDPOINT_ID --deployed-model-id=DEPLOYED_MODEL_ID --region="${REGION}" --project="${PROJECT_ID}"
# gcloud ai endpoints delete ENDPOINT_ID --region="${REGION}" --project="${PROJECT_ID}"

cd click-to-deploy
terraform destroy -var="project_id=${PROJECT_ID}" -var="data_location=${REGION}"
```

## Pillar 2: How to Access the Demo

### 2.1 Retrieving Entry Point URLs

This blueprint operates through Google Cloud Console and Vertex AI APIs:

* **Vertex AI Datasets:** `https://console.cloud.google.com/vertex-ai/datasets?project=YOUR_PROJECT_ID`
* **Vertex AI Custom Training Jobs:** `https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=YOUR_PROJECT_ID`
* **Vertex AI Model Registry:** `https://console.cloud.google.com/vertex-ai/models?project=YOUR_PROJECT_ID`
* **Vertex AI Endpoints:** `https://console.cloud.google.com/vertex-ai/online-prediction/endpoints?project=YOUR_PROJECT_ID`

### 2.2 Credentials & Service Accounts

* **Presenter Login:** Sign in to Google Cloud Console with your `@google.com` or Argolis admin account that holds `roles/viewer` or `roles/aiplatform.user` on the target project.
* **Runtime Service Accounts (provisioned by Terraform):**
  * `yolo-training-sa@<PROJECT_ID>.iam.gserviceaccount.com` — Runs Vertex AI Custom Training with `roles/aiplatform.user` and `roles/storage.objectAdmin` scoped to the artifacts bucket.
  * `yolo-serving-sa@<PROJECT_ID>.iam.gserviceaccount.com` — Serves Vertex AI CPR Endpoints with `roles/storage.objectViewer` and `roles/artifactregistry.reader`.

### 2.3 Expected Initial Landing State

Before starting the customer walkthrough, confirm in Vertex AI Console:

1. **Managed Dataset** appears under **Vertex AI → Datasets** with `Image (Object Detection)` data type and labeled retail shelf bounding boxes visible in the **Browse** tab.
2. **Completed Training Job** shows status **Succeeded** under **Vertex AI → Training → Custom Jobs**.
3. **Registered Model** appears under **Vertex AI → Model Registry** with version `1` and the custom CPR container URI (`...-docker.pkg.dev/.../yolo-automl-compat-predictor:...`).

## Demo Walkthrough (Pillar 3 & Pillar 4: Step-by-Step Walkthrough & Seller Talking Script)

### Opening Discovery Hook (Before Sharing Screen)

> **What to Say:** *"Many retail and manufacturing computer vision teams tell us their data scientists can train a high-accuracy YOLO model on a laptop in an afternoon, yet it takes 6 to 8 weeks to turn that notebook into a governed, repeatable training and serving pipeline in production. When you need to retrain on new seasonal SKUs or swap model architectures, how much glue code does your team have to rewrite today?"*

### Step 1: Retail Shelf Dataset Governance in Vertex AI Managed Datasets

* **What to Do (Click Sequence):**
  1. Open **Vertex AI → Datasets** in Cloud Console and click your `retail_shelf_detection_dataset`.
  2. Click the **Browse** tab and select 2–3 shelf images to display the bounding box annotations and train/validation/test split tags.
* **Expected Visual Outcome:** Annotated retail shelf images render with bounding boxes around product facings and stockout gaps, along with split distribution counts.
* **What to Say (Seller Script):** *"Everything starts with governed data. Using our dataset converter (`yolo2vertexCSV.py`), your existing YOLO-labeled images stay in Cloud Storage while Vertex AI Managed Datasets gives your team visual QA, automatic train/val/test splitting, and full data lineage—without duplicating image files."*
* **Why Google Cloud (Differentiator Callout):** *"Vertex AI Custom Training automatically mounts your project's GCS buckets at `/gcs/` via GCS FUSE, so `train.py` reads manifests and symlinks images directly without copying gigabytes of training data into the container."*

### Step 2: Reproducible Custom YOLO Training Job (`launch_training_job.py` & `train.py`)

* **What to Do (Click Sequence):**
  1. Navigate to **Vertex AI → Training → Custom Jobs** and click the completed `yolo_custom_training_train_...` job.
  2. Show the machine spec (`n1-standard-4`, `1 x NVIDIA_TESLA_T4`), container image (`pytorch-gpu.2-3.py310`), and click **View Logs** to highlight the epoch progress and final `best.pt` + `class_map.json` upload to GCS.
* **Expected Visual Outcome:** Training logs show Ultralytics YOLO epochs completing on GPU and uploading `best.pt`, `class_map.json`, and `results.csv` to `gs://<PROJECT_ID>-yolo-artifacts/yolo_artifacts_.../`.
* **What to Say (Seller Script):** *"Notice that we didn't have to manage a Kubernetes cluster or build a custom training base image from scratch. Vertex AI provisions the T4 GPU on demand, injects the dataset splits, runs our PyTorch training script, logs every metric, and tears down the compute the second training finishes."*
* **Why Google Cloud (Differentiator Callout):** *"Pay-per-second serverless GPU training combined with native GCS FUSE streaming means zero idle GPU cost and zero manual infrastructure orchestration."*

### Step 3: AutoML-Compatible Custom Prediction Routine (`cpr_src/predictor.py`) & Model Registry

* **What to Do (Click Sequence):**
  1. Navigate to **Vertex AI → Model Registry** and click the `yolo_custom_training_model_...` model, then click **Version 1 → Details**.
  2. Point out the **Artifact location** (`gs://.../yolo_artifacts_.../`) and **Container image** (`...-docker.pkg.dev/.../yolo-automl-compat-predictor:...`), as well as the `instance_schema_uri` and `prediction_schema_uri`.
* **Expected Visual Outcome:** The Model Registry details page shows the custom CPR container alongside standard AutoML Object Detection schemas.
* **What to Say (Seller Script):** *"Here is the secret weapon for fast production adoption. Normally, switching from AutoML to a custom YOLO model breaks every downstream retail analytics app because the JSON payload changes. Our Custom Prediction Routine (`YoloCompatiblePredictor`) translates YOLO bounding boxes into the exact `displayNames`, `confidences`, and normalized `[xMin, xMax, yMin, yMax]` format AutoML uses—so downstream apps don't change a single line of code."*
* **Why Google Cloud (Differentiator Callout):** *"Vertex AI CPR lets data scientists write pure Python preprocessing and postprocessing without maintaining Flask, FastAPI, or Gunicorn boilerplate—the Vertex AI SDK builds and optimizes the serving container automatically."*

### Step 4: Live Prediction Test on a Retail Shelf Image

* **What to Do (Click Sequence):**
  1. Navigate to **Vertex AI → Online Prediction → Endpoints** (or open Cloud Shell) and run the following verified prediction command against your deployed endpoint:

```bash
IMG_B64=$(base64 -w 0 sample_shelf.jpg)
cat <<EOF > /tmp/request.json
{
  "instances": [{"content": "${IMG_B64}"}],
  "parameters": {"confidenceThreshold": 0.5, "maxPredictions": 50}
}
EOF

curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ENDPOINT_ID}:predict" \
  -d @/tmp/request.json | jq .
```

* **Expected Visual Outcome:** The endpoint returns structured JSON detections within ~1–2 seconds:

```json
{
  "predictions": [
    {
      "displayNames": ["shelf_item", "stockout_gap"],
      "confidences": [0.94, 0.88],
      "bboxes": [
        [0.12, 0.34, 0.20, 0.55],
        [0.58, 0.72, 0.21, 0.54]
      ]
    }
  ]
}
```

* **What to Say (Seller Script):** *"In under a second, our custom YOLO model returns normalized bounding boxes and confidence thresholds in the exact AutoML contract, backed by least-privilege IAM and Artifact Registry container security."*

### Closing & Next-Step Transition

* **Common Customer Objection & Response:**
  * *Objection:* *"Can we use our own pre-trained weights or fine-tune newer YOLO architectures without rebuilding the whole pipeline?"*
  * *Response:* *"Yes—`train.py` accepts any `--model_variant` (either a standard Ultralytics weight or an existing `.pt` checkpoint on `/gcs/`), and `cpr_src/predictor.py` dynamically loads `class_map.json` alongside `best.pt` at startup."*
* **Call to Action:** *"We can deploy this exact Terraform and Vertex AI MLOps blueprint into a sandbox project with a sample of your labeled images in a 2-hour technical workshop. Which computer vision use case and engineering leads should we include next week?"*

## Troubleshooting & Live-Demo Recovery Tips

* **GCS FUSE Mount Error (`Manifest file not found at /gcs/...`):** Confirm the Vertex AI Managed Dataset's GCS bucket lives in the **same GCP project** as the Custom Training job. Cross-project buckets are not mounted to `/gcs/` by Vertex AI.
* **GPU Quota Exceeded (`NVIDIA_TESLA_T4`):** If `us-central1` has temporary T4 capacity constraints, update `REGION` in `launch_training_job.py` and `data_location` in `click-to-deploy/variables.tf` to `us-east4` or `europe-west4`, or test `train.py` on CPU (`device="cpu"`).
* **Docker Build / Push Permission Denied:** Re-run `gcloud auth configure-docker ${REGION}-docker.pkg.dev` before executing `launch_training_job.py` so `LocalModel.build_cpr_model()` can push the CPR image.
* **Endpoint Cold-Start Latency:** If presenting a live `:predict` call, send one warm-up request 5 minutes before the customer call so the CPR container and model weights are already loaded in memory.

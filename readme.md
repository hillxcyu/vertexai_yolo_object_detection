# Vertex AI Custom YOLO Training & Serving MLOps Blueprint

This project provides a complete end-to-end MLOps workflow to:

1. **Train** a custom Ultralytics YOLO object detection model on a Vertex AI Managed Dataset using a **Vertex AI Custom Training job** with GCS FUSE mounts.
2. **Build** a custom inference Docker container using a **Vertex AI Custom Prediction Routine (CPR)** (`cpr_src/predictor.py`) that formats predictions to match the standard AutoML Object Detection schema.
3. **Register & Serve** the trained model, container image, and prediction schemas in the **Vertex AI Model Registry** and **Vertex AI Endpoints**.

> For the full seller & deployment guide, see [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md).

## Repository Structure

* `launch_training_job.py` — End-to-end orchestration script: ensures Artifact Registry repo, copies Vertex AI schemas, runs custom training, builds/pushes the CPR image, and uploads the model to Vertex AI Model Registry.
* `train.py` — Custom training entrypoint executed on Vertex AI Training; converts Vertex AI JSONL manifests (via `/gcs/` FUSE) to YOLO format, trains the model, and uploads `best.pt` and `class_map.json` to GCS.
* `create_AR.py` — Helper to verify or create the Docker Artifact Registry repository.
* `cpr_src/` — Custom Prediction Routine (`predictor.py` + `requirements.txt`) for AutoML-compatible object detection inference.
* `yolo_dataset_tools/` — Utilities (`yolo2vertexCSV.py`, `create_dataset.py`) to convert local YOLO datasets into Vertex AI Managed Image Datasets.
* `click-to-deploy/` — C2D-compliant Terraform configuration and pinned Python requirements to provision required APIs, GCS bucket, Artifact Registry repository, and least-privilege service accounts.

## Prerequisites

* [Docker](https://docs.docker.com/get-docker/) installed and running (for local CPR container builds).
* Python 3.10+ and required packages:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Provision Infrastructure (Optional via Terraform)

```bash
cd click-to-deploy
terraform init
terraform apply -var="project_id=YOUR_PROJECT_ID" -var="data_location=us-central1"
```

### 2. GCP Authentication

Authenticate both your local SDK environment and application default credentials:

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 3. Dataset Location Requirement

Your Vertex AI Managed Dataset **must** be located in a GCS bucket within the **same Google Cloud Project** where the training job runs. Vertex AI Custom Training mounts all GCS buckets from the host project to `/gcs/` inside the training container; buckets from other projects will not be mounted.

### 4. Configure & Run

Set the environment variables (or update the defaults at the top of `launch_training_job.py`):

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export STAGING_BUCKET="gs://YOUR_GCS_BUCKET"
export DATASET_ID="YOUR_VERTEX_AI_DATASET_ID"
```

Then launch the end-to-end pipeline:

```bash
python3 launch_training_job.py
```

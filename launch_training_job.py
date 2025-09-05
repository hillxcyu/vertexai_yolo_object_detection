# launch_training_job.py (CPR Upload - Using LocalModel build - Simplified State Check - Cleaned)
# Builds custom CPR image, runs training, uploads model using LocalModel object.

import os
import sys
from google.cloud import aiplatform
from google.cloud.aiplatform.compat.types import job_state as gca_job_state
# Removed GAPIC client import as state check is simplified
try:
    from google.cloud.aiplatform.prediction import LocalModel
    # Dynamic predictor import happens after sys.path modification
except ImportError as e:
     logging.error(f"Import Error: {e}. Ensure SDK is installed.")
     exit()

from datetime import datetime
import logging
import argparse
from pathlib import Path
import time
import json
from typing import Optional, Type, List

from create_AR import ensure_artifact_registry_repo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# ==============================================================================
# Configuration Settings
# ==============================================================================
PROJECT_ID = "YOUR PROJECT ID"
REGION = "YOUR PROJECT REGION"
# REGION = "us-central1"
STAGING_BUCKET = "YOUR STAGING_BUCKET"
# STAGING_BUCKET = "gs://yolo_training_artifacts"
DATASET_ID = "YOUR DATASET_ID"
# DATASET_ID = "1214135766438390813"
BASE_JOB_NAME = "YOUR JOB NAME"
# BASE_JOB_NAME = "yolo12_object_detection_product"

DATASET_RESOURCE_NAME = f"projects/{PROJECT_ID}/locations/{REGION}/datasets/{DATASET_ID}"
ANNOTATION_SCHEMA_URI = aiplatform.schema.dataset.annotation.image.bounding_box
TRAINING_SCRIPT_PATH = "train.py"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
JOB_DISPLAY_NAME = f"{BASE_JOB_NAME}_train_{TIMESTAMP}"
MODEL_DISPLAY_NAME = f"{BASE_JOB_NAME}_model_{TIMESTAMP}"

# --- Artifact Output Path ---
GCS_MODEL_ARTIFACT_DIR = f"{STAGING_BUCKET}/yolo_artifacts_{TIMESTAMP}/"
logging.info(f"Training artifacts target GCS path: {GCS_MODEL_ARTIFACT_DIR}")

# --- Custom CPR Container Image Configuration ---
AR_REPOSITORY = "yolo-custom-cpr-images" # <--- !!! CHANGE THIS if needed !!!
CUSTOM_IMAGE_NAME = f"yolo-automl-compat-predictor"

USER_PREDICTOR_DIR = "cpr_src" # Directory containing predictor.py and requirements.txt
PREDICTOR_REQUIREMENTS_PATH = os.path.join(USER_PREDICTOR_DIR, "requirements.txt")
CUSTOM_IMAGE_TAG = TIMESTAMP
CUSTOM_DEPLOY_CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{AR_REPOSITORY}/{CUSTOM_IMAGE_NAME}:{CUSTOM_IMAGE_TAG}"
logging.info(f"Custom CPR prediction image target URI: {CUSTOM_DEPLOY_CONTAINER_URI}")

# --- Training Container Configuration ---
TRAIN_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest"
logging.info(f"Using training container as base for CPR build: {TRAIN_CONTAINER_URI}")

# --- Compute/Training Config (Single Replica) ---
MACHINE_TYPE = "n1-standard-4"; ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"; ACCELERATOR_COUNT = 1; REPLICA_COUNT = 1
TRAINING_ARGS = ["--output-artifact-uri", GCS_MODEL_ARTIFACT_DIR, "--model_variant=/gcs/yolo_training_artifacts/yolo_artifacts_20250416_105831/best.pt", "--epochs=40", "--batch_size=16", "--img_size=640"] # <--- !!! set training parameters here, use --model_variant and point to a existing weights file (.pt ) will initiate the training from existing weights.
REQUIREMENTS_TRAIN = ["ultralytics","google-cloud-storage","pyyaml"]

# --- Define Schema URIs directly as strings ---
INSTANCE_SCHEMA_URI = f"{STAGING_BUCKET}/schema/predict/instance/image_object_detection_1.0.0.yaml"
PREDICTION_SCHEMA_URI = f"{STAGING_BUCKET}/schema/predict/prediction/image_object_detection_1.0.0.yaml"

# --- Dynamic Predictor Import ---
predictor_dir_abs_path = os.path.abspath(USER_PREDICTOR_DIR)
if predictor_dir_abs_path not in sys.path:
    sys.path.insert(0, predictor_dir_abs_path)
    logging.info(f"Added '{predictor_dir_abs_path}' to sys.path for predictor import")

try:
    from predictor import YoloCompatiblePredictor
    logging.info(f"Successfully imported predictor class '{YoloCompatiblePredictor.__name__}'")
except ImportError as e:
     logging.error(f"Import Error: {e}. Could not import predictor class from '{predictor_dir_abs_path}'.")
     exit()
# ==============================================================================


def build_predictor_image():
    """Builds and pushes the custom CPR predictor image using LocalModel."""
    logging.info(f"Building CPR container image using source dir: {USER_PREDICTOR_DIR}")
    if not os.path.exists(PREDICTOR_REQUIREMENTS_PATH): logging.error(f"Predictor requirements file not found: {PREDICTOR_REQUIREMENTS_PATH}"); return None
    predictor_script_path = os.path.join(USER_PREDICTOR_DIR, "predictor.py");
    if not os.path.exists(predictor_script_path): logging.error(f"Predictor script not found: {predictor_script_path}"); return None

    logging.info("--- Starting Docker build process via LocalModel.build_cpr_model ---")
    logging.warning("Requires Docker daemon running locally and authenticated to Artifact Registry.")
    try:
        logging.info(f"Using base image: {TRAIN_CONTAINER_URI}")
        local_model_obj = LocalModel.build_cpr_model(
            src_dir=USER_PREDICTOR_DIR,
            output_image_uri=CUSTOM_DEPLOY_CONTAINER_URI,
            predictor=YoloCompatiblePredictor, # Use imported class
            base_image=TRAIN_CONTAINER_URI,
            requirements_path=PREDICTOR_REQUIREMENTS_PATH,
        )
        logging.info(f"Custom CPR container build/push initiated successfully for {CUSTOM_DEPLOY_CONTAINER_URI}.")
        return local_model_obj
    except Exception as e:
        logging.error(f"Failed to build or push CPR container image: {e}", exc_info=True)
        return None


def run_training_job():
    """Defines and runs the training job, returning job object on simple success check."""
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)
    try:
        vertex_ai_dataset = aiplatform.ImageDataset(dataset_name=DATASET_RESOURCE_NAME)
        logging.info(f"Using dataset: {vertex_ai_dataset.display_name}")
    except Exception as e:
        logging.error(f"Failed get dataset reference: {e}", exc_info=True); return None

    job = aiplatform.CustomTrainingJob(
        display_name=JOB_DISPLAY_NAME, script_path=TRAINING_SCRIPT_PATH,
        container_uri=TRAIN_CONTAINER_URI, requirements=REQUIREMENTS_TRAIN,
        staging_bucket=STAGING_BUCKET,
    )

    logging.info(f"Submitting training job '{JOB_DISPLAY_NAME}'...")
    try:
        job.run(
            dataset=vertex_ai_dataset, annotation_schema_uri=ANNOTATION_SCHEMA_URI,
            args=TRAINING_ARGS, replica_count=REPLICA_COUNT,
            training_filter_split="labels.aiplatform.googleapis.com/ml_use=training",
            validation_filter_split="labels.aiplatform.googleapis.com/ml_use=validation",
            test_filter_split="labels.aiplatform.googleapis.com/ml_use=test",
            machine_type=MACHINE_TYPE, accelerator_type=ACCELERATOR_TYPE,
            accelerator_count=ACCELERATOR_COUNT, sync=True
        )
        logging.info("Training run call completed. Waiting briefly before checking state...")
        time.sleep(10) # Optional delay
        job_state = job.state
        logging.info(f"Training job finished with state reported by job object: {job_state}")

        succeeded_state_value = gca_job_state.JobState.JOB_STATE_SUCCEEDED.value
        if job_state == gca_job_state.JobState.JOB_STATE_SUCCEEDED or job_state == succeeded_state_value:
            logging.info("Training job SUCCEEDED based on job object state.")
            return job
        else:
             logging.error(f"Training job did not succeed (State: {job_state}).")
             job_error = getattr(job, 'error', None)
             if job_error: logging.error(f"Job error attribute: {job_error}")
             return None # Indicate failure

    except Exception as e:
        logging.error(f"The training job failed: {e}", exc_info=True)
        return None


def upload_trained_model(local_model_for_upload: LocalModel):
    """Uploads the model using the LocalModel object and predefined artifact path."""
    logging.info(f"Uploading model '{MODEL_DISPLAY_NAME}' from {GCS_MODEL_ARTIFACT_DIR} using LocalModel object...")
    try:
        aiplatform.init(project=PROJECT_ID, location=REGION) # Ensure init

        model = aiplatform.Model.upload(
            local_model=local_model_for_upload,
            display_name=MODEL_DISPLAY_NAME,
            artifact_uri=GCS_MODEL_ARTIFACT_DIR, # GCS path where train.py saved artifacts
            instance_schema_uri=INSTANCE_SCHEMA_URI,
            prediction_schema_uri=PREDICTION_SCHEMA_URI,
            sync=True
        )
        logging.info("Model uploaded successfully using local_model!")
        logging.info(f"Model Resource Name: {model.resource_name}")
        logging.info(f"View Model: https://console.cloud.google.com/vertex-ai/locations/{REGION}/models/{model.name}?project={PROJECT_ID}")
        return model
    except Exception as e_upload:
        logging.error(f"Failed to upload model using local_model: {e_upload}", exc_info=True)
        return None

# ==============================================================================
# Main Workflow Orchestration
# ==============================================================================
if __name__ == "__main__":
    # --- Pre-run Checks ---
    if not os.path.exists(TRAINING_SCRIPT_PATH): logging.error(f"Script '{TRAINING_SCRIPT_PATH}' not found."); exit()
    if not os.path.exists(os.path.join(USER_PREDICTOR_DIR, "predictor.py")): logging.error(f"Predictor script 'predictor.py' not found in '{USER_PREDICTOR_DIR}'."); exit()
    if not os.path.exists(PREDICTOR_REQUIREMENTS_PATH): logging.error(f"Predictor requirements '{PREDICTOR_REQUIREMENTS_PATH}' not found."); exit()
    logging.info("Local file checks passed.")
    logging.warning("Ensure Docker is running and configured: `gcloud auth configure-docker {REGION}-docker.pkg.dev`")
    logging.warning(f"Ensure Artifact Registry repo '{AR_REPOSITORY}' exists in {PROJECT_ID}/{REGION}.")
    success = ensure_artifact_registry_repo(
        project_id=PROJECT_ID,
        region=REGION,
        repository_id=AR_REPOSITORY
    )

    if success:
        logging.info(f"Artifact Registry repository '{AR_REPOSITORY}' is ready.")
    else:
        logging.error(f"Failed to ensure Artifact Registry repository '{AR_REPOSITORY}' is ready.")
        exit(1)
    copy_cmd= (
        f'gcloud storage cp -R gs://google-cloud-aiplatform/schema {STAGING_BUCKET}/'
    )
    logging.info("Copy schema from public gcs bucket.")
    schema_copy_success=os.system(copy_cmd)

    # --- Step 1: Run Training Job ---
    logging.info("--- Starting Step 1: Training Job ---")
    completed_job = run_training_job()
    if not completed_job:
        logging.error("Training Job failed. Exiting.")
        exit(1)
    logging.info("--- Step 1: Training Job Completed Successfully ---")


    # --- Step 2: Build the Custom CPR Image ---
    logging.info("--- Starting Step 2: Build Predictor Image ---")
    local_model = build_predictor_image() # Returns the LocalModel object
    if not local_model:
        logging.error("Failed to build CPR predictor image. Exiting.")
        exit(1)
    local_model.push_image()
    logging.info(f"--- Step 2: Build Predictor Image Initiated/Completed Successfully ---")


    # --- Step 3: Upload Model ---
    logging.info("--- Starting Step 3: Upload Model to Vertex AI Registry ---")
    uploaded_model = upload_trained_model(local_model_for_upload=local_model)
    if uploaded_model:
        logging.info("--- Step 3: Upload Model Completed Successfully ---")
        logging.info("Workflow completed successfully.")
    else:
        logging.error("--- Step 3: Upload Model Failed ---")
        logging.error("Workflow failed during model upload.")
        exit(1)

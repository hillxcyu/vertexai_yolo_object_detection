# train.py (Single Replica - Fixed Artifact Path + CPR Prep)
# Uses GCS mount, symlinks, user's trainer.RANK fix.
# Saves artifacts to a specific GCS URI passed via arguments.
# Saves class_map.json.

import argparse # <--- Added
import logging
import os
import shutil
import sys
import json
import yaml
import fnmatch
from pathlib import Path
from google.cloud import storage

try:
    from ultralytics import YOLO
    import ultralytics.engine.trainer as yolo_trainer
    import torch
except ImportError as e:
    logging.error(f"Import Error: {e}. Ensure 'ultralytics', 'torch' installed.")
    sys.exit(1)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')

# --- Env Vars (Only for reading data now) ---
AIP_DATA_FORMAT = os.environ.get("AIP_DATA_FORMAT")
AIP_TRAINING_DATA_URI = os.environ.get("AIP_TRAINING_DATA_URI")
AIP_VALIDATION_DATA_URI = os.environ.get("AIP_VALIDATION_DATA_URI")
AIP_TEST_DATA_URI = os.environ.get("AIP_TEST_DATA_URI")
# We will IGNORE os.environ.get("AIP_MODEL_DIR")

# --- Constants ---
LOCAL_DATA_ROOT = Path("/tmp/yolo_data")
LOCAL_DATASET_DIR = LOCAL_DATA_ROOT / "datasets"
LOCAL_YAML_PATH = LOCAL_DATA_ROOT / "data.yaml"

# --- Global Variable for Class Names ---
all_class_names_global = []

# --- GCS Client ---
try:
    storage_client = storage.Client()
    logging.info("GCS client initialized.")
except Exception as e:
    logging.error(f"Failed to initialize GCS client: {e}", exc_info=True)
    storage_client = None

# --- Helper Functions (parse_gcs_pattern - unchanged) ---
def parse_gcs_pattern(gcs_pattern):
    # ... (keep the existing function) ...
    if not gcs_pattern or not gcs_pattern.startswith("gs://"):
        if gcs_pattern.startswith("/gcs/"): parts = Path(gcs_pattern).parts; bucket_name=parts[2]; prefix="/".join(parts[3:-1])+'/' if len(parts)>4 else "" if len(parts)==3 else "/".join(parts[3:-1]); pattern=parts[-1]; return bucket_name, prefix, pattern
        raise ValueError(f"Invalid GCS pattern format: {gcs_pattern}.")
    path_parts = gcs_pattern.replace("gs://", "").split("/", 1); bucket_name=path_parts[0]; full_path_pattern=path_parts[1] if len(path_parts)>1 else ""
    if not full_path_pattern or full_path_pattern.endswith('/'): prefix=full_path_pattern; pattern='*'
    elif '/' in full_path_pattern: prefix=str(Path(full_path_pattern).parent)+'/'; pattern=Path(full_path_pattern).name
    else: prefix=""; pattern=full_path_pattern
    logging.debug(f"Parsed GCS pattern: bucket='{bucket_name}', prefix='{prefix}', pattern='{pattern}'")
    return bucket_name, prefix, pattern


# --- Data Processing Function (process_jsonl_manifests - unchanged) ---
# Uses /gcs/ mount and symlinks. Modifies global all_class_names_global list.
def process_jsonl_manifests(jsonl_gcs_pattern, split_name, class_name_to_id):
    # ... (keep the existing function - identical to previous version) ...
    global all_class_names_global
    logging.info(f"Processing {split_name} data using pattern: {jsonl_gcs_pattern}")
    if not jsonl_gcs_pattern: logging.warning(f"No GCS URI pattern for {split_name}. Skipping."); return None
    local_split_dir = LOCAL_DATASET_DIR / split_name; local_images_dir = local_split_dir / "images"; local_labels_dir = local_split_dir / "labels"
    local_images_dir.mkdir(parents=True, exist_ok=True); local_labels_dir.mkdir(parents=True, exist_ok=True)
    processed_image_count = 0; manifest_files_processed = 0; processed_image_gcs_uris = set()
    manifest_blob_names = []; bucket_name = None
    try: # Find manifest files (wildcard or single)
        if '*' in jsonl_gcs_pattern or '?' in jsonl_gcs_pattern or '[' in jsonl_gcs_pattern:
            if not storage_client: raise ConnectionError(f"GCS client needed for wildcard '{jsonl_gcs_pattern}'")
            bucket_name, prefix, pattern = parse_gcs_pattern(jsonl_gcs_pattern)
            logging.info(f"Listing blobs in gs://{bucket_name}/{prefix} matching '{pattern}'...")
            all_blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
            matching_blobs = [ blob for blob in all_blobs if fnmatch.fnmatch(Path(blob.name).name, pattern) ]
            if not matching_blobs: logging.warning(f"No manifests found for '{pattern}' in gs://{bucket_name}/{prefix}"); return None
            manifest_blob_names = [blob.name for blob in matching_blobs]; logging.info(f"Found {len(manifest_blob_names)} manifest(s).")
        else:
            bucket_name, blob_prefix, blob_pattern = parse_gcs_pattern(jsonl_gcs_pattern); single_blob_name = blob_prefix + blob_pattern
            if not single_blob_name: logging.error(f"Could not determine blob name from {jsonl_gcs_pattern}"); return None
            manifest_blob_names = [single_blob_name]; logging.info(f"Processing single manifest: gs://{bucket_name}/{single_blob_name}")

        for i, blob_name in enumerate(manifest_blob_names): # Process each manifest
            manifest_files_processed += 1; manifest_fuse_path = Path(f"/gcs/{bucket_name}/{blob_name}")
            logging.info(f"Reading manifest shard {i+1}/{len(manifest_blob_names)} via path: {manifest_fuse_path}")
            if not manifest_fuse_path.is_file(): logging.error(f"Manifest file not found at {manifest_fuse_path}. Skipping shard."); continue
            try:
                with open(manifest_fuse_path, 'r') as f_shard:
                    for line_num, line in enumerate(f_shard, 1): # Process each line
                        try:
                             data = json.loads(line); image_gcs_uri = data.get("imageGcsUri")
                             if not image_gcs_uri: logging.warning(f"Manifest {blob_name}, L{line_num}: Skipping, missing 'imageGcsUri'."); continue
                             if image_gcs_uri in processed_image_gcs_uris: logging.debug(f"Img {Path(image_gcs_uri).name} processed. Skipping."); continue
                             try: # Image Handling (get /gcs/ path, create symlink)
                                 img_bucket, img_prefix, img_filename_pattern = parse_gcs_pattern(image_gcs_uri); img_full_blob_path = img_prefix + img_filename_pattern
                                 image_fuse_path = Path(f"/gcs/{img_bucket}/{img_full_blob_path}")
                                 if not image_fuse_path.is_file(): logging.error(f"Img not found at {image_fuse_path} (from {image_gcs_uri}). Skipping."); continue
                                 image_filename = image_fuse_path.name; local_image_symlink = local_images_dir / image_filename; local_label_path = local_labels_dir / f"{local_image_symlink.stem}.txt"
                                 try: # Symlink creation
                                     if local_image_symlink.exists() or local_image_symlink.is_symlink(): os.remove(local_image_symlink)
                                     os.symlink(str(image_fuse_path.resolve()), local_image_symlink); logging.debug(f"Created symlink: {local_image_symlink} -> {image_fuse_path}")
                                 except OSError as e_symlink: logging.error(f"Failed to create symlink {local_image_symlink}: {e_symlink}", exc_info=True); continue
                             except ValueError as e_parse_img: logging.error(f"Manifest {blob_name}, L{line_num}: Error parsing img URI '{image_gcs_uri}': {e_parse_img}. Skipping."); continue
                             # Annotation processing
                             annotations = data.get("boundingBoxAnnotations", []); yolo_labels = []
                             for ann in annotations:
                                 display_name = ann.get("displayName"); class_id = -1
                                 if display_name:
                                      if display_name not in class_name_to_id: new_id = len(all_class_names_global); class_name_to_id[display_name] = new_id; all_class_names_global.append(display_name); logging.info(f"New class: '{display_name}' ID: {new_id}"); class_id = new_id
                                      else: class_id = class_name_to_id[display_name]
                                 x_min, x_max, y_min, y_max = ann.get("xMin"), ann.get("xMax"), ann.get("yMin"), ann.get("yMax")
                                 if class_id !=-1 and None not in [x_min, x_max, y_min, y_max]:
                                      cx=max(0.0,min(1.0,(x_min+x_max)/2.0)); cy=max(0.0,min(1.0,(y_min+y_max)/2.0)); w=max(0.0,min(1.0,x_max-x_min)); h=max(0.0,min(1.0,y_max-y_min))
                                      yolo_labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                             with open(local_label_path, 'w') as label_file: label_file.write("\n".join(yolo_labels))
                             processed_image_gcs_uris.add(image_gcs_uri); processed_image_count += 1
                             if processed_image_count % 500 == 0: logging.info(f"Processed {processed_image_count} images across shards for {split_name}...")
                        except json.JSONDecodeError: logging.error(f"Manifest {blob_name}, L{line_num}: Invalid JSON.")
                        except Exception as e_line: logging.error(f"Manifest {blob_name}, L{line_num}: Error - {e_line}", exc_info=True)
            except IOError as e_io: logging.error(f"Error reading {manifest_fuse_path}: {e_io}", exc_info=True)
            except Exception as e_shard: logging.error(f"Error processing {manifest_fuse_path}: {e_shard}", exc_info=True)
        logging.info(f"Finished processing {manifest_files_processed} manifest(s), {processed_image_count} images symlinked for {split_name}.")
        return str(local_images_dir.resolve())
    except Exception as e: logging.error(f"Failed processing {split_name} pattern {jsonl_gcs_pattern}: {e}", exc_info=True); return None


# --- Main Data Preparation Orchestration Function ---
# Calls process_jsonl_manifests and generates data.yaml pointing to local structure
def prepare_data_and_get_yaml():
    """Orchestrates data preparation using /gcs/ mount and symlinks."""
    global all_class_names_global
    logging.info("--- Starting Data Preparation (Using GCS Mount & Symlinks) ---")
    if LOCAL_DATA_ROOT.exists(): logging.warning(f"Removing previous data at {LOCAL_DATA_ROOT}."); shutil.rmtree(LOCAL_DATA_ROOT, ignore_errors=True)
    try: LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as e: logging.error(f"Could not create {LOCAL_DATA_ROOT}: {e}", exc_info=True); raise
    uses_wildcards = any(uri and ('*' in uri or '?' in uri or '[' in uri) for uri in [AIP_TRAINING_DATA_URI, AIP_VALIDATION_DATA_URI, AIP_TEST_DATA_URI])
    if uses_wildcards and not storage_client: raise ConnectionError("GCS client needed for wildcards, but failed.")
    logging.info(f"Data format specified by Vertex AI: '{AIP_DATA_FORMAT}'")
    class_name_to_id = {}; all_class_names_global.clear() # Clear global list
    train_img_dir_local = process_jsonl_manifests(AIP_TRAINING_DATA_URI, "train", class_name_to_id)
    val_img_dir_local = process_jsonl_manifests(AIP_VALIDATION_DATA_URI, "val", class_name_to_id)
    test_img_dir_local = process_jsonl_manifests(AIP_TEST_DATA_URI, "test", class_name_to_id)
    logging.info("Generating data.yaml file...")
    if not train_img_dir_local: raise FileNotFoundError("Training data processing failed.")
    if not val_img_dir_local: logging.warning("Validation data processing failed or not provided.")
    yaml_content = {'path': str(LOCAL_DATASET_DIR.resolve()), 'train': str(Path(train_img_dir_local).resolve()), 'val': str(Path(val_img_dir_local).resolve()) if val_img_dir_local else None, 'test': str(Path(test_img_dir_local).resolve()) if test_img_dir_local else None, 'names': {i: name for i, name in enumerate(all_class_names_global)}}
    if yaml_content['val'] is None: del yaml_content['val']
    if yaml_content['test'] is None: del yaml_content['test']
    logging.info(f"Final data.yaml content:\n{yaml.dump(yaml_content, default_flow_style=False)}")
    try:
        with open(LOCAL_YAML_PATH, 'w') as f_yaml: yaml.dump(yaml_content, f_yaml, default_flow_style=False, sort_keys=False)
        logging.info(f"Successfully created data.yaml at: {LOCAL_YAML_PATH}")
    except IOError as e: logging.error(f"Failed to write {LOCAL_YAML_PATH}: {e}", exc_info=True); raise
    logging.info("--- Data Preparation Finished ---")
    return str(LOCAL_YAML_PATH.resolve())


# --- Main Training Execution ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Vertex AI YOLO Single Replica Training Script")
    # Add argument for the fixed output artifact URI
    parser.add_argument('--output-artifact-uri', type=str, required=True,
                        help='GCS URI where final model artifacts should be saved (e.g., gs://bucket/path/)')
    parser.add_argument('--model_variant', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--project', type=str, default='yolo_training')
    parser.add_argument('--name', type=str, default='vertex_ai_run_single')
    args = parser.parse_args()

    logging.info(f"Starting YOLO training script execution (Single Replica)...")
    logging.info(f"Received script arguments: {args}")
    # Log the TARGET artifact location, NOT AIP_MODEL_DIR
    logging.info(f"Target artifact output location: {args.output_artifact_uri}")

    data_yaml_path = None
    try:
        data_yaml_path = prepare_data_and_get_yaml() # Populates global all_class_names_global
        logging.info(f"Data preparation complete. Using data configuration: {data_yaml_path}")
    except Exception as e:
        logging.error(f"Critical error during data preparation: {e}", exc_info=True)
        sys.exit("Data preparation failed. Exiting.")

    # --- Model Training ---
    final_model_path = None
    run_dir = None
    try:
        # --- User's Fix for set_epoch error ---
        logging.info("Applying trainer.RANK = -1 fix for potential DDP issue...")
        yolo_trainer.RANK = -1
        yolo_trainer.LOCAL_RANK = -1
        # --- End Fix ---

        logging.info(f"Loading base YOLO model: {args.model_variant}")
        model = YOLO(args.model_variant)

        logging.info(f"Starting YOLO training with: epochs={args.epochs}, batch={args.batch_size}, imgsz={args.img_size}")
        # Train with explicit device, rank override no longer needed here
        results = model.train(
            data=data_yaml_path, epochs=args.epochs, batch=args.batch_size,
            imgsz=args.img_size, project=args.project, name=args.name, exist_ok=True,
            plots=True,
            # scale=0.75,
            # shear=5,
            device=0,     # Explicitly set target device
        )
        logging.info("YOLO training completed.")

        # Find best/last weights
        run_dir = Path(results.save_dir)
        weights_dir = run_dir / 'weights'
        best_model_path_local = weights_dir / 'best.pt'
        last_model_path_local = weights_dir / 'last.pt'
        if best_model_path_local.exists(): final_model_path = best_model_path_local
        elif last_model_path_local.exists(): final_model_path = last_model_path_local
        else: raise FileNotFoundError(f"No model weights found in {weights_dir}")
        logging.info(f"Model weights found locally: {final_model_path}")

    except Exception as e_train:
        logging.error(f"An error occurred during model training: {e_train}", exc_info=True)
        sys.exit("Model training failed. Exiting.")

    # --- Save Artifacts to GCS (using --output-artifact-uri) ---
    output_uri = args.output_artifact_uri
    if output_uri and output_uri.startswith("gs://"):
        logging.info(f"Preparing to save artifacts to specified GCS path: {output_uri}")
        if not storage_client:
            logging.error("Cannot save artifacts: GCS client not initialized.")
        elif not final_model_path or not final_model_path.exists():
             logging.error("Final model path not found. Skipping artifact upload.")
        else:
            try:
                # Use the provided URI directly
                model_bucket_name, model_blob_prefix = parse_gcs_pattern(output_uri)[0:2]
                # Ensure prefix ends with / if it's not empty (acting as a directory)
                if model_blob_prefix and not model_blob_prefix.endswith('/'):
                     model_blob_prefix += '/'
                model_bucket = storage_client.bucket(model_bucket_name)

                # 1. Upload primary weights
                model_filename = final_model_path.name
                blob_model_path = f"{model_blob_prefix}{model_filename}"
                blob_model = model_bucket.blob(blob_model_path)
                blob_model.upload_from_filename(str(final_model_path))
                logging.info(f"Uploaded primary weights '{model_filename}' to gs://{model_bucket_name}/{blob_model_path}")

                # 2. Save and Upload Class Map (using global all_class_names_global)
                if all_class_names_global is not None:
                    class_map = {i: name for i, name in enumerate(all_class_names_global)}
                    class_map_filename = "class_map.json"
                    local_class_map_path = run_dir / class_map_filename if run_dir else LOCAL_DATA_ROOT / class_map_filename
                    try:
                        with open(local_class_map_path, 'w') as f_map: json.dump(class_map, f_map, indent=2)
                        logging.info(f"Saved class mapping locally to {local_class_map_path}")
                        blob_map_path = f"{model_blob_prefix}{class_map_filename}"
                        blob_map = model_bucket.blob(blob_map_path)
                        blob_map.upload_from_filename(str(local_class_map_path))
                        logging.info(f"Uploaded class map '{class_map_filename}' to gs://{model_bucket_name}/{blob_map_path}")
                    except Exception as e_map: logging.warning(f"Failed to save or upload class map: {e_map}", exc_info=True)
                else: logging.warning("Class name list empty. Cannot save class map.")

                # 3. Upload optional artifacts (results.csv, plots)
                logging.info(f"Attempting to upload other artifact files from local run directory: {run_dir}")
                if run_dir and run_dir.is_dir(): # Check if run_dir exists and is a directory
                    uploaded_count = 0
                    for item_path in run_dir.iterdir(): # Iterate through items in run_dir
                        if item_path.is_file(): # Check if it is a file (skip directories like 'weights')
                            try:
                                blob_artifact_path = f"{model_blob_prefix}{item_path.name}"
                                blob_artifact = model_bucket.blob(blob_artifact_path)
                                blob_artifact.upload_from_filename(str(item_path))
                                logging.info(f"Uploaded artifact file: '{item_path.name}'")
                                uploaded_count += 1
                            except Exception as e_upload_artifact:
                                logging.warning(f"Failed to upload artifact file '{item_path.name}': {e_upload_artifact}")
                    logging.info(f"Uploaded {uploaded_count} additional artifact files from {run_dir}")
                else:
                    logging.warning(f"Local run directory '{run_dir}' not found or invalid, cannot upload optional artifacts.")

                # # 4. Optional ONNX Export/Upload
                # try:
                #     logging.info("Attempting ONNX export...")
                #     export_model = YOLO(final_model_path)
                #     onnx_path_local = Path(export_model.export(format='onnx', imgsz=args.img_size))
                #     if onnx_path_local.exists():
                #          blob_onnx_path = f"{model_blob_prefix}{onnx_path_local.name}"
                #          blob_onnx = model_bucket.blob(blob_onnx_path)
                #          blob_onnx.upload_from_filename(str(onnx_path_local)); logging.info(f"Uploaded ONNX model '{onnx_path_local.name}'")
                # except Exception as e_export:
                #     logging.warning(f"Failed ONNX export/upload: {e_export}", exc_info=True)

            except Exception as e_save:
                logging.error(f"Failed to save artifacts to {output_uri}: {e_save}", exc_info=True)
    else:
        logging.warning("No valid --output-artifact-uri provided or it's not a GCS path. Skipping GCS upload.")
        if run_dir: logging.info(f"Local model artifacts are available in: {run_dir}")

    logging.info("YOLO training script finished successfully.")
    sys.exit(0)

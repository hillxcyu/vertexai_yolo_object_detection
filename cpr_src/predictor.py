# predictor.py (Using CWD for artifacts - Simpler Load Method)
import os
import base64
import io
import logging
import json
from pathlib import Path # Keep Path for checks/loading
from typing import Any, Dict, List

# Third-party libraries
import numpy as np
from PIL import Image

# Vertex AI SDK imports
from google.cloud import aiplatform
try:
    from google.cloud.aiplatform.prediction.predictor import Predictor
    from google.cloud.aiplatform.utils import prediction_utils # Correct import path
except ImportError as e:
    logging.error(f"Import Error: {e}. Ensure google-cloud-aiplatform SDK is up-to-date.")
    Predictor = object # Basic fallback
    raise e

# Ultralytics and Torch imports
try:
    from ultralytics import YOLO
    import torch
except ImportError as e:
     logging.error(f"Import Error: {e}. Ensure 'ultralytics' and 'torch' installed.")
     raise e

logging.basicConfig(level=logging.INFO)

class YoloCompatiblePredictor(Predictor):
    """
    CPR Predictor class for YOLO models, compatible with AutoML Obj Det format.
    Uses prediction_utils.download_model_artifacts() which downloads to CWD.
    """
    def __init__(self):
        """Basic initializer."""
        self._model = None
        self._class_map = None
        self._device = None
        logging.info("Predictor instance created. Call load() to initialize model.")

    def load(self, artifacts_uri: str) -> None:
        """
        Loads artifacts using download_model_artifacts helper into the CWD.

        Args:
            artifacts_uri (str): The GCS URI (gs://...) directory containing model artifacts.
        """
        logging.info(f"Starting artifact loading from: {artifacts_uri}")
        if not artifacts_uri: raise ValueError("Invalid artifacts_uri provided.")

        # --- Define expected local filenames relative to CWD ---
        model_filename = "best.pt"
        class_map_filename = "class_map.json"
        # Use Path objects relative to the current directory
        local_model_path = Path(model_filename)
        local_class_map_path = Path(class_map_filename)

        # --- Use the SDK helper function to download to CWD ---
        logging.info(f"Calling prediction_utils.download_model_artifacts from {artifacts_uri} (downloads to CWD)...")
        try:
            # This function downloads contents of artifacts_uri into the CWD
            prediction_utils.download_model_artifacts(artifacts_uri)
            cwd = Path.cwd() # Get current working directory for logging
            logging.info(f"Finished download_model_artifacts. Checking for files in CWD ({cwd})...")

            # Verify files exist in CWD after download
            if not local_model_path.exists():
                 logging.error(f"Contents of CWD ({cwd}): {list(cwd.iterdir())}")
                 raise FileNotFoundError(f"Model file '{model_filename}' not found in CWD {cwd} after download.")
            if not local_class_map_path.exists():
                 logging.error(f"Contents of CWD ({cwd}): {list(cwd.iterdir())}")
                 raise FileNotFoundError(f"Class map file '{class_map_filename}' not found in CWD {cwd} after download.")

        except Exception as e:
            logging.error(f"Failed during artifact download or file check: {e}", exc_info=True)
            raise

        # --- Load the downloaded artifacts directly from CWD ---
        self._device = 0 if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading YOLO model from './{local_model_path}' onto device: {self._device}")
        self._model = YOLO(local_model_path) # Load using relative path

        logging.info(f"Loading class map from './{local_class_map_path}'...")
        with open(local_class_map_path, 'r') as f:
            self._class_map = {int(k): v for k, v in json.load(f).items()}
        logging.info(f"Class map loaded: {self._class_map}")

        logging.info("Predictor loaded successfully.")

    # preprocess, predict, postprocess methods remain unchanged
    def preprocess(self, prediction_input: Dict) -> Dict:
        # ...(Implementation remains the same)...
        instances = prediction_input.get("instances"); parameters = prediction_input.get("parameters", {})
        if instances is None: raise ValueError("Request missing 'instances'.")
        if not isinstance(instances, list): raise ValueError("'instances' must be a list.")
        images = []
        for i, instance in enumerate(instances):
            if not isinstance(instance, dict) or "content" not in instance: logging.warning(f"Instance {i}: Invalid format. Skipping."); images.append(None); continue
            try: image_bytes=base64.b64decode(instance["content"]); image=Image.open(io.BytesIO(image_bytes)).convert('RGB'); images.append(image)
            except Exception as e: logging.warning(f"Instance {i}: Failed decode/load: {e}. Skipping."); images.append(None)
        return {"pil_images": images, "parameters": parameters}

    def predict(self, instances: Dict) -> Dict:
        # ...(Implementation remains the same)...
        if not self._model: raise RuntimeError("Model not loaded.")
        pil_images = instances.get("pil_images", []); parameters = instances.get("parameters", {})
        valid_images = [img for img in pil_images if img is not None]; original_indices = [i for i, img in enumerate(pil_images) if img is not None]
        final_results = [None] * len(pil_images)
        if valid_images:
             logging.info(f"Running prediction on {len(valid_images)} valid image(s)...")
             yolo_results_valid = self._model.predict(valid_images, device=self._device, verbose=False)
             logging.info("Prediction finished.")
             for i, result in enumerate(yolo_results_valid): original_index = original_indices[i]; final_results[original_index] = result
        else: logging.info("No valid images found.")
        return {"yolo_results": final_results, "parameters": parameters}

    def postprocess(self, prediction_results: Dict) -> Dict:
        # ...(Implementation remains the same)...
        if not self._class_map: raise RuntimeError("Class map not loaded.")
        yolo_results = prediction_results.get("yolo_results", []); parameters = prediction_results.get("parameters", {})
        logging.info("Postprocessing results...")
        predictions_list = []
        confidence_threshold = parameters.get("confidenceThreshold", 0.5); max_predictions = parameters.get("maxPredictions", 100)
        for result in yolo_results:
            image_predictions = {"displayNames": [], "confidences": [], "bboxes": []}
            if result is None: predictions_list.append(image_predictions); continue
            boxes = result.boxes.cpu().numpy(); count = 0
            sorted_indices = np.argsort(boxes.conf)[::-1]
            for index in sorted_indices:
                if count >= max_predictions: break
                box_data = boxes[index]; confidence = float(box_data.conf)
                if confidence >= confidence_threshold:
                    class_id = int(box_data.cls); class_name = self._class_map.get(class_id, f"unknown_{class_id}"); box_coords_norm = box_data.xyxyn[0]
                    bbox_formatted = [float(box_coords_norm[0]), float(box_coords_norm[2]), float(box_coords_norm[1]), float(box_coords_norm[3])]
                    image_predictions["displayNames"].append(class_name); image_predictions["confidences"].append(confidence); image_predictions["bboxes"].append(bbox_formatted); count += 1
            predictions_list.append(image_predictions)
        logging.info("Postprocessing finished.")
        return {"predictions": predictions_list}

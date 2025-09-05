# create_AR.py
# Standalone script to check for and create a Google Artifact Registry repository for Docker images.

import logging
import argparse
try:
    from google.cloud import artifactregistry_v1
    from google.api_core import exceptions as api_exceptions
except ImportError:
    logging.error("Please install required libraries: `pip install google-cloud-artifact-registry google-api-core`")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def ensure_artifact_registry_repo(project_id: str, region: str, repository_id: str) -> bool:
    """
    Checks if an Artifact Registry repository exists, creates it if not.

    Args:
        project_id: Google Cloud project ID.
        region: Google Cloud region for the repository.
        repository_id: The desired ID for the Docker repository.

    Returns:
        True if the repository exists or was created successfully, False otherwise.
    """
    logging.info(f"Checking for Artifact Registry repository '{repository_id}' in {region}...")
    client = artifactregistry_v1.ArtifactRegistryClient()
    parent = f"projects/{project_id}/locations/{region}"
    repo_name_full = f"{parent}/repositories/{repository_id}"

    try:
        # Check if repository exists
        client.get_repository(name=repo_name_full)
        logging.info(f"Repository '{repository_id}' already exists in {region}.")
        return True
    except api_exceptions.NotFound:
        # Repository not found, attempt to create it
        logging.info(f"Repository '{repository_id}' not found. Attempting to create...")
        repository_obj = artifactregistry_v1.Repository(
            name=repo_name_full, # Name is set within the object for creation
            format_=artifactregistry_v1.Repository.Format.DOCKER, # Specify Docker format
            description="Repository for Custom Prediction Routine images", # Optional description
        )
        try:
            operation = client.create_repository(
                parent=parent,
                repository_id=repository_id,
                repository=repository_obj,
            )
            logging.info(f"Create repository operation initiated: {operation.operation.name}. Waiting...")
            # Wait for the operation to complete (adjust timeout if needed, e.g., 300 seconds)
            operation.result(timeout=300)
            logging.info(f"Successfully created repository '{repository_id}' in {region}.")
            return True
        except Exception as create_error:
            logging.error(f"Failed to create repository '{repository_id}': {create_error}", exc_info=True)
            logging.error("Please ensure you have 'artifactregistry.repositories.create' permission.")
            return False
    except Exception as get_error:
        # Handle errors during the get_repository call
        logging.error(f"Failed to check for repository '{repository_id}': {get_error}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create or verify an Artifact Registry Docker repository.")
    parser.add_argument("--project-id", required=True, help="Google Cloud Project ID.")
    parser.add_argument("--region", required=True, help="Google Cloud region for the repository (e.g., us-central1).")
    parser.add_argument("--repository-name", required=True, help="Name (ID) of the Docker repository to create/verify.")

    args = parser.parse_args()

    logging.info("Script starting...")
    logging.warning("Ensure you have authenticated via `gcloud auth application-default login` "
                   "and have necessary Artifact Registry permissions.")

    success = ensure_artifact_registry_repo(
        project_id=args.project_id,
        region=args.region,
        repository_id=args.repository_name
    )

    if success:
        logging.info(f"Artifact Registry repository '{args.repository_name}' is ready.")
        exit(0)
    else:
        logging.error(f"Failed to ensure Artifact Registry repository '{args.repository_name}' is ready.")
        exit(1)

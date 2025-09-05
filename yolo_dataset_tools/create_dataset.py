PROJECT_ID = "vital-octagon-19612"  
LOCATION =  "us-central1"  
BUCKET_URI = "gs://hill_yolo_training"
dataset_name = "hill_stockout_dataset"
from google.cloud import aiplatform
IMPORT_FILE = f"{BUCKET_URI}/vision/vertex_ai_single_row_index.csv"

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
dataset = aiplatform.ImageDataset.create(
    display_name=dataset_name,
    gcs_source=[IMPORT_FILE],
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
)

print(dataset.resource_name)

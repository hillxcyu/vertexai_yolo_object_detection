## Requirement

Need docker to be usable.
Install dependencies using pip install -r requirement.txt under this directory.
The dataset need to be located in the same project as the training job. This is
because the training job utilized the "gcs fues-mount" feature where all buckets
were mounted under /gcs of the project, while buckets created by other projects 
will not be mounted.

## Settings.
Change gcp settings in launch_training_job.py, make sure your environment is 
setup for gcp

gcloud auth login
gcloud auth application-default login


## Run

python3 launch_training_job.py

## What it does?

1. train a yolo model using managed dataset, via vertexai custom_training.
2. Build a custom inference docker image using local_model custom prediction routine, 
do input/output conversion to guarantee the model output complies with Automl 
trained model.
3. Upload the model to vertexai model registry with correct prediction and 
instance schema

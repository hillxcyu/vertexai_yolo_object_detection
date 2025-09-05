import os
import pandas as pd
from PIL import Image

def convert_yolo_to_vertexai_single_row_csv(local_data_dir, output_csv_file, gcs_bucket_name,img_folder):
    """
    Converts a YOLO dataset to a Vertex AI CSV index file with a single row
    per bounding box annotation.

    Args:
        local_data_dir (str): Path to the root directory of the YOLO dataset.
                                 Expected structure:
                                 <local_data_dir>/
                                 ├── images/
                                 │   ├── train/
                                 │   ├── val/
                                 │   └── test/
                                 └── labels/
                                     ├── train/
                                     ├── val/
                                     └── test/
        output_csv_file (str): Path to the output Vertex AI CSV index file.
        gcs_bucket_name (str): The name of your Google Cloud Storage bucket
                                 where the images will be stored (e.g., 'hill_yolo_training').
    """
    data = []
    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(local_data_dir, 'images', split)
        label_dir = os.path.join(local_data_dir, 'labels', split)

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Warning: Skipping '{split}' split as image or label directory is missing.")
            continue

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in image_files:
            image_gcs_uri = f"gs://{gcs_bucket_name}/{img_folder}/{split}/{image_file}"  # Adjust path if needed
            label_name_without_ext = os.path.splitext(image_file)[0]
            label_path = os.path.join(label_dir, f'{label_name_without_ext}.txt')

            try:
                image = Image.open(os.path.join(image_dir, image_file))
                width, height = image.size
            except Exception as e:
                print(f"Error opening image {os.path.join(image_dir, image_file)}: {e}. Skipping.")
                continue

            if os.path.exists(label_path):
                with open(label_path, 'r') as labelfile:
                    for line in labelfile:
                        try:
                            class_id, x_center, y_center, w, h = map(float, line.strip().split())

                            x_min = x_center - w / 2
                            y_min = y_center - h / 2
                            x_max = x_center + w / 2
                            y_max = y_center + h / 2

                            data.append([
                                image_gcs_uri,
                                str(int(class_id)),  # Assuming class ID needs to be a string
                                f"{x_min:.6f}",
                                f"{y_min:.6f}",
                                "",
                                "",
                                f"{x_max:.6f}",
                                f"{y_max:.6f}",
                                "",
                                ""
                            ])
                        except ValueError:
                            print(f"Warning: Invalid line in label file {label_path}: {line.strip()}. Skipping.")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_file, index=False, header=False)
    print(f"Successfully created Vertex AI CSV index file: {output_csv_file}")

local_data_directory = 'stockout'  # Replace with the actual path to your local dataset
img_folder=local_data_directory
output_csv_file = 'vertex_ai_single_row_index.csv'
gcs_bucket = 'hill_yolo_training'  # Replace with your GCS bucket name

convert_yolo_to_vertexai_single_row_csv(local_data_directory, output_csv_file, gcs_bucket,img_folder)

print(f"\nVertex AI CSV index file saved to: {output_csv_file}")
print(f"Ensure your images are uploaded to 'gs://{gcs_bucket}/{img_folder}/'.")

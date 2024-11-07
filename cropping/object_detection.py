import shutil
from ultralytics import YOLO
from pathlib import Path
import os
from tqdm import tqdm
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run predictions and save annotations to a new folder.')
parser.add_argument('val_images_dir', type=str, help='Path to validation images directory')
parser.add_argument('target_dir', type=str, help='Path to target annotations directory')
parser.add_argument('model_weights', type=str, help='Path to model weights file (e.g., best.pt)')

args = parser.parse_args()

# Convert the input paths to Path objects
val_images_dir = Path(args.val_images_dir)
target_dir = Path(args.target_dir)
model_weights = args.model_weights

# Create a new directory for the predicted annotations within the target directory
new_annotations_dir = target_dir / 'annotations'
new_annotations_dir.mkdir(parents=True, exist_ok=True)

# Load the trained model from the specified weights file
model = YOLO(model_weights)

# Run predictions on the validation set using the trained model
results = model.predict(source=str(val_images_dir), save_txt=True, save_conf=True)

# Class list (if needed for future reference)
classList = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]

# Convert the results to the desired annotation format and save them
for result in tqdm(results):
    # Get the corresponding image file name and create the annotation file name
    img_name = Path(result.path).stem
    ann_file = new_annotations_dir / f"{img_name}.txt"
    
    # Write annotations in the required format
    with open(ann_file, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)  # Get class ID
            bbox = box.xyxy[0].tolist()  # Get bounding box in absolute pixel format (xyxy)
            
            # Create the line with class and bounding box values formatted for the annotation file
            line = f"{cls_id} {' '.join(f'{coord:.6f}' for coord in bbox)}\n"
            f.write(line)

print(f"Predictions saved as annotations in {new_annotations_dir}")

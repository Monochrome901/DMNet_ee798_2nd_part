import os
import json
import cv2
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Function to load annotations from txt file
def load_annotations_from_txt(annotation_file):
    bboxes = []
    with open(annotation_file, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if len(elements) == 5:
                class_id, x_min, y_min, x_max, y_max = map(float, elements)
                bbox = {
                    'class_id': int(class_id),
                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min]  # converting to width, height format
                }
                bboxes.append(bbox)
    return bboxes

# Function to draw bounding boxes on the image
def draw_bboxes(img, bboxes, color, label):
    for bbox in bboxes:
        class_id = bbox['class_id']
        x_min, y_min, width, height = bbox['bbox']
        x_max, y_max = int(x_min + width), int(y_min + height)
        cv2.rectangle(img, (int(x_min), int(y_min)), (x_max, y_max), color, 2)
        cv2.putText(img, f"{label}: {class_id}", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to visualize and compare predictions and ground truth
def visualize_comparison(json_file, img_folder, annot_folder, num_samples=5, output_folder=None):
    # Load the predicted results from the JSON file
    with open(json_file, 'r') as f:
        predicted_data = json.load(f)

    # Sort image names alphabetically
    sorted_img_names = sorted(predicted_data.keys())[:num_samples]

    for img_name in sorted_img_names:
        # Load image
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image {img_name} not found!")
            continue

        # Load ground truth annotations
        annot_path = os.path.join(annot_folder, img_name.replace('.jpg', '.txt'))
        if not os.path.exists(annot_path):
            print(f"Annotation file {annot_path} not found!")
            continue

        ground_truth_bboxes = load_annotations_from_txt(annot_path)
        predicted_bboxes = predicted_data[img_name]

        # Draw ground truth (green) and predicted (blue) bounding boxes
        img_with_bboxes = img.copy()
        draw_bboxes(img_with_bboxes, ground_truth_bboxes, (0, 255, 0), 'GT')  # Ground truth in green
        draw_bboxes(img_with_bboxes, predicted_bboxes, (255, 0, 0), 'Pred')  # Predictions in blue

        # Convert to RGB for matplotlib
        img_with_bboxes_rgb = cv2.cvtColor(img_with_bboxes, cv2.COLOR_BGR2RGB)

        # Display the image with bounding boxes
        plt.figure(figsize=(10, 8))
        plt.imshow(img_with_bboxes_rgb)
        plt.title(f"Image: {img_name}")
        plt.axis('off')
        plt.show()

        # Save the result image if output folder is specified
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_img_path = os.path.join(output_folder, f"comparison_{img_name}")
            cv2.imwrite(output_img_path, img_with_bboxes)

    print(f"Predictions for {num_samples} images have been saved in {output_folder}")

# Example usage
json_file = "final_fusion_result.json.bbox.json"  # Path to the final JSON file
img_folder = "dataset/Global/val/images"  # Folder containing the images
annot_folder = "dataset/Global/val/ground_annotations"  # Folder containing the actual annotations
output_folder = "output"  # Folder to save output images (optional)

# Clean output directory
output_dir = Path(output_folder)
if output_dir.exists():
    shutil.rmtree(output_dir)

# Visualize and compare
visualize_comparison(json_file, img_folder, annot_folder, num_samples=5, output_folder=output_folder)

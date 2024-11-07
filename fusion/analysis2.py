import json
import os

def save_results_in_coco_format(fusion_result_file, output_json_file):
    with open(fusion_result_file, 'r') as f:
        final_detection_by_image = json.load(f)  # Load the JSON content as a dictionary
    
    annotations = []  # This will store all detection results
    annotation_id = 1  # Each annotation needs a unique ID

    for img_id, (img_name, detections) in enumerate(final_detection_by_image.items()):
        # Add annotations for each detection
        for detection in detections:
            class_id = detection['class_id']
            x_min, y_min, width, height = detection['bbox']
            score = detection.get('score', 1.0)  # Use score from detection, default to 1.0 if not available
            
            annotation = {
                "image_id": img_id,  # ID of the image
                "category_id": class_id,  # Category of the object
                "bbox": [x_min, y_min, width, height],  # Bounding box coordinates
                "score": score  # Add the score field here
            }
            annotations.append(annotation)
            annotation_id += 1

    # Write the results to a JSON file in COCO prediction format (just a list of annotations)
    with open(output_json_file, 'w') as json_file:
        json.dump(annotations, json_file, indent=4)

import os
import json

def convert_txt_to_coco(annotation_dir, output_json_file):
    """
    Convert the .txt ground truth annotation files in the format 
    'x_min, y_min, width, height, _, class, _, _' to a COCO-style JSON file.
    """
    images = []
    annotations = []
    annotation_id = 1  # COCO format expects each annotation to have a unique ID

    # Iterate through all annotation files in the directory
    for img_id, ann_file in enumerate(os.listdir(annotation_dir)):
        if ann_file.endswith('.txt'):
            img_name = ann_file.replace('.txt', '.jpg')
            img_info = {
                "file_name": img_name,
                "id": img_id  # Ensure each image has a unique ID
            }
            images.append(img_info)

            with open(os.path.join(annotation_dir, ann_file), 'r') as f:
                for line in f:
                    elements = line.strip().split(',')  # Split by commas
                    if len(elements) >= 6:  # Ensure at least 6 values
                        x_min = float(elements[0])
                        y_min = float(elements[1])
                        width = float(elements[2])
                        height = float(elements[3])
                        class_id = int(elements[5])  # Extract class ID from position 5

                        annotation = {
                            "image_id": img_id,  # Match the image ID
                            "id": annotation_id,
                            "category_id": class_id,  # Use extracted class ID
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0  # Assuming all annotations are not crowd annotations
                        }
                        annotations.append(annotation)
                        annotation_id += 1

    # Define COCO format dictionary
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "pedestrian"},
            {"id": 1, "name": "people"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "car"},
            {"id": 4, "name": "van"},
            {"id": 5, "name": "truck"},
            {"id": 6, "name": "tricycle"},
            {"id": 7, "name": "awning-tricycle"},
            {"id": 8, "name": "bus"},
            {"id": 9, "name": "motor"}
        ]
    }

    # Write out to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)


# Call this function for ground truth
convert_txt_to_coco('dataset/Global/val/ground_annotations', 'ground_truth_annotations_coco.json')
save_results_in_coco_format('final_fusion_result.json.bbox.json', 'final_fusion_result_coco.json')

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load the ground truth and predicted annotations
ground_truth_file = 'ground_truth_annotations_coco.json'
predicted_file = 'final_fusion_result_coco.json'

# Load the ground truth and predicted JSON files
coco_gt = COCO(ground_truth_file)
coco_pred = coco_gt.loadRes(predicted_file)

# Initialize COCOeval object
coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

# Evaluate
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Print mAP
mAP = coco_eval.stats[0]
print(f"mAP: {mAP}")


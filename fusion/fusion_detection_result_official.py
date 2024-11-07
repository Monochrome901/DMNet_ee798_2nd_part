import os
import glob
import copy, cv2
import numpy as np
from tqdm import tqdm
# from plot_utils import overlay_func, overlay_bbox_img
from eval_utils import resize_bbox_to_original, wrap_initial_result, results2json, nms, class_wise_nms
import argparse

"""
Modified Code for DMnet, Global-local fusion detection with custom dataset
Author: Modified by [Your Name]
Dataset structure:
dataset/
-- Global/
---- val/
------ annotations/ (txt files)
------ images/ (jpg files)
-- Density/
---- val/
------ annotations/ (txt files)
------ images/ (jpg files)
Each .txt file contains bounding box annotations in the format:
class_id x_min y_min x_max y_max

Sample command line to run:
python fusion_detection_result.py --root_dir dataset --mode val

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='DMNet -- Global-local fusion detection')
    parser.add_argument('--root_dir', default="dataset",
                        help='the path for source data')
    parser.add_argument('--mode', default="val", help='Indicate if you are working on train/val/test set')
    parser.add_argument('--truncate_threshold', type=float, default=0,
                        help='Threshold defined to select the cropped region')
    parser.add_argument('--iou_threshold', type=float, default=0.7,
                        help='Iou Threshold defined to filter out bbox, recommended value: 0.7')
    parser.add_argument('--TopN', type=int, default=500,
                        help='Only keep TopN bboxes with highest score, default value 500')
    parser.add_argument('--show', action='store_true', help='Need to keep original image?')
    args = parser.parse_args()
    return args

def load_annotations(annotation_file):
    bboxes = []
    with open(annotation_file, 'r') as file:
        for line in file:
            elements = line.strip().split()
            if len(elements) == 5:
                class_id, x_min, y_min, x_max, y_max = map(float, elements)
                bbox = {
                    'class_id': int(class_id),
                    'bbox': [x_min, y_min, x_max - x_min, y_max - y_min]
                }
                bboxes.append(bbox)
    return bboxes
def overlay_func(img_pth, raw_anns, classList, truncate_threshold, exclude_region, show):
    """
    Overlay bounding boxes, category for each bounding boxes and separate by each density region
    :param img_pth: The path to input image
    :param raw_anns: annotations for the given image in the format [class_id, x_min, y_min, x_max, y_max]
    :param classList: List of categories for the dataset
    :param truncate_threshold: amount of pixels to help filter out bounding boxes that
                               are close to the boundary of the image
    :param exclude_region: The density regions to analyze
    :param show: Whether to show the overlay map
    :return: processed annotations
    """
    I = cv2.imread(img_pth)
    cp_I = I[:]
    anns = []
    
    if len(I.shape) == 3:
        img_height, img_width = I.shape[:-1]
    elif len(I.shape) == 2:
        img_height, img_width = I.shape
    else:
        print("Unable to determine image shape! Exiting...")
        exit(0)

    for bbox in raw_anns:
        class_id = int(bbox['class_id'])
        x_min, y_min, x_max, y_max = map(int, bbox['bbox'])
        
        # Apply truncation threshold to filter bounding boxes near image borders
        if truncate_threshold > 0:
            if not (truncate_threshold <= x_min < x_max <= img_width - truncate_threshold) or \
                   not (truncate_threshold <= y_min < y_max <= img_height - truncate_threshold):
                continue  # Skip bounding boxes that are too close to the image border
        
        text = classList[class_id]
        anns.append([class_id, x_min, y_min, x_max, y_max])
        
        if show:
            # Draw rectangle and label on the image
            cp_I = cv2.rectangle(cp_I, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness=2)
            cv2.putText(cp_I, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

    if show:
        if exclude_region:
            text = "density_crop"
            for coord in exclude_region:
                x_min, y_min, width, height = coord
                x_max = x_min + width
                y_max = y_min + height
                cp_I = cv2.rectangle(cp_I, (x_min, y_min), (x_max, y_max), (0, 255, 255), thickness=2)
                cv2.putText(cp_I, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

        cv2.imshow("overlay_image", cp_I)
        cv2.waitKey(0)

    return anns

# Other utility functions like parse_args(), load_annotations(), overlay_func() remain the same

if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    show = args.show
    root = args.root_dir
    truncate_threshold = args.truncate_threshold
    classList = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]

    img_path = os.path.join(root, "Global", mode, "images")
    dens_path = os.path.join(root, "Density", mode, "images")
    img_annot_path = os.path.join(root, "Global", mode, "annotations")
    dens_annot_path = os.path.join(root, "Density", mode, "annotations")

    img_list = glob.glob(f'{img_path}/*.jpg')
    dens_list = glob.glob(f'{dens_path}/*.jpg')
    assert len(img_list) > 0, "Failed to find any global images!"
    assert len(dens_list) > 0, "Failed to find any density images!"

    # Initialize image detection result
    final_detection_result = []
    img_fusion_result_collecter = []

    final_detection_by_image = {}  # Dictionary to store results for all images

    for img_file in tqdm(img_list, total=len(img_list)):
        img_name = os.path.basename(img_file)
        img_annotation_file = os.path.join(img_annot_path, img_name.replace('.jpg', '.txt'))
        current_global_img_bbox = load_annotations(img_annotation_file)

        # Visualization of global detection result
        overlay_func(img_file, current_global_img_bbox, classList, truncate_threshold, exclude_region=None, show=show)

        img_density_detection_result = []
        matched_dens_files = [f for f in dens_list if img_name in f]
        exclude_region = []

        for dens_file in matched_dens_files:
            dens_img_name = os.path.basename(dens_file)
            dens_annotation_file = os.path.join(dens_annot_path, dens_img_name.replace('.jpg', '.txt'))
            dens_annotations = load_annotations(dens_annotation_file)

            # Resize bounding boxes to original scale (global image)
            crop_img = cv2.imread(dens_file)
            crop_img_h, crop_img_w = crop_img.shape[:2]
            start_x, start_y = map(int, dens_img_name.split('_')[:2])
            crop_bbox_to_original = resize_bbox_to_original(dens_annotations, start_x, start_y)

            img_density_detection_result.extend(crop_bbox_to_original)

            exclude_region.append([start_x, start_y, crop_img_w, crop_img_h])

        # Overlay final result (fusion of global and density)
        img_initial_fusion_result = current_global_img_bbox + img_density_detection_result
        img_fusion_result_collecter.append(img_initial_fusion_result)

        overlay_func(img_file, img_initial_fusion_result, classList, truncate_threshold, exclude_region=None, show=show)

        # Perform class-wise NMS to fuse bboxes
        nms_preprocess = wrap_initial_result(img_initial_fusion_result)
        keep = class_wise_nms(nms_preprocess, args.iou_threshold, args.TopN)
        class_wise_nms_result = [img_initial_fusion_result[k] for k in keep]

        # Store final detection result for each image
        final_detection_by_image[img_name] = class_wise_nms_result

    # Save fusion detection result to output json file
    output_file = os.path.join("final_fusion_result.json")
    results2json(final_detection_by_image, out_file=output_file)

    print(f"Saved final fusion results to {output_file}")



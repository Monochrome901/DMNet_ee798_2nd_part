import os
import numpy as np
from terminaltables import AsciiTable

def resize_bbox_to_original(bboxes, start_x, start_y):
    """
    Given bboxes from density crops, cast back coordinates to original images.
    :param bboxes: Bounding boxes from density crops.
    :param start_x: The starting x position in original images.
    :param start_y: The starting y position in original images.
    :return: Scaled annotations with coordinates matching the original image.
    """
    modify_bbox = []
    for bbox in bboxes:
        coord = bbox["bbox"]
        # Adjust the bounding box coordinates back to the global image coordinates
        coord[0] += start_x
        coord[1] += start_y
        bbox["bbox"] = coord
        modify_bbox.append(bbox)
    return modify_bbox

def wrap_initial_result(img_initial_fusion_result):
    """
    Given img_initial_fusion_result, wrap it into a numpy array for NMS.
    :param img_initial_fusion_result: Raw annotations from initial data collection.
    :return: Numpy array for NMS.
    """
    nms_process_array = []
    for anno in img_initial_fusion_result:
        nms_process_array.append([anno['class_id']] + anno['bbox'])
    return np.array(nms_process_array)

def class_wise_nms(current_nms_target_col, thresh, TopN):
    """
    Class-wise Non-Maximum Suppression (NMS) function.
    :param current_nms_target_col: Array containing class-wise detections.
    :param thresh: IoU threshold for NMS.
    :param TopN: Keep Top N bounding boxes based on scores.
    :return: Indices of bounding boxes to keep.
    """
    bbox_id = np.arange(len(current_nms_target_col))
    truncate_result = current_nms_target_col.copy()
    categories = current_nms_target_col[:, 0]
    keep = []

    for category in set(categories):
        # Filter by category
        mask = categories == category
        current_nms_target = current_nms_target_col[mask]

        if len(current_nms_target) == 0:
            continue

        # Sort by scores (assuming scores are at index 2)
        scores = current_nms_target[:, 4]
        order = scores.argsort()[::-1]

        # Bounding box coordinates
        x1 = current_nms_target[:, 1]
        y1 = current_nms_target[:, 2]
        x2 = current_nms_target[:, 3]
        y2 = current_nms_target[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        while order.size > 0:
            i = order[0]
            keep.append(int(bbox_id[i]))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

    # Truncate to TopN if necessary
    if len(keep) > TopN:
        scores = truncate_result[keep, 4]
        keep = scores.argsort()[::-1][:TopN]

    return keep

import json

def results2json(json_results, out_file):
    """
    Save fused annotations to a JSON file.
    :param json_results: List of fused results.
    :param out_file: Path to output file.
    """
    # Convert results to JSON format
    with open(f'{out_file}.bbox.json', 'w') as f:
        json.dump(json_results, f, indent=4)  # Use indent for pretty printing


def nms(dets, thresh):
    """
    Non-Maximum Suppression (NMS) implementation.
    :param dets: Numpy array of bounding boxes and scores.
    :param thresh: IoU threshold for suppression.
    :return: Indices of bounding boxes to keep.
    """
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3] + x1
    y2 = dets[:, 4] + y1
    scores = dets[:, 5]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def evaluate_classwise(annotations, categories):
    """
    Evaluate results class-wise using AP (Average Precision).
    :param annotations: List of bounding boxes.
    :param categories: List of categories.
    :return: AP for each category.
    """
    category_results = {}
    for category_id in categories:
        # Filter annotations for the current category
        category_bboxes = [ann for ann in annotations if ann['class_id'] == category_id]
        
        if not category_bboxes:
            continue

        # Perform NMS for this category
        nms_target = wrap_initial_result(category_bboxes)
        keep = class_wise_nms(nms_target, thresh=0.5, TopN=500)

        category_results[category_id] = keep

    # Display results as table
    results_per_category = [(cat, len(category_results[cat])) for cat in category_results]
    headers = ['Category', 'AP']
    table_data = [headers] + results_per_category
    table = AsciiTable(table_data)
    print(table.table)

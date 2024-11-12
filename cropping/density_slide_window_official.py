from tqdm import tqdm
import glob
import os
import argparse
from plot_utils import overlay_image
from density_slide_utils import split_overlay_map, save_cropped_result
from eval_utils import measure_hit_rate
from density_slide_utils import calculate_avg_bbox_size  # Ensure this function calculates avg bbox size for one file

"""
Code for DMnet, density crops generation
Author: Changlin Li
Code revised on : 7/16/2020

Given dataset(train/val/test) generate density crops for given dataset.
Default format for source data: The input images are in jpg format and raw annotations are in txt format 
(Based on Visiondrone 2018/19/20 dataset)

The data should be arranged in following structure before you call any function within this script:
dataset(Train/val/test)
--------images
--------dens (short for density map)
--------Annotations (Optional, but not available only when you conduct inference steps)

Sample running command:
python density_slide_window_official.py . height_width threshold --output_folder output_folder --mode val
"""


def get_dynamic_threshold(annotation_file):
    """
    Determine a dynamic threshold based on the maximum average bounding box size
    from the 'annotations' folder. The threshold increases linearly from 0.02 to 0.1
    as max(width, height) changes from 0 to 100.
    
    :param annotation_file: Path to the annotation file in 'annotations'
    :return: Adjusted threshold value
    """
    avg_width, avg_height = calculate_avg_bbox_size(annotation_file)

    # Calculate max dimension of the average bounding box
    max_dimension = max(avg_width, avg_height)

    # Linearly interpolate threshold between 0.02 and 0.1
    threshold = 0.02 + (0.1 - 0.02) * (max_dimension / 100)
    
    # Clamp the threshold to the range [0.02, 0.1]
    threshold = max(0.02, min(0.1, threshold))
    
    return threshold



def measure_hit_rate_on_data(file_list, window_size, output_dir, mode="train"):
    """
    Serve as a function to measure how many bboxs we missed for DMNet. It helps estimate the performance of
    bounding boxes using 'ground_annotations'.
    :param file_list: List of image files
    :param window_size: Kernel size for sliding
    :param output_dir: Directory to save results
    :param mode: Dataset to use
    :return:
    """
    count_data = total_data = 0

    for file in tqdm(file_list, total=len(file_list)):
        overlay_map = overlay_image(file, window_size, 0.08, output_dir)
        result = split_overlay_map(overlay_map)
        annotation_file = file.replace("images", "ground_annotations").replace(".jpg", ".txt")
        count, total = measure_hit_rate(annotation_file, result, mode)
        count_data += count
        total_data += total

    print("Hit rate is: " + str(round(count_data / total_data * 100.0, 2)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='DMNet--Generate density crops from given density map')
    parser.add_argument('root_dir', default="dataset", help='The path for source data')
    parser.add_argument('window_size', help='The size of kernel, format: h_w')
    parser.add_argument('--output_folder', help='Directory to save generated images and annotations')
    parser.add_argument('--mode', default="val", help='Indicate if you are working on train/val/test set')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    mode = args.mode
    root_dir = args.root_dir
    folder_name = args.output_folder
    data_root = f'{root_dir}/{mode}'  # Set to dataset/val

    # Collect image and annotation files from dataset/val
    img_array = glob.glob(f'{data_root}/images/*.jpg')
    annotation_files = glob.glob(f'{data_root}/annotations/*.txt')  # For threshold calculation

    # Create output directories
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    # Window size
    window_size = args.window_size.split("_")
    window_size = (int(window_size[0]), int(window_size[1]))

    # Define output directories
    output_img_dir = os.path.join(folder_name, mode, "images")
    output_anno_dir = os.path.join(folder_name, mode, "annotations")

    # Ensure output directories exist
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_anno_dir, exist_ok=True)

    # Process each image
    for image_file in tqdm(img_array, total=len(img_array)):
        # Get corresponding annotation file for threshold calculation
        annotation_file = image_file.replace("images", "annotations").replace(".jpg", ".txt")

        # Dynamically calculate threshold
        threshold = get_dynamic_threshold(annotation_file)

        # Save cropped results using 'ground_annotations' for hit rate calculations
        save_cropped_result([image_file], window_size, threshold, None, output_img_dir, output_anno_dir, mode=mode)

    # Evaluate hit rate (optional step)
    # measure_hit_rate_on_data(img_array, window_size, output_img_dir, mode=mode)

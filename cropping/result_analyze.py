import glob
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

"""
Code for DMnet, conduct post analysis for generated density crops

**You don't need this script to use DMNet**

Author: Changlin Li
Code revised on : 7/18/2020

Given dataset(train/val/test) conduct post analysis for generated density crops
Default format for source data: The input images are in jpg format and raw annotations are in txt format 
(Based on Visiondrone 2018/19/20 dataset)

The data should be arranged in following structure before you call any function within this script:

Original_dataset(Train/val/test)
--------images
--------Annotations 

Density crops
--------images
--------Annotations 

Sample code to run:
python result_analyze.py data_folder crop_folder --mode val

"""


def compute_result_resolution(filepath, crop_filepath):
    """
    Given crop images (at most m*n splits, depend on how you split and whether u keep background crops or not)
    conduct analysis to see how large you crops from original image.
    This might be a good indicator to see if you hyper-parameters are good or not.
    If you crop something too large(almost is the original image, for example), then you may need to double check
    if papameters are suitable or maybe there are some bugs in your data.
    :param filepath: The original image to analysis
    :param crop_filepath: The crops made from original image
    :return: dict that records the largest percentage of crops compare with original images
    """
    # start from cropped data and scan them.
    # infer original data accordingly
    record = {}
    for file in tqdm(filepath, total=len(filepath)):
        img = cv2.imread(file)
        img_area = img.shape[0] * img.shape[1]
        crop_img_area = 0
        for crop_file in crop_filepath:
            if file.split("/")[-1] in crop_file:
                crop_img = cv2.imread(crop_file)
                crop_img_area = max(crop_img_area, crop_img.shape[0] * crop_img.shape[1])
        assert crop_img_area > 0, "this file has 0 area: " + str(file)
        record[file] = round(crop_img_area / img_area * 100.0, 2)
    return record

def compute_partition_distribution(filepath, crop_filepath):
    """
    Give an analysis of distributions for generated crops. How many of them are small/medium/large?
    Distribution can help you identify LOTS of possible issues.
    :param filepath: The original image to analysis
    :param crop_filepath: The crops made from original image
    :return: distribution
    """
    dist = []
    sparse, common, cluster = 0, 0, 0
    assert len(filepath) > 0, "No original images found"
    assert len(crop_filepath) > 0, "No crop images found"
    
    # Extract crop filenames, removing the first 4 underscores to get the original file name
    crop_filenames_clean = {('_'.join(crop_file.split('_')[4:])).replace('.jpg', '.txt'): crop_file for crop_file in crop_filepath}

    for file in tqdm(filepath, total=len(filepath)):
        original_filename = os.path.basename(file).replace('.jpg', '.txt')  # Get the original file name without the extension
        
        # Check if the annotation file exists for the original image
        anno_file = file.replace("images", "annotations").replace("jpg", "txt")
        
        if not os.path.exists(anno_file):
            print(f"Annotation file not found for {file}")
            continue
        
        with open(anno_file, "r") as fileloader:
            txt_list = fileloader.readlines()
        total_object = len(txt_list)
        
        if total_object == 0:
            print(f"No annotations found for {file}")
            continue
        
        found_match = False  # Flag to track if any crops matched this original file
        
        # Check if the original filename exists in the cleaned crop filenames
        if original_filename in crop_filenames_clean:
            found_match = True
            crop_file = crop_filenames_clean[original_filename]
            anno_crop_file = crop_file.replace("images", "annotations").replace("jpg", "txt")
            
            # Check if the annotation file exists for the crop
            if not os.path.exists(anno_crop_file):
                print(f"Annotation file not found for crop {crop_file}")
                continue
            
            with open(anno_crop_file, "r") as fileloader:
                txt_list_crop = fileloader.readlines()
            total_crop_object = len(txt_list_crop)
            ratio = round(total_crop_object / total_object, 2)
            print(ratio)
            if ratio < 0.33:
                sparse += 1
            elif 0.33 <= ratio < 0.65:
                common += 1
            elif 0.65 <= ratio <= 1.00:
                cluster += 1
            dist.append(ratio)
        
        if not found_match:
            print(f"No crop found for original image {file}")

    total_crops = sparse + common + cluster
    
    if total_crops > 0:
        print("The proportion of sparse crop is: ", round(sparse / total_crops, 2))
        print("The proportion of common crop is: ", round(common / total_crops, 2))
        print("The proportion of cluster crop is: ", round(cluster / total_crops, 2))
    else:
        print("No valid crops found for any images.")
    
    # Plot the distribution only if there are values
    if dist:
        plt.title("Distribution by annotation counts")
        plt.hist(dist, 100)
        plt.xlabel("percentage of objects per crops")
        plt.ylabel("count")
        plt.show()
    
    return dist




def parse_args():
    parser = argparse.ArgumentParser(
        description='DMNet--Image cropping result analysis')
    parser.add_argument('root_dir', default=".",
                        help='the path for source data')
    parser.add_argument('crop_root_dir', default=".",
                        help='the path for density crops data')
    parser.add_argument('--mode', default="train", help='Indicate if you are working on train/val/test set')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    mode = args.mode
    data_root = args.root_dir
    crop_data_root = args.crop_root_dir
    img_array = glob.glob(f'{data_root}/{mode}/images/*.jpg')
    crop_img_array = glob.glob(f'{crop_data_root}/{mode}/images/*.jpg')
    compute_partition_distribution(img_array, crop_img_array)

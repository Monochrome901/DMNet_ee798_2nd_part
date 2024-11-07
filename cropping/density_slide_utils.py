import cv2
import os
from anno_utils import format_label
from plot_utils import overlay_image
from tqdm import tqdm
import numpy as np
from scipy.ndimage import label


def calculate_avg_bbox_size(file_path):
    """
    Calculate the average bounding box size (width and height) for a single annotation file.
    :param file_path: Path to the annotation file
    :return: Tuple (avg_width, avg_height)
    """
    total_width, total_height, count = 0, 0, 0

    # Read the annotation file
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line to extract width and height
            parts = line.strip().split()
            if len(parts) == 5:  # Ensure valid format
                x_min = float(parts[1])
                y_min = float(parts[2])
                x_max = float(parts[3])
                y_max = float(parts[4])

                width = (x_max - x_min)
                height = (y_max - y_min)
                total_width += width
                total_height += height
                count += 1

    # Calculate the average width and height
    if count > 0:
        avg_width = total_width / count
        avg_height = total_height / count
    else:
        avg_width, avg_height = 0, 0  # No bounding boxes in this file

    return avg_width, avg_height



def split_overlay_map(grid):
    """
    Conduct eight-connected-component methods on grid to connect all pixels within a similar region.
    Uses `scipy.ndimage.label` for faster connected component analysis.
    :param grid: density mask to connect
    :return: merged regions for cropping purpose
    """
    if grid is None or grid.shape[0] == 0:
        grid = np.array(grid)
    labeled_array, num_features = label(grid == 255)

    results = []
    for region in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == region)
        if coords.size == 0:
            continue
        top, left = coords.min(axis=0)
        bot, right = coords.max(axis=0)
        pixel_area = (right - left) * (bot - top)
        results.append([region, (left, top), (right, bot), pixel_area])

    return results

# def split_overlay_map(grid):
#     """
#     Conduct eight-connected-component methods on grid to connnect all pixel within the similar region
#     :param grid: desnity mask to connect
#     :return: merged regions for cropping purpose
#     """
#     if grid is None or grid[0] is None:
#         return 0
#     # Assume overlap_map is a 2d feature map
#     m, n = grid.shape
#     visit = [[0 for _ in range(n)] for _ in range(m)]
#     count, queue, result = 0, [], []
#     for i in range(m):
#         for j in range(n):
#             if not visit[i][j]:
#                 if grid[i][j] == 0:
#                     visit[i][j] = 1
#                     continue
#                 queue.append([i, j])
#                 top, left = float("inf"), float("inf")
#                 bot, right = float("-inf"), float("-inf")
#                 while queue:
#                     i_cp, j_cp = queue.pop(0)
#                     top = min(i_cp, top)
#                     left = min(j_cp, left)
#                     bot = max(i_cp, bot)
#                     right = max(j_cp, right)
#                     if 0 <= i_cp < m and 0 <= j_cp < n and not visit[i_cp][j_cp]:
#                         visit[i_cp][j_cp] = 1
#                         if grid[i_cp][j_cp] == 255:
#                             queue.append([i_cp, j_cp + 1])
#                             queue.append([i_cp + 1, j_cp])
#                             queue.append([i_cp, j_cp - 1])
#                             queue.append([i_cp - 1, j_cp])
#                 count += 1
#                 assert top < bot and left < right, "Coordination error!"
#                 pixel_area = (right - left) * (bot - top)
#                 result.append([count, (max(0, left), max(0, top)), (min(right, n), min(bot, m)), pixel_area])
#                 # compute pixel area by split_coord
#     return result


def gather_split_result(img_path, result, output_img_dir,
                        output_anno_dir, mode="train"):
    """
    Collect split results after we run eight-connected-components
    We need to extract coord from merging step and output the cropped images together with their annotations
    to output image/anno dir
    :param img_path: The path for image to read-in
    :param result: merging result from eight-connected-component
    :param output_img_dir: the output dir to save image
    :param output_anno_dir: the output dir to save annotations
    :param mode: The dataset to process (Train/val/test)
    :return:
    """
    # obtain output of both cropped image and cropped annotations
    # Ensure the output directories exist
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir, exist_ok=True)  # Corrected directory creation
    if not os.path.exists(output_anno_dir):
        os.makedirs(output_anno_dir, exist_ok=True)

    img = cv2.imread(img_path)
    anno_path = img_path.replace("images", "ground_annotations").replace("jpg", "txt")
    txt_list = format_label(anno_path, mode) if mode != "test-challenge" else []

    for count, top_left_coord, bot_right_coord, pixel_area in result:
        (left, top), (right, bot) = top_left_coord, bot_right_coord

        cropped_image = img[top:bot, left:right]
        cropped_image_resolution = cropped_image.shape[0] * cropped_image.shape[1]

        if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0 or cropped_image_resolution < 70 * 70:
            continue

        # Save cropped annotations
        if mode != "test-challenge":
            anno_filename = f"{top}_{left}_{bot}_{right}_{os.path.basename(img_path).replace('.jpg', '.txt')}"
            anno_file_path = os.path.join(output_anno_dir, anno_filename)
            with open(anno_file_path, 'w') as filerecorder:
                for bbox_left, bbox_top, bbox_right, bbox_bottom, raw_coord in txt_list:
                    if left <= bbox_left and right >= bbox_right and top <= bbox_top and bot >= bbox_bottom:
                        raw_coord = raw_coord.split(",")
                        raw_coord[0], raw_coord[1] = str(int(raw_coord[0]) - left), str(int(raw_coord[1]) - top)
                        raw_coord = ",".join(raw_coord)
                        filerecorder.write(raw_coord + "\n")

        # Save cropped images
        img_filename = f"{top}_{left}_{bot}_{right}_{os.path.basename(img_path)}"
        img_file_path = os.path.join(output_img_dir, img_filename)
        status = cv2.imwrite(img_file_path, cropped_image)
        if not status:
            print(f"Failed to save image: {img_file_path}")
            exit()


def save_cropped_result(img_array, window_size, threshold, output_dens_dir,
                        output_img_dir, output_anno_dir, mode="train"):
    """
    A wrapper to conduct all necessary operation for generating density crops
    :param img_array: The input image to crop on
    :param window_size: The kernel selected to slide on images to gather crops
    :param threshold: determine if the crops are ROI, only crop when total pixel sum exceeds threshold
    :param output_dens_dir: The output dir to save density map
    :param output_img_dir: The output dir to save images
    :param output_anno_dir: The output dir to save annotations
    :param mode: The dataset to operate on (train/val/test)
    :return:
    """
    for img_file in tqdm(img_array, total=len(img_array)):
        overlay_map = overlay_image(img_file, window_size, threshold, output_dens_dir,
                                    mode=mode,
                                    overlay_bbox=False, show=False, save_overlay_map=False)
        result = split_overlay_map(overlay_map)
        gather_split_result(img_file, result, output_img_dir, output_anno_dir, mode=mode)

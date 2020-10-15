"""
Module providing functions for evaluating automation results.
"""
import os
import re
import glob
from typing import Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

from glabel.analysis import utils
from glabel.nn import postprocessing
from glabel.gui import gui_utils


def get_validation_files(val_dir, data_dir):
    """
    Get the .rois files for the files used in the specified `val_dir`. This is intended as an easy way to get access
    to the manual annotations made for the files used for validation of an automatioin process.

    :param val_dir: Path to directory with files used in validation process of automation step.
    :type val_dir: str
    :param data_dir: Path to directory with .rois files.
    :type data_dir: str
    :return: List of paths to .rois files.
    """
    def fsplit(f):
        """Split the frame number and patch side off of a validation filename."""
        return re.split('_\d{1,3}_\D.\w{2,3}$', f)[0]

    def proc_f(p, f):
        """Create the correct .rois filename from a passed filename used in validation."""
        return os.path.join(p, os.path.basename(fsplit(f)) + '.rois')

    # Initial search pattern is to take any file within the specified `val_dir` to find correct .rois file, if the
    # specified directory does contain sub-directories, look for the files within those instead.
    search_ptr = '*'
    probe_content = glob.glob(os.path.join(val_dir, search_ptr))[0]
    if os.path.isdir(probe_content):
        search_ptr = '**/*'

    val_files = list(set([proc_f(data_dir, f) for f in glob.glob(os.path.join(val_dir, search_ptr))]))

    return val_files


def get_bbox(rois, size=None) -> Tuple[int]:
    """
    Returns bounding box as [left, top, width, height].

    :param rois: ROI coordinates of a **single patch**!
    :type rois: np.ndarray
    :param size: Fixed size of calculated bounding box with center at center of ROIs. If not specified,
        a tight fitting bounding box is returned.
    :type size: tuple
    :return: Tuple of bounding box parameters as (left edge, top edge, widht, height)
    """
    if rois is not np.ndarray:
        rois = np.asarray(rois, dtype=np.float)
    rois[rois == -1] = np.nan  # Replace unplaced ROIs with NaN values

    # Break in case no ROI is placed
    if np.all(np.isnan(rois)):
        return [-1, -1, -1, -1]

    else:
        if not size:
            min_x, min_y = np.around(np.nanmin(rois, axis=0).ravel()).astype(int)
            max_x, max_y = np.around(np.nanmax(rois, axis=0).ravel()).astype(int)
        else:
            mean_roi = np.nanmean(rois, axis=0)
            min_x = int(mean_roi[0] - size // 2)
            max_x = int(mean_roi[0] + size // 2)
            min_y = int(mean_roi[1] - size // 2)
            max_y = int(mean_roi[1] + size // 2)

        width = max_x - min_x
        height = max_y - min_y

        return min_x, min_y, width, height


def get_suture_map(rois, peak_r=2, blob=False, blob_sigma=1.0, size=224) -> np.ndarray:
    """
    Create a suture map from specified ROI positions.

    Create an artificial suture map as returned by the suture finding segmentation network but based on manual
    annotations. Is intended to be used in order to compare predicted and ground truth segmentation maps to evaluate
    prediction metrics.

    :param rois:  ROI coordinates for a single patch.
    :type rois: np.ndarray
    :param peak_r: Radius of suture location markings on segmentation map
    :type peak_r: int
    :param blob: Boolean setting determining if suture location markings are gaussian blobs or binary squares.
    :type blob: bool
    :param blob_sigma: Sigma value used when gaussian blobs are used for markings.
    :type blob_sigma: float
    :param size: Dimension of square segmentation map. Defaults to 224.
    :type size: int
    :return: Suture map created from passed ROI positions.
    """
    if rois is not np.ndarray:
        rois = np.asarray(rois, dtype=np.float)
    fake_map = np.zeros((768, 768))
    bbox = get_bbox(rois, size)
    rois = rois[~np.isnan(rois).any(axis=1)]  # Remove any rows containing nan values

    fake_map = utils.mark_map(fake_map, rois, peak_r, blob, blob_sigma)
    gt_map = fake_map[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]  # Crop map down to (224, 224) size

    return gt_map


# def get_gt_segmap(rois, size=224, peak_r=2, blob=False, blob_sigma=1.0):
#     """
#     Universal and flexible function for creating segmentation map from ground truth annotations.
#
#     :param rois:
#     :param size:
#     :param peak_r:
#     :param blob:
#     :param blob_sigma:
#     :return:
#     """
#     if rois is not np.ndarray:
#         rois = np.asarray(rois, dtype=np.float)
#
#     # Check if passed ROIs coordinates are relative to the patch size or in global coordinates
#     rel_coords = True if rois.max() <= 1.0 else False
#     if rel_coords:
#         rois = rois * [size, size]
#     # Create the marking for the specified settings
#     peak_fill = np.ones((peak_r*2+1, peak_r*2+1)) if not blob else utils.get_gaussian_peak_fill(peak_r, blob_sigma)
#
#     segmap = np.zeros((size, size))
#
#


def intersection_over_union(box_a, box_b):
    """
    Calculate the intersection over union metric between two bounding boxes.

    :param box_a: Bounding Box specifiying [left edge, top edge, width, height]
    :type box_a: list[int]
    :param box_b: Bounding Box specifiying [left edge, top edge, width, height]
    :type box_b: list[int]
    :return: Intersection over union (IoU) metric for the bounding boxes.
    """
    inter_w = min(box_a[0] + box_a[2], box_b[0] + box_b[2]) - max(box_a[0], box_b[0])
    inter_h = min(box_a[1] + box_a[3], box_b[1] + box_b[3]) - max(box_a[1], box_b[1])
    inter_area = max(0, inter_w) * max(0, inter_h)

    areaA = box_a[2] * box_a[3]
    areaB = box_b[2] * box_b[3]

    union_area = float(areaA + areaB - inter_area)

    return inter_area / union_area


def img_iou(img_a, img_b, threshold=0.0):
    """
    Calculate the intersection over union metric pixel-wise for two images.

    :param img_a: Grayscale image [0, 1] with marked suture locations.
    :type img_a: np.ndarray
    :param img_b: Grayscale image [0, 1] with marked suture locations.
    :type img_b: np.ndarray
    :param threshold: Absolute threshold above which a pixel is counted as marked. Reduces noise in predictions.
    :return: Intersection over unioin (IoU) metric for the two images.
    """
    assert img_a.shape == img_b.shape, "Non-equal image shapes"

    cnt_inter = 0
    cnt_union = 0
    # Iterate over all pixel locations
    for y in range(img_a.shape[0]):
        for x in range(img_a.shape[1]):
            # If the pixel is marked in both images -> Count up the intersection and union area
            if (img_a[y, x] > threshold) and (img_b[y, x] > threshold):
                cnt_inter += 1
                cnt_union += 1

            # If the pixel differs between images -> Count up the union area only
            elif (img_a[y, x] > threshold) != (img_b[y, x] > threshold):
                cnt_union += 1

    return cnt_inter / cnt_union


def process_sorting_predictions(sorting_preds, img_data):
    """
    Process the raw output of the inference pipeline to be more readable and easier to use for evaluation purposes.

    :param sorting_preds: The result of the sorting inference.
    :type sorting_preds: list[nn.suture_detection.SortingPrediction]
    :param img_data: Image data.
    :type img_data: np.ndarray
    :return: Processed sorting predictions
    """
    from glabel.nn.suture_detection import SortingPrediction

    num_frames = img_data.shape[0]
    sorting_predictions = [[] for _ in range(num_frames)]

    manip_preds = deepcopy(sorting_preds)
    # Group the received predictions to their correct frames
    for pred in manip_preds:
        sorting_predictions[pred.frame].append(pred)

    # ------------------------------------------------------------------------
    # Process all of the prediction data
    pred_rois = [[] for _ in range(num_frames)]  # Empty list for each frame
    discarded_sortings = []  # List for storing which suture predictions could not be sorted

    # Iterate over full list of predictions frame by frame
    for frame_ctr, frame_sortings in enumerate(sorting_predictions):
        # If the current frame was not used in inference -> Skip to the next one
        if len(frame_sortings) == 0:
            continue

        left_sortings = [None] * 35
        right_sortings = [None] * 35

        # Iterate over individual predictions of the frame
        for pred in frame_sortings:
            # If all position probabilities for the suture have been reduced to below 0 -> Add it to the discarded ones
            # and continue with the next prediction
            if np.max(pred.probabilities) <= 0.0:
                discarded_sortings.append(pred)
                continue

            id = pred.pred_id  # Position with highest probability
            sort_list = left_sortings if pred.side == 'left' else right_sortings

            # If the position is not filled yet just put it at the predicted position
            if not sort_list[id]:
                sort_list[id] = pred

            # Otherwise, compare it to the prediction that exists at that spot and exchange it if the new prediction has
            # a higher confidence
            else:
                this_prob = pred.probabilities[id]  # Currently investigated
                that_prob = sort_list[id].probabilities[id]  # The one previously placed at the position
                if this_prob > that_prob:
                    # Exchange the predictions but negate the probability of the one we removed from its position in
                    # in order to have it sorted at its second highest confidence position in the next iteration
                    that_pred = sort_list[id]
                    that_pred.probabilities[id] *= -1
                    new_id = np.argmax(that_pred.probabilities)  # Find position that now has the highest probability
                    # Create a new namedtuple object with the new information from the exchanged sorting
                    mod_pred = SortingPrediction(new_id, that_pred.y, that_pred.x, that_pred.frame, that_pred.side,
                                                 that_pred.probabilities)
                    # Add the newly found, higher confident prediction at that position
                    sort_list[id] = pred
                    # Add the removed
                    frame_sortings.append(mod_pred)
                else:
                    pred.probabilities[id] *= -1
                    new_id = np.argmax(pred.probabilities)
                    mod_pred = SortingPrediction(new_id, pred.y, pred.x, pred.frame, pred.side, pred.probabilities)
                    frame_sortings.append(mod_pred)

        proc_rois = [[p.x, p.y] if p else [-1, -1] for p in left_sortings + right_sortings]
        pred_rois[frame_ctr] = proc_rois

    return pred_rois


def get_sort_eval_df(gt_rois, pred_rois):
    """
    Create a pandas DataFrame for evaluation data of suture sorting results.

    The created DataFrame rows correspond to individual sutures with the following columns:

    * 'file_id': Frame number
    * 'side': ['l', 'r'] depending on stereoscopic view side
    * 'gt_id': Ground truth grid ID of suture
    * 'gt_y': Ground truth vertical pixel position
    * 'gt_x': Ground truth horizontal pixel position
    * 'pred_y': Predicted vertical pixel position
    * 'pred_x': Predicted horizontal pixel position
    * d_y': Vertical distance between between ground truth and prediction (gt - pred)
    * d_x': Horizontal distance between between ground truth and prediction (gt - pred)
    * 'gt_placed': Ground truth if suture is visible
    * 'pred_placed': Prediction if suture is visible
    * 'placed_conf': ['TP', 'FP', 'FN', 'TN'] Confusion value for suture being placed.

    :param np.ndarray gt_rois: Numpy array containing ground truth suture locations. Must be of shape [#frames, #grid
        positions, 2].
    :param list[list] pred_rois: Predicted suture sortings for each frame. Outer list is over frames,
        inner list over grid positions.
    :return: Pandas DataFrame with each row corresponding to a single suture.
    """
    evaluations = []

    for frame_id, (gt_file_rois, pred_file_rois) in enumerate(zip(gt_rois, pred_rois)):
        for roi_id, (gt_roi, pred_roi) in enumerate(zip(gt_file_rois, pred_file_rois)):

            gt_placed = False if any(coord == -1 for coord in gt_roi) else True
            pred_placed = False if any(coord == -1 for coord in pred_roi) else True
            if gt_placed and pred_placed:
                placed_conf = 'TP'
            elif not gt_placed and pred_placed:
                placed_conf = 'FP'
            elif gt_placed and not pred_placed:
                placed_conf = 'FN'
            else:
                placed_conf = 'TN'

            evaluations.append({
                'frame_id': frame_id,
                'side': 'l' if roi_id < 35 else 'r',
                'gt_id': roi_id if roi_id < 35 else roi_id - 35,
                'gt_y': gt_roi[1],
                'gt_x': gt_roi[0],
                'pred_y': pred_roi[1],
                'pred_x': pred_roi[0],
                'd_y': gt_roi[1] - pred_roi[1],
                'd_x': gt_roi[0] - pred_roi[0],
                'gt_placed': gt_placed,
                'pred_placed': pred_placed,
                'placed_conf': placed_conf
            })

    return pd.DataFrame(evaluations)


def get_total_inference_df(region_boxes, region_confs, suture_maps, suture_boxes, pred_rois, gt_rois):
    """
    Create a pandas DataFrame for evaluation data suture predictions.

    The created DataFrame rows correspond to individual sutures with the following columns:

    * 'frame_id': Frame number
    * 'side': ['l', 'r'] Stereoscopic view side
    * 'yolo_box_left', 'yolo_box_top', 'yolo_box_width', 'yolo_box_height', 'yolo_box_conf': Values for suture grid
      region bounding box
    * 'gt_yolo_box_left', 'gt_yolo_box_top', 'gt_yolo_box_width', 'gt_yolo_box_height', 'gt_yolo_box_conf': Values for
      ground truth suture grid region bounding box
    * 'yolo_IOU': Intersection over Union between predicted and ground truth suture grid region bounding boxes
    * 'unet_box_left', 'unet_box_top', 'unet_box_width', 'unet_box_height', 'unet_box_conf': Values for expanded
      bounding box for suture map creation
    * 'unet_IOU': Intersection over Union between predicted suture probability map and ground truth segmentation map
    * 'pred_roi_x': Horizontal pixel coordinate of suture
    * 'pred_roi_y': Vertical pixel coordinate of suture
    * 'gt_roi_x': Ground truth horizontal pixel coordinate of suture
    * 'gt_roi_y': Ground truth vertical pixel coordinate of suture
    * 'roi_id': Suture grid position

    :param list[list[Box]] region_boxes: Suture grid region bounding boxes.
    :param list[list[float]] region_confs: Bounding box confidence values.
    :param np.ndarray suture_maps: Predicted suture probability maps. Of shape[#frames*2, map width, map height].
    :param list[list[Box]] suture_boxes: Expanded suture grid region bounding boxes used for creating suture maps.
    :param list[list] pred_rois: Processed suture sorting predictions.
    :param list[list] gt_rois: Ground truth sutures.
    :return: Pandas DataFrame with each row corresponding to a single suture.
    """
    inferences = []

    # Double loop for basically iterating over individual **patches**
    for frame_idx in range(len(region_boxes)):
        for side in ['left', 'right']:
            side_idx = 0 if side == 'left' else 1

            # YOLO inference information for this specific patch
            y_box = region_boxes[frame_idx][side_idx]
            y_conf = region_confs[frame_idx][side_idx]
            y_box_list = [y_box.left, y_box.top, y_box.width, y_box.height]
            assert y_box.frame == frame_idx and y_box.side == side, "YOLO Box sanity check failed!"

            # UNet inference information for this specific patch
            u_prob_map = suture_maps[frame_idx * 2 + side_idx]
            u_box = suture_boxes[frame_idx][side_idx]
            assert u_box.frame == frame_idx and u_box.side == side, "UNet Box sanity check failed!"

            # EfficientNet sorting information for this specific patch
            e_frame_rois = pred_rois[frame_idx]
            e_patch_rois = e_frame_rois[:35] if side == 'left' else e_frame_rois[35:]

            # Ground truth information
            gt_frame_rois = gt_rois[frame_idx]
            gt_patch_rois = gt_frame_rois[:35] if side == 'left' else gt_frame_rois[35:]
            gt_tight_box = get_bbox(gt_patch_rois)
            gt_big_box = get_bbox(gt_patch_rois, 224)
            gt_prob_map = get_suture_map(gt_patch_rois, 2, False)  # Binary peaks of radius 2

            # Create long-form dataframe by creating a **single row for each of the found ROIs**
            for roi_idx in range(35):
                inferences.append({
                    'frame_id': frame_idx,
                    'side': side,

                    'yolo_box_left': y_box.left,
                    'yolo_box_top': y_box.top,
                    'yolo_box_width': y_box.width,
                    'yolo_box_height': y_box.height,
                    'yolo_box_conf': y_conf,
                    'gt_yolo_box_left': gt_tight_box[0],
                    'gt_yolo_box_top': gt_tight_box[1],
                    'gt_yolo_box_width': gt_tight_box[2],
                    'gt_yolo_box_height': gt_tight_box[3],
                    'yolo_IOU': intersection_over_union(gt_tight_box, y_box_list),

                    'unet_box_top': u_box.top,
                    'unet_box_left': u_box.left,
                    'unet_box_width': u_box.width,
                    'unet_box_height': u_box.height,
                    'unet_IOU': img_iou(gt_prob_map, u_prob_map),

                    'pred_roi_x': e_patch_rois[roi_idx][0],
                    'pred_roi_y': e_patch_rois[roi_idx][1],
                    'gt_roi_x': gt_patch_rois[roi_idx][0],
                    'gt_roi_y': gt_patch_rois[roi_idx][1],
                    'roi_id': roi_idx
                })

    return pd.DataFrame(inferences)


def get_placement_confusion(ser, normalize=True) -> np.ndarray:
    """
    Create confusion matrix from pandas Series containing confusion values.

    Confusion values means that the type of confusion for a observation is listed using one of ['TP', 'FP', 'FN', 'TN'].

    :param pd.Series ser: Pandas Series containing confusion values.
    :param bool normalize: Set to True to normalize confusion matrix values to range [0, 1].
    :return: Confusion matrix as numpy array of shape [2, 2].
    """
    conf_dict = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    for res in ser:
        conf_dict[res] += 1

    conf_mat = np.zeros((2, 2))
    conf_mat[0, 0] = conf_dict['TP']
    conf_mat[1, 0] = conf_dict['FP']
    conf_mat[0, 1] = conf_dict['FN']
    conf_mat[1, 1] = conf_dict['TN']
    if normalize:
        conf_mat = conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]

    return conf_mat


def save_inferences(dirpath, results, settings):
    """
    Save inference results to files.

    :param str dirpath: Path to directory in which the result files are created.
    :param dict results: Dictionary of inference results.
    :param dict settings: Dictionary of settings used for inference.
    """
    os.makedirs(dirpath, exist_ok=True)

    # Save region inference
    if all(key in list(results.keys()) for key in ['region_boxes', 'region_confs']):
        proc_region_boxes = postprocessing.process_region_predictions(
            results['region_boxes'],
            results['region_confs'],
            settings['total_frames'],
            False)
        gui_utils.save_region_inference(proc_region_boxes, dirpath, settings)

    # Save suture probability map inference
    if all(key in list(results.keys()) for key in ['suture_maps', 'suture_boxes']):
        proc_prob_maps = postprocessing.process_suture_maps(
            results['suture_maps'],
            results['suture_boxes'],
            settings['total_frames']
        )
        gui_utils.save_suture_map_inference(proc_prob_maps, dirpath, settings)

    # Save peak finding results
    if 'suture_peaks' in list(results.keys()):
        gui_utils.save_suture_peaks(results['suture_peaks'], dirpath, settings)

    # Save suture sorting inference
    if 'sorted_sutures' in list(results.keys()):
        proc_sortings, _ = postprocessing.process_suture_sortings(
            results['sorted_sutures'],
            settings['total_frames'],
            70,
            True
        )
        gui_utils.save_sorted_peaks(proc_sortings, dirpath, settings)


def get_settings_dict(total_frames, network_paths) -> dict:
    """
    Create default settings for automatic suture identification and sorting.

    :param int total_frames: Total number of frames in the image data for which inference is run.
    :param dict network_paths: Dictionary of neural network files to use for inference.
    :return: Settings dictionary which can be passed to :func:`nn.suture_detection.auto_annotate` to run inference.
    """
    input_settings = {
        'from_frame': 0,
        'to_frame': total_frames,
        'total_frames': total_frames
    }

    region_settings = {
        'run_region_detect': True,
        'region_frame_delta': 1,
        'region_network_path': network_paths['region_cfg'],
        'region_weights_path': network_paths['region_weights'],
        'load_regions': False,
        'regions_file': None
    }

    suture_map_settings = {
        'run_suture_find': True,
        'suture_find_network': network_paths['suture_finding'],
        'suture_find_batch': 1,
        'load_maps': False,
        'maps_file': None
    }

    peak_find_settings = {
        'run_peak_find': True,
        'peak_find_distance': 3,
        'peak_find_threshold': 0.5,
        'load_peaks': False,
        'peaks_file': None
    }

    peak_sort_settings = {
        'run_peak_sort': True,
        'suture_sort_network': network_paths['suture_sort'],
        'suture_sort_batch': 1,
        'load_sortings': False,
        'sorting_file': None
    }

    settings_dict = {}
    settings_dict.update(input_settings)
    settings_dict.update(region_settings)
    settings_dict.update(suture_map_settings)
    settings_dict.update(peak_find_settings)
    settings_dict.update(peak_sort_settings)

    return settings_dict


def generate_gt_labels(n_frames, n_rows, n_cols, scope='id') -> np.ndarray:
    """
    Create an array of ground truth annotation labels that specify the grid position (`id`), column- (`col`) or
    row-membership (`row`) corresponding to a sorted ROI array of specified dimensions.

    .. caution:: The ground truth labels are generated under the assumption that the peaks are stored in suture grid
        order as is done everywhere within the project. If the passed peaks are not sorted according to their grid
        position, the returned labels will not correspond!

    :param int n_frames: Number of frames in the data.
    :param int n_rows: Number of rows in the suture grid.
    :param int n_cols: Number of columns in the suture grid.
    :param str scope: ['id', 'row', 'col'] The type of labels to create. `id` will generate grid position labels, while
        `row` and `col` will generate labels for row- and column-memberships respectively.
    :return: (#frames, 70, 1) numpy array with generated ground truth labels.
    """
    if scope == 'id':
        patch_labels = np.arange(n_rows * n_cols)
    elif scope == 'row':
        patch_labels = np.repeat(np.arange(n_rows), n_cols)
    elif scope == 'col':
        patch_labels = np.tile(np.arange(n_cols), n_rows)
    else:
        raise ValueError(f"Invalid value {scope} for parameter 'id' encountered!")

    frame_labels = np.tile(patch_labels, 2)
    gt_labels = np.tile(frame_labels, (n_frames, 1))

    return gt_labels


def create_classification_maps(rois, map_size, peak_r) -> np.ndarray:
    """
    Create classification maps for sorting the passed suture coordinates
    using an EfficientNetB0 (or other network trained on same input data).

    :param np.ndarray rois: (N, 2) Suture coordinates of a single patch.
    :param tuple size: (height, width) dimensions of created maps.
    :param int peak_r: Radius of suture markings.
    :return: (N, height, width) np.ndarray built from N classification maps.
    """
    peak_fill = np.ones((peak_r*2, peak_r*2))

    mask_val = -1 if rois.max() <= 1 else -224
    scale_factor = 224 if rois.max() <= 1 else 1
    
    masked_rois = np.ma.masked_equal(rois, mask_val)
    valid_rois = masked_rois.compressed().reshape(-1, 2)
    
    class_maps = np.zeros((len(valid_rois), 224, 224, 3))
    
    for idx, roi in enumerate(valid_rois):
        x = int(roi[0] * scale_factor)
        y = int(roi[1] * scale_factor)
        # Mark the base channel for all maps
        class_maps[:, y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r, 0] = 1
        # Mark the individual channels only for this map
        class_maps[idx, y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r, 1:] = 1
        
    return class_maps


def get_larynx_list(rois_files):
    """
    Create a list of all individual hemi-larynx identifiers from the list of .rois files.

    :param list rois_files: List of paths to .rois files.
    """
    clean_files = [os.path.basename(f) for f in rois_files]
    parts = [f.split('_') for f in clean_files]
    ids = [p[1] if p[0] == 'Human' else p[0] for p in parts]

    return ids


def kfold_iterator(x: list, k: int):
    """
    Generator yielding (val, train) tuples of split list elements.
    """
    n = len(x)
    s = n // k
    r = n % k
    i = 0
    while i < n:
        if i == 0 and r != 0:
            incl = x[i:i+s+r]
            excl = [v for v in x if v not in incl]
            yield (incl, excl)
            i += s+r
        else:
            incl = x[i:i+s]
            excl = [v for v in x if v not in incl]
            yield (incl, excl)
            i += s


def create_sort_png_mappings(rois, peak_r=2, blob=False, blob_sigma=1.0):
    l_maps = []
    l_idcs = []
    r_maps = []
    r_idcs = []

    if not blob:
        peak_fill = 255 * np.ones((peak_r*2, peak_r*2))
    else:
        x, y = np.meshgrid(np.linspace(-1, 1, peak_r*2), np.linspace(-1, 1, peak_r*2))
        d = np.sqrt(x * x + y * y)
        d -= np.min(d)  # Makes sure that the central pixels of the peak_fill will have value 255
        sigma, mu = blob_sigma, 0.0
        peak_fill = (np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))) * 255).astype(np.uint8)

    # Create the base maps that make up the first channel of each input
    l_base = np.zeros((224, 224))
    r_base = np.zeros((224, 224))
    for idx, roi in enumerate(rois):
        if any(coord == -1 for coord in roi):
            continue
        base = l_base if idx < 35 else r_base
        x = int(roi[0] * 224)
        y = int(roi[1] * 224)
        base[y-peak_r:y+peak_r, x-peak_r:x+peak_r] = peak_fill

    # Create the inputs by adding the individual map channels behind the base map
    for idx, roi in enumerate(rois):
        # If the current ROI is not placed, do not create a map for it and skip to the next one
        if any(coord == -1 for coord in roi):
            continue
        else:
            side = 'l' if idx < 35 else 'r'
            input_map = np.zeros((224, 224, 3), dtype=np.uint8)
            input_map[:, :, 0] = l_base if side == 'l' else r_base
            # Create the individual channel map
            x = int(roi[0] * 224)
            y = int(roi[1] * 224)
            input_map[y-peak_r:y+peak_r, x-peak_r:x+peak_r, 1:] = np.dstack([peak_fill, peak_fill])

            if side == 'l':
                l_maps.append(input_map)
                l_idcs.append(idx)
            else:
                r_maps.append(input_map)
                r_idcs.append(idx-35)

    return l_maps, r_maps, l_idcs, r_idcs


def create_segmap(rois, classtype, peak_r):
    halfpoint = len(rois) // 2
    l_rois = rois[:halfpoint]
    r_rois = rois[halfpoint:]

    if classtype == 'single':
        l_map = np.zeros((224, 224), dtype=np.uint8)
        r_map = np.zeros((224, 224), dtype=np.uint8)
    elif classtype == 'column':
        l_map = np.zeros((224, 224, 5), dtype=np.uint8)
        r_map = np.zeros((224, 224, 5), dtype=np.uint8)
    else:
        l_map = np.zeros((224, 224, 35), dtype=np.uint8)
        r_map = np.zeros((224, 224, 35), dtype=np.uint8)

    for map, rois in zip([l_map, r_map], [l_rois, r_rois]):
        for class_id, roi in enumerate(rois):
            if not any(coord <= -1 for coord in roi):
                scale_fac = 224 if roi.max() < 1 else 1
                x = int(roi[0] * scale_fac)
                y = int(roi[1] * scale_fac)

                # For singleclass maps, all ROI positions are marked on the single channel
                if classtype == 'single':
                    map[y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r] = 255

                # For column classes, each column membership gets its own channel,
                # each having all sutures belonging to that column marked.
                elif classtype == 'column':
                    column_id = class_id % 5
                    map[y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r, column_id] = 255

                # For multiclass maps, each ROI is marked on its own channel
                else:
                    map[y-peak_r-1:y+peak_r, x-peak_r-1:x+peak_r, class_id] = 255

    return l_map, r_map

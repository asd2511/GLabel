from collections import namedtuple
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QPointF

SuturePrediction = namedtuple('SuturePrediction', ['left_map', 'right_map', 'left_bbox', 'right_bbox'])
SortingPrediction = namedtuple('SortingPrediction', ['pred_id', 'y', 'x', 'frame', 'side', 'probabilities'])


def process_region_predictions(region_boxes, region_confs, total_frames, qt=True):
    """
    Process the suture region predictions to prepare them for display in the GUI or for saving to file.

    The predicted bounding boxes will be converted into pyqtgraph.RectROI objects which can easily be displayed in the
    GUI. When saving, the bounding box information can be easily extracted from the ROI objects.

    :param region_boxes: Predicted suture region bounding boxes.
    :param region_confs: Prediction confidences for suture region bounding boxes.
    :param total_frames: Total number of frames in the image data. May be different from number of frames used in
        prediction so this value is used to correctly connect the inferred bounding boxes to their frames.
    :return: List of list holding the two bounding boxes for each frame.
    """
    bbox_stack = [[] for _ in range(total_frames)]

    for i in range(len(region_boxes)):
        l_vals = region_boxes[i][0]
        r_vals = region_boxes[i][1]
        l_conf = region_confs[i][0]
        r_conf = region_confs[i][1]

        if qt:
            l_box = pg.RectROI((l_vals.left, l_vals.top), (l_vals.width, l_vals.height))
            r_box = pg.RectROI((r_vals.left, r_vals.top), (r_vals.width, r_vals.height))
        else:
            l_box = [l_vals.left, l_vals.top, l_vals.width, l_vals.height]
            r_box = [r_vals.left, r_vals.top, r_vals.width, r_vals.height]

        bbox_entry = [l_box, l_conf, r_box, r_conf]
        bbox_stack[l_vals.frame] = bbox_entry

    return bbox_stack


def process_suture_maps(suture_maps, suture_boxes, total_frames):
    """
    Process the suture probability maps and corresponding expanded bounding boxes to prepare them for display in the GUI
    or for saving to file.

    The probability maps and corresponding bounding boxes will be converted into a single `SuturePrediction` namedtuple
    object for each frame. The probability maps and location information for placement on the frame can be easily
    extracted when displaying or saving the data.

    :param suture_maps: Probability maps for suture locations.
    :param suture_boxes: Bounding boxes with information about locations of probability maps on frames.
    :param total_frames: Total number of frames in the image data. May be different from number of frames used in
        prediction so this value is used to correctly connect the inferred bounding boxes to their frames.
    :return: List holding a `SuturePrediction` namedtuple object for each frame with all inference information.
    """
    suture_predictions = [[] for _ in range(total_frames)]

    for i in range(len(suture_boxes)):
        pred_idx = i * 2
        l_idx = 0 if suture_boxes[i][0].side == 'left' else 1
        prediction = SuturePrediction(suture_maps[pred_idx], suture_maps[pred_idx + 1],
                                      suture_boxes[i][l_idx], suture_boxes[i][1 - l_idx])
        suture_predictions[suture_boxes[i][l_idx].frame] = prediction

    return suture_predictions


def process_suture_sortings(sorting_preds, total_frames, total_places, qt=True, fix_cols=True, fix_rows=True):
    sorting_predictions = [[] for _ in range(total_frames)]
    # Group the received predictions to their correct frames
    for pred in sorting_preds:
        sorting_predictions[pred.frame].append(pred)
    manip_preds = deepcopy(sorting_predictions)

    # ------------------------------------------------------------------------
    # Process all of the prediction data
    if qt:
        pred_rois = [[QPointF(-1, -1)] * total_places for _ in range(total_frames)]
    else:
        pred_rois = [[] for _ in range(total_frames)]  # Empty list for each frame
    discarded_sortings = []  # List for storing which suture predictions could not be sorted

    # Iterate over full list of predictions frame by frame
    for frame_ctr, frame_sortings in enumerate(manip_preds):
        # If the current frame was not used in inference -> Skip to the next one
        if len(frame_sortings) == 0:
            continue

        left_sortings = [None] * (total_places // 2)
        right_sortings = [None] * (total_places // 2)

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
                    new_id = np.argmax(
                        that_pred.probabilities)  # Find position that now has the highest probability
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

        if qt:
            proc_rois = [QPointF(p.x, p.y) if p else QPointF(-1, -1) for p in left_sortings + right_sortings]
        else:
            proc_rois = [[p.x, p.y] if p else [-1, -1] for p in left_sortings + right_sortings]
        pred_rois[frame_ctr] = proc_rois

    # sorting_predictions: Raw and unprocessed, but grouped into their correct frames
    # pred_rois: Processed sorted peaks with each frame having 70 positions and the peaks sorted into them
    if fix_rows or fix_cols:
        pred_rois = fixup_sortings(pred_rois, 5, 7, fix_cols=fix_cols, fix_rows=fix_rows)
    return sorting_predictions, pred_rois


def fixup_sortings(pred_rois, num_cols, num_rows, stereo=True, fix_cols=True, fix_rows=True):
    # For each frame separately, the sutures of a column are investigated to be in the correct order w.r.t to their
    # vertical coordinate.
    total_placements = num_rows * num_cols * 2 if stereo else num_rows * num_cols
    side_placements = num_rows * num_cols
    fixed_sortings = [[QPointF(-1, -1) for _ in range(total_placements)] for _ in range(len(pred_rois))]

    for frame_idx, frame_rois in enumerate(pred_rois):
        l_rois = frame_rois[:num_cols*num_rows]
        r_rois = frame_rois[num_cols*num_rows:]

        # First, fix the ordering within columns to be vertically sorted
        if fix_cols:
            for (side_rois, offset) in zip([l_rois, r_rois], [0, side_placements]):
                fix_col(num_cols, side_rois, offset, fixed_sortings[frame_idx])

        # Second, fix the ordering within rows to be horizontally sorted
        if fix_rows:
            for (side_rois, offset) in zip([l_rois, r_rois], [0, side_placements]):
                fix_row(num_rows, num_cols, side_rois, offset, fixed_sortings[frame_idx])

    return fixed_sortings


def fix_col(num_cols, side_rois, offset, fixed_sortings):
    for col in range(num_cols):
        col_rois = [(idx, roi) for idx, roi in enumerate(side_rois) if (idx % num_cols == col)
                    and (not any(coord == -1 for coord in [roi.y(), roi.x()]))]
        vert_sorted_col = sorted(col_rois, key=lambda x: x[1].y(), reverse=True)
        for pre_sort, post_sort in zip(col_rois, vert_sorted_col):
            fixed_sortings[pre_sort[0]+offset] = post_sort[1]


def fix_row(num_rows, num_cols, side_rois, offset, fixed_sortings):
    for row in range(num_rows):
        row_rois = [(idx, roi) for idx, roi in enumerate(side_rois) if (idx // num_cols == row)
                    and (not any(coord == -1 for coord in [roi.y(), roi.x()]))]
        hor_sorted_row = sorted(row_rois, key=lambda x: x[1].x())
        for pre_sort, post_sort in zip(row_rois, hor_sorted_row):
            fixed_sortings[pre_sort[0]+offset] = post_sort[1]

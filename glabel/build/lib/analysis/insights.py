import os

import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

from glabel.analysis import utils
from glabel.analysis.column_clustering import kmeans_cluster


def kmeans_2nd_step_df(peaks_npz, rois_files, z_normalize=True, aspect_ratio='xy') -> pd.DataFrame:
    """
    Build a dataframe to gain insights into possible features for creating a 2nd step classifier for KMeans column
    clustering.

    :param peaks_npz: Path to .npz file with totality of manually labeled peaks.
    :type peaks_npz: str
    :param rois_files: List of paths to .rois file to use for gaining insights.
    :type rois_files: List[str]
    :param z_normalize: Boolean setting to set z-score normalization of peaks used for statistic calculations
    :type z_normalize: bool
    :param aspect_ratio: Setting for how the aspect ratio of variances is computed. Defaults to 'xy'.
        ``'xy'``: Will compute the aspect ratio as x-variance over y-variance.
        ``'yx'``: Will compute the aspect ratio as y-variance over x-variance.
    :type aspect_ratio: {'xy', 'yx'} or None
    :return: DataFrame holding insights into possible features for 2nd step KMeans classifier.
    """
    # Load the saved ground truth labeled peaks from the .npz file
    d = np.load(peaks_npz, allow_pickle=True)
    l_peaks_col = d['l_peaks_col']  # All peaks from all left patches
    r_peaks_col = d['r_peaks_col']  # All peaks from all right patches

    if z_normalize:
        # ------------------------------------------------------------------------
        # Normalize the peaks by subtracting mean and dividing by standard deviation
        l_peaks_col = [utils.normalize_peaks(l_peaks_col[i]) for i in range(5)]
        r_peaks_col = [utils.normalize_peaks(r_peaks_col[i]) for i in range(5)]

    # ------------------------------------------------------------------------
    # Get information about ground truth column membership and average covariances/variances for each column based on
    # the ground truth labels
    # Get the covariances for each of the gt columns
    l_covs = [np.cov(l_peaks_col[i], rowvar=False) for i in range(5)]
    r_covs = [np.cov(r_peaks_col[i], rowvar=False) for i in range(5)]
    # Get the variances for each of the gt columns
    l_vars_x = [np.var(np.asarray(l_peaks_col[i])[:, 0]) for i in range(5)]
    r_vars_x = [np.var(np.asarray(r_peaks_col[i])[:, 0]) for i in range(5)]
    l_vars_y = [np.var(np.asarray(l_peaks_col[i])[:, 1]) for i in range(5)]
    r_vars_y = [np.var(np.asarray(r_peaks_col[i])[:, 1]) for i in range(5)]
    # Get the aspect ratio for each of the gt columns
    ar_nums = (l_vars_x, r_vars_x) if aspect_ratio == 'xy' else (l_vars_y, r_vars_y)
    ar_dens = (l_vars_y, r_vars_y) if aspect_ratio == 'xy' else (l_vars_x, r_vars_x)
    l_ars = [ar_nums[0][i] / ar_dens[0][i] for i in range(5)]
    r_ars = [ar_nums[1][i] / ar_dens[1][i] for i in range(5)]

    # ------------------------------------------------------------------------
    # Get the total average covariance and variances for each side (aggregating over all columns of a side)
    # Get the mean of each of the 4 covariance entries by creating a 3D array and average depth-wise
    avg_l_cov = np.mean(np.dstack(l_covs), axis=2)
    avg_r_cov = np.mean(np.dstack(r_covs), axis=2)
    # Get the average variances in x- and y-direction
    avg_l_var_x = np.mean(l_vars_x)
    avg_r_var_x = np.mean(r_vars_x)
    avg_l_var_y = np.mean(l_vars_y)
    avg_r_var_y = np.mean(r_vars_y)
    # Get the average aspect ratios
    avg_l_ar = np.mean(l_ars)
    avg_r_ar = np.mean(r_ars)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    # Start to analyze the data to gain insights
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    dlist = []  # List object for holding iterative dicts --> Will eventually be converted into the insight dataframe

    # Iterate over the specified .rois files to use as validation data
    for file in tqdm(rois_files, desc='files'):
        # Get the peak coordinates and their gt column membership labels from the file
        peaks, gt_labels = utils.get_peaks_col_labeled(file)

        # Iterate over each patch (alternating left/right) and run detection and evaluation for each patch
        for patch_idx, (frame_peaks, frame_gt_labels) in enumerate(zip(peaks, gt_labels)):
            side = 'l' if patch_idx % 2 == 0 else 'r'

            # ------------------------------------------------------------------------
            # Run kmeans clustering on the peaks
            pred_labels = kmeans_cluster(frame_peaks, [1, 0])

            if z_normalize:
                frame_peaks = utils.normalize_peaks(frame_peaks)

            # ------------------------------------------------------------------------
            # Group the predictions into their column membership and extract the covariance information for each column
            # Group the peak coordinates into the columns
            pred_col_peaks = [peaks[patch_idx][np.where(pred_labels == i)] for i in range(5)]
            pred_col_peaks_norm = [frame_peaks[np.where(pred_labels == i)] for i in range(5)]
            # Calculate the covariances and variances from the normalized peaks for each column
            pred_col_covs = [np.cov(pred_col_peaks_norm[i], rowvar=False) for i in range(5)]

            # ------------------------------------------------------------------------
            # Count the total number of misclassifications for the patch
            num_misclassifications = np.sum(frame_gt_labels != pred_labels)

            # ------------------------------------------------------------------------
            # Iterate over the data from individual columns and calculate statistics/differences of features
            for col_idx in range(5):
                # Count the number of misclassifications that happened for the individual columns
                gt_col_idcs = np.where(frame_gt_labels == col_idx)
                pred_col_idcs = np.where(pred_labels == col_idx)
                col_misclassifications = np.sum(frame_gt_labels[gt_col_idcs] != pred_labels[gt_col_idcs])

                # ------------------------------------------------------------------------
                # Compare the covariance of the predicted column with the gt average column covariance and the overall
                # gt covariance for the complete side (all columns of that side)
                # Get reference to the correct average side values
                avg_side_cov = avg_l_cov if side == 'l' else avg_r_cov
                avg_col_covs = l_covs if side == 'l' else r_covs
                # Compute the difference between covariances
                col_norm_cov_diff = np.linalg.norm(avg_col_covs[col_idx] - pred_col_covs[col_idx])
                side_norm_cov_diff = np.linalg.norm(avg_side_cov - pred_col_covs[col_idx])

                # ------------------------------------------------------------------------
                # Do the same for the variances
                # Get the gt variances
                col_vars_x = l_vars_x if side == 'l' else r_vars_x
                col_vars_y = l_vars_y if side == 'l' else r_vars_y
                avg_var_x = avg_l_var_x if side == 'l' else avg_r_var_x
                avg_var_y = avg_l_var_y if side == 'l' else avg_r_var_y
                # Get the prediction variances
                pred_var_x = np.var(pred_col_peaks_norm[col_idx][:, 0])
                pred_var_y = np.var(pred_col_peaks_norm[col_idx][:, 1])
                # Compute the differences between variances
                col_var_x_diff = np.abs(col_vars_x[col_idx] - pred_var_x)
                col_var_y_diff = np.abs(col_vars_y[col_idx] - pred_var_y)
                side_var_x_diff = np.abs(avg_var_x - pred_var_x)
                side_var_y_diff = np.abs(avg_var_y - pred_var_y)

                # ------------------------------------------------------------------------
                # Do the same for the aspect ratios
                # Get the gt aspect ratios
                col_ars = l_ars if side == 'l' else r_ars
                avg_ar = avg_l_ar if side == 'l' else avg_r_ar
                # Get the prediction aspect ratio
                ar_num = pred_var_x if aspect_ratio == 'xy' else pred_var_y
                ar_den = pred_var_y if aspect_ratio == 'xy' else pred_var_x
                pred_ar = ar_num / ar_den
                # Compute the difference between aspect ratios
                col_ar_diff = np.abs(col_ars[col_idx] - pred_ar)
                side_ar_diff = np.abs(avg_ar - pred_ar)

                # ------------------------------------------------------------------------
                # ------------------------------------------------------------------------
                # Append the information as the next row to the data list
                dlist.append({
                    'file': os.path.basename(file),
                    'patch_idx': patch_idx,
                    'side': side,
                    'column': col_idx,
                    'patch_misclassifications': num_misclassifications,
                    'col_misclassifications': col_misclassifications,
                    'col_norm_cov_diff': col_norm_cov_diff,  # Diff between pred and gt of this column's covariance
                    'side_norm_cov_diff': side_norm_cov_diff,  # Diff between pred cov of this column and avg cov of this side
                    'col_var_x_diff': col_var_x_diff,
                    'col_var_y_diff': col_var_y_diff,
                    'side_var_x_diff': side_var_x_diff,
                    'side_var_y_diff': side_var_y_diff,
                    'col_ar_diff': col_ar_diff,
                    'side_ar_diff': side_ar_diff
                })

    return pd.DataFrame(dlist)


def save_labeled_peaks_npz(files, savename) -> None:
    """
    Load and save all manually labled peaks as .npz format.

    All manually labeled peaks for the specified files will be loaded, grouped into their patch side and columns, and
    saved in .npz format.

    The .npz archive will contain the data saved under 4 keys:

    =========== =============================================================================================
    **Key**     **Data**
    l_peaks     All labeled peaks from left patches of the files
    r_peaks     All labeled peaks from the right patches of the files
    l_peaks_col All labeled peaks from the left patches of the files, grouped into the 5 different columns
    r_peaks_col All labeled peaks from the right patches of the files, grouped into the 5 different columns
    =========== =============================================================================================

    :param files: List of .rois files for which to load and summarize peak locations.
    :type files: List[str]
    :param savename: Path to .npz save file
    :type savename: str
    """
    l_peaks = []
    r_peaks = []
    l_peaks_col = [[] for n in range(5)]
    r_peaks_col = [[] for n in range(5)]

    for f in tqdm(files, desc='files'):
        if not 'daten/fuer_nn' in f:
            f = os.path.join('daten', 'fuer_nn', f)
        rois = utils.load_rois_patch_relative(f, (224, 224))

        for frame_rois in rois:
            l_rois = frame_rois[:35]
            r_rois = frame_rois[35:]

            for side in ['l', 'r']:
                side_list = l_peaks if side == 'l' else r_peaks
                col_list = l_peaks_col if side == 'l' else r_peaks_col
                side_rois = l_rois if side == 'l' else r_rois

                for idx, roi in enumerate(side_rois):
                    # Skip unplaced/unlabeled ROIs
                    if any(coord == -1 for coord in roi):
                        continue
                    side_list.append(roi)
                    col = idx % 5
                    col_list[col].append(roi)

    if not savename.endswith('.npz'):
        savename += '.npz'
    np.savez(savename, l_peaks=l_peaks, r_peaks=r_peaks, l_peaks_col=l_peaks_col, r_peaks_col=r_peaks_col)

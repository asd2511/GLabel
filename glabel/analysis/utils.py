"""
Module providing utilities for analysis and overall data handling. Functions can be used to easily load and process
portions of annotation data. Functions are also heavily used in preparation process of neural network training data.
"""
import os
import json
import glob
import re
from typing import List, Tuple, Union

import flammkuchen as fl
import numpy as np
from PyQt5.QtCore import QPointF
from scipy.interpolate import griddata, Rbf, bisplrep, bisplev
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


def to_uint8(img_data) -> np.ndarray:
    """
    Convert image data from its usual 10bit format (after conversion from CINE format) to 8bit.

    Conversion follows:

    .. math::

        \\frac{data - data.min()}{data.max() - data.min()} * 255

    :param np.ndarray img_data: Image data
    :return: The image data converted to 8bit datatype.
    """
    # The data is of type np.uint16, but the actual values were retrieved from 10bit data
    # ==> We need to manually rescale the values to the 8bit range before converting the datatype to preserve the
    # image contents
    # TODO: This is stupid slow but the only method that I got working quickly --> Find better solution!
    img_data = ((img_data - img_data.min()) / (img_data.max() - img_data.min())) * 255
    return img_data.astype(np.uint8)


def load_image_data(file, uint8=False, symrange=False, only_labeled=False) -> np.ndarray:
    """
    Load the image data from a .rois or HDF5 path.

    The loaded .h5 path must contain one of ['images', 'ims'] as key for the image data.

    :param file: Path to the path.
    :type file: str
    :param uint8: Boolean setting to change datatype of returned image data to unsigned 8bit. This will convert the data
        by rescaling the unsigned 16bit (or actually 10bit) to the [0, 255] range and then change the underlying type.
    :type uint8: bool
    :param symrange: Boolean setting to change the value range of the returned image data to [-1, 1]. This can be used
        for training neural networks that might have disadvantages with dark values being close to 0.
    :type symrange: bool
    :param only_labeled: Boolean setting for loading only the image frames that have labels assigned to them as read
        from the passed .rois file. This will only work if a .rois file has been specified (not for a .h5 file)
    :type only_labeled: bool
    :return: numpy.ndarray containing the image data.

    :raises KeyError: if no valid dataset key is found in the loaded .h5 dataset.
    :raises AssertionError: if the specified file is no .rois file.
    """
    if only_labeled:
        assert '.rois' in file, "You can only limit image loading to labeled data by specifying a .rois file!"

    num_frames_labeled = None
    if '.rois' in file:
        with open(file, 'r') as f:
            save_data = json.load(f)
        file = save_data['image_file']
        if not os.path.isfile(file):
            backup_file = os.path.split(file)[1].replace('.rois', '.h5')
            if os.path.isfile(backup_file):
                file = backup_file

        if only_labeled:
            # Count the number of frames that have at least one ROI placed to load only that many frames from the file
            num_frames_labeled = sum(
                1 if any(i['roi_positions'][roi]['placed'] for roi in range(len(i['roi_positions']))) else 0
                for i in save_data['frames'])

    datasets = fl.meta(file)
    keys = list(datasets.keys())
    if 'images' in keys:
        img_data = fl.load(file, '/images', sel=fl.aslice[:num_frames_labeled])
    elif 'ims' in keys:
        img_data = fl.load(file, '/ims', sel=fl.aslice[:num_frames_labeled])
    else:
        raise KeyError(f"{file} does not contain any of ['images', 'ims'] as a dataset key!")

    if uint8:
        return to_uint8(img_data)
    elif symrange:
        # Rescale to [-1, 1]
        img_data = 2 * ((img_data - img_data.min()) / (img_data.max() - img_data.min())) - 1
        return img_data
    else:
        return img_data


def load_rois_array(file) -> np.ndarray:
    """
    Load the ROI placements from a .rois path into a numpy.ndarray of shape [frames, num_rois, 2].

    Positions of unplaced ROIs will be listed as [-1, -1] entries in the array.

    :param file: Path to the path to load ROI placements from.
    :type file: str
    :return: np.ndarray of shape [frames, num_rois, 2], with num_rois being the total number of possible ROI placements
        (typically num_rows * num_cols * 2 for stereoscopic grid)
    """
    with open(file, 'r') as f:
        data = json.load(f)['frames']

    num_frames = len(data)
    # Total number of rois could also be inferred from values of 'num_rows' and 'num_columns' in .rois dictionary.
    num_rois = len(data[0]['roi_positions'])
    coord_keys = list(data[0]['roi_positions'][0]['pos'].keys())  # Should be ['x', 'y']
    num_coords = len(coord_keys)

    roi_array = np.empty((num_frames, num_rois, num_coords))
    for frame_idx in range(num_frames):
        for roi_idx in range(num_rois):
            for coord_idx, coord_key in enumerate(coord_keys):
                roi_array[frame_idx][roi_idx][coord_idx] = \
                    data[frame_idx]['roi_positions'][roi_idx]['pos'][coord_key]

    return roi_array


def load_rois_qpoint_list(file) -> list:
    """
    Load the ROI placements from a .rois file into a list of lists of QPointF objects.

    :param str file: Path to the .rois file to load.
    :return: List of lists of QPointF objects.
    """
    rois_array = load_rois_array(file)
    n_frames, n_rows, n_cols = rois_array.shape

    rois_list = []
    for frame_rois in rois_array:
        frame_list = [QPointF(r[0], r[1]) for r in frame_rois]
        rois_list.append(frame_list)

    return rois_list


def load_rois_patch_relative(file, patch_size=(224, 224)) -> np.ndarray:
    """
    Load ROI placements from a .rois file with coordinates being relative to the specified patch size around the center
    of the annotations.

    This is especially useful in creation of segmentation maps for training the segmentation network to find sutures.
    E.g., for a segmentation map of size 224x224 px, an image region of that size, centered on the mean central
    coordinate will be determined. The coordinates of ROI annotations made on a 768x768 px image are then converted to
    the smaller coordinate system in order to easily mark an 224x224 image with the annotations.

    :param str file: Filepath to the .rois file to load annotations from.
    :param tuple patch_size: Tuple of patch size as (width, height). Defaults to (224, 224).
    :return: Numpy array of shape [frames, sutures per frame, 2] with coordinates converted to patch-sized coordinate
        system.
    """
    with open(file, 'r') as f:
        save_dict = json.load(f)
    data = save_dict['frames']  # Get the saved ROI placements
    num_rois = save_dict['num_rows'] * save_dict['num_columns']

    rect_frame_idcs, rects = get_rois_rect(file, 0, 'wh')  # Get the bounding boxes of the suture regions

    num_frames = len(rects)  # Number of labeled(!) frames (not the actual number of frames in the data)
    roi_array = np.empty((num_frames, num_rois*2, 2))
    for frame_idx, frame_rects in zip(rect_frame_idcs, rects):
        pos = data[frame_idx]['roi_positions']

        l_rect = frame_rects[0]
        l_xc = l_rect[0] + l_rect[2] // 2  # Central x coordinate
        l_yc = l_rect[1] + l_rect[3] // 2  # Central y coordinate
        # Offset as [delta_y, delta_x] of top-left corner of bounding box
        l_offset = (l_yc - patch_size[0]//2, l_xc - patch_size[1]//2)  # Top-left of patch

        r_rect = frame_rects[1]
        r_xc = r_rect[0] + r_rect[2] // 2
        r_yc = r_rect[1] + r_rect[3] // 2
        # Analogous to left side
        r_offset = (r_yc - patch_size[0]//2, r_xc - patch_size[1]//2)

        # Iterate over ROIs from left side
        for roi_idx in range(num_rois):
            if pos[roi_idx]['placed']:
                # Make saved ROI placements relative to the desired bounding box size
                roi_y = (pos[roi_idx]['pos']['y'] - l_offset[0]) / patch_size[0]
                roi_x = (pos[roi_idx]['pos']['x'] - l_offset[1]) / patch_size[1]
            else:
                roi_y = roi_x = -1
            roi_array[frame_idx][roi_idx] = roi_x, roi_y
        # Do the same for the ROIs on the right side
        for roi_idx in range(num_rois, num_rois*2):
            if pos[roi_idx]['placed']:
                roi_y = (pos[roi_idx]['pos']['y'] - r_offset[0]) / patch_size[0]
                roi_x = (pos[roi_idx]['pos']['x'] - r_offset[1]) / patch_size[1]
            else:
                roi_x = roi_y = -1
            roi_array[frame_idx][roi_idx] = roi_x, roi_y

    return roi_array


def get_rois_rect(file, padding=5, mode='corners') -> Tuple[List, List]:
    """
    Find the rectangles containing all placed ROIs from each frame of a .rois path.

    The rectangle will enclose the centers of all placed ROIs, with additional padding applied to all sides equally.
    The returned rectangles can then be used to extract the image regions relevant for training of the suture detection
    model.

    Frames containing no placed ROIs in the data will be skipped and no rectangle will be listed for them. This causes
    the total length of the returned list to be `number of frames - number of empty frames`.

    ==========  =========================================================================
    **Mode**    **Rectangle values**
    'corners'   Two opposing corners as [min_x, min_y, max_x, max_y]
    'wh'        Top-left corner along width and height as [min_x, min_y, width, height]
    ==========  =========================================================================

    :param file: Path to the .rois path from which to load positions.
    :type file: str
    :param padding: defaults to 5. Padding in pixels applied to all sides of the patches to ensure full visibility of
        the sutures located at edges.
    :type padding: int
    :param mode: defaults to 'corners'. One of ['corners', 'wh']. Setting the type of values returned for the
        rectangles.
    :type mode: str
    :return: Two lists of equal length. First list holds frame indices for rectangles in second list at same position.
        Rectangles list is a list of lists containing the coordinates of upper-left and lower-right corner or height and
        width of rectangles. Dimension of list is [#frames-#empty, 2, 4]. First entry for left view, second for right
        view. Each rectangle is described by 4 values depending on mode parameter.

    :raises ValueError: if invalid mode is specified.
    :raises AssertionError: if specified padding is < 0.
    """
    assert padding >= 0, f"Passed padding parameter has to be greater than 0! (encountered {padding})"

    # Get ROI placements and grid dimensionality
    with open(file, 'r') as f:
        data = json.load(f)
        rows = data['num_rows']
        cols = data['num_columns']

    # ROI positions of all frames
    rois = load_rois_array(file)

    # Extract rectangles showing all placed ROIs for both left and right view from each frame
    frame_indices = []
    rects = []
    for frame_idx in range(rois.shape[0]):
        pos = rois[frame_idx]
        pos[pos == [-1, -1]] = np.nan  # Replace unplaced ROIs with NaN values
        if np.all(np.isnan(pos)):
            continue

        # Find the enclosing rectangle holding all ROIs for each view
        left_pos = pos[:rows * cols]  # ROIs in left view
        right_pos = pos[rows * cols:]  # ROIs in right view

        l_min_x, l_min_y = np.around(np.nanmin(left_pos, axis=0).ravel()).astype(int)
        l_max_x, l_max_y = np.around(np.nanmax(left_pos, axis=0).ravel()).astype(int)

        r_min_x, r_min_y = np.around(np.nanmin(right_pos, axis=0).ravel()).astype(int)
        r_max_x, r_max_y = np.around(np.nanmax(right_pos, axis=0).ravel()).astype(int)

        frame_indices.append(frame_idx)
        if mode == 'corners':
            rects.append(
                [[l_min_x - padding, l_min_y - padding, l_max_x + padding, l_max_y + padding],
                 [r_min_x - padding, r_min_y - padding, r_max_x + padding, r_max_y + padding]])
        elif mode == 'wh':
            l_width = l_max_x - l_min_x + 2 * padding
            l_height = l_max_y - l_min_y + 2 * padding
            r_width = r_max_x - r_min_x + 2 * padding
            r_height = r_max_y - r_min_y + 2 * padding
            rects.append(
                [[l_min_x - padding, l_min_y - padding, l_width, l_height],
                 [r_min_x - padding, r_min_y - padding, r_width, r_height]])
        else:
            raise ValueError(f"Unknown mode {mode} encountered in get_rois_rect!")

    return frame_indices, rects


def extract_rois_patch(file, padding=5, unisize=None, uint8=False) -> List[np.ndarray]:
    """
    Extract the image patch(es) containing all placed ROIs from each frame.

    :param file: Path to the .rois path from which to load positions.
    :type file: str
    :param padding: defaults to 5. Padding in pixel applied to all sides of the patches to ensure full visibility of the
        sutures located at edges.
    :type padding: int
    :param unisize: Universal dimension that all of the returned patches should have as [width, height].
    :type unisize: Tuple[int]
    :param uint8: Boolean setting to change datatype of returned image data to unsigned 8bit. This will convert the data
        by rescaling the unsigned 16bit (or actually 10bit) to the [0, 255] range and then change the underlying type.
    :type uint8: bool
    :return: List of numpy.ndarrays of length #frames (\*2 for stereo grid) containing the extracted image patches from
        the original image data. Each numpy will have varying dimensions based on the size of the respective patch.

    :raises ValueError: if the original image data path is not a .h5 path.
    :raises AssertionError: if the specified unisize is not a 2-value tuple/list.

    .. todo:: Extend the valid path types to include the same as for opening in GUI.
    """
    if unisize:
        assert len(unisize) == 2, "Specified size must be of format [width, height]!"
    # Get the enclosing rectangle coordinates
    frames, rects = get_rois_rect(file, padding, 'wh')  # Returns lists of [min_x, min_y, width, height]

    # Find path to the original image data path
    with open(file, 'r') as f:
        # data_file = json.load(f)['image_file']
        data_file = file.replace('.rois', '.h5')
    if '.h5' not in data_file:
        raise ValueError(f"Function can currently only handle .h5 (hdf5) files! (tried to load {file})")

    # Load all the frames as image data
    img_data = load_image_data(data_file)

    patches = []
    for idx, frame_idx in enumerate(frames):
        l_rect = rects[idx][0]
        r_rect = rects[idx][1]
        img = img_data[frame_idx]

        # Extract both patches from the image data
        if not unisize:
            l_left = l_rect[1]
            l_right = l_rect[1] + l_rect[2]
            l_top = l_rect[0]
            l_bottom = l_rect[0] + l_rect[3]

            r_left = r_rect[1]
            r_right = r_rect[1] + r_rect[2]
            r_top = r_rect[0]
            r_bottom = r_rect[0] + r_rect[3]
        else:
            w_offset = unisize[0] // 2
            h_offset = unisize[1] // 2
            l_center = [l_rect[0] + l_rect[2] // 2, l_rect[1] + l_rect[3] // 2]
            l_left = l_center[0] - w_offset
            l_right = l_center[0] + w_offset
            l_top = l_center[1] - h_offset
            l_bottom = l_center[1] + h_offset

            r_center = [r_rect[0] + r_rect[2] // 2, r_rect[1] + r_rect[3] // 2]
            r_left = r_center[0] - w_offset
            r_right = r_center[0] + w_offset
            r_top = r_center[1] - h_offset
            r_bottom = r_center[1] + h_offset

        l_patch = img[l_top:l_bottom, l_left:l_right]
        r_patch = img[r_top:r_bottom, r_left:r_right]

        patches.append(l_patch)
        patches.append(r_patch)

    if uint8:
        return to_uint8(np.asarray(patches))
    else:
        return patches


def normalize_peaks(peaks) -> np.ndarray:
    r"""
    Normalize the coordinates of peaks/2d coordinates.

    .. note::

        Normalization follows the standard score/z-score calculation.


    .. math::

        z = \frac{x-\mu}{\sigma}


    :param peaks: The peaks/2d coordinates to normalize.
    :type peaks: Union[List[float], np.ndarray]
    :return: The peaks normalized according to the z-score.
    """
    peak_mean = np.mean(peaks, axis=0)
    peak_std = np.std(peaks, axis=0)
    peaks_normed = (peaks - peak_mean) / peak_std

    return peaks_normed


def bbox_pca_transform_peaks(peaks) -> np.ndarray:
    """
    Transform peaks using PCA on the convex hull bounding box.

    The bounding box of the convex hull is used instead of the peak coordinates themselves
    because some suture grids are rotated in such a way that the principle components of the
    peak coordinates point along the diagonals of the grid. Performing raw PCA on such points
    does not align the grid well with the coordinate system.

    Instead, the convex hull of the peaks is calculated and the best fitting bounding box
    determined. The found bounding box is then "squished" vertically to artificially enforce
    the first principle component to be along the horizontal. PCA is calculated for the
    bounding box and peaks are transformed according to it.

    An additional check is performed that catches cases where the PCA transform would flip
    the peaks along one of the axes. In such a case, the flip is reversed before returning
    the transformed peaks.

    :param np.ndarray peaks: (N, 2) Peak coordinates array
    :return: The PCA transformed peaks.
    """
    # "Squishing" the peaks vertically bounding box with principle component along horizontal
    bbox = minimum_bounding_rectangle(peaks * [1, 0.5])
    pca = PCA().fit(bbox)

    # Check if the transform inverts any of the axes by looking at
    # the transformed bbox. If it does, revert the inversion on the
    # peaks after they have been transformed.
    pca_bbox = pca.transform(bbox)
    flip_x = any(np.sign(pca_bbox[c, 0]) != np.sign(bbox[c, 0]) for c in range(4))
    flip_y = any(np.sign(pca_bbox[c, 1]) != np.sign(bbox[c, 1]) for c in range(4))

    pca_peaks = pca.transform(peaks)
    if flip_x:
        pca_peaks = pca_peaks * [-1, 1]
    if flip_y:
        pca_peaks = pca_peaks * [1, -1]

    return pca_peaks


def get_gt_col_labels(peaks) -> np.ndarray:
    """
    Return the ground-truth column membership for specified peaks.

    .. caution::
        The list of peaks must be ordered according to their suture ids in
        order to find their column membership by using their index.
        Loading peaks using utils.get_rois... returns labeled sutures in
        this correct way. The suture starts at 0 at the lower-left corner of
        sutures and increases linearly going row-wise.

    :param peaks: List of 2D peak/suture coordinates. Must be of shape (35, 2)
    :type peaks: List[float]
    :return: Labels indicating column membership for each peak.
    """
    assert len(peaks) == 35, "Full suture coordinates must be passed to generate corresponding column labels!"

    return np.tile(np.arange(5), 7)


def get_peaks_col_labeled(file) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Get the list of manually labeled sutures and their column membership labels from the specified file.

    The peaks correspond to what will be the result of peak finding on UNet probability maps for the sutures.

    :param file: Path to the .rois file to load labels from.
    :type file: str
    :return: Manually labeled suture coordinates and their corresponding column label.
        List entries are alternating left and right views of each frame in correspondence with the return structure of
        utils.extract_rois_patch()
    """
    peaks = []
    labels = []
    rois = load_rois_patch_relative(file, (224, 224))
    for frame in rois:
        for side in [frame[:35], frame[35:]]:
            valid_indices = ~np.any(side == -1, axis=1)
            valid_rois = side[valid_indices]
            valid_rois = valid_rois * 224
            valid_labels = get_gt_col_labels(side)[valid_indices]

            peaks.append(valid_rois)
            labels.append(valid_labels)

    return peaks, labels


def get_same_filesplit(directory) -> Tuple[List[str], List[str]]:
    """
    Get the same filesplit as in the specified directory.

    This is intended to be used for copying the file split that was used for e.g. training one of the neural networks.
    Specifying the top directory for the data directory of one of the networks, this function will search for the
    basenames of files that were used in creating the data for the network and return the .rois files from them.

    .. caution::

        This function will only work as intended if the specified data directory contains the directories 'train' and
        'validation' as direct subdirectories!

    :param directory: The top-level data directory containing training and validation files.
    :type directory: str
    :return: Tuple of lists. First list for files in the training set, second for files in the validation set.
    """
    # Function that splits off the frame number, side and file ending from an image or h5 file
    fsplit = lambda f: re.split('_\d{1,3}_\D.\w{2,3}$', f)[0]

    # Process a file path by splitting off frame number, side and file ending and join it back together with the
    # correct data directory and the .rois file ending
    proc_f = lambda p, f: os.path.join(p, os.path.basename(fsplit(f))+'.rois')

    # Find all files in the trianing and validation directories
    train_files = glob.glob(os.path.join(directory, 'train', '**/*'))
    val_files = glob.glob(os.path.join(directory, 'validation', '**/*'))

    unique_train_files = list(set([proc_f('daten/fuer_nn', f) for f in train_files]))
    unique_val_files = list(set([proc_f('daten/fuer_nn', f) for f in val_files]))

    return unique_train_files, unique_val_files


def get_gaussian_peak_fill(peak_r, blob_sigma) -> np.ndarray:
    """
    Creates a gaussian blob usable for marking peaks on suture maps for training or evaluation.

    The peak fill matrix will be of dimensions [peak_r*2+1, peak_r*2+1] in order to stay consistent with non-gaussian
    peak fill matrices.

    :param peak_r: Radius of peak fill matrix
    :type peak_r: int
    :param blob_sigma: Sigma parameter for gaussian distribution creating the peak fill matrix.
    :return: [peak_r*2+1, peak_r*2+1] matrix holding the gaussian peak fill.
    """
    x, y = np.meshgrid(np.linspace(-1, 1, peak_r * 2 + 1), np.linspace(-1, 1, peak_r * 2 + 1))
    d = np.sqrt(x * x + y * y)
    d -= np.min(d)  # Makes sure that the central pixels of the peak_fill will have value 1
    sigma, mu = blob_sigma, 0.0
    peak_fill = (np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2))))

    return peak_fill


def get_binary_peak_fill(peak_r) -> np.ndarray:
    """
    Creates a binary, rectangular peak fill matrix for marking peaks on suture maps for training or evaluation.

    The peak fill matrix will be of dimensions [peak_r*2+1, peak_r*2+1] to have center pixel align exactly with marked
    ROI positions.

    :param peak_r: Radius of peak fill matrix.
    :return: [peak_r*2+1, peak_r*2+1] matrix holding the binary peak fill.
    """
    return np.ones((peak_r * 2 + 1, peak_r * 2 + 1))


def mark_map(segmap, rois, peak_r, gaussian=False, sigma=1.0) -> np.ndarray:
    """
    Mark the passed segmentation map at the ROI locations.

    :param segmap: Matrix used as the map to be marked with suture locations.
    :type segmap: np.ndarray
    :param rois: ROI locations of a single patch to be marked on map.
    :type rois: np.ndarray
    :param peak_r: Radius for marking peaks on map.
    :type peak_r: int
    :param gaussian: Boolean setting if markings should be gaussian blobs or binary rectangles.
    :type gaussian: bool
    :param sigma: Variance of gaussian to be used when marking as gaussian blobs.
    :return: Map with suture locations marked according to parameters.
    """
    if gaussian:
        peak_fill = get_gaussian_peak_fill(peak_r, sigma)
    else:
        peak_fill = get_binary_peak_fill(peak_r)

    for roi in rois:
        x = int(roi[0])
        y = int(roi[1])
        segmap[y - peak_r - 1:y + peak_r, x - peak_r - 1:x + peak_r] = peak_fill

    return segmap


def interpolate_missing_rois(rois, num_rows, num_cols, use_nearest=True, interp_function='bispline'):
    """
    Interpolate any non-placed ROIs in order to allow display of 3D surface mesh data.

    Interpolation is performed using scipy's `griddata` [#]_ interpolation method using the `'nearest'` method. This will
    interpolate any missing placements with adding it as the closest available placement. This should definitely be
    changed in the future, but data is too sparse for interpolation with b-splines.

    .. [#] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    :param rois:
    :param num_rows:
    :param num_cols:
    :param bool use_nearest:
    :param str interp_function:
    :return:
    """
    interp_rois = np.empty_like(rois)
    frame_cutoff = None
    for frame_idx in range(rois.shape[0]):
        grid_rois = rois[frame_idx].reshape(num_rows, num_cols, 4)
        # Cancel execution when an unlabeled frame is encountered
        if np.all(np.isnan(grid_rois)):
            frame_cutoff = frame_idx
            break

        missing_rows, missing_cols = np.where(np.any(np.isnan(grid_rois), axis=2))
        placed_points = np.vstack(np.where(~np.any(np.isnan(grid_rois), axis=2))).T

        x = placed_points[:, 1]
        y = placed_points[:, 0]
        interp_grid = np.copy(grid_rois)
        for axis in range(4):
            values = grid_rois[:, :, axis][y, x]
            ## ----Using interp2d---- => Error because of bad parameters ##
            # f = interp2d(x, y, values, kind=method)
            # Returns evaluation over the cross-product of x and y coordinates, but we only want the values at the
            # specific coordinates -> Take first column
            # interp_values = f(missing_cols, missing_rows)[0]
            ## ----Using bispline directly---- => Error because of bad parameters ##
            # tck = bisplrep(x, y, values, s=len(x))
            # interp_values = bisplev(missing_cols, missing_rows, tck)
            # if len(missing_cols) > 1:
            #     interp_values = interp_values[:, 0]
            if use_nearest:
                # -- Using griddata to interpolate using nearest point -- #
                points = np.vstack([y, x]).T
                interp_values = griddata(points, values, (missing_rows, missing_cols), method='nearest')
                interp_grid[missing_rows, missing_cols, axis] = interp_values
            else:
                # -- Using radial basis functions to interpolate data using cubic interpolation -- #
                rbf = Rbf(x, y, values, function=interp_function)
                interp_values = rbf(missing_cols, missing_rows)
                interp_grid[missing_rows, missing_cols, axis] = interp_values

        interp_rois[frame_idx] = interp_grid.reshape(-1, 4)

    if frame_cutoff is not None:
        interp_rois = interp_rois[:frame_cutoff]

    return interp_rois


def id_pos_translate(val, rows, cols, stereo, unify_stereo=True) -> Union[int, Tuple[int]]:
    """
    Translate between a 1D ID and the 2D grid position of an ROI.

    Depending on whether the passed `val` is a single integer or a tuple the function translates to the other
    identification format respectively.

    :param val: Identification value to translate. Can be either a single integer or a tuple of integers. Translation
        will result in the other identification format.
    :type val: Union[int, Tuple[int]]
    :param rows: Number of rows in the grid the identification belongs to.
    :type rows: int
    :param cols: Number of columns in the grid the identification belongs to.
    :type cols: int
    :param stereo: If the grid is stereoscopic or not.
    :type stereo: bool
    :param unify_stereo: Boolean setting to translate positions of the right stereoscopic view to the same row and col
        identifiers as the corresponding position of the left view. E.g.: For a Grid of dimensions 7 rows x 5 columns,
        both identifiers 0 and 35 are translated to (0, 0) using this setting.
    :return: The translated identification. Either a single integer specifying the grid_id, or a tuple of (row, column)
        membership.
    :raises NotImplementedError: if a type other than int or tuple is passed for `val`.
    """
    if isinstance(val, (int, np.int64, np.int32, np.int)):
        total_placements = rows * cols if not stereo else rows * cols * 2
        assert 0 <= val <= total_placements, f"Passed ID is outside of valid range for passed number of rows and cols!" \
                                             f"({val} outside [0, {total_placements}])"

        col_offset = 0
        if stereo and val >= rows * cols:
            val -= rows * cols
            if not unify_stereo:
                col_offset = cols

        row = val // cols
        col = (val % cols) + col_offset
        return (row, col)

    elif isinstance(val, (tuple, list)):
        assert 0 <= val[0] <= rows, "Passed row value is outside of range [0, #rows]!"
        col_lim = cols if not stereo else cols * 2
        assert 0 <= val[1] <= col_lim, "Passed column value is outside of range [0, #colums]!"

        offset = 0 if val[1] < cols else rows * cols

        return cols * val[0] + val[1] + offset

    else:
        raise NotImplementedError(f"Grid identifier translation for type {type(val)} not implemented! Please pass an"
                                  f"integer or a tuple of integers.")


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates

    https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

"""
Module providing functions for applying clustering methods to suture coordinates to assign grid positions.
"""
from typing import Union, List

import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def approx_col_centroids(ps) -> np.ndarray:
    """
    Approximate horizontal/x centroids of each column at vertical position of total mean.

    The total mean is calculated from all specified peaks. After having found the total mean, the column centroids are
    approximated by creating 5 centroids that are evenly spread between the leftmost and rightmost peak found on the
    same vertical level as the total centroid.

    :param ps: Peak coordinates of sutures as [x, y].
    :type ps: np.ndarray
    :return: List of the 5 approximated column centroids.
    """
    mean_coord = np.mean(ps, axis=0)  # Total mean of all peaks is initial centroid
    # Find the horizontal extremes in the data --> These are the initial assumptions for spreading the centroids
    min_x = ps[np.argmin(ps[:, 0])]
    max_x = ps[np.argmax(ps[:, 0])]
    # Build centroids based on the vertical mean and the horizontal spread limits
    xs = np.linspace(min_x[0], max_x[0], 5)
    centroids = np.array([xs, [mean_coord[1]] * 5]).T
    # Find the actually closest peak to the limiting points while favoring horizontally close points by a factor of 2
    closest_left = ps[abs((centroids[0] - ps) * [1, 2]).sum(axis=1).argmin()]
    closest_right = ps[abs((centroids[-1] - ps) * [1, 2]).sum(axis=1).argmin()]
    # Update the centroids with the new horizontal limits
    close_xs = np.linspace(closest_left[0], closest_right[0], 5)
    close_centroids = np.array([close_xs, [mean_coord[1]] * 5]).T

    return close_centroids


def get_covar(angle=2.148, w=51.5089, h=14.9394) -> np.ndarray:
    """
    Construct a covariance matrix from specified properties.

    If any parameter is not specified, it defaults to the value for the total average covariance calculated from the
    manually labeled suture data:

        angle=2.148
        w=51.5089
        h=14.9394

    .. note::

        Based on math from:
        https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

    :param angle: The rotation angle of the covariance matrix in degrees. (Positive = counter-clockwise)
    :type angle: float
    :param w: Width of the covariance.
    :type w: float
    :param h: Height of the covariance.
    :type h: float
    :return: Covariance matrix as 2x2 numpy array
    """
    theta = -angle * (np.pi / 180)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    S = np.array([[h / 2, 0], [0, w / 2]])
    T = R @ S

    return T @ T.T


def kmeans_cluster(peaks, fac) -> np.ndarray:
    """
    Cluster the specified peaks into columns using k-means.

    :param peaks: Peak coordinates of sutures.
    :type peaks: Union[List[float], np.ndarray]
    :param fac: Distance weighting factor for weighting horizontal and vertical distances. (Ordering is [wx, wy])
        If only a single value is passed, it is assumed as wy, with wx being set to 1.
    :type fac: Union[tuple, list, int]
    :return: Labels for predicted column-membership of each peak.
    """
    if type(fac) != tuple and type(fac) != list:
        fac = [1, fac]
    centroids = approx_col_centroids(peaks)
    X = peaks * fac
    kmeans = KMeans(5, centroids * fac, 1)
    labels = kmeans.fit_predict(X)

    return labels


def gmm_cluster(peaks: list, fac=None, angle=None, width=None, height=None, init_covars=True) -> tuple:
    """
    Cluster the specified peaks into columns using a GMM.

    :param peaks: Peak coordinates of sutures.
    :param list: Distance weighting factor for weighting horizontal and vertical distances.
    :return: Labels for column-membership of each peak.
    """
    centroids = approx_col_centroids(peaks)

    if fac is None:
        fac = [1., 1.]
    elif not isinstance(fac, tuple) or not isinstance(fac, list):
        fac = [1, fac]

    # If any of the covariance variables is missing, we replace it by default with the average
    # covariance variable calculated from all manually labeled sutures
    if not angle:
        angle = 2.1487
    if not width:
        width = 51.5089 * fac[0]
    if not height:
        height = 14.9394 * fac[1]

    if init_covars:
        covariance = get_covar(angle, width, height)
        precisions = np.tile(np.linalg.inv(covariance), (5, 1, 1))
    else:
        precisions = None

    X = peaks * fac
    gmm = GaussianMixture(5, 'full', means_init=centroids * fac, precisions_init=precisions)
    labels = gmm.fit_predict(X)

    return labels, gmm


def postprocess_col_clusters(clustered_rois, n_rows, alpha=1.5) -> list:
    """
    Perform post-processing clean up on column clustered ROIs.

    The clean up process will search for any column clusters with more than `n_rows` ROIs assigned
    to it. All found columns have ROIs reassigned to a neighboring column with less than the maximum number of
    rows. If both neighbors have missing ROIs, assign to the column with the closest other ROI.

    :param list clustered_rois: List of np.ndarrays. Each list entry corresponds to a column. The (N, 2) array
        within each list contains the ROI coordinates assigned to that column.
    :param int n_rows: Number of rows in the suture grid.
    :param float alpha: Factor for considering horizontal outliers in column clustering results.
    :return: List of np.ndarrays in same ordering with lists having a max length of `n_rows`.
    """
    rois_copy = clustered_rois.copy()
    n_cols = len(rois_copy)
    col_lengths = [len(rois_copy[i]) for i in range(n_cols)]

    for i in range(n_cols):
        if col_lengths[i] > n_rows:

            # Find a candidate ROI to reassign
            reassign_candidate_idx = find_reassign_roi(rois_copy[i])
            reassign_candidate = rois_copy[i][reassign_candidate_idx]

            to_left = False
            to_right = False
            # Check if the neighboring columns have any free rows
            if i > 0:
                to_left = True if col_lengths[i-1] < n_rows else False
            if i < n_cols - 1:
                to_right = True if col_lengths[i+1] < n_rows else False

            # Trivial case if only one column is a candidate
            if to_left != to_right:
                to_idx = i-1 if to_left else i+1
            # If both are valid candidates, choose the one that has the closest ROI to the
            # reassigned ROI.
            elif to_left == to_right is True:
                l_dists = np.linalg.norm(rois_copy[i-1] - reassign_candidate)
                r_dists = np.linalg.norm(rois_copy[i+1] - reassign_candidate)
                closest = np.argmin([np.min(l_dists), np.min(r_dists)])
                to_idx = i-1 if closest == 0 else i+1
            # If no neighboring column has any free rows, something went terribly wrong in clustering
            else:
                raise NotImplementedError(f"Both neighboring columns of column {i} are full! Something went "
                                          f"terribly wrong in clustering!")

            # Swap the ROI
            other_col = rois_copy[to_idx].tolist()
            other_col.append(reassign_candidate.tolist())
            rois_copy[to_idx] = np.asarray(other_col)
            rois_copy[i] = np.delete(rois_copy[i], reassign_candidate_idx, axis=0)

    # Try to fix horizontal outliers in the columns
    rois_copy = fix_horizontal_outliers(rois_copy, alpha)

    return rois_copy


def find_reassign_roi(rois) -> int:
    """
    Find a candidate ROI in `rois` considered to be an outlier based on distance to its neighbors.

    The ROI with the greatest horizontal displacement to its vertically closest neighbor is chosen as the
    outlier to be reassigned.

    :param np.ndarray rois: (N, 2) array of N ROIs. One ROI from this array is chosen to be reassigned.
    :return: Index of the ROI in `rois` chosen to be reassigned.
    """
    # Sort the ROIs by their vertical coordinate
    sort_idcs = np.argsort(rois[:, 1])
    v_sort = rois[sort_idcs]
    # Compute differences in horizontal position between neighbors
    h_diff = np.diff(v_sort[:, 0])
    # The candidate to be reassigned is the ROI with the largest difference to its neighbor
    # (+1 because np.diff removes first element from array)
    return sort_idcs[np.argmax(h_diff) + 1]


def fill_col_gaps(rois, n_rows, beta=1.5):
    """
    Fill any gaps in the sorted ROIs.

    :param list rois: List of (N, 2) np.ndarrays of ROI coordinates clustered into columns and already sorted by row.
    :param int n_rows: Number of rows in the suture grid.
    :param float beta: Factor for which a vertical outlier is assumed.
    :return: List of np.ndarrays where potential gaps in the columns have been filled with invalid [-1, -1] point.
    """
    rois_copy = rois.copy()

    # Search for possible gaps in the rows
    col_lengths = [len(c) for c in rois_copy]
    n_cols = len(rois_copy)

    # Determine the average vertical distance between ROIs
    avg_v = np.mean([np.mean(np.diff(col[:, 1])) for col in rois_copy])

    for i in range(n_cols):
        if col_lengths[i] < n_rows:
            # A gap in the column is identified by searching for a vertical distance that is significantly larger
            # than the mean vertical distance between rows.
            col = rois_copy[i]
            v_dists = np.diff(col[:, 1])
            above_gap_idcs = np.where(abs(v_dists) > beta*abs(avg_v))[0] + 1  # +1 because np.diff removes first element
            # Skip if no large vertical difference was found
            if len(above_gap_idcs) == 0:
                continue

            for agi in above_gap_idcs:
                # We need to insert the invalid [-1, -1] point before the found index
                fill = np.ones_like(rois_copy[i][0]) * -1
                rois_copy[i] = np.insert(rois_copy[i], agi, fill, axis=0)

    return rois_copy


def sort_into_rows(clustered_rois, n_rows, fill_gaps=True, beta=1.5):
    """
    Sort the column-clustered ROIs into rows using vertical relationships between them.

    :param list clustered_rois: List of (N, 2) np.ndarrays, where N is the number of ROIs in the column. Each column
        may have a different number of ROIs in it.
    :param int n_rows: Number of rows in the suture grid.
    :param bool fill_gaps: Toggle trying to identify gaps in columns and filling them with invalid [-1, -1] ROIs.
    :param float beta: Factor for determining vertical outliers for which a missing grid position is assumed.
    :return: List of (N, 2) np.ndarrays. Same shape/logic as the passed `clustered_rois` but column entries are sorted
        according to row membership.
    """
    rois_copy = clustered_rois.copy()

    # The row index of each ROI is simply its position within the vertically sorted column.
    rows = [np.argsort(rois_copy[i][:, 1])[::-1] for i in range(len(rois_copy))]
    sorted_rois = [rois_copy[i][rows[i]] for i in range(len(rois_copy))]

    if fill_gaps:
        sorted_rois = fill_col_gaps(sorted_rois, n_rows, beta)

    return sorted_rois


def fix_horizontal_outliers(clustered_rois, alpha=1.5):
    """
    Try to identify and fix any ROIs in the first row that are clustered into the wrong column.

    Fixing these "column base" ROIs is important because the final grid sort is based on the vertical positioning of
    ROIs within the same column. If the first ROI is wrong, all grid labels for that column will be offset by 1.

    :param list clustered_rois: List of (N, 2) np.ndarrays clustering ROIs into columns.
    :param float alpha: Factor for determining horizontal column outliers.
    :return: List of (N, 2) np.ndarrays with same clustering of ROIs into columns but with any found mistakes in column
        bases resolved.
    """
    rois_copy = clustered_rois.copy()
    n_cols = len(rois_copy)

    # Determine the average horizontal distance between ROIs
    col_med_h_pos = [np.median(col[:, 0]) for col in rois_copy]  # Median to better handle outliers
    col_avg_h_dist = [np.median(abs(np.diff(col[:, 0]))) for col in rois_copy]
    avg_h_dist = np.mean(col_avg_h_dist)

    for i in range(n_cols):
        col_rois = rois_copy[i]
        # Determine difference between horizontal distances and median distance. If a ROI has a distance of double the
        # mean distance, regard it as an outlier.
        h_med_offset = abs(col_rois - col_med_h_pos[i])[:, 0]
        outlier_candidate_idcs = np.where(h_med_offset > alpha*abs(col_avg_h_dist[i]))[0]

        # Reassign the outlier ROIs to the column with closest horizontal mean
        for outlier_idx in outlier_candidate_idcs:
            outlier = col_rois[outlier_idx]
            closest_col_mean = np.argmin(abs(outlier[0] - np.asarray(col_med_h_pos)))
            # If the outlier is nevertheless closest to the current column, leave it there
            if closest_col_mean == i:
                continue
            # Swap the ROI between the columns
            rois_copy[i] = np.delete(rois_copy[i], outlier_idx, axis=0)
            outlier_candidate_idcs -= 1  # Adjust remaining indices to reflect removed ROI
            rois_copy[closest_col_mean] = np.insert(rois_copy[closest_col_mean], 0, outlier, axis=0)

    return rois_copy


def grid_sort(x, col_y, n_rows=7, catch_gaps=True, alpha=1.5, beta=1.5, unprocessed=False) -> np.ndarray:
    """
    Sort ROIs into grid structure using previously obtained column membership labels.

    The sorting mechanism relies solely on the vertical relationship between ROIs in each column. The algorithm tries
    to catch any grid positions missing in the middle of columns, but this will not work super reliably.

    :param np.ndarray x: (N, 2) np.ndarray of data point coordinates.
    :param np.ndarray col_y: (N, 1) np.ndarray of column labels for data points.
    :param int n_rows: Number of rows in the grid. (Number of columns is inferred from the column labels).
    :param bool catch_gaps: If True, try to detect columns with missing grid positions.
    :param float alpha: Factor for determining horizontal distance outliers. Used in cleaning up column clustering
        results. Each ROI that has a horizontal distance > beta*(col_median-ROI) is considered an outlier and
        reassigned to a closer column.
    :param float beta: Factor for determining vertical distance outliers. For each outlier it is assumed that a grid
        position is not filled. Each ROI that has a horizontal distance > alpha*median_distance to its bottom column
        neighbor is considered an outlier and a missing grid posiiton is placed between them.
    :param bool unprocessed: Skip all attempts to clean up the grid sorting and perform raw vertical sorting per
        column.
    :return: (N, 1) np.ndarray of grid ID labels.
    """
    data = np.c_[x, np.arange(x.shape[0])]  # Add index column for later sorting of grid ids
    col_clustered = [data[col_y == c] for c in np.unique(col_y)]  # List of np.ndarrays grouping points into columns
    if not unprocessed:
        col_clustered = postprocess_col_clusters(col_clustered, n_rows, alpha)  # Fix columns with too many entries
    n_cols = len(col_clustered)

    row_sorted = sort_into_rows(col_clustered, n_rows, fill_gaps=(not unprocessed), beta=beta)

    grid_labels = []
    for c_idx, col in enumerate(row_sorted):
        btm_id = c_idx
        top_id = c_idx + len(col) * n_cols
        labels = np.arange(btm_id, top_id, n_cols)
        grid_labels.append(labels)

    labels_arr = np.asarray([l for col in grid_labels for l in col])
    row_sorted_arr = np.asarray([c for col in row_sorted for c in col])
    row_sorted_ma = np.ma.masked_equal(row_sorted_arr, -1)
    # Try-except for when no values are masked (which causes .mask to be False)
    try:
        labels_arr = labels_arr[~row_sorted_ma.mask[:, 0]]
    except IndexError:
        pass

    sorted_data = row_sorted_ma.compressed().reshape(-1, 3)
    sort_idcs = np.argsort(sorted_data[:, 2])
    grid_y = labels_arr[sort_idcs]

    return grid_y


def fit_poly_lines(x, y, deg=3):
    poly_funcs = []
    for col_idx in np.unique(y):
        col_points = x[y == col_idx].copy()
        col_points = col_points[np.argsort(col_points[:, 1])]  # Sort by vertical position
        fit = np.polyfit(col_points[:, 1], col_points[:, 0], deg=deg)  # Fit to predict the horizontal position
        poly = np.poly1d(fit)
        poly_funcs.append(poly)

    return poly_funcs

def build_poly_grid(x, y):
    polys = fit_poly_lines(x, y, 3)
    top = np.max(x[:, 1])
    bottom = np.min(x[:, 1])
    n_rows = min(np.bincount(y).max(), 7)
    row_ys = np.linspace(bottom, top, n_rows)

    # col_xs = np.array([np.median(x[y == col], axis=0)[0] for col in np.unique(y)])
    col_xs = [[] for _ in np.unique(y)]
    for col in np.unique(y):
        col_xs[col] = polys[col](row_ys)

    grid_coords = np.zeros((n_rows, 5, 2))
    for col in np.unique(y):
        grid_coords[:, col] = np.c_[col_xs[col], row_ys]

    return grid_coords


def fit_to_poly_grid(x, y):
    """
    Construct a grid based on the existing column clustering results and fit all ROIs to it to gain column and row
    memberships.

    The predicted grid will have its column positions created by a polynomial fit of the existing column clustering
    results. The row positions are created as equidistant in the range of the available data.

    Initial assignment:
        Each ROI is initially assigned to its closest grid position. Assignment order is according to distance to grid
        positions, meaning that ROIs already close to grid positions get assigned before distant ones. Only grid
        positions that are closer than 50% of the median distance between grid positions are considered. Already filled
        grid positions cannot be assigned a new ROI. ROIs with all close grid positions already filled will be left
        unassigned.

    Refining assignments:
        All initially unassigned ROIs will be assigned to their closest grid position by reassigning filled grid
        positions recursively. For the ROI assigned to the closest grid position of an unplaced ROI, the second-closest
        grid position will be considered. If that is also already filled, perform "second-best search" recursively until
        a ROI can be assigned to a free grid position.

    TODO: More testing!

    :param np.ndarray x: (N, 2) np.ndarray specifying N ROI coordinates to be assigned to grid positions.
    :param np.ndarray y: (N, 1) np.ndarray with initial column labels for ROIs.
    :return: (I, J) np.ndarray, with I and J being number of rows and columns in the grid respectively. Array entry
        at (v, w) specifies ROI coordinate for grid position at row v and column w.
    """
    grid_coords = build_poly_grid(x, y)  # Build grid based on vertical polynomials and horizontal equidistant lines
    flat_grid = grid_coords.reshape(-1, 2)

    # Construct a KDTree for both the grid positions and the available ROIs to find distances between them
    grid_tree = KDTree(flat_grid)
    rois_tree = KDTree(x)

    # The distance matrix between the KDTrees has its indices constructed as:
    # Row = Poly Grid ID, Column = ROI ID
    median_col_h_dists = np.array([np.median(abs(np.diff(grid_coords[:, c]))) for c in range(grid_coords.shape[1])])
    median_row_v_dists = np.array([np.median(abs(np.diff(grid_coords[r, :]))) for r in range(grid_coords.shape[0])])
    median_grid_dist = np.median(np.hstack((median_col_h_dists, median_row_v_dists)))
    # median_grid_dist = np.median(abs(np.diff(grid_coords)))  # Determine median distance between grid points as cutoff
    distmat = grid_tree.sparse_distance_matrix(rois_tree, 10).toarray()  # Masking distance matrix manually to
    distmat[np.where(distmat > .5*median_grid_dist)] = -1   # still keep distances of 0 as valid distance
    # Get the matrix coordinates of distances in ascending order
    asc_idcs = np.vstack(np.unravel_index(distmat.argsort(axis=None), distmat.shape)).T

    # New grid-shaped array which will hold the ROI coordinates for each grid position
    grid_assignments = np.zeros(grid_coords.shape)
    grid_ids = np.ones(grid_coords.shape[:2]) * -1
    # Keep track of which Grid positions have been filled and which ROIs have been assigned. All ROIs which could not
    # be assigned because their closest grid positions had other ROIs closer to them will be handled extra.
    filled_grid_idcs = []
    placed_roi_idcs = []
    # Iterate through all Grid ID - ROI Id pairs ordered in increasing distance. This makes sure that grid positions
    # are primarily filled with the ROI closest to them before any other ROI has the chance to fill the spot.
    for idx in asc_idcs:
        grid_idx, roi_idx = idx  # idx is index into distance matrix (row=grid, col=roi)
        # Only assign unplaced ROIs to unfilled grid positions. Additionally, the masking prevents assignments that
        # have a distance greater than 50% of the median distance between grid positions.
        if grid_idx not in filled_grid_idcs and roi_idx not in placed_roi_idcs and distmat[grid_idx, roi_idx] != -1:
            gc = flat_grid[grid_idx]
            roi = x[roi_idx]
            d = gc - roi
            grid_assignments[np.unravel_index(grid_idx, grid_coords.shape[:2])] = roi
            grid_ids[np.unravel_index(grid_idx, grid_coords.shape[:2])] = roi_idx
            # Flag the grid position and ROI as having been assigned.
            filled_grid_idcs.append(grid_idx)
            placed_roi_idcs.append(roi_idx)
    # plt.gca().invert_yaxis()

    # fix_missed_placements(x, grid_assignments, grid_tree, distmat, placed_roi_idcs, plot=True)
    unplaced_roi_idcs = set(range(distmat.shape[1])) - set(placed_roi_idcs)
    # Sort unpalaced ROIs by their vertical coordinate - start reassignment with lowest ROIs
    unplaced_roi_idcs = sorted(unplaced_roi_idcs, key=lambda idx: x[idx][1], reverse=True)
    touched_g_ids = [None]
    assignments_backup = grid_assignments.copy()
    ids_backup = grid_ids.copy()
    for unplaced_idx in unplaced_roi_idcs:
        recursive_assign_search(x, unplaced_idx, grid_assignments, grid_ids, grid_tree, placed_roi_idcs, touched_g_ids)
        touched = np.where(assignments_backup.reshape(-1, 2)[:, 0] != grid_assignments.reshape(-1, 2)[:, 0])[0]
        touched_g_ids.extend(touched)

    flipped_ids = np.flip(grid_ids, axis=0)
    flat_ids = flipped_ids.reshape(-1).astype(np.int)
    masked_ids = np.ma.masked_equal(flat_ids, -1)

    return np.argsort(masked_ids)[:len(x)]


def direction_filter_candidates(coords, direction, candidate_idcs, grid_assignments):
    # Disregard all grid positions that do not lie in search direction
    same_direction = np.zeros_like(candidate_idcs).astype(np.bool)
    for i, cand_idx in enumerate(candidate_idcs):
        candidate_coords = grid_assignments.reshape(-1, 2)[cand_idx]
        # Determine the vertical distance if the candidate grid position is filled. Otherwise, if the position is
        # vacant, make sure it is considered as valid by assigning it the `direction` value.
        candidate_vert_dist = (candidate_coords - coords)[1] if (candidate_coords != [0, 0]).all() else direction
        candidate_direction = np.sign(candidate_vert_dist)
        same_direction[i] = candidate_direction == direction

    if not any(same_direction):
        # raise ValueError("No grid positions found in the specified direction!")
        return False

    return candidate_idcs[same_direction]


def find_closest_grid_idx(coords, g_id, grid_tree, grid_assignments, direction, searched_g_ids):
    # Find the closest grid positions to this ROI
    # Different conditions exist for the closest positions:
    #   * If this ROI is unassigned --> Find the two closest positions (second as alternative search if first is
    #       rejected)
    #   * If this ROI is assigned to the grid --> Disregard its current grid position
    #   * If this ROI is a border ROI and it is assigned to its closest grid position --> Reject reassignment
    #   * If the request for reassignment came from an assigned ROI, disregard its grid position in all cases
    _, closest_grid_idcs = grid_tree.query(coords, 8)
    # Sort the found closest grid positions while forcing to stay in the column of the closest grid position
    closest_col = closest_grid_idcs[0] % grid_assignments.shape[1]
    closest_grid_idcs = closest_grid_idcs[closest_grid_idcs % grid_assignments.shape[1] == closest_col]
    # Remove any already touched grid positions
    for prev_id in searched_g_ids:
        closest_grid_idcs = closest_grid_idcs[closest_grid_idcs != prev_id]

    if g_id is not None:
        # Remove own grid position from the candidates
        closest_grid_idcs = closest_grid_idcs[closest_grid_idcs != g_id]

    closest_grid_idcs = direction_filter_candidates(coords, direction, closest_grid_idcs, grid_assignments)
    if closest_grid_idcs is False:
        return False

    if len(closest_grid_idcs) == 0:
        raise ValueError("No valid grid position found! All candidates removed by conditions!")

    return closest_grid_idcs[0]


def recursive_assign_search(x, roi_idx, grid_assignments, grid_ids, grid_tree, placed_idcs, searched_g_ids, try_alt=True,
                            direction=0):
    # Get all necessary info about the ROI to be assigned
    # What is needed is:
    #   * Is it already assigned?
    #   * If yes, which grid position is it assigned to?
    #   * Where in the 2D space is it located?
    #   * Is it in the top or bottom row?
    assigned = True if roi_idx in placed_idcs else False
    this_coords = x[roi_idx]
    this_row, this_col = [c[0] for c in np.where(grid_assignments == x[roi_idx])[:2]] if assigned else (None, None)
    this_g_id = np.ravel_multi_index((this_row, this_col, 0), grid_assignments.shape[:2] + (1,)) if assigned else None
    border = True if this_row in [0, grid_assignments.shape[0]-1] else False

    # Stop search if current suture is assigned to a border position. The border positions are mostly assigned
    # correctly, so we do not want to change them.
    if border:
        return False

    # If this is an unplaced suture, start by looking towards the bottom of the grid
    direction = 1 if direction == 0 else direction

    # Find the candidate grid position for this suture
    try:
        reassign_to_idx = find_closest_grid_idx(this_coords, this_g_id, grid_tree, grid_assignments, direction,
                                                searched_g_ids)
    except ValueError as e:
        print(e)
        return False
    reassign_to_row, reassign_to_col = np.unravel_index(reassign_to_idx, grid_assignments.shape[:2])

    # If the position is already filled --> Begin recursive search for a free grid position
    if not np.all(grid_assignments[reassign_to_row, reassign_to_col] == [0, 0]):
        to_reassign = grid_assignments[reassign_to_row, reassign_to_col]
        to_reassign_roi_idx = np.where(x == to_reassign)[0][0]
        if searched_g_ids == [None]:
            searched_g_ids = [this_g_id]
        else:
            searched_g_ids.append(this_g_id)

        if reassign_to_idx is not False:
            success = recursive_assign_search(x, to_reassign_roi_idx, grid_assignments, grid_ids, grid_tree,
                                              placed_idcs, searched_g_ids, try_alt, direction)
        else:
            success = False

        # If the recursive search in the direction of the closest grid position failed, try the second best position
        # (second closest position is enforced by passing the closest position as `requester_g_id`)
        if not success and not assigned and try_alt:
            direction *= -1
            _ = recursive_assign_search(x, roi_idx, grid_assignments, grid_ids, grid_tree, placed_idcs,
                                                  [reassign_to_idx], False, direction)
            # Return if reassignment worked or not. The original reassignment branch ends here because if the
            # alternative is successfull in reassigning, it will happen in place and no action is necessary in the
            # original path.
            # If both are unsuccessful treat this ROI as non-assignable?!
            return False
        elif not success and (assigned or not try_alt):
            return False

    # If the grid position is empty, the recursive search is finished and all ROIs along the chain can be reassigned
    grid_assignments[reassign_to_row, reassign_to_col] = this_coords
    grid_ids[reassign_to_row, reassign_to_col] = roi_idx
    return True

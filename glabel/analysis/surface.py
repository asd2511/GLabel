"""
Module with functionalities centered around 3D reconstruction using the method described in
`M Döllinger, DA Berry; "Computation of the three-dimensional medial surface dynamics of the vocal folds";
Journal of Biomechanics, 2006` https://doi.org/10.1016/j.jbiomech.2004.11.026.
"""
from typing import List

import numpy as np
from scipy.optimize import minimize


def init_F(calibration) -> np.ndarray:
    r"""
    Build the initialized matrix F.

    The matrix will be initialized with the user-clicked vectors received during the calibration process as described
    in the paper. The matrix will be 4-dimensional, built with the following system:

    .. math::

        F = \begin{bmatrix}
                x_1 & x_2 & x_3 & 0 \\
                y_1 & y_2 & y_3 & 0 \\
                x'_1 & x'_2 & x'_3 & 0 \\
                y'_1 & y'_2 & y'_3 & y'_4
            \end{bmatrix}

    :param calibration: The dictionary holding the calibration vectors. Should be the dictionary as emitted by the
        calibration process of  :class:`~gui.calibration_window.CalibrationWindow`.
    :type calibration: dict
    :return: The initialized 4x4 matrix F as a np.ndarry
    """
    # Initialize F using the user-clicked vectors from the calibration as initial values
    F = np.zeros((4, 4))
    for col in range(3):
        F[0][col] = np.round(calibration['v'][f'v{col+1}']['x'])
        F[1][col] = np.round(calibration['v'][f'v{col+1}']['y'])
        F[2][col] = np.round(calibration['vp'][f'v{col+1}']['x'])
        F[3][col] = np.round(calibration['vp'][f'v{col+1}']['y'])
    F[3][3] = np.round(calibration['phys']['b4'][3])

    return F


def get_vecs_3d(calibration) -> dict:
    """
    Return a dictionary holding the vectors representing the physical dimensions of the calibration cube. Paper notation
    is v_3D.

    This will return the vectors b_1, b_2, b_3 and b_4 as described in the paper. Each vector will represent one of the
    physical world coordinates, i.e. b_1 will be set to [val1, 0, 0, 0] to represent the span the first dimension, b_2
    the second and so on. The fourth vector b_4 will span the artificial 4th dimension which ensures mathematical
    stability.

    :param calibration: The dictionary holding the calibration vectors. Should be the dictionary as emitted by the
        calibration process of  :class:`~gui.calibration_window.CalibrationWindow`.
    :type calibration: dict
    :return: Dictionary holding the 4 dimension spanning vectors b_1, b_2, b_3 and b_4 with entries according to
        calibration.
    """
    return calibration['phys']


def get_vecs_rec(calibration) -> dict:
    r"""
    Return a dictionary holding the vectors representing the calibration cube edges in image domain. Paper notation is
    v_rec.

    This will return the vectors v_1, v_2, v_3, v'_1, v'_2 and v'_3 as described in the paper. The dictionary will
    contain only 3 vectors, each built as follows:

    .. math::

        v = \begin{pmatrix}
                x_1 \\
                y_1 \\
                x'_1 \\
                y'_1 \\
            \end{pmatrix}

    This directly corresponds to the notation from the paper:

    .. math:: f(b_1) = (v_1, v'_1) = (x_1, y_1, x'_1, y'_1)

    :param calibration: The dictionary holding the calibration vectors. Should be the dictionary as emitted by the
        calibration process of  :class:`~gui.calibration_window.CalibrationWindow`.
    :type calibration: dict
    :return: Dictionary holding 3 vectors, each representing a calbration cube edge in both view angles.
    """
    vecs_rec = dict.fromkeys(['v1', 'v2', 'v3'])
    for vec in list(vecs_rec.keys()):
        v = calibration['v'][vec]
        vp = calibration['vp'][vec]
        vecs_rec[vec] = [v['x'], v['y'], vp['x'], vp['y']]

    return vecs_rec


def obj_F(F, calibration) -> float:
    r"""
    Objective function for entries of F to be minimized for 3D reconstruction of surface.

    .. math::

        min ||F \cdot \vec{v}_{3D} - \vec{v}_{rec} ||_2

    .. caution:: The matrix F must be passed as a 1D flattened array by e.g. calling F.ravel() on it. This is a
        restriction set by scipy.optimize.

    :param F: 4x4 Matrix F to be optimized.
    :type F: np.ndarray
    :param calibration: The dictionary holding the calibration vectors. Should be the dictionary as emitted by the
        calibration process of  :class:`~gui.calibration_window.CalibrationWindow`.
    :type calibration: dict
    :return: Reconstruction error as sum of euclidean distances.
    """
    F = F.reshape((4, 4))  # Reshape from 1D back to 4x4 matrix
    vecs_3d = get_vecs_3d(calibration)
    vecs_rec = get_vecs_rec(calibration)

    diffs = []
    for i in range(1, 4):
        diffs.append(
            np.linalg.norm(
                np.dot(F, vecs_3d[f'b{i}']) - vecs_rec[f'v{i}']
            )
        )

    return np.sum(diffs)


def obj_yp4(yp4, F) -> float:
    """
    Objective function for entry y'_4 of matrix F to be minimized.

    The value of y'_4 is minimized separately to minimize the condition number of matrix F to ensure invertibility.

    :param yp4: The entry of y'_4 of matrix F:
    :type yp4: float
    :param F: 4x4 Matrix F for which the entry y'_4 should be optimized to minimize the condition number.
    :type F: np.ndarray
    :return: Condition number of matrix F with current value y'_4.
    """
    F[3][3] = yp4  # Insert the (new) value for y'_4 in the matrix
    return np.linalg.cond(F)


def get_optimized_F(calibration) -> np.ndarray:
    """
    Build, optimize and return the 4x4 matrix F from a given calibration.

    The optimization for F and its entry y'_4 will both be performed using the Nelder-Mead algorithm of
    scipy.optimize.minimize .

    :param calibration: The dictionary holding the calibration vectors. Should be the dictionary as emitted by the
        calibration process of  :class:`~gui.calibration_window.CalibrationWindow`.
    :type calibration: dict
    :return: The optimized 4x4 matrix F for reconstructing suture positions as a 3D surface.
    """
    F = init_F(calibration)  # Get the initialized matrix

    # Optimize for F
    res_F = minimize(obj_F, F.ravel(), (calibration), 'nelder-mead', options={'maxfev': 1e6})
    opt_F = res_F.x.reshape((4, 4))  # Reshape the optimization result back to 4x4 matrix

    # Optimize for entry y'_4
    t = np.zeros((4, 4))
    t[:, :3] = opt_F[:, :3]
    t[3, 3] = F[3, 3]
    res_yp4 = minimize(obj_yp4, opt_F[3, 3], (opt_F), 'nelder-mead', options={'maxfev': 1e6})
    opt_yp4 = res_yp4.x

    # Finalize the matrix by inserting the optimized value for y'_4
    # final_F = opt_F
    # final_F[3, 3] = opt_yp4
    final_F = np.zeros((4, 4))
    final_F[:, :3] = opt_F[:, :3]
    final_F[3, 3] = opt_yp4

    return final_F


def convert_rois_to_vec(rois, calibration, rows, cols):
    r"""
    Convert a list of suture ROI placements to 4D vectors necessary for reconstructing the surface.

    Creation of 4D vectors :math:`\vec{v}_{rec}` is necessary in order to compute
    :math:`\vec{v}_{3D} = F^{-1}\cdot\vec{v}_{rec}`.
    The conversion from 2D vectors to the necessary 4D vectors is achieved by simple concatenation of coordinate values:

    .. math:: \left( x_1, y_1 \right), \left( x_2, y_2 \right) \rightarrow \left( x_1, y_1, x_2, y_2 \right)

    :param List[List[QPointF]] rois: Annotated ROIs marking sutures. Must be a List of list of :pyqt:`QPointF <qpointf>`
        objects. First list runs over each frame of the data. Second list runs over each possible suture grid position,
        with entries being ROI coordinates.
    :param dict calibration: Calibration for reconstruction of points in 3D. Used here to modify the suture ROIs to have
        their origin align with the calibration cube origins.
    :param int rows: Number of rows in suture grid.
    :param int cols: Number of columns in suture grid.
    """
    num_frames = len(rois)
    num_grid_positions = rows * cols
    l_origin = calibration['v']['origin']
    r_origin = calibration['vp']['origin']

    vecs = np.empty((num_frames, num_grid_positions, 4))

    for frame_c, frame_rois in enumerate(rois):
        for idx, (l_roi, r_roi) in enumerate(zip(frame_rois[:num_grid_positions], frame_rois[num_grid_positions:])):
            # If an ROI is missing from one or both of the views it is marked in **both views** as missing using value
            # -1 as invalid identifier
            if any(coord == -1 for coord in [l_roi.x(), l_roi.y(), r_roi.x(), r_roi.y()]):
                vecs[frame_c][idx] = np.array([-1]*4)

            # Otherwise, the ROI coordinates are converted to the calibration cube coordinate system for both view sides
            # separately and concatenated to a 4D vector [x, y, x', y'] (prime = right view side)
            else:
                l_x = l_roi.x() - l_origin['x']
                l_y = l_roi.y() - l_origin['y']
                r_x = r_roi.x() - r_origin['x']
                r_y = r_roi.y() - r_origin['y']
                vecs[frame_c][idx] = np.array([l_x, l_y, r_x, r_y])

    return vecs


def get_3d_points(rois, F, calibration, num_rows, num_cols):
    """
    Calculate the three-dimensional information from a list of placed suture ROIs.

    The finalized calculation of 3D points from annotated sutures in stereoscopic views uses the 2D coordinates and the
    previously calculated matrix F for reconstruction.

    :param List[List[QPointF]] rois: List (over frames) of list (over grid positions) of 2D suture ROI coordinates.
    :param np.ndarray F: Reconstruction matrix F calculated from calibration.
    :param int num_rows: Number of rows in suture grid.
    :param int num_cols: Number of columns in suture grid.
    """
    # Get the 4D vectors of corresponding suture views
    rois_vecs = convert_rois_to_vec(rois, calibration, num_rows, num_cols)

    # Calculate the 3D points from the marked suture ROIs
    # rois_3d = [[None]*num_rows*num_cols for _ in range(len(rois))]
    rois_3d = np.empty((len(rois), num_rows*num_cols, 4))
    for frame_idx, frame_vecs in enumerate(rois_vecs):
        for roi_id, vec in enumerate(frame_vecs):
            if not any(coord == -1 for coord in vec):
                vec_3d = np.dot(np.linalg.inv(F), vec)
                rois_3d[frame_idx][roi_id] = vec_3d
            else:
                rois_3d[frame_idx][roi_id] = [np.nan]*4

    return rois_3d

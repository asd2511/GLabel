"""
Utility functions for data relevant to the GUI.
Functions for saving and loading of calibration and inference data, manipulating annotations, as well as handling GUI
selections made during inference settings selection.
"""
import os
import json
import h5py
from typing import List
from itertools import product

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QSpinBox, QComboBox, QFormLayout, QDialog, QLabel, QVBoxLayout, QPushButton, \
    QDialogButtonBox
from PyQt5.QtCore import Qt, QPointF

filenames = {
    'regions': 'suture_regions.json',
    'suture_maps': 'suture_maps.h5',
    'map_peaks': 'found_peaks.json',
    'sortings': 'sorting_inference.json'
}  #: # Hard-coded default filenames for saving inference data


def save_inferences(mw) -> None:
    """
    Save all made inferences currently added to the passed main_window object.

    Saving functionalities are distributed across :func:`save_region_inference`, :func:`save_suture_map_inference`,
    :func:`save_suture_peaks` and :func:`save_sorted_peaks`.

    :param gui.main_window.Main mw: Main window object.
    """
    assert mw.inference_settings is not None, "No inference settings were saved! Saving inferences should only" \
                                                "happen after an inference was made."

    dirname = create_inference_dir(mw)

    if mw.inference_settings['run_region_detect']:
        save_region_inference(mw.image_stack.bbox_stack, dirname, mw.inference_settings)
    if mw.inference_settings['run_suture_find']:
        save_suture_map_inference(mw.image_stack.suture_predictions, dirname, mw.inference_settings)
    if mw.inference_settings['run_peak_find']:
        save_suture_peaks(mw.image_stack.prediction_peaks, dirname, mw.inference_settings)
    if mw.inference_settings['run_peak_sort']:
        save_sorted_peaks(mw.image_stack.sorting_predictions, dirname, mw.inference_settings)


def get_inference_dir(mw) -> str:
    """
    Create the directory path for the directory that will contain the saved inference data.

    :param mw: Main window object.
    :type mw: gui.main_window.Main
    :return: Path to the directory that will hold the inference data.
    """
    data_file = os.path.abspath(mw.filename)
    parent_path = os.path.dirname(data_file)
    dirname = os.path.join(parent_path, 'suture_inferences', os.path.splitext(os.path.basename(data_file))[0])

    return dirname


def create_inference_dir(mw) -> str:
    """
    Create the directory 'suture_inferences' for holding the saved inferences and return its path.

    The directory will be made in the same directory as the data file for which the inferences were made. It will have
    the same name as the data file.

    :param mw: Main window object.
    :type mw: gui.main_window.Main
    :return: Path to the newly crated directory
    """
    dirname = get_inference_dir(mw)

    # Create the directory if not already existing
    os.makedirs(dirname, exist_ok=True)

    return dirname


def save_region_inference(bbox_stack, dirname, mw_settings) -> None:
    """
    Save the suture region inference to a .json file.

    The first saved dictionary element is `settings`, listing all used inference settings for the region prediction.
    For each inferred frame, the bounding box parameters `x` (left edge), `y` (top edge), `w` (width), `h` (height) and
    `conf` (confidence) are saved under the key `{frame index}_{patch side}`, with the patch_side value being either 'l'
    or 'r'.

    :param bbox_stack: Suture region bounding boxes.
    :type bbox_stack: List[List[pyqtgraph.graphicsitems.ROI.RectROI, float, pyqtgraph.graphicsitems.ROI.RectROI, float]
    :param dirname: Directory path for saving the .json file to.
    :type dirname: str
    :param mw_settings: Used inference settings from the main_window.Main object.
    :type mw_settings: dict
    :return:
    """
    region_dict = {}

    # Add the used settings to the saved dictionary
    region_dict.update({
        'settings': {
            'from_frame': mw_settings['from_frame'],
            'to_frame': mw_settings['to_frame'],
            'total_frames': mw_settings['total_frames'],
            'region_frame_delta': mw_settings['region_frame_delta'],
            'region_network_path': os.path.abspath(mw_settings['region_network_path']),
            'region_weights_path': os.path.abspath(mw_settings['region_weights_path'])
        }
    })

    for frame_idx, frame in enumerate(bbox_stack):
        if frame:  # Only save frames with any added prediction
            for box, conf, side in zip(bbox_stack[frame_idx][::2], bbox_stack[frame_idx][1::2], ['l', 'r']):
                if type(box) == list:
                    box_x = box[0]
                    box_y = box[1]
                    box_w = box[2]
                    box_h = box[3]
                else:
                    box_x, box_y = box.pos()
                    box_w, box_h = box.size()
                region_dict.update({
                    f"{frame_idx}_{side}": {
                        'x': box_x,
                        'y': box_y,
                        'w': box_w,
                        'h': box_h,
                        'conf': conf
                    }
                })

    fpath = os.path.join(dirname, filenames['regions'])
    with open(fpath, 'w') as f:
        json.dump(region_dict, f, indent=4)


def save_suture_map_inference(predictions, dirname, mw_settings) -> None:
    """
    Save the suture probability map inferences to a .h5 file.

    For each inferred frame the probability map is saved as a 2D numpy matrix of shape (224x224) along with the bounding
    box specifications for each probability map.

    .. note::

        The bounding box specifications are saved as a json string dictionary as a dataset in the HDF5 file. In order to
        read the dictionary again access it using json and numpy indexing, e.g.:
        ``bbox_dict = json.loads(h5_file['0_r']['box'][()])``

    :param predictions: Suture predictions.
    :type predictions: List[gui.image_widget.SuturePrediction]
    :param dirname: Directory path for saving the .h5 file to
    :type dirname: str
    :param mw_settings: Used inference settings from the main_window.Main object.
    :type mw_settings: dict
    """
    # Create dictionary with the used settings
    suture_settings = {
        'from_frame': mw_settings['from_frame'],
        'to_frame': mw_settings['to_frame'],
        'total_frames': mw_settings['total_frames'],
        'suture_find_network': os.path.abspath(mw_settings['suture_find_network']),
        'suture_find_batch': mw_settings['suture_find_batch']
    }

    fpath = os.path.join(dirname, filenames['suture_maps'])
    with h5py.File(fpath, 'w') as f:
        # Save the settings as a json converted dictionary
        f.create_dataset("settings", data=json.dumps(suture_settings))
        # Iterate over all inferred frames
        for frame_idx, frame in enumerate(predictions):
            if frame:  # Only save frames with any added prediction
                for map, box, side in zip(predictions[frame_idx][:2], predictions[frame_idx][2:], ['l', 'r']):
                    assert box.frame == frame_idx and box.side == 'left' if side == 'l' else box.side == 'right', \
                        "Sanity check for probability frame index and patch side failed while iterating in saving!"

                    box_dict = {
                        'x': box.left,
                        'y': box.top,
                        'w': box.width,
                        'h': box.height,
                        'frame': box.frame,
                        'side': box.side
                    }
                    grp = f.create_group(f"{frame_idx}_{side}")
                    grp.create_dataset('map', data=map)
                    grp.create_dataset('box', data=json.dumps(box_dict))


def save_suture_peaks(found_peaks, dirname, mw_settings) -> None:
    """
    Save the coordinates of the found peaks on the suture probability maps as .json file.

    For each frame, the coordinates of all found peaks will be saved as a list of lists of "shape" [-1, 2]. The first
    listed coordinate is for x/horizontal position and the second for y/vertical position.

    :param found_peaks: Peak coordinates found on the predicted probability maps using peak finding method.
    :type found_peaks: List[nn.suture_detection.Peak]
    :param dirname: Directory path for saving the .h5 file to
    :type dirname: str
    :param mw_settings: Used inference settings from the main_window.Main object.
    :type mw_settings: dict
    """
    peak_dict = {}

    # Add the used settings to the saved dictionary
    peak_dict.update({
        'settings': {
            'from_frame': mw_settings['from_frame'],
            'to_frame': mw_settings['to_frame'],
            'total_frames': mw_settings['total_frames'],
            'peak_find_distance': mw_settings['peak_find_distance'],
            'peak_find_threshold': mw_settings['peak_find_threshold']
        }
    })

    dict_keys = list(product(range(mw_settings['from_frame'], mw_settings['to_frame']), ['left', 'right']))
    peak_dict.update({
        '_'.join(map(str, dict_keys[i])): [] for i in range(len(dict_keys))
    })

    for peak in found_peaks:
        peak_key = '_'.join((str(peak.frame), str(peak.side)))
        # Coordinates are stored as int32 before this step which is not JSON serializable, so we convert to regular int
        peak_dict[peak_key].append([int(peak.x), int(peak.y)])

    fpath = os.path.join(dirname, filenames['map_peaks'])
    with open(fpath, 'w') as f:
        json.dump(peak_dict, f, indent=4)


def save_sorted_peaks(sortings, dirname, mw_settings) -> None:
    """
    Save the end result of inference pipeline, the sorted found sutures, as a .json file.

    For each frame, the coordinates, the highest probable position ID and the probabilities for all available suture
    positions are saved.

    :param sortings: Sorting predictions containing suture coordinates, most probable position ID and overall
        probabilities for all available suture positions.
    :type sortings: List[List[nn.suture_detection.SortingPrediction]]
    :param dirname: Directory path for saving the .h5 file to
    :type dirname: str
    :param mw_settings: Used inference settings from the main_window.Main object.
    :type mw_settings: dict
    """
    sorted_dict = {}

    # Add the used settings to the saved dictionary
    sorted_dict.update({
        'settings': {
            'from_frame': mw_settings['from_frame'],
            'to_frame': mw_settings['to_frame'],
            'total_frames': mw_settings['total_frames'],
            'suture_sort_network': os.path.abspath(mw_settings['suture_sort_network']),
            'suture_sort_batch': mw_settings['suture_sort_batch']
        }
    })

    for idx, frame in enumerate(sortings):
        if frame:  # Only save frames with any added prediction
            sorted_dict.update({f"{idx}_left": []})
            sorted_dict.update({f"{idx}_right": []})
            for suture in frame:
                # Extract suture coordinates based on passed datatype (either QPointF or List with [x, y]
                x = suture[0] if type(suture) == list else suture.x
                y = suture[1] if type(suture) == list else suture.y
                # Values need to be converted from int64 and int32 to regular int to be JSON serializable
                sorted_dict[f"{idx}_{suture.side}"].append({
                    'pred_id': int(suture.pred_id),
                    'x': int(x),
                    'y': int(y),
                    'probabilities': suture.probabilities.tolist()
                })

    fpath = os.path.join(dirname, filenames['sortings'])
    with open(fpath, 'w') as f:
        json.dump(sorted_dict, f, indent=4)


def search_existing_inferences(mw) -> dict:
    """
    Search for any existing saved inferences for the currently opened file of the main window object.

    The function expects any saved inferences to be located in the sub-directory `suture_inferences` within the
    directory of the opened data file, which must contain a directory with the same name as the opened data file.
    That inference directory will be searched for the saved inference files, with the names as defined at the top of
    this script.

    :param mw: Main window object.
    :type mw: gui.main_window.Main
    :return: Dictionary telling which inferences exist as saved files.
    """
    inference_dir = get_inference_dir(mw)

    available_inferences = dict.fromkeys(filenames.keys(), False)
    for inference, save_file in filenames.items():
        filepath = os.path.join(inference_dir, save_file)
        if os.path.isfile(filepath):
            available_inferences[inference] = os.path.abspath(filepath)

    return available_inferences


def process_combo_select(combobox: QComboBox) -> None:
    """
    Process the '...select file' option available when making inference settings. (Functions as pyqtslot)

    When the user selects the '...select file' option of a :pyqt:`QComboBox <qcombobox>` in the inference settings
    window, a new dialog for selecting a file is opened. The selection is added to the list of options for the
    :pyqt:`QComboBox <qcombobox>` and set as the current selected option.

    :param QComboBox combobox: QComboBox for which selecting the '...select file' option should be handled.
    """
    if combobox.currentText() == '...select file':
        file = QFileDialog.getOpenFileName(directory='./')[0]
        if file:
            combobox.addItem(os.path.abspath(file))
            combobox.setCurrentText(os.path.abspath(file))
        else:
            # Move the selection off the '...select file' item such that it can be re-selected again
            combobox.setCurrentIndex(0)


def process_reuse_select(combobox, existing_inference, formlayout) -> None:
    """
    Process ComboBox selection for reloading or newly creating inference data for one of the pipeline blocks.

    If the user wants to use the automatically found existing inference data ('{data_directory}/suture_inferences/
    {data_file_name}'), the formlayout displays the settings applied for that inference data and locks the user out
    of making any changes to them.
    The same behaviour applies when the user selects to load the inference data from a valid inference file other than
    the automatically found.

    When the user wants to create new inference data, all previously locked settings should be made available again for
    change by the user.

    :param combobox: QComboBox for selecting inference data behaviour.
    :type combobox: QComboBox
    :param existing_inference: Path to file if an existing inference file was found at the default location, otherwise
        will be False.
    :type existing_inference: Union[str, bool]
    :param formlayout: QFormLayout being responsible for customizing inference settings of the currently active pipeline
        block.
    :type formlayout: QFormLayout
    """
    # If the user wants to use use the inference saved at the default save location, load the used settings, apply them
    # to the form fields and disable the selection boxes from changing the settings again
    if combobox.currentText() == 'Use existing inference' and existing_inference:
        set_formlayout_enabled(formlayout, False)
        settings = read_inference_settings(existing_inference)
        apply_inference_settings(settings, formlayout)

    elif combobox.currentText() == '...load from file':
        file = QFileDialog.getOpenFileName(directory='./')[0]
        if file:
            combobox.addItem(os.path.abspath(file))
            combobox.setCurrentText(os.path.abspath(file))
            set_formlayout_enabled(formlayout, False)
            settings = read_inference_settings(file)
            apply_inference_settings(settings, formlayout)
        else:
            # Move the selection off the '...select file' item such that it can be re-selected again
            combobox.setCurrentIndex(0)

    else:
        set_formlayout_enabled(formlayout, True)


def set_formlayout_enabled(formlayout, enabled) -> None:
    """
    Set a whole :pyqt:`QFormLayout <qformlayout>` as enabled or disabled.

    Setting a :pyqt:`QFormLayout <qformlayout>` to enabled allows editing the form fields contained within.

    :param QFormLayout formlayout: Layout for which form fields are enabled.
    :param bool enabled: Boolean setting to enable or disable the form layout.
    """
    # Iterate over form items (right column only) but skip the first entry, which is the inference load combobox
    for i in range(1, formlayout.rowCount()):
        formlayout.itemAt(i*2).widget().setEnabled(enabled)


def read_inference_settings(inference_file) -> dict:
    """
    Read only the used settings parameters from the specified inference file.

    If the file is a .json file, the saved dictionary must contain the key `'settings'` which specifies all used settings.
    In case of a .h5 file, the file must contain a `'settings'` dataset from which to load used settings.

    :param inference_file: Path to the file to read settings from.
    :type inference_file: str
    :return: Dictionary holding used inference settings.
    :raises NotImplementedError: if the filetype of the specified file is not understood.
    """
    if inference_file.endswith('.json'):
        line = True
        settings_found = False
        with open(inference_file, 'r') as f:
            while line:
                # Search for line declaring the beginning of the settings entry
                line = f.readline()
                if "\"settings\":" in line:
                    settings_found = True
                    # If the settings are found, extract all the values until the closing bracket is encountered
                    settings_line = "{" + f.readline()
                    while "}," not in settings_line:
                        settings_line += f.readline()
                    break
        if settings_found:
            # The trailing "," character needs to be removed for parsing the dictionary
            settings_line = settings_line.strip(',\n')
            # Parse the json string as a dictionary
            settings_dict = json.loads(settings_line)
            return settings_dict
        else:
            raise KeyError("No settings entry found in specified .json file!")

    elif inference_file.endswith('.h5'):
        with h5py.File(inference_file, 'r') as f:
            if 'settings' not in f.keys():
                raise KeyError("No settings dataset found in specified .h5 file!")
            else:
                settings_line = f['settings'][()]
        # Parse the json string as a dictionary
        settings_dict = json.loads(settings_line)
        return settings_dict

    else:
        filetype = os.path.splitext(inference_file)[1]
        raise NotImplementedError(f"Reading of inferences saved as {filetype} is not implemented!")


def apply_inference_settings(settings, formlayout) -> None:
    """
    Apply inference settings to a formlayout.

    :param settings: Dictionary holding the settings as they are saved in the inference files.
    :type settings: dict
    :param formlayout: QFormLayout for applying settings to.
    :type formlayout: QFormLayout
    """
    d = [(key, value) for key, value in settings.items()]  # Changing the dictionary to a list to enable slicing

    # Slice off the first three entries ('from_frame', 'to_frame', 'total_frames') before iterating over settings and
    # applying them to the form layout fields in order of appearance
    for idx, (key, value) in enumerate(d[3:]):
        # When iterating over the FormLayout widgets, we are only interested in the "right column" widgets, which are
        # the actual boxes and edit fields. Additionally, we jump over the first index, which is the ComboBox for
        # reusing saved inference data.
        # +1: For skipping over Reuse ComboBox
        cur_widget = formlayout.itemAt(idx+1, 1).widget()

        # If the current widget is a SpinBox, the setting value is a numerical and we need to use the `setValue` method
        if type(cur_widget) is QSpinBox:
            cur_widget.setValue(value)

        # If the current widget is a ComboBox, the setting value is a string and we need to add the setting value as a
        # new option to the box and set it as the currently selected option
        elif type(cur_widget) is QComboBox:
            cur_widget.addItem(value)
            cur_widget.setCurrentText(value)


def edit_roi_assignment(rois, cur_id, new_id, method='replace'):
    """
    Modifies a passed list of ROIs inplace using the specified method.
    """
    if method == 'replace':
        rois[new_id] = rois[cur_id]
        rois[cur_id] = QPointF(-1, -1)

    elif method == 'swap':
        t = rois[new_id].copy()
        rois[new_id] = rois[cur_id]
        rois[cur_id] = t


def save_cube_calibration(calibration, calib_fname) -> None:
    """
    Save calibration data to file using JSON formatting.

    :param dict calibration: The calibration data as a dictionary
    :param str calib_fname: Filename for saving the calibration. Will strip off any filetype ending and replace it with
        '.calib' before saving.
    """
    calib_file = calib_fname.split('.')[0] + '.calib'
    calibration['calibration_file'] = calib_fname
    with open(calib_file, 'w') as f:
        json.dump(calibration, f, indent=4)


def load_cube_calibration(calib_file):
    """
    Load calibration data from a JSON formatted file.

    :param str calib_file: Path to file to load calibration data from. The file contents must be a JSON formatted
        string.
    """
    with open(calib_file, 'r') as f:
        calibration = json.load(f)
    calibration_file = calibration.pop('calibration_file')

    return calibration, calibration_file


def save_3d_data_csv(surf_points, fpath) -> None:
    """
    Save the calculated 3D surface points as .csv format.

    :param np.ndarray surf_points: Reconstructed 3D surface points.
    :param str fpath: Filepath to save data to.
    """
    with open(fpath, 'w') as f:
        f.write(f'# 3D surface data array shape: {surf_points.shape}\n')

        for f_idx, frame in enumerate(surf_points):
            f.write(f'# Frame {f_idx}:\n')
            np.savetxt(f, frame, fmt='%.3f', delimiter=',')


def save_3d_data_json(surf_points, fpath) -> None:
    """
    Save the calculated 3D surface points as .json format.

    :param np.ndarray surf_points: Reconstructed 3D surface points.
    :param str fpath: Filepath to save data to.
    """
    surf_dict = {f'frame {i}': fd.tolist() for i, fd in enumerate(surf_points)}

    with open(fpath, 'w') as f:
        json.dump(surf_dict, f, indent=4)


def save_3d_data_npy(surf_points, fpath) -> None:
    """
    Save the calculated 3D surface points as .npz format.

    :param np.ndarray surf_points: Reconstructed 3D surface points.
    :param str fpath: Filepath to save data to.
    """
    np.save(fpath, surf_points, allow_pickle=False)


class OptionDialog(QDialog):
    """
    **Bases:** :pyqt:`Qdialog <qdialog>`

    Simple Dialog window asking the user to decide for one of the displayed options.

    Window will display a single message string along with a button for each of the specified options. Calling
    :meth:`exec_` on this OptionDialog will return the chosen option or False if the user clicked on `Cancel`.
    """
    def __init__(self, msg, options):
        """
        Create and show a Dialog window to the user displaying *msg* and asking for an option select or cancellation.

        :param msg: Message to display in the dialog window.
        :type msg: str
        :param options: List of options available to the user as individual buttons.
        :type options: list[str]
        """
        super().__init__()

        self.setWindowTitle("Select Option")

        text = QLabel(msg)
        qbtn = QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(qbtn)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(text, alignment=Qt.AlignLeft)
        for idx, option in enumerate(options):
            option_btn = QPushButton(option, self)
            option_btn.clicked.connect(lambda x, r=idx: self.done(r+1))  # Emits (option idx)+1 (0 reserved for cancel)
            self.layout.addWidget(option_btn, alignment=Qt.AlignHCenter)
        self.layout.addWidget(self.buttonBox, alignment=Qt.AlignRight)

        self.setLayout(self.layout)

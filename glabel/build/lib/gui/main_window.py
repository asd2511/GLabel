#!/usr/bin/env python
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor, QScreen
from PyQt5.QtCore import Qt, QPointF, QThreadPool
from pyqtgraph import mkPen

import imageio as io
from skimage.io import imread
import cv2
import numpy as np

import h5py
import flammkuchen as fl
import json
import os
import re
import getpass
from datetime import datetime
import hashlib
import sys

from glabel.gui import (image_widget, config, grid_widget, worker, gui_utils, settings_window, calibration_window,
                        convert_cine_to_h5)
from glabel.analysis import surface, evaluation_utils


class Main(QMainWindow):
    """
    **Bases:** `QMainWindow <https://doc.qt.io/qt-5/qmainwindow.html>`_

    Main Window containing the GLable application. Holds and distributes values as necessary between widgets.
    """
    def __init__(self, screen_size: QScreen, debug: bool = False):
        """
        Instantiate the main window for the GLable application.

        :param screen_size: Object holding information about the current screen width and height. Should be the QScreen
            object relating to the currently used screen for the application.
        :param debug: Boolean toogle used for debugging purposes. When set to True, the initial question for a Username
            will be skipped and a default data path will be opened on startup. The default image that is opened is
            hard-coded and must be changed depending on your directory structure.
        """
        super().__init__()

        requires_login = True if not debug else False
        self.user = None if not debug else 'Debug'  #: Username as entered in Login Window
        if requires_login:
            self.ask_login()

        config = self.load_config()
        self.settings = config[0]  #: Settings loaded by :func:`load_config()`
        self.keybinds = config[1]  #: Keybinds for shortcuts loaded by :func:`load_config()`
        self.inference_settings = None

        # File-Management
        self.filename = self.settings['Filename']  #: str : Path to opened path
        self.rois_file = None  #: Path reference to .rois path if placements were imported
        self.save_file = None  #: Path reference to last used save path. Will be reset when opening new image data
        self.save_mode = self.settings['SaveMode']  #: 0 = new, timestamped path; 1 = overwriting existing path
        self.save_confirm = self.settings['SaveConfirm']  #: Boolean for setting if user must confirm before saving
        self.calibration_file = None
        self.calibration = None

        self.stack_dimensions = self.settings['StackDimensions']  # Shape of loaded image data

        # Default settings concerning GridWidget
        self.num_rows = self.settings['NumRows']
        self.num_cols = self.settings['NumCols']
        self.stereo_grid = self.settings['StereoGrid']
        self.grid_style = self.settings['GridStyle']
        # Variable for auto-progression mode: 0=free, 1=left->right, 2=bottom->top
        self.auto_mode = self.settings['AutoMode']
        self.auto_snaking = self.settings['AutoSnaking']
        self.begin_top = self.settings['BeginTop']
        self.auto_copy = self.settings['AutoCopy']
        self.num_rows_track = self.settings['NumRowsTrack']
        self.active_progress_mode = self.settings['ActiveProgressMode']
        self.frame_clicking = self.settings['FrameClicking']
        # Appearance variables
        self.roi_color = QColor(self.settings['ROIColor'])  #: Color for placed, but not selected ROI
        image_widget.ROI.ROI_color = self.roi_color
        self.active_color = QColor(self.settings['ActiveColor'])  #: Color for currently selected ROI
        grid_widget.PixelGrid.PixelColors['active'] = self.active_color.getRgb()[:3]
        image_widget.ROI.Active_color = self.active_color
        self.show_closeup = self.settings['ShowCloseup']  #: Boolean setting if closeup view should be displayed
        self.roi_snap = self.settings['ROISnap']  #: Boolean setting if ROIs should snap to full pixel positions
        image_widget.ROI.TranslateSnap = self.roi_snap
        self.show_crosshair = self.settings["ShowCrosshair"]
        # Enhancement variables
        self.gaussian = self.settings['Gaussian']
        # Statusbar and menubar
        self.status = self.statusBar()
        self.menu = self.menuBar()
        # `File` menu
        self.file_m = self.menu.addMenu("&File")
        self.file_m.addAction("&Open...", self.open, Qt.Key_O | Qt.ControlModifier)
        self.file_m.addAction("&Convert .CINE file...", lambda: convert_cine_to_h5.run_as_modal(self))
        self.file_m.addAction("&Save", self.save, Qt.Key_S | Qt.ControlModifier)
        self.file_m.addAction("Save &as...", self.save_as)
        self.file_m.addAction("&Export segmentation map...", self.export, Qt.Key_E | Qt.ControlModifier)
        self.file_m.addAction("Save inference...", lambda: gui_utils.save_inferences(self))
        self.file_m.actions()[4].setEnabled(False)
        # `Settings` menu
        self.settings_m = self.menu.addMenu("&Settings")
        self.settings_m.addAction("&Grid", self.show_grid_settings)
        self.settings_m.addAction("&Shortcuts", self.show_shortcut_settings)
        self.settings_m.addAction("&Automation", self.show_automation_settings)
        self.settings_m.addAction("App&earance", self.show_appearance_settings)
        self.settings_m.addAction("&Properties", self.show_properties_settings)
        self.settings_m.addAction("&Enhancement", self.show_enhancement_settings)
        # `Analyze` menu
        self.analyze_m = self.menu.addMenu("&Analyze")
        self.analyze_m.addAction("&Calibration", self.show_calibration_window)
        self.analyze_m.addAction("Reconstruct 3D Surface", self.show_3d_surf)
        self.analyze_m.actions()[-1].setEnabled(False)
        # 'NN' menu
        self.nn_m = self.menu.addMenu("&Magic")
        self.region_m = self.nn_m.addMenu("Predict Suture &Regions")
        self.region_m.addAction("YOLO", lambda: self.find_suture_regions('yolo'))
        self.region_m.addAction("Tiny YOLO", lambda: self.find_suture_regions('tiny'))
        self.suture_m = self.nn_m.addMenu("Predict &individual sutures")
        self.suture_m.addAction("Single Frame", lambda: self.find_individual_sutures('single'))
        self.suture_m.addAction("Recurrent", lambda: self.find_individual_sutures('recurrent'))
        self.nn_m.addAction("Automated labeling", self.run_suture_detection)
        self.nn_m.setEnabled(False)
        # Widget references
        self.image_stack = None  #: :class:`~gui.image_widget.ImageStack` for displaying and handling the image data
        self.grid_widget = None  #: :class:`~gui.grid_widget.GridWidget` displaying the grid

        # Window Setup
        self.setGeometry(100, 50, screen_size.width() - 200, screen_size.height() - 300)
        self.title = 'GLable'
        self.setWindowTitle(self.title)

        self.threadpool = QThreadPool()

        if debug:
            self.open("../Daten/Fuer_NN/Human_33-17_Phantom_Cam_16904_Cine1_100frames.rois")

    def ask_login(self):
        """
        Show :class:`LoginWindow` dialog asking user for Username.

        The entered username is used and set as the :py:attr:`user` attribute.
        """
        login_dlg = LoginWindow()
        if not login_dlg.exec_():
            sys.exit("No login")
        self.user = login_dlg.name_edit.text()

    def save_config(self):
        """
        Save the current meta-data, applied settings and used shortcuts to a .json path called `config.json` to the
        default directory at `./.suturelab`.

        This method is by default only called when the main window of the GLable application is closed in order to save
        all configurations of the user for the next startup of the application.
        """
        try:
            os.mkdir('.suturelab/')
        except FileExistsError:
            pass
        file = '.suturelab/config.json'
        with open(file, 'w') as f:
            json.dump({
                'meta_data': {
                          'time (UTC)': datetime.utcnow().strftime('%d.%m.%y %H:%M:%S'),
                          'user': self.user,
                          'account': os.getlogin()
                        },
                'config': {
                    'keybinds': self.keybinds,
                    'settings': {
                        "Filename": self.filename,
                        "SaveMode": self.save_mode,
                        "SaveConfirm": self.save_confirm,
                        "StackDimensions": len(self.image_stack.roi_stack) if self.image_stack else "",
                        "NumRows": self.num_rows,
                        "NumCols": self.num_cols,
                        "StereoGrid": self.stereo_grid,
                        "GridStyle": self.grid_style,
                        "AutoMode": self.auto_mode,
                        "AutoSnaking": self.auto_snaking,
                        "BeginTop": self.begin_top,
                        "AutoCopy": self.auto_copy,
                        "ActiveProgressMode": self.active_progress_mode,
                        "FrameClicking": self.frame_clicking,
                        "NumRowsTrack": self.num_rows_track,
                        "Gaussian": self.gaussian,
                        "ROIColor": self.roi_color.name(),
                        "ActiveColor": self.active_color.name(),
                        "ShowCloseup": self.show_closeup,
                        "ROISnap": self.roi_snap,
                        "ShowCrosshair": self.show_crosshair
                    }
                }
            }, f, indent=4)

    def load_config(self) -> tuple:
        """
        Search for and load previously saved configuration path.

        Will search for the default save directory *.suturelab* in this directory and for the path *config.json* within
        it. This directory and path are automatically created when :func:`closeEvent` and thereby :func:`save_config`
        is called.

        This method is initially called by :func:`__init__` and will restore all previously user-customized settings
        from their last session if available.

        :return: Tuple (settings, keybinds) holding dictionaries for the settings and keybinds loaded from the
            configuration path.
        """
        if not os.path.isdir('.suturelab/') or not os.path.isfile('.suturelab/config.json'):
            # Return the default settings and keybinds stored in config.py
            return config.settings, config.keybinds
        else:
            with open('.suturelab/config.json') as f:
                data = json.load(f)
            config_user = data['meta_data']['user']
            if config_user != self.user:
                if image_widget.ConfirmDialog(f"Load and use configuration from User {config_user} created at "
                                              f"{data['meta_data']['time (UTC)']} (UTC)?"
                                              f"\nSelecting Cancel loads default configuration").exec_():
                    pass
                else:
                    return config.settings, config.keybinds
            settings = data['config']['settings']
            keybinds = data['config']['keybinds']

            # Check if saved configuration path is missing some newly implemented keys
            if set(settings.keys()) != set(config.settings.keys()) or\
                set(keybinds.keys()) != set(config.keybinds.keys()):
                if image_widget.ConfirmDialog("Outdated configuration path found!\nIn order to use the application the "
                                              "existing configuration must be reset to its default state!").exec_():
                    return config.settings, config.keybinds
                else:
                    sys.exit("Invalid configuration path!")

            return settings, keybinds

    def open(self, file: str = None):
        """
        Open image path to be displayed. Will instantiate a :class:`ImageStack <gui.image_widget.ImageStack>` object
        holding the image data and handling setting of ROIs.

        Supported filetypes:
            .h5, .png, .tiff, .npz, .mp4, .avi, .rois

        Files associated with image data (even if its multi-frame) will call :func:`read_image` to load data. This
        concerns files of type .h5, .png, .tiff and .npz.

        For files associated with video data :func:`read_video` is called to load their data. This concerns files of
        type .mp4 and .avi.

        Loading a .rois path will first import the saved ROI placements and then load the image data for which the ROIs
        were originally placed. The reference to the original data is saved in the .rois path as well.

        The newly opened path will be saved in memory as the :py:attr:`.filename` attribute. Loading a .rois path will
        save its path in the :py:attr:`rois_file` attribute and the path to the original image data in
        :py:attr:`filename` attribute. If a path to a :py:attr:`save_file` is in memory, it will be erased.

        :param file: Path to image to be opened. If not specified, a
            `QFileDialog <https://doc.qt.io/qt-5/qfiledialog.html>`_ is opened for the user to select the path.

        :raises ValueError: if the path type of the path to be opened is not supported/unknown.
        """
        if not file:
            if self.filename:
                latest_directory = os.sep.join(self.filename.split(os.sep)[0:-1])
            else:
                latest_directory = ""

            file = QFileDialog.getOpenFileName(directory=latest_directory)[0]
            if file == '':
                return

        self.rois_file = None
        self.save_file = None  # needs to be reset, otherwise the old save location is still present

        self.status.showMessage(f"Loading {file} ...")  # Show path path in statusBar

        self.filename = file
        if any(ftype in self.filename for ftype in ['.h5', '.png', '.tiff', '.npz']):
            img_data = self.read_image(self.filename)
            roi_data = None

        elif any(ftype in self.filename for ftype in ['.mp4', '.avi']):
            img_data = self.read_video(self.filename)
            roi_data = None

        # Loading previously saved ROI data
        elif '.rois' in self.filename:
            self.save_file = self.filename
            self.rois_file = self.filename  # Save the .rois path as the rois_file
            self.status.showMessage(f"Importing ROIs from {self.filename}")
            with open(self.filename) as file:
                loaded_data = json.load(file)
            self.filename = os.path.normpath(loaded_data['image_file'])  # Extract the 'real' filename from the ROI data
            self.num_rows = loaded_data['num_rows']
            self.num_cols = loaded_data['num_columns']
            self.stereo_grid = loaded_data['stereo_grid']
            # Load the original image data belonging to the clicked ROIs
            if os.path.isfile(self.filename):
                pass
            elif os.path.isfile(loaded_data['backup_path']):
                self.status.showMessage(f"Could not find path specified in .rois path ... Trying backup path at "
                                        f"{loaded_data['backup_path']}")
                self.filename = loaded_data['backup_path']
            else:
                self.status.showMessage(f"Could not find path specified at backup location ...")
                notice_msg = "Could not find the original image data to the opened ROI positions. \nPlease specify " \
                             "the original image data the ROIs were clicked for."
                if image_widget.ConfirmDialog(notice_msg).exec_():
                    self.filename = QFileDialog.getOpenFileName(caption=notice_msg, directory="../daten")[0]
                else:
                    return

            # Compare sha hashes
            saved_sha = loaded_data['sha_hash']
            file_sha = self.get_sha_string(self.filename)  # Get sha hash of the path trying to be opened
            if saved_sha != file_sha:
                if not image_widget.ConfirmDialog("The image data set as the source data for the saved ROI positions "
                                                  "has been changed since!"
                                                  "\nUse the changed image data anyway?").exec_():
                    return

            if any(ftype in self.filename for ftype in ['.h5', '.png', '.tiff', '.npz']):
                img_data = self.read_image(self.filename)
            elif any(ftype in self.filename for ftype in ['.avi', '.mp4']):
                img_data = self.read_video(self.filename)

            roi_data = [[QPointF(roi['pos']['x'], roi['pos']['y'])
                         for roi in frame['roi_positions']]
                        for frame in loaded_data['frames']]
        else:
            raise ValueError("Trying to open unknown path type!")

        if img_data is not None:
            self.stack_dimensions = img_data.shape
            self.image_stack = image_widget.ImageStack(img_data, rois=roi_data,
                                                       num_rows=self.num_rows, num_cols=self.num_cols,
                                                       stereo_grid=self.stereo_grid, show_closeup=self.show_closeup,
                                                       auto_mode=self.auto_mode, auto_snaking=self.auto_snaking,
                                                       auto_copy=self.auto_copy, num_rows_track=self.num_rows_track,
                                                       shortcuts=self.keybinds, grid_style=self.grid_style,
                                                       begin_top=self.begin_top,
                                                       gaussian=self.gaussian,
                                                       show_crosshair=self.show_crosshair,
                                                       active_progress=self.active_progress_mode,
                                                       frame_clicking=self.frame_clicking)

            self.setCentralWidget(self.image_stack)
            self.grid_widget = self.image_stack.grid_widget
            self.status.showMessage(f"Displaying {self.filename}")
            self.setWindowTitle(self.title + ' - ' + self.filename)
            # Make the NN menu available
            self.nn_m.setEnabled(True)

        else:
            raise ValueError("The path to be opened could not be read!")

    def reopen_data(self) -> None:
        """
        Re-open the last opened image data.

        Intended purpose is to reset data after changing the grid settings.

        .. note:: Currently not used!

        """
        if self.rois_file is not None:
            self.open(self.rois_file)
        else:
            self.open(self.filename)

    @staticmethod
    def read_image(filename: str) -> np.ndarray:
        """
        Read image data from path and return it as an array.

        Supported filetypes are:
            .png, .tiff, .h5, .npz

        ==================  =========================
        **Filetype**        **Used function**
        .png                `imageio.imread <https://imageio.readthedocs.io/en/stable/userapi.html#imageio.imread>`_
        .tiff               `skimage.imgio.imread <`https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread>`_
        .h5                 `flammkuchen.load <https://github.com/portugueslab/flammkuchen/blob/master/io.rst>`_
        .npz                `np.load <https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html>`_
        ==================  =========================

        .. note:: For .h5 data, there has to be one of *images*, *ims* or *seg_map* datasets included to be loaded.
            For .npz files, a dataset *seg_map* is expected.

        :param filename: Path to the path to open.
        :return: The loaded image data as an array. The dimensionality of the array depending on the data can be between
            2D (single b&w image) and 4D (multiple colored video frames).

        :raises ValueError: if unknown/unsupported filetype is encountered.
        """
        # Handling of various path formats
        if '.png' in filename:
            img_data = np.asarray(io.imread(filename))  # Load a single image
        elif '.tiff' in filename:
            img_data = imread(filename)  # Load an image stack # TODO: Implement support for .tif files! (single image)
        elif '.h5' in filename:
            datasets = fl.load(filename)
            if 'images' in list(datasets.keys()):
                img_data = datasets['images']
            elif 'seg_map' in list(datasets.keys()):
                img_data = datasets['seg_map'] * 255
            elif 'ims' in list(datasets.keys()):
                img_data = datasets['ims']
        elif '.npz' in filename:
            data = np.load(filename)
            if 'seg_map' in data.files:
                img_data = data['seg_map'] * 255
        else:
            raise ValueError(f"Invalid path type of {filename} encountered in Main.read_image()!")

        return img_data

    @staticmethod
    def read_video(filename: str) -> np.ndarray:
        """
        Read image data frame-by-frame and return it as array.

        Supported filetypes are:
            .mp4, .avi

        All filetypes are using
        `cv2.VideoCapture <https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture>`_
        to read data.

        :param filename: Path to the path to open.
        :return: The loaded video frames as an array. Returned array will be 4D with axes being [frames, y, x, color].

        :raises AssertionError: if unknown/unsupported filetype is encountered.

        .. todo:: Untested for b&w video data!
        """
        assert any(ft in 'test.mp4' for ft in ['.mp4', '.avi']), f"Unknown filetype of {filename} encountered in " \
                                                                 f"Main.read_video()!"

        capture = cv2.VideoCapture(filename)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_data = np.empty((frame_count, frame_height, frame_width, 3), dtype=np.uint8)

        f_count = 0
        ret = True
        while f_count < frame_count and ret:
            ret, video_data[f_count] = capture.read()
            f_count += 1

        return video_data

    def convert_cine(self) -> None:
        """
        Convert a .cine file to .h5 (HDF5) format.

        PyQtSlot called when selecting **File->Convert .CINE file...**.
        """
        # prog_dlg = QProgressDialog("Converting .CINE file...", "Abort", 0, 2000, self, Qt.Dialog)
        # prog_dlg.setWindowTitle("Converter")
        # prog_dlg.setWindowModality(Qt.WindowModal)
        # prog_dlg.setValue(0)
        # prog_dlg.show()

        thread_worker = worker.Worker(convert_cine_to_h5.start_converter)
        self.threadpool.start(thread_worker)

    def save(self) -> None:
        """
        Manage save path location before finally calling :func:`save_as` to save the currently placed ROI data.

        Called when selecting **File->Save** or using shortcut **CTRL+S**.

        If no :attr:`save_file` is currently set, :func:`save_as` will be called without passing a filename to force
        user to specify a save location.
        Otherwise, depending on :attr:`save_mode`, the current :attr:`save_file` will be used as is or extended with
        the current timestamp and afterwards passed on to :func:`save_as`.

        Additionally, depending on :attr:`save_confirm` the user gets presented with a
        :class:`~gui.image_widget.ConfirmDialog` asking for confirmation of the correct path to save under before doing
        so. If the user cancels or closes the dialog, the method will immediately return without saving data.

        :raises ValueError: if :attr:`save_mode` has invalid value.
        """
        if not self.save_file:
            self.save_as()
        else:
            # save_mode 0: Create new, timestamped path
            if self.save_mode == 0:
                savefile = self.get_ts_savefile(self.save_file)
                if self.save_confirm:
                    if not image_widget.ConfirmDialog(
                            f"Create new save path \n{savefile}\nfor placed ROIs?"
                    ).exec_():
                        return
            # save_mode 1: Overwrite existing save path
            elif self.save_mode == 1:
                savefile = self.save_file
                if self.save_confirm:
                    if not image_widget.ConfirmDialog(
                            f"Overwrite current save path \n{self.save_file}\nfor placed ROIs?"
                    ).exec_():
                        return
            else:
                raise ValueError(f"Encountered invalid value {self.save_mode} for 'save_mode' in main_window!")
            self.save_as(savefile)

    def save_as(self, filename: str = None) -> None:
        """
        Save all currently placed ROIs to a .rois path as dictionary compiled by :func:`get_save_dict`.

        Show a `QFileDialog <https://doc.qt.io/qt-5/qfiledialog.html>`_ for the user to select a save path if
        **filename** is not passed, which will suggest a save path based on the currently opened data path. If user does
        not specify a save location, the method will return without saving any data.

        If path to a path is passed as **filename**, that path will be used for saving.

        Saving is handled by `json.dump <https://docs.python.org/3/library/json.html#json.dump>`_.

        :param filename: Path to save path location.
        """
        if self.image_stack is not None:

            if not filename:
                try:
                    f_path = os.path.splitext(self.filename)[0]
                except TypeError:
                    image_widget.ConfirmDialog("Error in filename encountered!\nData saved as backup as backup.rois "
                                               "in current working directory").exec_()
                    f_path = 'backup'
                if self.save_mode == 0:
                    file_suggest = self.get_ts_savefile(f_path)
                else:
                    file_suggest = f_path + '.rois'
                filename, _ = QFileDialog.getSaveFileName(caption="Save File",
                                                          directory=file_suggest,
                                                          filter="ROI Positions (*.rois)")
                if filename == '':
                    return

                # Split off the timestamp from the internally saved save path path
                if re.search('_\d{6}-\d{6}.rois', filename):
                    self.save_file = ''.join(re.split('_\d{6}-\d{6}', filename))
                else:
                    self.save_file = filename

            self.status.showMessage(f"Saving ROIs to {filename} ...")

            self.image_stack.save_cur_rois()  # Save ROIs of current frame

            with open(filename, 'w') as file:
                json.dump(self.get_save_dict(), file, indent=4)

        self.status.showMessage(f"ROIs saved to {filename}", 5000)
        self.status.showMessage(f"Displaying {self.filename}")

    @staticmethod
    def get_ts_savefile(filename: str) -> str:
        """
        Extend the passed **filename** with the current timestamp.

        :param filename: Filepath to extend with timestamp.
        :return: Filepath extended with current timestamp in format (path)_ddmmyy-HHMMSS.rois
        """
        if '.rois' in filename:
            filename = filename.split('.rois')[0]
        cur_time = datetime.now().strftime("%d%m%y-%H%M%S")

        return filename + '_' + cur_time + '.rois'

    def export(self) -> None:
        """
        Export currently placed ROI placements as a segmentation map.

        Opens a `QFileDialog <https://doc.qt.io/qt-5/qfiledialog.html>`_ to ask user for save location and filetype.
        Available filetypes for segmentation maps are ``.h5``, ``.npz`` and ``.tiff``.

        ============    ===============================     ====================================================
        **Filetype**    **size (for single test path)**     **compression**
        .npz            ~220kB                              :func:`np.savez_compressed`
        .h5             ~500kB                              :func:`h5py.File().create_dataset(compression=gzip,
                                                            compression_opts=9)`
        .tiff           ~720kB                              uncompressed using :func:`imageio.mimwrite`
        ============    ===============================     ====================================================
        """
        if self.image_stack is not None:
            filename, frmt = QFileDialog().getSaveFileName(caption="Save File",
                                                           directory=os.path.splitext(self.filename)[0] + '_segmap',
                                                           filter="Raw data (compressed) (*.h5);;"
                                                                  "Numpy arrays (compressed) (*.npz);;"
                                                                  "Images (*.tiff)",
                                                           initialFilter='*.h5')
            if filename == '':
                return

        self.status.showMessage(f"Exporting ROIs to {filename}...", 0)

        export_dimensions = self.stack_dimensions if len(self.stack_dimensions) == 3 else self.stack_dimensions[:-1]
        export_stack = np.empty(export_dimensions).astype(bool)
        self.image_stack.save_cur_rois()
        rois = self.image_stack.roi_stack

        for frame_idx, frame in enumerate(export_stack):
            for roi in rois[frame_idx]:
                x = int(roi.x())
                y = int(roi.y())
                export_stack[frame_idx][y-1:y+1, x-1:x+1] = True if x != -1 or y != -1 else False

        if '.h5' in frmt:
            with h5py.File(filename, 'w') as file:
                file.create_dataset('seg_map', data=export_stack, compression='gzip', compression_opts=9)
        elif '.npz' in frmt:
            np.savez_compressed(filename, seg_map=export_stack)
        elif '.tiff' in frmt:
            io.mimwrite(filename, export_stack*255)

        self.status.showMessage("Export done!", 2000)
        self.status.showMessage(self.filename)

    def get_save_dict(self) -> dict:
        """
        Construct dictionary used for saving current data and setup.

        ====================    ==================================================================
        **'meta_data'**         Dictionary containing following meta-data
        -> 'time (UTC)'         Current timestamp in format dd.mm.yy HH:MM:SS
        ->'user'                Username as entered in :class:`LoginWindow` by user
        -> 'account'            Account name as read by :func:`os.getlogin`

        **'image_file'**        Path to the opened image data
        **'backup_path'**       Backup path constructed by assuming *'Daten'* directory exists
        **'sha_has'**           SHA256 string of image data returned by :func:`get_sha_string`
        **'num_rows'**          :attr:'num_rows' attribute
        **'num_cols'**          :attr:'num_cols' attribute
        **'stereo_grid'**       :attr:'stereo_grid' attribute
        **'frames'**            List of dictionaries holding frame data
        -> 'frame'              Number of frame
        -> 'roi_positions'      List of dictionaries holding actual ROI placement data
        ->->*'id'*              Id of grid position for ROI
        ->->*'pos'*             Dictionary holding 'x' and 'y' coordinates of ROI
        ->*'placed'*            Boolean indicating if ROI is placed
        ====================    ==================================================================

        :return: Dictionary holding all meta-data, current settings and ROI placement data necessary for reconstructing
            the current state.
        """
        rois = self.image_stack.roi_stack
        data = {
                'meta_data': {
                    'time (UTC)': datetime.utcnow().strftime('%d.%m.%y %H:%M:%S'),
                    'user': self.user,
                    'account': os.getlogin()
                },
                'image_file': os.path.abspath(self.filename),
                'backup_path': os.path.normpath('Daten/' + self.filename.split('Daten')[-1]),  # Taking a guess
                'sha_hash': self.get_sha_string(self.filename),
                'num_rows': self.num_rows,
                'num_columns': self.num_cols,
                'stereo_grid': self.stereo_grid,
                'frames': [{
                    'frame': frame_id,
                    'roi_positions': [{
                        'id': j,
                        'pos': {
                            'x': roi.x(),
                            'y': roi.y()},
                        'placed': roi.x() != -1 or roi.y() != -1
                    } for j, roi in enumerate(frame)]
                } for frame_id, frame in enumerate(rois)]
            }

        return data

    @staticmethod
    def get_sha_string(filename: str) -> str:
        """
        Read the contents of the specified path and return a SHA256-hash string representation.

        The path specified by **filename** is read and iterated over in blocks of 4KB. For each block, the data is fed
        to the hash function. After completion, the hash is returned in string representation.

        :param filename: Path to path for which the SHA256-hash is to be built.
        :return: String object containing only hexadecimal digits representing the SHA256-hash of the path.
        """
        sha256_hash = hashlib.sha256()
        with open(filename, 'rb') as file:
            # Read the data in blocks of 4K and update the hash <- Solution for potentially large files
            for byte_block in iter(lambda: file.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    def show_grid_settings(self) -> None:
        """
        Open settings window to change grid settings. Creates a :class:`GridSettings <settings_window.GridSettings()>`
        object.

        Calls :func:`set_grid_settings` to apply changed settings.

        .. warning:: **If User changes settings for the GridWidget, all currently existing Widgets will be reset.**
        """
        # TODO: Allow changing of grid settings only before placing an ROI. After the first ROI has been placed, the
        #  option to change settings should not be available.

        sw = settings_window.GridSettings(self.num_rows, self.num_cols, self.stereo_grid, self.grid_style)
        # If execution finishes with "Accepted" status
        if sw.exec_():
            self.set_grid_settings(sw)

    def set_grid_settings(self, sw: settings_window.GridSettings) -> None:
        """
        Apply the newly entered grid settings to the GUI.

        Is automatically called from :func:`show_grid_settings` after the user exits the settings window.
        Will overwrite the :attr:`num_rows`, :attr:`num_cols`, :attr:`stereo_grid` and :attr:`grid_style` with their
        newly received values.

        .. warning:: If the GUI is already initialized with an :class:`ImageStack` and any of the grid settings are
            changed, the currently opened data path will be re-opened.This will delete all currently placed ROIs from
            the data path.

        :param sw: Reference to the GridSettings object presented to the user.

        .. todo:: The implementation of resetting the ImageStack is somewhat messy. The main reason for doing it, is
            that references will get lost when the GridWidget is re-initialized with new settings. Possible solutions
            would be to save and re-open the data automatically, or finding a way to re-connect the references of all
            widgets correctly in the ImageStack class.
        """
        # TODO: Messy implementation -> Should be changed to allow re-styling the grid without resetting the path!
        #   Reference to grid_widget within image_stack gets lost when trying to reinstantiate it as the other class.
        # The whole image_stack contents should be reset when the grid dimensions have been changed by the user
        new_rows = int(sw.rows)
        new_cols = int(sw.cols)
        new_stereo = sw.stereo
        new_style = sw.grid_style
        if self.image_stack is not None and (new_rows != self.num_rows or new_cols != self.num_cols or
                                             new_stereo != self.stereo_grid or new_style != self.grid_style):
            if image_widget.ConfirmDialog(f"Changing the Grid Settings will force the application to re-open the loaded"
                                          f"data."
                                          f"\n\nThis will delete any unsaved ROI placements!"
                                          f"\n\nChange the Grid and re-open your current path anyway?").exec_():
                reset = True
            else:
                # Do not change anything
                return
        else:
            # Parameters can be changed without resetting the image_stack
            reset = False
        self.num_rows = new_rows
        self.num_cols = new_cols
        self.stereo_grid = new_stereo
        self.grid_style = new_style
        # If there is already an ImageStack object, reset the currently displayed GridWidget (along with the possibly
        # changed GridStyle
        if self.image_stack is not None and reset:
            # self.reopen_data()
            self.open(self.filename)
        # # If the dimensions stayed the same -> Change the style if necessary
        # elif self.image_stack is not None and new_style != self.grid_style:
        #     self.grid_style = new_style
        #     self.image_stack.update_grid_style(self.grid_style)
        #     setattr(self, 'grid_widget', self.image_stack.grid_widget)

    def show_shortcut_settings(self) -> None:
        """
        Open settings window to change keyboard shortcut settings. Creates a
        :class:`ShortcutSettings <settings_window.ShortcutSettings()>` object.

        Calls :func:`set_shortcut_settings` to apply changed settings.
        """
        sw = settings_window.ShortcutSettings(self.keybinds)
        if sw.exec_():
            self.set_shortcut_settings(sw)

    def set_shortcut_settings(self, sw: settings_window.ShortcutSettings) -> None:
        """
        Apply the newly entered shortcuts to the GUI.

        Is automatically called from :func:`show_shortcut_settings` after the user exits the settings window.
        Will overwrite the :attr:`keybinds` with the newly received dictionary of keybinds.

        If the GUI is already initialized with an :class:`ImageStack` object, :func:`ImageStack.update_shortcuts`
        is called to use the newly entered key combinations.

        :param sw: The :class:`ShortcutSettings` object presented to the user for customizing shortcuts.
        """
        self.keybinds = sw.shortcut_dict
        if self.image_stack:
            self.image_stack.keybinds = self.keybinds
            self.image_stack.update_shortcuts()

    def show_automation_settings(self):
        """
        Open settings window to change auto-progress settings. Creates a
        :class:`AutomationSettings <settings_window.AutomationSettings()>` object.

        Calls :func:`set_automation_settings` to apply changed settings.
        """
        sw = settings_window.AutomationSettings(self.num_rows, self.auto_mode, self.auto_snaking, self.auto_copy,
                                                self.num_rows_track, self.begin_top, self.active_progress_mode,
                                                self.frame_clicking)
        if sw.exec_():
            self.set_automation_settings(sw)

    def set_automation_settings(self, sw: settings_window.AutomationSettings):
        """
        Apply the newly entered automation settings to the GUI.

        Is automatically called from :func:`show_automation_settings` after the user exits the settings window.
        Will overwrite the following attributes with their new values:

        * :attr:`auto_mode`
        * :attr:`auto_snaking`
        * :attr:`auto_copy`
        * :attr:`num_rows_track`
        * :attr:`begin_top`
        * :attr:`active_progress_mode`
        * :attr:`frame_clicking`

        If an :class:`~gui.image_widget.ImageStack` or :class:`~gui.grid_widget.GridWidget` is already initialized in
        the GUI, the applied settings are passed on to them.

        :param sw: The :class:`AutomationSettings` object presented to the user.
        """
        self.auto_mode = sw.mode
        self.auto_snaking = sw.snaking
        self.auto_copy = sw.auto_copy
        self.num_rows_track = sw.num_rows_track
        self.begin_top = sw.begin_top
        self.active_progress_mode = sw.active_progress_mode
        self.frame_clicking = sw.frame_clicking
        if self.image_stack is not None:
            self.image_stack.auto_copy = self.auto_copy
            self.image_stack.num_rows_track = self.num_rows_track
            self.image_stack.active_progress = self.active_progress_mode
            self.image_stack.frame_clicking = self.frame_clicking
        if self.grid_widget is not None:
            self.grid_widget.set_mode(self.auto_mode)
            self.grid_widget.set_snaking(self.auto_snaking)
            self.grid_widget.set_begin_top(self.begin_top)

    def show_appearance_settings(self) -> None:
        """
        Open settings window to change appearance settings. Creates a
        :class:`AppearanceSettings <settings_window.AppearanceSettings()>` object.

        Calls :func:`set_appearance_settings` to apply changed settings.
        """
        sw = settings_window.AppearanceSettings(self.roi_color, self.active_color, self.show_closeup, self.roi_snap,
                                                self.show_crosshair)
        if sw.exec_():
            self.set_appearance_settings(sw)

    def set_appearance_settings(self, sw: settings_window.AppearanceSettings) -> None:
        """
        Apply the newly entered appearance settings to the GUI.

        Is automatically called from :func:`show_appearance_settings` after the user exits the settings window.

        If the GUI is already initialized with an :class:`ImageStack <gui.image_widget.ImageStack>`,
        :func:`gui.image_widget.ImageView.draw_rois` and
        `ROI.setPen <http://www.pyqtgraph.org/documentation/graphicsItems/roi.html?highlight=roi#pyqtgraph.ROI.setPen>`_
        are called to apply the newly set colors.

        The affected attributes are

        * :attr:`roi_color`
        * :attr:`active_color`
        * :attr:`show_closeup`
        * :attr:`roi_snap`.

        :param sw: The :class:`AppearanceSettings <gui.settings_window.AppearanceSettings>` object used to enter the new
            parameter values.
        """
        self.roi_color = image_widget.ROI.ROI_color = sw.roi_color
        self.active_color = image_widget.ROI.Active_color = sw.active_color
        grid_widget.PixelGrid.PixelColors['active'] = sw.active_color.getRgb()[:3]
        self.show_closeup = sw.closeup_check.isChecked()
        self.roi_snap = image_widget.ROI.TranslateSnap = sw.snap_check.isChecked()
        self.show_crosshair = sw.crosshair_check.isChecked()

        # Show or hide crosshair
        if self.show_crosshair:
            self.image_stack.img_view.show_crosshair()
        else:
            self.image_stack.img_view.hide_crosshair()

        # Re-draw the ROIs in their new color
        if self.image_stack is not None:
            self.image_stack.img_view.draw_rois()  # Redraw ROIs with new color-scheme
            self.image_stack.closeup_roi.setPen(mkPen(color=self.roi_color, width=3))  # Recolor the closeup ROI

    def show_properties_settings(self) -> None:
        """
        Open settings window to change property settings. Creates a
        :class:`PropertiesSettings <settings_window.PropertiesSettings()>` object.

        Calls :func:`set_properties_settings` to apply changed settings.
        """
        sw = settings_window.PropertiesSettings(self.save_mode, self.save_confirm)
        if sw.exec_():
            self.set_properties_settings(sw)

    def set_properties_settings(self, sw: settings_window.PropertiesSettings) -> None:
        """
        Apply the newly entered property settings to the GUI.

        The affected attributes are :attr:`save_mode` and :attr:`save_confirm`.

        :param sw: The :class:`~settings_window.PropertiesSettings` object used to enter the new parameter values.
        """
        self.save_mode = sw.save_mode
        self.save_confirm = sw.confirm

    def show_calibration_window(self) -> None:
        """
        Create and display a :class:`CalibrationWindow <gui.calibration_window.CalibrationWindow()>` for creating a
        calibration used in 3D reconstruction.

        The method will open a :pyqt:`QFileDialog <qfiledialog>` to allow selecting of an image file used in creating
        the calibration, or an `.calib` file with an already saved calibration. The calibration is further processed by
        a call to :func:`set_calibration` after the :class:`CalibrationWindow <calibration_window.CalibrationWindow()>`
        has been closed.
        """
        file = QFileDialog.getOpenFileName(caption="Select image for calibration.", directory="../daten")[0]
        if file == '':
            return
        elif file.endswith('.calib'):
            calibration, self.calibration_file = gui_utils.load_cube_calibration(file)
            self.set_calibration(calibration)
        else:
            self.calibration_file = file
            calibration_img = self.read_image(file)

            sides = 2 if self.stereo_grid else 1
            cal_win = calibration_window.CalibrationWindow(calibration_img, sides, self)
            cal_win.show()

            cal_win.calibration_set_sig.connect(lambda calib: self.set_calibration(calib))

    def set_calibration(self, calibration) -> None:
        """
        Apply the passed calibration to be used in the 3D reconstruction algorithm.

        The passed calibration is used to calculate the reconstruction matrix F by calling
        :func:`surface.get_optimized_F`, which is saved as attribute :attr:`surface_f`.

        The passed calibration is automatically saved to file with the filename being the same as the image file used
        for creating the calibration, with file ending .calib (see :func:`gui_utils.save_cube_calibration()`

        :param calibration: Calibration data as returned created by :class:`~calibration_window.CalibrationWindow()`.
        :type calibration: dict
        """
        self.calibration = calibration
        self.image_stack.surface_f = surface.get_optimized_F(calibration)
        self.analyze_m.actions()[-1].setEnabled(True)
        gui_utils.save_cube_calibration(calibration, self.calibration_file)

    def show_3d_surf(self) -> None:
        """
        Initialize calls to calculate and subsequently display reconstructed 3D data from currently present annotations.

        The calculation and display of the 3D data is handled by functions :func:`ImageStack.calculate_surface()` and
        :func:`ImageStack.show_surface()`
        """
        self.image_stack.calculate_surface(self.calibration)
        self.image_stack.show_surface()

    def show_enhancement_settings(self) -> None:
        """
        Open settings window to change enhancement settings. Creates a
        :class:`EnhancementSettings <settings_window.EnhancementSettings()>` object.

        Calls MainWindow.set_enhancement_settings to apply changed settings.
        """
        sw = settings_window.EnhancementSettings(self.gaussian)
        if sw.exec_():
            self.set_enhancement_settings(sw)
            if self.image_stack:
                self.image_stack.gaussian = self.gaussian
                self.image_stack.update_stack()

    def set_enhancement_settings(self, sw) -> None:
        """
        Sets enhancement specific attributes.
        """
        self.gaussian = sw.gaussian

    def find_suture_regions(self, net) -> None:
        """
        Method for exclusively running inference for suture grid regions.

        .. warning:: Deprecated! Is replaced with more general method :func:`full_suture_detection`

        .. todo:: Delete this method!

        :param string net: Which YOLO network to use for inference. One of 'YOLO' or 'Tiny'.
        """
        assert self.image_stack is not None, "The ImageStack must be initialized before running any inference"

        num_frames = self.image_stack.stack.shape[0]
        frame_delta = self.ask_frame_delta(num_frames, net)
        if frame_delta:
            prog_dlg = QProgressDialog("Detecting regions...", "Abort", 0, num_frames//frame_delta,
                                       self, Qt.Dialog)
            prog_dlg.setWindowTitle("Detecting suture regions")
            prog_dlg.setWindowModality(Qt.WindowModal)
            prog_dlg.setValue(0)

            # Late import to reduce overhead at startup
            from glabel.nn.suture_detection import run_region_detect
            # Set up separate thread to handle the inference to prevent locking up the GUI
            thread_worker = worker.Worker(run_region_detect, self.image_stack.stack, net, frame_delta)
            thread_worker.signals.progress.connect(lambda x: prog_dlg.setValue(prog_dlg.value() + x))
            thread_worker.signals.message.connect(lambda msg: prog_dlg.setLabelText(msg))
            thread_worker.signals.result.connect(lambda x: self.image_stack.add_bboxes(*x))
            # Start the thread
            self.threadpool.start(thread_worker)

            # bboxes, confidences = run_region_detect(self.image_stack.stack, net, 25, prog_dlg)
            # self.image_stack.add_bboxes(bboxes, confidences)

    def find_individual_sutures(self, net) -> None:
        """
        Exclusively run inference for identifying individual sutures.

        .. warning:: Deprecated! Replaced by more general :func:`full_suture_detection`.

        .. todo:: Remove this method!

        :param str net: Which network to use.
        """
        assert self.image_stack is not None, "The ImageStack must be initialized before running any inference"

        num_frames = self.image_stack.stack.shape[0]
        frame_delta = self.ask_frame_delta(num_frames, net)
        if frame_delta:
            prog_dlg = QProgressDialog("Detecting individual sutures...", "Abort", 0,
                                       num_frames//frame_delta+num_frames,  # First for YOLO detection, then for UNet
                                       self, Qt.Dialog)
            prog_dlg.setWindowTitle("Finding individual sutures")
            prog_dlg.setWindowModality(Qt.WindowModal)
            prog_dlg.setValue(0)

            # Late import to reduce overhead at startup
            from glabel.nn.suture_detection import run_suture_detect
            # Set up separate thread to handle the inference to prevent locking up the GUI
            thread_worker = worker.Worker(run_suture_detect, self.image_stack.stack, net, frame_delta, 10)
            thread_worker.signals.progress.connect(lambda x: prog_dlg.setValue(prog_dlg.value() + x))
            thread_worker.signals.message.connect(lambda msg: prog_dlg.setLabelText(msg))
            thread_worker.signals.result.connect(lambda x: self.distribute_inferences(*x))
            # Start the thread
            self.threadpool.start(thread_worker)

            # yolo_boxes, yolo_confs, pred_maps, bboxes = run_suture_detect(self.image_stack.stack, net,
            #                                                                                frame_delta, prog_dlg)
            # self.image_stack.add_bboxes(yolo_boxes, yolo_confs)
            # self.image_stack.add_suture_predictions(pred_maps, bboxes)

    def full_suture_detection(self) -> None:
        """
        Run full suture detection and sorting process.

        .. warning:: Deprecated! Replaced in favor of more general method :func:`run_suture_detection`

        .. todo:: Remove this method!
        """
        assert self.image_stack is not None, "The ImageStack must be initialized before running any inference!"

        num_frames = self.image_stack.stack.shape[0]

        inference_settings = self.ask_inference_settings(num_frames)
        if inference_settings is not None:
            prog_dlg = QProgressDialog("Preparing automation...", "Abort", 0,
                                       num_frames//inference_settings['region_frame_delta'] +  # YOLO
                                       num_frames * 2 // inference_settings['suture_find_batch'] +  # RUNet
                                       num_frames * 70 // inference_settings['suture_sort_batch'],  # EfficientNet guess TODO: Bad implementation
                                       self, Qt.Dialog)
            prog_dlg.setWindowTitle("Finding and sorting sutures")
            prog_dlg.setWindowModality(Qt.WindowModal)
            prog_dlg.setValue(0)
            # Force immediate showing of dialog to give user instant feedback before starting long import and inference
            prog_dlg.show()

            # Late import to reduce overhead at GUI startup
            from glabel.nn.suture_detection import run_detect_and_sort

            # Set up separate thread to handle the inference to prevent locking up the GUI
            thread_worker = worker.Worker(run_detect_and_sort, self.image_stack.stack,
                                          inference_settings['region_frame_delta'],
                                          inference_settings['suture_find_batch'],
                                          inference_settings['suture_sort_batch'])
            thread_worker.signals.progress.connect(lambda x: prog_dlg.setValue(prog_dlg.value() + x))
            thread_worker.signals.message.connect(lambda msg: prog_dlg.setLabelText(msg))
            thread_worker.signals.result.connect(lambda x: self.distribute_inferences(*x))
            # Start the thread
            self.threadpool.start(thread_worker)

            # We only guessed the number of steps needed for the final suture sorting (we cannot know the step number
            # until the RUNet has finished its inference), so the progress dialog needs to be finished manually in case
            # it hasn't yet.
            # TODO: This behavior of guessing the number of steps should be changed in the future!
            prog_dlg.reset()

    def run_suture_detection(self) -> None:
        """
        Entry-point for running inference for suture grid data.

        This method is called as the user clicks the option to automatically find sutures in the data. Inference
        settings will be asked from the user by call to :func:`ask_inference_settings`. The entered settings will be
        used to approximate the number of calculation steps in the inference and will be passed onward to the main
        inference function.

        A :pyqt:`QProgressDialog <qprogressdialog>` is used to give feedback about inference to the user.

        In order to not block the execution of this window and the progress dialog, a new worker thread is created in
        which the inference is executed. The new thread uses a :class:`~worker.Worker()` object to handle parallel
        execution and communication.

        After the inference thread has finished execution, the resulting inference data is passed on to
        :func:`distribute_inferences()` for further processing and display.

        :raises AssertionError: if no :class:`ImageStack` object has been initialized.
        :raises AssertionError: if the number of frames for running inference on is negative
        """
        assert self.image_stack is not None, "The ImageStack must be initialized before running any inference!"

        num_frames = self.image_stack.stack.shape[0]

        inference_settings = self.ask_inference_settings(num_frames)
        if inference_settings is not None:
            assert inference_settings['to_frame'] >= inference_settings['from_frame'], \
                "Inference `from` and `to` frame range invalid (`to` larger than `from`)!"

            total_steps = self.calculate_inference_steps(inference_settings)
            prog_dlg = QProgressDialog("Preparing automation...", "Abort", 0, total_steps, self, Qt.Dialog)
            prog_dlg.setWindowTitle("Finding and sorting sutures")
            prog_dlg.setWindowModality(Qt.WindowModal)
            prog_dlg.setValue(0)
            prog_dlg.show()

            from glabel.nn.suture_detection import auto_annotate

            if inference_settings['from_frame'] == inference_settings['to_frame']:
                # If both values are the same, limit the data to that single frame
                data_part = self.image_stack.stack[inference_settings['from_frame']]
                # Restore the removed first dimension which is the number of frames
                data_part = np.expand_dims(data_part, axis=0)
            else:
                data_part = self.image_stack.stack[inference_settings['from_frame']:inference_settings['to_frame']]

            thread_worker = worker.Worker(auto_annotate, data_part, inference_settings)
            thread_worker.signals.progress.connect(lambda x: prog_dlg.setValue(prog_dlg.value() + x))
            thread_worker.signals.message.connect(lambda msg: prog_dlg.setLabelText(msg))
            thread_worker.signals.result.connect(lambda x: self.distribute_inferences(**x))
            thread_worker.signals.finished.connect(lambda: prog_dlg.setValue(total_steps))

            self.threadpool.start(thread_worker)

    def distribute_inferences(self,
                              region_boxes=None, region_confs=None,
                              suture_maps=None, suture_boxes=None,
                              suture_peaks=None,
                              sorted_sutures=None) -> None:
        """
        Distribute inference results to correct destinations for processing, reference-keeping and display to the user.

        The raw inference results are distributed to the correct destination classes. They will be processed and saved
        there accordingly. This class will not keep references to the results itself.

        The destinations for the inference results are:

        ==================================  =============================================
        **Inference result**                **Distributed to**
        Region boxes & confidences          :func:`ImageStack.add_bboxes`
        Probability maps & expanded bboxes  :func:`ImageStack.add_suture_predictions`
        Found peaks                         :attr:`ImageStack.prediction_peaks`
        Suture sortings                     :func:`ImageStack.add_sorted_predictions`
        ==================================  =============================================

        :param List[List[Box]] region_boxes: List of lists of :class:`Box` namedtuples. First list is over frames in
            data, with each contained list holding 2 boxes (left and right view). These bounding boxes are the irregular
            sized boxes predicted by the first automation step.
        :param List[List[float]] region_confs: List of lists holding bounding box confidences for left and right view
            of each frame (order is left view, right view).
        :param np.ndarray suture_maps: Numpy array of shape (#frames*2, dim, dim), with `dim` being the dimension of the
            expanded bounding box size for suture maps. (Default dimension is 224 x 224)
        :param List[List[Box]] suture_boxes: List of lists of :class:`Box` namedtuples. Same concept as for the
            `region_boxes` list, just for the expanded suture identification bounding boxes. (Default boxes are of size
            224 x 224).
        :param List[Peak] suture_peaks: List of unordered :class:`Peak` namedtuples as found by peak finding algorithm
            on suture probability maps. Reference to frame and view side a frame belongs to is stored in the namedtuple.
        :param List[nn.suture_detection.SortingPrediction] sorted_sutures: List of
            :class:`SortingPrediction <nn.suture_detection.SortingPrediction>` namedtuples. The same peaks as stored in
            the `suture_peaks` variable, but with added sorting inference data from last automation step. The namedtuple
            objects contain references to frame and view side for each suture, as well as membership probabilities for
            each individual grid position.
        """
        if region_boxes is not None and region_confs is not None:
            self.image_stack.add_bboxes(region_boxes, region_confs)
        if suture_maps is not None and suture_boxes is not None:
            self.image_stack.add_suture_predictions(suture_maps, suture_boxes)
        if suture_peaks is not None:
            self.image_stack.prediction_peaks = suture_peaks
        if sorted_sutures is not None:
            # from analysis import evaluation_utils
            evaluation_utils.process_sorting_predictions(sorted_sutures, self.image_stack.stack)
            self.image_stack.add_sorted_predictions(sorted_sutures)

        self.file_m.actions()[4].setEnabled(True)

        # self.image_stack.init_shortcuts()

    def ask_frame_delta(self, num_frames, net):
        """
        .. warning:: Deprecated! No longer in use! Was only called by outdated methods for separate inference of regions
            and sutures.

        .. todo: Remove this method!
        """
        frame_delta, ok = QInputDialog().getInt(self, f"{net} settings",
                                                "Inference only every nth frame:",
                                                value=1, min=1, max=num_frames, step=1)
        if ok and frame_delta:
            return frame_delta
        else:
            return None

    def calculate_inference_steps(self, inference_settings) -> int:
        """
        Calculate/Estimate the number of steps necessary for running inference.

        In general, a single step is regarded as a single run of any inference process. This means every single
        prediction by a neural network is considered a step. This definition makes it easy to broadcast an update from
        the worker thread to the GUI thread after each step.

        Total number of necessary inference steps for given inference settings is calculated as best as possible. The
        number of steps can be accurately calculated for finding suture regions, running suture identification and peak
        finding. Inference steps in these processes is based on the number of frames for which inference is run.

        The number of steps for sorting the found peaks can only be estimated, because the number of found peaks is
        only available after peak finding. In order to still accommodate for the process in the calculation of steps,
        all suture positions are assumed to be found for each frame. E.g.: For the default stereo suture grid with 7
        rows and 5 columns, this assumes 70 sutures to be sorted for each frame. In general, this leads to an
        overestimate for the number of inference steps, as very often a portion of sutures is not visible because of
        occlusion.

        :param dict inference_settings: The inference settings to be used.
        :return: Estimate for total number of steps in inference process.
        """
        inf_frames = inference_settings['to_frame'] - inference_settings['from_frame']
        total_steps = 0
        # Region detect inference happens only every `frame_delta`th frame, so the number of steps can be easily
        # calculated from the total number of frames
        if inference_settings['run_region_detect']:
            total_steps += inf_frames // inference_settings['region_frame_delta']
        # Creating suture probability maps happens twice for every single frame, but happens in batches where each batch
        # is considered a single step
        if inference_settings['run_suture_find']:
            total_steps += inf_frames * 2 // inference_settings['suture_find_batch']
        # Peak detection happens twice for every frame
        if inference_settings['run_peak_find']:
            total_steps += inf_frames * 2
        # The steps for peak sorting can only be guessed as we do not know beforehand how many peaks there are in the
        # data.
        if inference_settings['run_peak_sort']:
            total_steps += inf_frames * 70 // inference_settings['suture_sort_batch']

        return total_steps

    def ask_inference_settings(self, num_frames) -> dict:
        """
        Create dialog window for asking user about settings to be used for automatic annotation process.

        A :class:`InferenceSettings` window is created which allows entering settings and customization of the inference
        process.
        The dialog summarizes all entered settings as a dictionary which is returned. See :class:`InferenceSettings` for
        a detailed list of settings.

        :param int num_frames: Total number of frames available. Necessary for giving dialog window information about
            the upper limit of frames for which inference can be run.
        :return: Dictionary containing all settings entered by the user.
        """
        existing_inferences = gui_utils.search_existing_inferences(self)

        dialog = settings_window.InferenceSettings(existing_inferences, num_frames, parent=self)
        if dialog.exec_():
            self.inference_settings = dialog.settings
            return dialog.settings
        else:
            return None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        Overrides method from `QWidget <https://doc.qt.io/qt-5/qwidget.html#closeEvent>`_. Called upon closing this
        QMainWindow instance.

        Will save the currently applied configuration to path by calling :func:`save_config` before passing the event to
        PyQt5 for handling.

        :param event: The `QCloseEvent <https://doc.qt.io/qt-5/qcloseevent.html>`_ called by the
            `QMainWindow <https://doc.qt.io/qt-5/qmainwindow.html>`_ when closing the window.
        """
        self.save_config()
        super().closeEvent(event)

    def show_error(self, message: str) -> None:
        """
        Show an error message on the statusbar of the main window.

        :param message: The message displayed to the user.
        """
        self.status.setStyleSheet("QStatusBar{background:red;}")
        self.status.showMessage(message, msecs=5000)
        self.status.setStyleSheet("QStatusBar{;}")


class LoginWindow(QDialog):
    """
    **Bases:** :class:`QDialog <PyQt5.QtWidgets.QDialog>`

    Show a dialog asking the user for an Username used throughout the GUI.
    """
    def __init__(self):
        """
        Initialize the layout and display the login dialog.
        """
        super().__init__()

        self.setWindowTitle("Login")

        self.layout = QGridLayout()

        self.name_label = QLabel("Username: ")

        self.name_edit = QLineEdit(getpass.getuser())

        self.name_edit.textChanged.connect(self.text_entered)
        self.layout.addWidget(self.name_label, 0, 0)
        self.layout.addWidget(self.name_edit, 0, 1)

        self.button = QPushButton("Login")
        self.button.setDisabled(True)
        self.text_entered()
        self.button.clicked.connect(self.accept)
        self.layout.addWidget(self.button, 1, 1)

        self.setLayout(self.layout)

    def text_entered(self) -> None:
        """
        Slot triggered on :func:`textChanged` signal of :attr:`name_edit`.

        Will check if user currently has valid Username entered and if so, will make *Login* button clickable to allow
        user to proceed with application.
        """
        if self.name_edit.text() != '':
            self.button.setDisabled(False)
        else:
            self.button.setDisabled(True)

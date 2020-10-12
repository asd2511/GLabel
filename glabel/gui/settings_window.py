#!/usr/bin/env python
import glob

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.Qt import QKeySequence

from glabel.gui import gui_utils


class GridSettings(QDialog):
    """
    Custom :pyqt:`QDialog <qdialog>` for displaying and changing settings relating to the :class:`~gui.grid_widget.GridWidget` class.
    """

    #: Dictionary mapping integer values to style of used :class:`~gui.grid_widget.GridWidget`.
    GridStyles = {0: 'Buttons',
                  1: 'Pixelmap'}

    def __init__(self, cur_rows, cur_cols, cur_stereo, cur_style):
        """
        Initialize new :pyqt:`QDialog <qdialog>` for setting :class:´~gui.grid_widget.GridWidget` settings.

        :param int cur_rows: Currently set number of grid rows.
        :param int cur_cols: Currently set number of grid columns.
        :param bool cur_stereo: Current setting for stereoscopic grid.
        :param int cur_style: Currently set style for displaying grid widget. Available values determined by :attr:`GridStyles`.
        """
        super().__init__()
        self.setWindowTitle("Grid Settings")

        self.rows = cur_rows  #: Set number of rows
        self.cols = cur_cols  #: Set number of columns
        self.stereo = cur_stereo  #: Set stereoscopy of grid
        assert cur_style in self.GridStyles, "Invalid GridStyle encountered when initializing GridSettings!"
        self.grid_style = self.GridStyles[cur_style]  #: Set style of grid widget

        # self.layout = QVBoxLayout()
        self.layout = QGridLayout()

        # Create LineEdit fields for number of rows and columns in grid
        self.layout.addWidget(QLabel("Number of rows:"), 0, 0)
        self.row_edit = QLineEdit()
        self.row_edit.setText(str(self.rows))
        self.row_edit.textChanged.connect(self.text_entered)
        self.layout.addWidget(self.row_edit, 0, 1)

        self.layout.addWidget(QLabel("Number of columns:"), 1, 0)
        self.col_edit = QLineEdit()
        self.col_edit.setText(str(self.cols))
        self.col_edit.textChanged.connect(self.text_entered)
        self.layout.addWidget(self.col_edit, 1, 1)

        # Add Checkbox for stereoscopic grid settings
        self.layout.addWidget(QLabel("Enable stereoscopic grid"), 2, 0)
        self.stereo_check = QCheckBox()
        self.stereo_check.setChecked(self.stereo)
        self.layout.addWidget(self.stereo_check, 2, 1)

        # Add selection for switching between button and pixelmap appearance
        self.layout.addWidget(QLabel("Grid-Style:"), 3, 0)
        self.gridstyle_box = QComboBox(self)
        self.gridstyle_box.addItems([self.GridStyles[key] for key in list(self.GridStyles.keys())])
        self.gridstyle_box.setCurrentIndex(cur_style)
        self.layout.addWidget(self.gridstyle_box, 3, 1)

        # Create accept and cancel buttons
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.setDisabled(False)
        self.buttonBox.accepted.connect(self.confirm)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox, 4, 0, 1, 2)

        self.setLayout(self.layout)

    def text_entered(self):
        """
        Function acting as pyqtslot for enabling/disabling the accept button.

        The settings window should not accept settings that have the number of rows or columns set to nothing.
        This slot makes sure that the `accept` button is only available when valid values are entered.
        """
        if self.row_edit.text() != '' and self.col_edit.text() != '':
            self.buttonBox.setDisabled(False)
        else:
            self.buttonBox.setDisabled(True)

    def confirm(self):
        """
        Pyqtslot being called upon user confirmation of settings.

        Will update :attr:`rows`, :attr:`cols`, :attr:`stereo` and :attr:`grid_style` attributes to
        allow grabbing the updated settings.
        """
        self.rows = self.row_edit.text()
        self.cols = self.col_edit.text()
        self.stereo = self.stereo_check.isChecked()
        self.grid_style = self.gridstyle_box.currentIndex()

        self.accept()


class ShortcutSettings(QDialog):
    """
    Custom :pyqt:`QDialog <qdialog>` for displaying and changing settings relating to all keyboard shortcuts used in :meth:`~gui.image_widget.ImageStack.init_shortcuts`.
    """

    #: Dictionary mapping shorcut keys to descriptor texts used in the settings window.
    SC_Descs = {
        "DeleteActiveROI": "Delete currently selected ROI",
        "DeleteAllROI": "Delete all ROIs on the current frame",
        "ActivateRightROI": "Activate right Grid position",
        "ActivateLeftROI": "Activate left Grid position",
        "ActivateUpROI": "Activate upwards Grid position",
        "ActivateDownROI": "Activate downwards Grid position",
        "ActivateNextROI": "Activate next Grid position in order",
        "ActivatePreviousROI": "Activate previous Grid position in order",
        "MoveROIUp": "Move selected ROI 1px up",
        "MoveROIDown": "Move selected ROI 1px down",
        "MoveROIRight": "Move selected ROI 1px right",
        "MoveROILeft": "Move selected ROI 1px left",
        "NextFrame": "Show next frame",
        "PreviousFrame": "Show previous frame",
        "CopyNextROI": "Copy ROIs from next frame",
        "CopyPreviousROI": "Copy ROIs from previous frame",
        "CopyROIsToAll": "Copy ROIs from this frame to all other frames",
        "ToggleROIVisibility": "Toggle ROI visibility",
        "CycleROIColorCoding": "Cycle color coding of placed ROIs [Off, Rows, Columns]",
        "FindNewSuturePositions": "Copy ROIs from previous frame and track sutures using template matching",
        "FindNearestDark": "Copy ROIs from previous frame and track using darkest pixel",
        "FindNearestLight": "Copy ROIs from previous frame and track using lightest pixel",
        "FirstFrame": "Return to the first frame in the data",
        "LockView": "Lock the view (=disable mouse panning/zooming) and enable placing ROIs without holding CTRL",
        "Test": "Test Shortcut (no function)"
    }

    def __init__(self, shortcuts):
        """
        Initialize new :pyqt:`QDialog <qdialog>` for setting shortcut settings.

        :param dict shortcuts: Currently set dictionary mapping keyboard keys to shortcuts.
        """
        super().__init__()
        self.setWindowTitle("Shortcut Settings")

        self.shortcut_dict = shortcuts  #: Dictionary holding the updated shortcut mappings for grabbing from external

        self.layout = QGridLayout()

        self.layout.addWidget(QLabel(
            "Click on edit-field for a shortcut to activate editing.\nAfter pressing key-combination, wait for `...` to"
            "disappear from the edit-field to set desired combination."
        ), 0, 1, 1, 1)

        for row, action in enumerate(list(self.shortcut_dict.keys()), 1):
            # self.layout.addWidget(QLineEdit(QKeySequence(self.shortcut_dict[action]).toString()), row, 0)
            self.layout.addWidget(QKeySequenceEdit(self.shortcut_dict[action]), row, 0)
            self.layout.addWidget(QLabel(self.SC_Descs[action]), row, 1)

        # Create accept and cancel buttons
        qbtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(qbtn)
        self.buttonBox.accepted.connect(self.set_shortcuts)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox, row+1, 0, 1, 2)

        self.setLayout(self.layout)

    def set_shortcuts(self):
        """
        Function acting as pyqtslot for when user accepts currently applied shortcut settings.
        """
        action_names = list(self.shortcut_dict.keys())

        self.shortcut_dict = {}
        for edit, action in ((self.layout.itemAt(e_idx).widget(), action_names[a_idx])
                             for a_idx, e_idx in enumerate(range(self.layout.count()-1)[1::2])):
            self.shortcut_dict[action] = edit.keySequence().toString()

        self.accept()


class AutomationSettings(QDialog):
    """
    Custom :pyqt:`QDialog <qdialog>` for displaying and changing settings relating to automation settings for
    semi-automatic and manual labeling methods.
    """

    #: Dictionary mapping int values to available modes for changing active grid position during manual labeling.
    ActiveProgessModes = {0: 'On empty frame and ROI placement',
                          1: 'Only on empty frame',
                          2: 'Only on ROI placement',
                          3: 'Never'}

    def __init__(self, max_rows, mode, snaking, auto_copy, num_rows_track, begin_top, cur_active_progress,
                 frame_clicking):
        """
        Initialize new :pyqt:`QDialog <qdialog>` for setting automation settings.

        :param int max_rows: Number of rows available in grid.
        :param int mode: Mode for grid pathing. Currently supports only values {0, 1, 2} representing {Free, Row-wise,
            Column-wise}, respectively.
        :param bool snaking: Boolean setting for enabling snaking progression through grid.
        :param bool auto_copy: Boolean setting for enabling automatic copying of all set ROIs to next empty frame upon
            switching to it.
        :param int num_rows_track: Number of rows that try to track their annotated suture when using `copy and track`
            features.
        :param bool begin_top: Boolean setting enabling grid pathing to begin at top left instead of bottom left.
        :param int cur_active_progress: Currently set progress mode for active grid position. Currently supports only
            values {0, 1, 2, 3} as described by :attr:`ActiveProgressModes`.
        :param bool frame_clicking: Boolean setting enabling "frame clicking", the fastest found method for manual
            suture annotation.
        """
        super().__init__()
        self.setWindowTitle("Automation Settings")

        self.mode = mode  #: Mode for grid pathing progression
        self.snaking = snaking  #: Boolean value for enabling snaking grid pathing
        self.auto_copy = auto_copy  #: Boolean value for enabling automatic copying of all ROIs to next empty frame
        self.num_rows_track = num_rows_track  #: Number of rows that track sutures using copy-and-track features
        self.begin_top = begin_top  #: Boolean value for enabling grid pathing to begin at top-left
        self.active_progress_mode = cur_active_progress  #: Value for active grid position progression mode
        self.frame_clicking = frame_clicking  #: Boolean value for enabling "frame clicking" mode

        self.layout = QGridLayout()

        self.layout.addWidget(QLabel("Set mode for automatic progression through grid:"), 0, 0, 1, 2)
        # Radio selection for progression mode
        self.mode_buttons = QButtonGroup()
        self.free_mode_button = QRadioButton()
        self.lr_mode_button = QRadioButton()
        self.bt_mode_button = QRadioButton()
        self.mode_buttons.addButton(self.free_mode_button, 0)
        self.mode_buttons.addButton(self.lr_mode_button, 1)
        self.mode_buttons.addButton(self.bt_mode_button, 2)
        self.mode_buttons.buttonClicked[int].connect(lambda btn_id: self.set_mode(btn_id))
        if self.mode == 0:
            self.free_mode_button.setChecked(True)
        elif self.mode == 1:
            self.lr_mode_button.setChecked(True)
        elif self.mode == 2:
            self.bt_mode_button.setChecked(True)
        else:
            raise ValueError("Automation mode can only be set to 0 (free), 1 (left-right) or 2 (bottom-top)!")

        # Checkbox for switching between linear and snaking progression
        self.snake_box = QCheckBox()
        self.snake_box.clicked.connect(self.set_snaking)
        self.snake_label = QLabel("Enable \"snaking\" mode")
        if self.snaking:
            self.snake_box.setChecked(True)
        if self.mode == 0:
            self.snake_box.setDisabled(True)
            self.snake_label.setDisabled(True)

        # Checkbox for switching between beginning of pathing at top- or bottom-left
        self.begin_box = QCheckBox()
        self.begin_box.clicked.connect(self.set_begin)
        self.begin_label = QLabel("Begin pathing for progression at top-left (disabled = bottom-left)")
        if self.begin_top:
            self.begin_box.setChecked(True)
        if self.mode == 0:
            self.begin_box.setDisabled(True)
            self.begin_label.setDisabled(True)

        # Checkbox for switching between automatic copying of ROIs to empty frames or no copying
        self.copy_box = QCheckBox()
        self.copy_box.clicked.connect(self.set_autocopy)
        if self.auto_copy:
            self.copy_box.setChecked(True)

        # ComboBox for switching between modes of automatic progression of currently active grid position
        self.active_progress_box = QComboBox(self)
        self.active_progress_box.addItems([self.ActiveProgessModes[key] for key in list(self.ActiveProgessModes.keys())])
        self.active_progress_box.setCurrentIndex(cur_active_progress)
        self.active_progress_box.currentIndexChanged.connect(self.set_active_progress)

        # CheckBox for enabling automatic frame switching when placing ROI
        self.frame_click_box = QCheckBox(self)
        # Only allow setting of frame clicking if the current active progress mode is set to 'Never'
        self.frame_click_box.setDisabled(False if self.active_progress_mode == 3 else True)
        self.frame_click_box.setChecked(True if self.active_progress_mode == 3 and self.frame_clicking else False)
        self.frame_click_box.clicked.connect(self.set_frame_click)

        # SpinBox for setting how many rows (from bottom) should be automatically tracked when using auto-track-copy
        self.auto_track_box = QSpinBox(self)
        self.auto_track_box.setRange(0, max_rows)
        self.auto_track_box.setSuffix(" rows")
        self.auto_track_box.setValue(self.num_rows_track)
        self.auto_track_box.setFixedWidth(75)
        self.auto_track_box.valueChanged.connect(self.set_num_rows_track)

        self.layout.addWidget(self.free_mode_button, 1, 0)
        self.layout.addWidget(QLabel("Progress freely"), 1, 1)
        self.layout.addWidget(self.lr_mode_button, 2, 0)
        self.layout.addWidget(QLabel("Progress row-wise through grid (left to right)"), 2, 1)
        self.layout.addWidget(self.bt_mode_button, 3, 0)
        self.layout.addWidget(QLabel("Progress column-wise through grid (bottom to top"), 3, 1)
        self.layout.addWidget(self.snake_box, 4, 0)
        self.layout.addWidget(QLabel("Enable \"snaking\" mode"), 4, 1)
        self.layout.addWidget(self.begin_box, 5, 0)
        self.layout.addWidget(QLabel("Begin pathing for progression at top-left (disabled = bottom-left)"), 5, 1)
        self.layout.addWidget(self.copy_box, 6, 0)
        self.layout.addWidget(QLabel("Enable automatic copying of ROIs to next frame if empty"), 6, 1)
        self.layout.addWidget(self.active_progress_box, 7, 0)
        self.layout.addWidget(QLabel("Mode for automatic progression of active grid position"), 7, 1)
        self.layout.addWidget(self.frame_click_box, 8, 0)
        self.layout.addWidget(QLabel("Activate frame clicking: Move to next frame after placement of ROI "
                                     "(only available if progress mode set to 'Never')"), 8, 1)
        self.layout.addWidget(self.auto_track_box, 9, 0)
        self.layout.addWidget(QLabel("Number of rows that auto-track (from bottom) when using copy-and-track command"), 9, 1)

        # Dialog Buttons
        qbtns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        qbtns.accepted.connect(self.accept)
        qbtns.rejected.connect(self.reject)

        self.layout.addWidget(qbtns, 10, 0, 1, 2)

        self.setLayout(self.layout)

    def set_mode(self, mode):
        """
        Custom PyQtSlot for catching and handling user changes in the selection of grid progression mode.

        This slot will receive the updated value for the selected progression mode, update :attr:`mode` and handle
        special cases for disabling setting options based on the selection.

        :param int mode: New value for the progression mode. This will be emitted by the selected radio button.
        """
        self.mode = mode

        if self.mode == 0:
            self.snake_box.setDisabled(True)
            self.snake_label.setDisabled(True)
            self.begin_box.setDisabled(True)
            self.begin_label.setDisabled(True)

        else:
            self.snake_box.setEnabled(True)
            self.snake_label.setEnabled(True)
            self.begin_box.setEnabled(True)
            self.begin_label.setEnabled(True)

    def set_snaking(self):
        """
        Slot for updating :attr:`snaking` with new selected value.
        """
        self.snaking = self.snake_box.isChecked()

    def set_autocopy(self):
        """
        Slot for updating :attr:`auto_copy` with new selected value.
        """
        self.auto_copy = self.copy_box.isChecked()

    def set_num_rows_track(self):
        """
        Slot for updating :attr:`num_rows_track` with new selected value.
        """
        self.num_rows_track = self.auto_track_box.value()

    def set_begin(self):
        """
        Slot for updating :attr:`begin_top` with new selected value.
        """
        self.begin_top = self.begin_box.isChecked()

    def set_active_progress(self):
        """
        Slot for updating :attr:`active_progress_mode` with new selected value.

        Will also handle enabling/disabling mode selection for "frame clicking" based on the selection made.
        """
        self.active_progress_mode = self.active_progress_box.currentIndex()
        if self.active_progress_mode == 3:
            self.frame_click_box.setEnabled(True)
        else:
            self.frame_click_box.setChecked(False)
            self.frame_click_box.setDisabled(True)
            self.set_frame_click()

    def set_frame_click(self):
        """
        Slot for updating :attr:`frame_clicking` with new selected value.
        """
        self.frame_clicking = self.frame_click_box.isChecked()


class AppearanceSettings(QDialog):
    def __init__(self, roi_color, active_color, closeup, snap, crosshair):
        super().__init__()
        self.setWindowTitle("Appearance Settings")

        self.roi_color = roi_color
        self.active_color = active_color

        self.layout = QGridLayout()

        self.layout.addWidget(QLabel("Color of placed ROIs"), 0, 0)
        self.roi_button = QPushButton()
        self.style_button(self.roi_button, self.roi_color)
        # self.roi_button.clicked.connect(self.pick_roi_color)
        self.roi_button.clicked.connect(
            lambda state, btn=self.roi_button, clrvar=self.roi_color: self.pick_color(btn, clrvar))
        self.layout.addWidget(self.roi_button, 0, 1)

        self.layout.addWidget(QLabel("Color of active ROI"), 1, 0)
        self.active_button = QPushButton()
        self.style_button(self.active_button, self.active_color)
        self.active_button.clicked.connect(
            lambda state, btn=self.active_button, clrvar=self.active_color: self.pick_color(btn, clrvar))
        self.layout.addWidget(self.active_button, 1, 1)

        self.layout.addWidget(QLabel("Show ROI CloseUp"), 2, 0)
        self.closeup_check = QCheckBox()
        self.closeup_check.setChecked(closeup)
        self.layout.addWidget(self.closeup_check, 2, 1)

        self.layout.addWidget(QLabel("Snap ROIs to closest full pixel position"), 3, 0)
        self.snap_check = QCheckBox()
        self.snap_check.setChecked(snap)
        self.layout.addWidget(self.snap_check, 3, 1)

        self.layout.addWidget(QLabel("Show crosshair"), 4, 0)
        self.crosshair_check = QCheckBox()
        self.crosshair_check.setChecked(crosshair)
        self.layout.addWidget(self.crosshair_check, 4, 1)

        # Dialog Buttons
        qbtns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        qbtns.accepted.connect(self.accept)
        qbtns.rejected.connect(self.reject)
        self.layout.addWidget(qbtns, 5, 0, 1, 2)

        self.setLayout(self.layout)

    def pick_roi_color(self):
        color_picker = QColorDialog()
        self.roi_color = color_picker.getColor(initial=self.roi_color)
        self.style_button(self.roi_button, self.roi_color)

    def pick_color(self, button, color):
        color_picker = QColorDialog()
        color = color_picker.getColor(initial=color)

        if button is self.roi_button:
            self.roi_color = color
        elif button is self.active_button:
            self.active_color = color
        else:
            raise ValueError("Unknown button encountered while trying to re-set colors!")

        self.style_button(button, color)

    @staticmethod
    def style_button(button, color):
        button.setStyleSheet(
            "border-width: 2px; " +
            "border-color: beige; " +
            "background-color: " + color.name())


class PropertiesSettings(QDialog):
    def __init__(self, mode, confirm):
        super().__init__()
        self.setWindowTitle("Properties")

        self.save_mode = mode
        self.confirm = confirm

        self.layout = QGridLayout()

        self.layout.addWidget(QLabel("Set mode for saving behavior when pressing 'Ctrl+S' or using 'File->Save':"),
                              0, 0, 1, 2)
        self.save_mode_btns = QButtonGroup()
        self.save_ts_btn = QRadioButton()  # Selects saving as new timestamped path
        save_ts_label = QLabel("Create and save in new path each time. (Filename will be extended with timestamp)")
        save_ts_label.setAlignment(Qt.AlignLeft)
        self.save_over_btn = QRadioButton()  # Selects saving by overwriting existing path
        save_over_label = QLabel("Overwrite existing savefile each time. (No timestamp in filename)")
        save_over_label.setAlignment(Qt.AlignLeft)
        self.save_mode_btns.addButton(self.save_ts_btn, 0)
        self.save_mode_btns.addButton(self.save_over_btn, 1)
        self.save_mode_btns.buttonClicked[int].connect(lambda btn_id: self.set_save_mode(btn_id))
        if self.save_mode == 0:
            self.save_ts_btn.setChecked(True)
        elif self.save_mode == 1:
            self.save_over_btn.setChecked(True)
        else:
            raise ValueError("Save mode can only be set to 0 (timestamped path) or 1 (overwriting path)!")

        self.confirm_box = QCheckBox()
        self.confirm_box.setChecked(self.confirm)
        self.confirm_box.clicked.connect(self.set_confirm)
        confirm_label = QLabel("Ask to confirm save location each time before writing to save path.")
        confirm_label.setAlignment(Qt.AlignLeft)

        self.layout.addWidget(self.save_ts_btn, 1, 0)
        self.layout.addWidget(save_ts_label, 1, 1)
        self.layout.addWidget(self.save_over_btn, 2, 0)
        self.layout.addWidget(save_over_label, 2, 1)
        self.layout.addWidget(self.confirm_box, 3, 0)
        self.layout.addWidget(confirm_label, 3, 1)

        qbtns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        qbtns.accepted.connect(self.accept)
        qbtns.rejected.connect(self.reject)
        self.layout.addWidget(qbtns, 4, 0, 1, 2)

        self.setLayout(self.layout)

    def set_save_mode(self, mode):
        self.save_mode = mode

    def set_confirm(self):
        self.confirm = self.confirm_box.isChecked()


class EnhancementSettings(QDialog):
    """
    **Bases** :class:`QDialog <PyQt5.QtWidgets.QDialog>`

    The settings window for image enhancement.
    """
    def __init__(self, gaussian):
        super().__init__()
        self.setWindowTitle("Enhancement Settings")

        self.gaussian = gaussian

        self.layout = QGridLayout()

        # Checkbox for switching between linear and snaking progression
        self.gaussian_box = QCheckBox()
        self.gaussian_box.clicked.connect(self.set_gaussian)
        self.gaussian_label = QLabel("Enable gaussian filter in all frames")
        if self.gaussian:
            self.gaussian_box.setChecked(True)

        # Layout
        self.layout.addWidget(self.gaussian_box, 1, 0)
        self.layout.addWidget(self.gaussian_label, 1, 1)

        # Dialog Buttons
        qbtns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        qbtns.accepted.connect(self.accept)
        qbtns.rejected.connect(self.reject)
        self.layout.addWidget(qbtns, 2, 0, 1, 2)

        self.setLayout(self.layout)

    def set_gaussian(self):
        """
        Checks the checkbox.
        """
        self.gaussian = self.gaussian_box.isChecked()


class InferenceSettings(QDialog):
    def __init__(self, existing_inferences, total_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.existing_inferences = existing_inferences
        self.total_frames = total_frames
        self.saved_darknet_networks = glob.glob('./nn/*.cfg')
        self.saved_darknet_weights = glob.glob('./nn/*.weights')
        self.saved_h5_networks = glob.glob('./nn/*.h5')

        self.settings = {}

        self.setLayout(self.build())

    def build(self) -> QVBoxLayout:
        # -----------------
        # Finalizing layout
        layout = QVBoxLayout()

        btnbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btnbox.accepted.connect(self.accept)
        btnbox.rejected.connect(self.reject)

        layout.addWidget(self.build_input_data_grp(), alignment=Qt.AlignLeft)
        layout.addWidget(self.build_region_grp(), alignment=Qt.AlignLeft)
        layout.addWidget(self.build_suture_maps_grp(), alignment=Qt.AlignLeft)
        layout.addWidget(self.build_peak_find_grp(), alignment=Qt.AlignLeft)
        layout.addWidget(self.build_peak_sort_grp(), alignment=Qt.AlignLeft)
        layout.addWidget(btnbox, alignment=Qt.AlignRight)

        return layout

    def build_input_data_grp(self) -> QGroupBox:
        # ====================================
        # Settings for limiting input data
        input_settings = QGroupBox('Limit input data', self)
        input_settings.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        input_layout = QFormLayout(input_settings)
        input_layout.setSizeConstraint(QLayout.SetNoConstraint)

        # - Option for selective excluding frames
        frame_start_box = QSpinBox()
        frame_start_box.setMaximum(self.total_frames)
        frame_start_box.setMinimum(0)
        frame_start_box.setMaximum(self.total_frames)
        frame_start_box.setValue(0)
        frame_start_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        frame_start_box.setObjectName('frame_start_box')

        frame_end_box = QSpinBox()
        frame_end_box.setMinimum(0)
        frame_end_box.setMaximum(self.total_frames)
        frame_end_box.setValue(self.total_frames)
        frame_end_box.setObjectName('frame_end_box')
        frame_end_box.valueChanged.connect(lambda val: self.police_value(val, frame_start_box, 'maximum'))

        # -- Add to grouped layout
        input_layout.addRow("From frame:", frame_start_box)
        input_layout.addRow("To frame:", frame_end_box)
        input_settings.setLayout(input_layout)

        return input_settings

    def build_region_grp(self) -> QGroupBox:
        # ====================================
        # Settings for suture region detection
        region_detect_settings = QGroupBox('Suture region detection settings', self)
        region_detect_layout = QFormLayout(region_detect_settings)

        # - Option for running inference only every nth frame
        frame_delta_box = QSpinBox()
        frame_delta_box.setValue(25)
        frame_delta_box.setMinimum(1)
        frame_delta_box.setMaximum(self.total_frames)
        frame_delta_box.setObjectName('frame_delta_box')

        # - Option for selecting which trained network to use
        region_network_select = QComboBox(self)
        region_network_select.addItems(self.saved_darknet_networks)
        region_network_select.addItem('...select file')
        region_network_select.setCurrentIndex(0)
        region_network_select.currentIndexChanged.connect(lambda i: gui_utils.process_combo_select(region_network_select))
        region_network_select.setObjectName('region_network_select')

        # - Option for selecting which network weights to use
        region_weights_select = QComboBox(self)
        region_weights_select.addItems(self.saved_darknet_weights)
        region_weights_select.addItem('...select file')
        region_weights_select.setCurrentIndex(0)
        region_weights_select.currentIndexChanged.connect(lambda i: gui_utils.process_combo_select(region_weights_select))
        region_weights_select.setObjectName('region_weights_select')

        # - Option for re-using previously saved inference
        region_reload_box = QComboBox(self)
        if self.existing_inferences['regions']:
            region_reload_box.addItem("Use existing inference")
        region_reload_box.addItem("Run new inference")
        region_reload_box.addItem('...load from file')
        region_reload_box.setMinimumWidth(100)
        region_reload_box.setObjectName('region_reload_box')

        # -- Add to grouped layout
        region_detect_layout.addRow("", region_reload_box)
        region_detect_layout.addRow("Inference only every nth frame:", frame_delta_box)
        region_detect_layout.addRow("Neural Network:", region_network_select)
        region_detect_layout.addRow("Network weights:", region_weights_select)
        region_detect_settings.setLayout(region_detect_layout)
        region_detect_settings.setCheckable(True)
        region_detect_settings.setChecked(True)
        region_detect_settings.setObjectName('region_detect_settings')

        # -- Setting additional parameters
        region_detect_settings.setFixedWidth(1000)

        # ----> Logic for handling reusing of saved inference file
        # Process selection once to have correct enable/disable behaviour when inference file is available
        gui_utils.process_reuse_select(region_reload_box, self.existing_inferences['regions'], region_detect_layout)
        region_reload_box.currentIndexChanged.connect(
            lambda i: gui_utils.process_reuse_select(region_reload_box,
                                                     self.existing_inferences['regions'],
                                                     region_detect_layout)
        )

        return region_detect_settings

    def build_suture_maps_grp(self) -> QGroupBox:
        # ====================================
        # Settings for individual suture detection
        suture_find_settings = QGroupBox('Suture identification settings', self)
        suture_find_layout = QFormLayout(suture_find_settings)

        # - Option for selecting which trained network to use
        find_network_select = QComboBox(self)
        find_network_select.addItems(self.saved_h5_networks)
        find_network_select.addItem('...select file')
        find_network_select.setCurrentIndex(0)
        find_network_select.currentIndexChanged.connect(lambda i: gui_utils.process_combo_select(find_network_select))
        find_network_select.setObjectName('find_network_select')

        # - Option for network batch size
        unet_batch_box = QSpinBox(self)
        unet_batch_box.setValue(1)
        unet_batch_box.setMinimum(1)
        unet_batch_box.setMaximum(128)
        unet_batch_box.setObjectName('unet_batch_box')

        # - Option for re-using previously saved inference
        find_reload_box = QComboBox(self)
        if self.existing_inferences['suture_maps']:
            find_reload_box.addItem("Use existing inference")
        find_reload_box.addItem("Run new inference")
        find_reload_box.addItem('...load from file')
        find_reload_box.setMinimumWidth(100)
        find_reload_box.setObjectName('find_reload_box')

        # -- Add to grouped layout
        suture_find_layout.addRow("", find_reload_box)
        suture_find_layout.addRow("Neural Network:", find_network_select)
        suture_find_layout.addRow("Batch size for RUNet:", unet_batch_box)
        suture_find_settings.setLayout(suture_find_layout)
        suture_find_settings.setCheckable(True)
        suture_find_settings.setChecked(True)
        suture_find_settings.setObjectName('suture_find_settings')

        # -- Setting additional parameters
        suture_find_settings.setFixedWidth(1000)

        # ----> Logic for handling reusing of saved inference file
        # Process selection once to have correct enable/disable behaviour when inference file is available
        gui_utils.process_reuse_select(find_reload_box, self.existing_inferences['suture_maps'], suture_find_layout)
        find_reload_box.currentIndexChanged.connect(
            lambda i: gui_utils.process_reuse_select(find_reload_box,
                                                     self.existing_inferences['suture_maps'],
                                                     suture_find_layout)
        )

        return suture_find_settings

    def build_peak_find_grp(self) -> QGroupBox:
        # ===================================
        # Settings for peak detection on probability maps
        peak_find_settings = QGroupBox('Peak finding settings')
        peak_find_layout = QFormLayout(peak_find_settings)
        # - Set the minimal distance between peaks
        peak_dist_box = QSpinBox(self)
        peak_dist_box.setValue(3)
        peak_dist_box.setMinimum(1)
        peak_dist_box.setMaximum(100)
        peak_dist_box.setObjectName('peak_dist_box')

        # - Set the absolute threshold required for peaks
        peak_thresh_box = QDoubleSpinBox(self)
        peak_thresh_box.setValue(0.5)
        peak_thresh_box.setMinimum(0.0)
        peak_thresh_box.setMaximum(1.0)
        peak_thresh_box.setObjectName('peak_thresh_box')

        # - Option for re-using previously saved inference
        peak_reload_box = QComboBox(self)
        if self.existing_inferences['map_peaks']:
            peak_reload_box.addItem("Use existing inference")
        peak_reload_box.addItem("Run new inference")
        peak_reload_box.addItem('...load from file')
        peak_reload_box.setMinimumWidth(100)
        peak_reload_box.setObjectName('peak_reload_box')

        # -- Add to grouped layout
        peak_find_layout.addRow("", peak_reload_box)
        peak_find_layout.addRow("Minimum peak distance:", peak_dist_box)
        peak_find_layout.addRow("Absolute threshold:", peak_thresh_box)
        peak_find_settings.setLayout(peak_find_layout)
        peak_find_settings.setCheckable(True)
        peak_find_settings.setChecked(True)
        peak_find_settings.setObjectName('peak_find_settings')

        # -- Setting additional parameters
        peak_find_settings.setFixedWidth(1000)

        # ----> Logic for handling reusing of saved inference file
        # Process selection once to have correct enable/disable behaviour when inference file is available
        gui_utils.process_reuse_select(peak_reload_box, self.existing_inferences['map_peaks'], peak_find_layout)
        peak_reload_box.currentIndexChanged.connect(
            lambda i: gui_utils.process_reuse_select(peak_reload_box,
                                                     self.existing_inferences['map_peaks'],
                                                     peak_find_layout)
        )

        return peak_find_settings

    def build_peak_sort_grp(self) -> QGroupBox:
        # ===================================
        # Settings for sorting found sutures
        peak_sort_settings = QGroupBox('Suture sorting settings')
        peak_sort_layout = QFormLayout(peak_sort_settings)

        # - Option for selecting which network to use
        sort_network_select = QComboBox(self)
        sort_network_select.addItems(self.saved_h5_networks)
        sort_network_select.addItem('...select file')
        sort_network_select.setCurrentIndex(0)
        sort_network_select.currentIndexChanged.connect(lambda i: gui_utils.process_combo_select(sort_network_select))
        sort_network_select.setObjectName('sort_network_select')

        # - Option for network batch size
        effnet_batch_box = QSpinBox()
        effnet_batch_box.setValue(1)
        effnet_batch_box.setMinimum(1)
        effnet_batch_box.setMaximum(128)
        effnet_batch_box.setObjectName('effnet_batch_box')

        # - Option for re-using previously saved inference
        sort_reload_box = QComboBox(self)
        if self.existing_inferences['sortings']:
            sort_reload_box.addItem("Use existing inference")
        sort_reload_box.addItem("Run new inference")
        sort_reload_box.addItem('...load from file')
        sort_reload_box.setMinimumWidth(100)
        sort_reload_box.setObjectName('sort_reload_box')

        # -- Add to grouped layout
        peak_sort_layout.addRow("", sort_reload_box)
        peak_sort_layout.addRow("Neural Network:", sort_network_select)
        peak_sort_layout.addRow("Batch size for EfficientNet:", effnet_batch_box)
        peak_sort_settings.setLayout(peak_sort_layout)
        peak_sort_settings.setCheckable(True)
        peak_sort_settings.setChecked(True)
        peak_sort_settings.setObjectName('peak_sort_settings')

        # -- Setting additional parameters
        peak_sort_settings.setFixedWidth(1000)

        # ----> Logic for handling reusing of saved inference file
        # Process selection once to have correct enable/disable behaviour when inference file is available
        gui_utils.process_reuse_select(sort_reload_box, self.existing_inferences['sortings'], peak_sort_layout)
        sort_reload_box.currentIndexChanged.connect(
            lambda i: gui_utils.process_reuse_select(sort_reload_box,
                                                     self.existing_inferences['sortings'],
                                                     peak_sort_layout)
        )

        return peak_sort_settings

    def get_input_settings(self) -> dict:
        input_settings = {
            'from_frame': self.findChild(QWidget, 'frame_start_box').value(),
            'to_frame': self.findChild(QWidget, 'frame_end_box').value(),
            'total_frames': self.total_frames
        }

        return input_settings

    def get_region_settings(self) -> dict:
        region_reload_box = self.findChild(QWidget, 'region_reload_box')
        region_settings = {
            'run_region_detect': self.findChild(QWidget, 'region_detect_settings').isChecked(),
            'region_frame_delta': self.findChild(QWidget, 'frame_delta_box').value(),
            'region_network_path': self.findChild(QWidget, 'region_network_select').currentText(),
            'region_weights_path': self.findChild(QWidget, 'region_weights_select').currentText(),
            'load_regions':
                False if region_reload_box.currentText() == 'Run new inference'
                else True,
            'regions_file':
                self.existing_inferences['regions'] if region_reload_box.currentText() == 'Use existing inference'
                else region_reload_box.currentText()
        }

        return region_settings

    def get_suture_map_settings(self) -> dict:
        find_reload_box = self.findChild(QWidget, 'find_reload_box')
        suture_map_settings = {
            'run_suture_find': self.findChild(QWidget, 'suture_find_settings').isChecked(),
            'suture_find_network': self.findChild(QWidget, 'find_network_select').currentText(),
            'suture_find_batch': self.findChild(QWidget, 'unet_batch_box').value(),
            'load_maps':
                False if find_reload_box.currentText() == 'Run new inference'
                else True,
            'maps_file':
                self.existing_inferences['suture_maps'] if find_reload_box.currentText() == 'Use existing inference'
                else find_reload_box.currentText()
        }

        return suture_map_settings

    def get_peak_find_settings(self) -> dict:
        peak_reload_box = self.findChild(QWidget, 'peak_reload_box')
        peak_find_settings = {
            'run_peak_find': self.findChild(QWidget, 'peak_find_settings').isChecked(),
            'peak_find_distance': self.findChild(QWidget, 'peak_dist_box').value(),
            'peak_find_threshold': self.findChild(QWidget, 'peak_thresh_box').value(),
            'load_peaks':
                False if peak_reload_box.currentText() == 'Run new inference'
                else True,
            'peaks_file':
                self.existing_inferences['map_peaks'] if peak_reload_box.currentText() == 'Use existing inference'
                else peak_reload_box.currentText()
        }

        return peak_find_settings

    def get_peak_sort_settings(self) -> dict:
        sort_reload_box = self.findChild(QWidget, 'sort_reload_box')
        peak_sort_settings = {
            'run_peak_sort': self.findChild(QWidget, 'peak_sort_settings').isChecked(),
            'suture_sort_network': self.findChild(QWidget, 'sort_network_select').currentText(),
            'suture_sort_batch': self.findChild(QWidget, 'effnet_batch_box').value(),
            'load_sortings':
                False if sort_reload_box.currentText() == 'Run new inference'
                else True,
            'sorting_file':
                self.existing_inferences['sortings'] if sort_reload_box.currentText() == 'Use existing inference'
                else sort_reload_box.currentText()
        }

        return peak_sort_settings

    def accept(self) -> None:
        self.settings.update(self.get_input_settings())
        self.settings.update(self.get_region_settings())
        self.settings.update(self.get_suture_map_settings())
        self.settings.update(self.get_peak_find_settings())
        self.settings.update(self.get_peak_sort_settings())

        super().accept()

    def police_value(self, value, control_widget, control_type):
        """
        Police the value entered in `control_widget` based on the passed `value` and `control_type`.

        Examples:
        ---------
        This will set the valid maxmimum of the passed QSpinBox object to the value 50.

        >>> police_value(50, QSPinBox(), 'maximum')

        :param value: Numerical value to base policing on.
        :type value: Union[int, float]
        :param control_widget: QWidget whose valid value range will be policed based on the passed value. The controlled
            property will be manipulated directly in this method.
        :type control_widget: QWidget
        :param control_type: Type of policing performed. Valid options are {'maximum', 'minimum'}.
        :type control_type: str
        """
        if control_type == 'maximum':
            control_widget.setMaximum(value)

        elif control_type == 'minimum':
            control_widget.setMinimum(value)

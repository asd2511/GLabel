from PyQt5.QtGui import QDoubleValidator, QRegExpValidator
from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QSizePolicy, QSplitter, QLineEdit, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QLocale, QRegExp
import pyqtgraph as pg


default_volume = 125  #: Default calibration cube volume [mm³] used (=> edge lengths = 5mm)


class CalibrationWindow(QMainWindow):
    """
    Window used for setting the calibration parameters for the GLable application.

    **Bases**: :pyqt:`QMainWindow <qmainwindow>`.

    Will be opened as a separate window when user selects *Analyze --> Calibration* from the
    :class:`Main <gui.main_window.Main`.
    This calibration window is intended to be used to set calibration data for reconstructing medial vocal fold surfaces
    in 3D by using a calibration cube image.
    """

    calibration_set_sig = pyqtSignal(dict)  #: Signal emitting dictionary with calibration values

    def __init__(self, img, sides=None, parent=None):
        """
        Setting the CalibrationWindow with an image to display.

        :param img: The image to open in the new CalibrationWindow window.
        :type img: numpy.ndarray
        :param sides: The number of views of the calibration object. Set to 2 for stereoscopic view and 1 for mono view.
            Can be omitted, giving user the option to manually add calibration cubes for mono or stereo views.
        :type sides: int
        :param parent: Parent object of the CalibrationWindow. Should be the main window opening this calibration window
            by default
        :type parent: QMainWindow
        """
        super().__init__(parent)

        self.img = img  #: Image displayed for setting calibration data

        self.setWindowTitle("Calibration")
        # New window position with small offset to main window and size to just fit the image and necessary widgets
        p_geometry = parent.geometry()
        self.setGeometry(p_geometry.x() + 50, p_geometry.y() + 50, self.img.shape[1] * 1.5, self.img.shape[0])

        #: Widget of class :class:`CalibrationWidget` holding the logic for creating calibration values
        self.cal_widget = CalibrationWidget(self.img)

        #: Menu allowing addition of calibration cube widgets
        self.cube_m = self.menuBar().addMenu("&Add calibration")
        self.cube_m.addAction("&Stereo view", self.cal_widget.add_stereo_cubes)
        self.cube_m.addAction("&Mono view", self.cal_widget.add_mono_cube)

        #: Menu action for removing all added calibration cube widgets
        self.rm_cubes_a = self.menuBar().addAction("&Remove calibrations", self.cal_widget.remove_cubes)
        self.rm_cubes_a.setEnabled(False)

        #: Menu action for setting the currently placed calibration cubes as the active calibration for GLable
        self.use_calibration_a = self.menuBar().addAction("&Use calibration", self.set_calibration)
        self.use_calibration_a.setEnabled(False)

        # Handling of menu/action availability
        self.cal_widget.added_sig.connect(lambda: self.rm_cubes_a.setEnabled(True))
        self.cal_widget.added_sig.connect(lambda: self.cube_m.setEnabled(False))
        self.cal_widget.valid_sig.connect(lambda: self.use_calibration_a.setEnabled(True))
        self.cal_widget.invalid_sig.connect(lambda: self.use_calibration_a.setEnabled(False))

        if sides == 1:
            self.cal_widget.add_mono_cube()
        elif sides == 2:
            self.cal_widget.add_stereo_cubes()
        else:
            raise ValueError(f"Incompatible number of `sides` {sides} received in CalibrationWindow!")

        self.setCentralWidget(self.cal_widget)

    def set_calibration(self) -> None:
        """
        Accept the currently placed calibration cubes as the valid calibration and make the values available to be used
        by GLable.

        The calibration data is processed, formatted and emitted as a dictionary via the :attr:`calibration_set_sig`
        signal.

        Calibration dictionary entries are:

        ==================  =================================================================================
        **Key**             **Entry**
        ['v']               Dictionary with calibration data from **left** view side
        ['vp']              Dictionary with calibration data from **right** view side
        [('v'/'vp')]['v1']  Calibration vector from top-left to top-right cube corner (drawn in magenta)
        [('v'/'vp')]['v2']  Calibration vector from top-left to bottom-left cube corner (drawn in green)
        [('v'/'vp')]['v3']  Calibration vector from top-left to (back)top-left cube corner (drawn in cyan)
        ==================  =================================================================================
        """
        sides = self.cal_widget.sides  # Number of sides depending on number of calibration cubes added
        segments = {'v': self.cal_widget.cubes.cube_v.segments}  # Segments of the left/single cube
        # We need to capture the offset of the origin of the CalibrationEdges objects such that we can find the
        # true position of their handles. Otherwise the positions would always be relative to the originally placed
        # origin point of the object.
        global_offset = {'v': self.cal_widget.cubes.cube_v.pos()}
        if sides == 2:
            segments['vp'] = self.cal_widget.cubes.cube_vp.segments  # Segments of the right cube
            global_offset['vp'] = self.cal_widget.cubes.cube_vp.pos()

        edgenames = ['v1', 'v2', 'v3']  # Names according to publication
        calibration = dict.fromkeys(segments)
        for cube in list(calibration.keys()):
            s = segments[cube]
            offset = global_offset[cube]
            calibration[cube] = dict.fromkeys(edgenames)  # Create new dictionary with one entry for each edge
            for segment, edgename in zip(s, edgenames):
                center_point = segment.handles[0]['pos'] + offset
                end_point = segment.handles[1]['pos'] + offset
                # calibration[cube][edgename] = {'x_center': center_point.x(), 'y_center': center_point.y(),
                #                                'x_end': end_point.x(), 'y_end': end_point.y()}
                calibration[cube][edgename] = {'x': end_point.x() - center_point.x(),
                                               'y': end_point.y() - center_point.y()}
            # Add the center point of all handles as origin of that view
            calibration[cube]['origin'] = {'x': center_point.x(),
                                           'y': center_point.y()}

        calibration['phys'] = {}
        edge_lens = []
        for idx, edge in enumerate(self.cal_widget.b_edits):
            edge_len = float(edge.text())
            edge_lens.append(edge_len)
            calibration['phys'][f'b{idx+1}'] = [edge_len if c == idx else 0.0 for c in range(4)]
        # Use the mean edge length to fill the fourth "dummy" dimension
        calibration['phys']['b4'] = [0.0, 0.0, 0.0, sum(edge_lens) / len(edge_lens)]

        self.calibration_set_sig.emit(calibration)


class CalibrationWidget(QWidget):
    """
    Widget populating :class:`CalibrationWindow` holding further widgets and logic for creating the calibration cube
    calibration data.

    **Bases** :pyqt:`QWidget <qwidget>`
    """

    added_sig = pyqtSignal()  #: Triggered when a calibration cube widget is added to the scene
    removed_sig = pyqtSignal()  #: Triggered when the current calibration cube widgets are removed from the scene
    valid_sig = pyqtSignal()  #: Triggered when currently entered parameters for physical dimensions are acceptable
    invalid_sig = pyqtSignal()  #: Triggered when invalid configuration is present (i.e. no calibration widgets present)

    def __init__(self, img):
        """
        Initialize calibration widget for setting calibration data.

        :param img: The image data to be displayed for calibration.
        :type img: numpy.ndarray
        """
        super().__init__()

        self.cubes = None  #: Holding references to created calibration cube widgets of class :class:`CubeCalibrator`
        self.sides = 0  #: Number of sides to be calibrated --> 1 for mono view, 2 for stereoscopic view

        self.layout = QGridLayout()  #: :pyqt:`QGridLayout <qgridlayout>` of this widget
        self.splitter = QSplitter(Qt.Horizontal)  #: :pyqt:`QSplitter <qsplitter>` separating image from edge info

        #: Displays the image as a :class:`pyqtgraph.ImageItem` object
        self.cal_item = pg.ImageItem(img)
        #: :class:`pyqtgraph.ImageView` for holding :attr:`self.cal_item` and :class:`CubeCalibrator` objects once
        #: initialized
        self.cal_view = pg.ImageView(self, imageItem=self.cal_item)
        self.cal_view.getView().setMenuEnabled(False)
        self.cal_view.ui.menuBtn.hide()
        self.cal_view.ui.roiBtn.hide()
        self.cal_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        #: :pyqt:`QWidget <qwidget>` for entering physical cube dimensions
        self.phys_params = QWidget()
        self.phys_layout = QGridLayout()
        locale = QLocale(QLocale.English, QLocale.UnitedStates)
        locale.setNumberOptions(QLocale.RejectGroupSeparator)
        # Capture up to 3 digits, may be followed by a dot, and up to two decimals
        phys_validator = QRegExpValidator(QRegExp("^\d{0,3}(\.\d{0,2})?$"))
        phys_validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        #: Field for entering/showing cube volume
        self.v_edit = QLineEdit()
        self.v_edit.setFixedWidth(50)
        self.v_edit.setValidator(phys_validator)
        self.v_edit.setText(str(default_volume))
        self.v_edit.textEdited.connect(self.set_lengths)
        self.v_edit.textEdited.connect(self.check_valid)
        self.phys_layout.addWidget(QLabel("Cube volume [mm³]"), 0, 0)
        self.phys_layout.addWidget(self.v_edit, 0, 1)
        #: Fields for entering/showing individual edge lengths
        self.b_edits = [QLineEdit(), QLineEdit(), QLineEdit()]
        b_colors = ['magenta', 'lime', 'cyan']
        for idx, edit in enumerate(self.b_edits):
            edit.setFixedWidth(50)
            edit.setValidator(phys_validator)
            edit.textEdited.connect(self.set_volume)
            edit.textEdited.connect(self.check_valid)
            self.phys_layout.addWidget(QLabel(f"Length of edge b_{idx+1} [mm]"), idx+1, 0)
            self.phys_layout.addWidget(edit, idx+1, 1)
            edit.setStyleSheet(f'border-color: {b_colors[idx]}')
        self.set_lengths()

        self.phys_params.setLayout(self.phys_layout)
        self.phys_layout.setAlignment(Qt.AlignTop)
        self.phys_params.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.splitter.addWidget(self.cal_view)
        self.splitter.addWidget(self.phys_params)
        self.splitter.setSizes([10e6, self.phys_params.sizeHint().width()])

        self.layout.addWidget(self.splitter, 0, 0)
        self.setLayout(self.layout)

    def add_stereo_cubes(self) -> None:
        """
        Add two :class:`CubeCalibrator` objects to :attr:`cal_view`.

        Acts as Slot for :attr:`CalibrationWindow.cube_m` action for setting stereo cubes.
        Populates :attr:`cubes` with initialized :class:`CubeCalibrator` object and emits empty :attr:`added_sig`.
        """
        self.cubes = CubeCalibrator(2, self.cal_view)
        self.sides = 2
        self.added_sig.emit()
        self.check_valid()

    def add_mono_cube(self) -> None:
        """
        Add one :class:`CubeCalibrator` object to :attr:`cal_view`.

        Acts as slot for :attr:`CalibrationWindow.cube_m` action for setting mono cube.
        Populates :attr:`cubes` with initialized :class:`CubeCalibrator` object and emits empty :attr:`added_sig`.
        :return:
        """
        self.cubes = CubeCalibrator(1, self.cal_view)
        self.sides = 1
        self.added_sig.emit()
        self.check_valid()

    def remove_cubes(self) -> None:
        """
        Remove any added :class:`CubeCalibrator` objects currently present.

        Sets :attr:`cubes` to `None`, deletes any present calibration widgets and emits empty :attr:`invalid_sig` and
        :attr:`removed_sig` signals.
        """
        self.cubes = None
        self.sides = 0
        # Go backwards through items, because calling .removeItem() drops the item from the list and re-initializes
        # the list as one element shorter ==> Normal iteration would skip the "backfilling" item
        for item in self.cal_view.getView().addedItems[::-1]:
            if type(item) == CalibrationEdges:
                self.cal_view.getView().removeItem(item)
        self.invalid_sig.emit()
        self.removed_sig.emit()

    def set_volume(self) -> None:
        """
        Calculate the cube volume based on the currently entered cube edge lengths.

        If any of the edge lengths are not set, volume will be set to invalid ("-").
        Acts as Slot for `textEdited` signal of :attr:`b_edits` so any time user changes an edge length the volume is
        updated.
        """
        vol = 1
        for edit in self.b_edits:
            val = edit.text()
            if val == '':
                self.v_edit.setText("-")
                return
            else:
                vol *= float(val)
        self.v_edit.setText(str(round(vol, 2)))

    def set_lengths(self) -> None:
        r"""
        Calculate the edge lengths based on the currently entered cube volume.

        Will set values for each edge length to :math:`V_{cube}^{\frac{1}{3}}` under assumption of quadratic cube. Will
        invalidate edge lengths as 0.0 if no volume is currently entered.
        Acts as a Slot for `textEdited` signal of :attr:`v_edit` so any time user changes the cube volume, all edge
        lengths are updated.
        """
        val = self.v_edit.text()
        if val == '':
            len = 0.0
        else:
            len = float(val)**(1/3)
        for edit in self.b_edits:
            edit.setText(str(round(len, 2)))

    def check_physicals(self) -> bool:
        """
        Validate that all entered values for cube edge lengths are acceptable values.

        Criterion for acceptable value is value range of [0.00 to 999.99].

        :return: True if all cube edge lengths entered in :attr:`b_edits` have valid values. False otherwise.
        """
        if all(edit.hasAcceptableInput() for edit in self.b_edits):
            return True
        else:
            return False

    def check_valid(self) -> None:
        """
        Validate that entered physical edge lengths are acceptable and that :attr:`cubes` is populated so that vectors
        can be grabbed. 
        """
        if self.check_physicals() and self.cubes is not None:
            self.valid_sig.emit()
        else:
            self.invalid_sig.emit()


class CubeCalibrator(pg.GraphicsObject):
    """
    Intermediary class for managing variable number of calibration objects.

    **Bases**: :class:`pyqtgraph.GraphicsObject`

    .. note::

        Initial intention was flexible implementation for general use of various calibration methods. Currently, only
        the calibration for stereoscopic suture video footage is implemented, which uses a stereo vision of a
        calibration cube.

    """
    def __init__(self, num_views, image_view):
        """
        Initialize a CubeCalibrator object that can be used to set the calibration for a cube object.

        The use of this CubeCalibrator class is useful to add multiple actual calibration objects to an image scene.
        It will instantiated one or multiple :class:`CalibrationEdges <CalibrationEdges>` objects that can be
        interactively set to match the edges of a calibration cube by the user.

        :param num_views: The number of views shown in the image data. Should be 1 for a monoscopic view and 2 for a
            stereoscopic view.
        :type num_views: int
        :param image_view: The ImageView used to display the image data.
        :type image_view: pyqtgraph.ImageView
        """
        super().__init__()

        img_shape = image_view.getImageItem().image.shape
        img_center = (int(img_shape[0]/2), int(img_shape[1]/2))

        v_offset = 100 if num_views == 2 else 0
        v_x = img_center[0] - v_offset
        v_y = img_center[1]
        center_v = (v_x, v_y)
        # Ordering for cube edges is: v_1, v_2, v_3
        endings_v = [(v_x + 50, v_y), (v_x, v_y + 50), (v_x - 25, v_y - 25)]
        #: :class:`CalibrationEdges` object for calibrating left view or for monoscopic cube view
        self.cube_v = CalibrationEdges(center=center_v, positions=endings_v)
        image_view.getView().addItem(self.cube_v)

        if num_views == 2:
            vp_x = img_center[0] + 100
            vp_y = img_center[1]
            center_vp = (vp_x, vp_y)
            endings_vp = [(vp_x + 50, vp_y), (vp_x, vp_y + 50), (vp_x + 25, vp_y - 25)]
            #: :class:`CalibrationEdges` object for calibrating right view. Is only instantiated in case of stereoscopy.
            self.cube_vp = CalibrationEdges(center=center_vp, positions=endings_vp)
            image_view.getView().addItem(self.cube_vp)


class CalibrationEdges(pg.PolyLineROI):
    """
    Widget consisting of 3 edges connected at a central point for aligning with calibration cube edges.

    Based on :class:`pyqtgraph.PolyLineROI`.

    **For visualization of used reference see:**

    *Döllinger, M. and Berry, D.; Computation of the three-dimensional medial surface dynamics of the vocal folds;
    Journal of Biomechanis; 39; 2006; 269-274*

    https://doi.org/10.1016/j.jbiomech.2004.11.026

    Re-modeling the PolyLineROI to have a center point with three "extremities" or handles attached.
    Used to mark the edges of the calibration cube in the stereoscopic hemilarynx data. Each handle is color coded
    according to its intended edge of the cube.

    * Magenta --> v_1
    * Lime --> v_2
    * Cyan --> v_3
    """
    def __init__(self, center, positions, **args):
        """
        Instantiate a single calibration object consisting of 3 edges connected at a center point.

        :param tuple center: Initial center position as coordinates tuple (x, y).
        :param List[tuple] positions: List of coordinates at which to place the edge ends at as (x, y). Order is 0:
            magenta/v_1, 1: lime/v_2, 2: cyan/v_3.
        :param args: Optional additional positional arguments are passed on to call to :class:`pyqtgraph.PolyLineROI`
        """
        self.center = center  #: Center coordinate of widget as (x, y) tuple
        self.edges = {}
        pg.PolyLineROI.__init__(self, positions, **args)

    def setPoints(self, positions, closed=False) -> None:
        """
        Overwrites `setPoints` of :class:`pyqtgraph.PolyLineROI` in order to build calibration tool with three 3 edges
        connected at one common central point.

        Is called by parent class upon initialization. Overwrites behavior to create poly-line of wanted structure.

        :param List[tuple] positions: List of (x, y) coordinate tuples at which to place edge ends. Order is 0:
            magenta/v_1, 1: lime/v_2, 2: cyan/v_3.
        :param bool closed: Boolean setting to make poly-line connect at start of first and end of last segment.
            Defaults here to False, as we do not want our edges to be connected except for at the center point.
        """
        self.clearPoints()

        self.addFreeHandle(self.center)
        for idx, p in enumerate(positions):
            self.addFreeHandle(p)
            self.addSegment(self.handles[0]['item'], self.handles[-1]['item'])
            if idx == 0:
                # Edge v_1 (magenta)
                self.segments[-1].setPen(pg.mkPen(color=(255, 0, 255), width=2))
            elif idx == 1:
                # Edge v_2 (green)
                self.segments[-1].setPen(pg.mkPen(color=(0, 255, 0), width=2))
            elif idx == 2:
                # Edge v_3 (blue)
                self.segments[-1].setPen(pg.mkPen(color=(0, 255, 255), width=2))

    def segmentClicked(self, segment, ev=None, pos=None) -> None:
        """
        Overriding method to prevent clicking from creating a new handle on ROI object.

        **Critical to ensure correct behavior for calibration!**
        """
        return  # DO NOT DELETE



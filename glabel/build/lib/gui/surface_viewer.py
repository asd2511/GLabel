import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QWidgetAction, QComboBox, QLineEdit, QToolBar, QWidget, QGridLayout, \
    QHBoxLayout, QVBoxLayout, QLabel, QSpinBox, QSplitter, QGraphicsEllipseItem
from PyQt5.QtCore import Qt, pyqtSignal

from glabel.analysis import utils


class SurfaceViewer(QMainWindow):
    """Window for displaying surface data in 3D"""

    data_changed = pyqtSignal(np.ndarray)
    frame_changed = pyqtSignal(int)
    close_signal = pyqtSignal()

    def __init__(self, rows, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Surface Viewer")
        # self.setWindowFlag(Qt.WindowStaysOnTopHint, on=True)
        self.setWindowFlags(self.windowFlags() | Qt.Dialog)
        self.setGeometry(100, 100, 768, 768)

        self.widget = QWidget(self)
        self.layout = QHBoxLayout(self)

        self.view = gl.GLViewWidget()
        self.view.setMinimumSize(768, 768)
        self.disp_plot = None  # type: DisplacementPlot

        # Add a grid to the 3D view for orientation and scale
        self.grid = None
        self.toggle_grid()

        self.rows = rows
        self.cols = cols
        self.frame_id = None  # Currently shown frame
        self.raw_points = None  # Raw, unprocessed points used for resetting current data
        self.surf_verts = None  # Surface points used as mesh vertices
        self.surf_faces = None  # Faces based on the vertices making up the mesh surface
        self.mesh = None  # Mesh object added to the view
        self.column_line = None  # LinePlot highlighting a single column for which the displacement is shown
        self.roi_scatter = None  # 3D scatter points to visualize ROI placements
        self.roi_colors = None  # List of (N, 4) floats for coloring the 3D scatterpoints indicating ROIs

        #: Boolean setting if missing ROIs should be interpolated. If set to False, missing grid positions will be
        #: filled with their nearest neighbor coordinate still allow building the surface mesh.
        self.interpolation = False
        #: Interpolation function/method. All except 'bispline' used by scipy.interpolate.Rbf to interpolate missing
        #: ROIs
        self.interp_function = 'cubic'

        self.toolbar = self.add_toolbar_actions()

        self.layout.addWidget(self.view)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def process_data(self, surf_points):
        points = utils.interpolate_missing_rois(surf_points, self.rows, self.cols, use_nearest=not self.interpolation,
                                                interp_function=self.interp_function)
        points = points - np.mean(points, axis=1)[:, None, :]  # Subtract mean from each x, y, z point
        points = points[:, :, :3]
        points = ((points / points.max()) * 10)
        # Switch columns to have mesh-data be shown in more intuitive direction (making y vertical, x horizontal and z
        # the other horizontal)
        points[:, :, 1] *= -1
        points[:, :, [2, 1]] = points[:, :, [1, 2]]

        return points

    def set_data(self, surf_points):
        self.raw_points = surf_points.copy()
        surf_points = self.process_data(surf_points)
        self.surf_verts = surf_points
        self.surf_faces = self.calculate_face_idcs()
        self.data_changed.emit(self.surf_verts)

    def show_frame(self, frame_id):
        self.frame_id = frame_id
        self.frame_changed.emit(self.frame_id)
        self.make_mesh()
        self.update_scatter()

    def make_mesh(self):
        # Instantiate or update the surface mesh
        if self.mesh is None:
            self.mesh = gl.GLMeshItem(vertexes=self.surf_verts[self.frame_id], faces=self.surf_faces, drawEdges=True,
                                      shader='shaded')
            self.view.addItem(self.mesh)
        else:
            self.mesh.setMeshData(vertexes=self.surf_verts[self.frame_id], faces=self.surf_faces)
            self.mesh.meshDataChanged()

    def update_scatter(self):
        # Update the ROI scatter on frame change, but only if it is currently shown
        if self.roi_scatter:
            self.roi_scatter.setData(pos=self.surf_verts[self.frame_id])

    def calculate_face_idcs(self) -> np.ndarray:
        """
        Calculate the indices into the array of vertices that form faces of the 3D surface

        The indices making up the faces of the mesh are built by relying on the spatial ordering within the passed
        surface points array. A short explanation of how the faces are constructed is given using the values rows = 7,
        cols = 5.

        :attr:`surf_verts` is an array with shape [#frames, #rows*#cols, 2]. Within the list of points of a single
        frame, the ordering of points follows the 1D ordering used in the rest of the GUI (linear ordering row-wise
        without snaking).
        The indices into grid positions look like this:

        .. math::

            \begin{bmatrix}
                30 & 31 & 32 & 33 & 34 \\
                25 & 26 & 27 & 28 & 29 \\
                20 & 21 & 22 & 23 & 24 \\
                15 & 16 & 17 & 18 & 19 \\
                10 & 11 &132 & 13 & 14 \\
                 5 &  6 &  7 &  8 &  9 \\
                 0 &  1 &  2 &  3 &  4 \\
            \end{bmatrix}

        The faces are constructed row-wise by connecting two rows with alternating oriented triangles. E.g.: The first
        two rows of the above grid are filled with faces using the indices:

        .. math::

            \begin{bmatrix}
                (5, 6, 1) & (6, 7, 2) & (7, 8, 3) & (8, 9, 4) \\
                (5, 0, 1) & (6, 1, 2) & (7, 2, 3) & (8, 3, 4)
            \end{bmatrix}

        The calculation of these indices happens by each of the three indices for each face as a series, which are
        constructed separately and concatenated into the correct format afterwards. The resulting series constructed for
        the above example are:
        >>> a = [0, 1, 2, 3, 5, 6, 7, 8]
        >>> b = [1, 2, 3, 4, 6, 7, 8, 9]
        >>> c = [5, 6, 7, 8, 1, 2, 3, 4]
        >>> face_idcs = np.vstack([a, b, c]).T
        """
        ids = np.arange(self.rows * self.cols).reshape(self.rows, self.cols)

        a = np.array([ids[i:i + 2, :-1].flatten() for i in range(ids.shape[0] - 1)]).flatten()
        b = np.array([ids[i:i + 2, 1:].flatten() for i in range(ids.shape[0] - 1)]).flatten()
        c_idcs = np.array([[i + 4, i] for i in range(1, self.rows * (self.cols - 1), self.cols)]).flatten()
        c = np.array([np.arange(i, i + self.cols - 1) for i in c_idcs]).flatten()
        face_idcs = np.vstack([a, b, c]).T

        return face_idcs

    def add_toolbar_actions(self):
        interp_box = QComboBox(self)
        interp_box.setObjectName('interp_box')
        interp_box.addItems(['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'])
        interp_box.setCurrentText(self.interp_function)
        interp_box.currentTextChanged.connect(lambda t: self.change_interpolation(t))
        interp_box.setEnabled(self.interpolation)

        toolbar = QToolBar(self)
        toolbar.addAction('Show ROIs', self.show_rois)
        toolbar.addAction('Enable interpolation', self.toggle_interpolation)
        toolbar.addWidget(interp_box)
        toolbar.addAction('Toogle grid', self.toggle_grid)
        toolbar.addAction('Change background', self.toggle_background)
        toolbar.addAction('Show displacement', self.show_displacement)

        self.addToolBar(toolbar)

        return toolbar

    def show_rois(self):
        if self.roi_scatter is None:
            self.roi_colors = np.ones((self.surf_verts.shape[0], 4))
            self.roi_colors[:, 3] = 0.5
            self.roi_scatter = gl.GLScatterPlotItem(pos=self.surf_verts[self.frame_id], color=self.roi_colors)
            self.roi_scatter.setObjectName("roi scatter")
        else:
            self.roi_scatter.setData(pos=self.surf_verts[self.frame_id])
        self.view.addItem(self.roi_scatter)

        # Change the button to hide the rois on next click
        action = self.sender()
        action.disconnect()
        action.setText("Hide ROIs")
        action.triggered.connect(self.hide_rois)

    def hide_rois(self):
        self.view.removeItem(self.roi_scatter)

        # Change the button to show the rois on next click
        action = self.sender()
        action.disconnect()
        action.setText("Show ROIs")
        action.triggered.connect(self.show_rois)

    def highlight_column(self, col):
        column_points = self.surf_verts[self.frame_id, col::self.cols]
        if self.column_line is None:
            self.column_line = gl.GLLinePlotItem(pos=column_points, color=(1.0, 0.0, 1.0, 0.75), width=3)
            self.view.addItem(self.column_line)
        else:
            self.column_line.setData(pos=column_points, color=(1.0, 0.0, 1.0, 0.75), width=3)

    def toggle_interpolation(self):
        self.interpolation = not self.interpolation
        self.findChild(QComboBox, 'interp_box').setEnabled(self.interpolation)

        # Update the shown data with the new interpolation method
        self.reset_data()

        # Update the button text to give feedback to user
        action = self.sender()
        text = 'Disable interpolation' if self.interpolation else 'Enable interpolation'
        action.setText(text)

    def toggle_grid(self):
        if self.grid is not None:
            self.view.removeItem(self.grid)
            self.grid = None
        else:
            self.grid = gl.GLGridItem()
            self.grid.scale(2, 2, 1)
            self.grid.setDepthValue(10)
            self.view.addItem(self.grid)

    def toggle_background(self):
        if self.view.opts['bgcolor'] == (0.0, 0.0, 0.0, 1.0):
            self.view.opts['bgcolor'] = (1.0, 1.0, 1.0, 1.0)
        else:
            self.view.opts['bgcolor'] = (0.0, 0.0, 0.0, 1.0)
        self.view.update()

    def change_interpolation(self, function):
        self.interp_function = function
        self.reset_data()  # Update the data to use the newly set function

    def show_displacement(self):
        splitter = QSplitter(Qt.Horizontal)

        self.disp_plot = DisplacementPlot(parent=self, data=self.surf_verts, rows=self.rows, cols=self.cols)
        self.disp_plot.change_frame(self.frame_id)
        self.disp_plot.column_shown.connect(lambda c: self.highlight_column(c))
        self.disp_plot.plot_cur_col()

        splitter.addWidget(self.view)
        splitter.addWidget(self.disp_plot)
        self.layout.addWidget(splitter)

    def reset_data(self):
        # Reset the interpolated data by re-setting the data
        self.set_data(self.raw_points)
        self.make_mesh()
        self.update_scatter()

    def color_roi(self, roi_id, color):
        if self.roi_scatter:
            self.roi_colors[roi_id] = color
            self.roi_scatter.setData(color=self.roi_colors)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.close_signal.emit()
        super().closeEvent(a0)


class SpaghettiViewer(SurfaceViewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.line_data = None

    def set_data(self, surf_points):
        surf_points = self.process_data(surf_points)
        self.line_data = surf_points

    def show_frame(self, frame_id):
        self.frame_id = frame_id
        for roi_id in range(self.line_data.shape[1]):
            points = self.line_data[:, roi_id]
            for frame_number, frame_points in enumerate(points):
                frame_points[0] = frame_points[0] - frame_number
            plt = gl.GLLinePlotItem(pos=points)
            self.view.addItem(plt)


class DisplacementPlot(QWidget):

    column_shown = pyqtSignal(int)

    def __init__(self, data, rows, cols, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data
        self.rows = rows
        self.cols = cols
        self.active_col = 0
        self.frame = None
        self.col_plot = None

        self.parent().data_changed.connect(lambda d: self.set_data(d))
        self.parent().frame_changed.connect(lambda f: self.change_frame(f))

        self.pw = self._init_plot()
        self.layout = self._init_layout()
        self.setLayout(self.layout)

    def _init_layout(self) -> QVBoxLayout:
        """
        Initialize the vertical PyQt layout for holding the plot widget.

        :return: :pyqt:`QVBoxLayout <qvboxlayout>` with a toolbar created by :func:`_init_toolbar` and plot widget added
            to it.
        """
        layout = QVBoxLayout(self)
        toolbar = self._init_toolbar()
        layout.addWidget(toolbar)
        layout.addWidget(self.pw)

        return layout

    def _init_toolbar(self) -> QToolBar:
        """
        Initialize the toolbar at the top of the widget to give control over the plots.

        The toolbar contains widgets and actions for setting which column to plot suture positions for, and what type
        of data is plotted (positions of current frame / mean and cov. across all frames).

        :return: :pyqt:`QToolBar <qtoolbar>` containing control elements already connected to the receiving methods.
        """
        toolbar = QToolBar(self)
        toolbar.addWidget(QLabel("Show column: "))

        col_box = QSpinBox(self)
        col_box.setMinimum(0)
        col_box.setMaximum(self.cols-1)
        col_box.setValue(self.active_col)
        col_box.valueChanged.connect(lambda col: self.change_column(col))
        toolbar.addWidget(col_box)

        toolbar.addAction("Toggle mean/cov", self.toggle_mean_cov)

        return toolbar

    def _init_plot(self) -> pg.PlotWidget:
        """
        Initialize the plotting widget.

        The plotting widget is a :pyqtgraph:`PlotWidget` which has its limits set to the data limits (across all frames)
        and automatic rescaling of the view enabled.

        :return: :pyqtgraph:`PlotWidget` for plotting of 2D data.
        """
        pw = pg.PlotWidget(self)
        pw.setMinimumSize(300, 300)
        pw.resize(768, 768)
        xlims, ylims = self.find_plot_lims()
        pw.setXRange(*xlims)
        pw.setYRange(*ylims)
        pw.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        return pw

    def find_plot_lims(self):
        """
        Determine the horizontal and vertical limits of the data across all frames/timesteps.

        This is used for setting the plot limits to show all of the data, no matter which frame or which column is
        plotted, giving more data overview to the user when switching between frames or columns.
        """
        min_x = self.data[:, :, 1].min()
        max_x = self.data[:, :, 1].max()
        min_y = self.data[:, :, 2].min()
        max_y = self.data[:, :, 2].max()

        return (min_x, max_x), (min_y, max_y)

    def set_data(self, datapoints) -> None:
        """
        Set the data for which the suture columns are shown.

        This sets the global numpy data array from which the data of individual columns is extracted.

        :param np.ndarray datapoints: Numpy array holding the 3D surface data. Must be of shape [#frames, #sutures, 3]
        """
        self.data = datapoints
        self.update_plot()

    def change_frame(self, frame_id) -> None:
        """
        Change the currently active frame.

        :param int frame_id: Frame number to switch to.
        """
        self.frame = frame_id
        self.update_plot()

    def change_column(self, col) -> None:
        """
        Switch the currently active and shown column.

        :param int col: New active column.
        """
        self.active_col = col
        self.update_plot()

    def update_plot(self) -> None:
        """
        Update the plot by re-calling the plotting function.
        """
        # For initial plot when nothing is initialized yet
        if not self.pw.plotItem.curves:
            self.plot_cur_col()

        if self.pw.plotItem.curves[0].name() == 'column':
            self.plot_cur_col()
        else:
            self.plot_mean_cov()

    def plot_cur_col(self) -> None:
        """
        Plot the currently selected column positions in the coronal plane.

        The suture positions for the current :attr:`active_col` are plotted as vertices of a line. For the plot, only
        the lateral (y) and vertical (z) coordinate is plotted, creating a view of the medial vocal fold surface in
        the coronal plane.
        """
        col_data = self.get_col_data(self.active_col)
        if self.col_plot is None:
            self.pw.clear()
            self.col_plot = self.pw.plot(col_data[:, 1]*-1, col_data[:, 2], name="column")
        else:
            self.col_plot.setData(col_data[:, 1]*-1, col_data[:, 2], name="column")

        self.column_shown.emit(self.active_col)

    def get_col_data(self, col, all_frames=False) -> np.ndarray:
        """
        Extract the positional data for a single column from the data set as :attr:`data`.

        :param int col: Column number in range [0, :attr:`cols`] for which to extract data. Columns are numbered from
            left to right corresponding to image data.
        :param bool all_frames: If set to True, the data for all frames will be returned. Otherwise, the data for the
            current frame only is returned.
        :return: Numpy matrix of shape [:attr:`rows`, 3] if `all_frames` is False, [#frames, :attr:`rows`, 3] otherwise.
        """
        if not all_frames:
            return self.data[self.frame][col::self.cols]
        else:
            return self.data[:, col::self.cols, :]

    def plot_displacements(self) -> None:
        """
        Plot the time-varying displacement of each suture of the shown column.

        For each suture, a separate line is plotted indicating the suture position across all availble time steps.

        .. note:: Currently not connected to any window functionality, so display can not be triggered through GUI.
        """
        col_data = self.get_col_data(self.active_col, all_frames=True)
        for row in range(self.rows):
            self.pw.plot(col_data[:, row, 1]*-1, col_data[:, row, 2])

    def plot_mean_separation(self) -> None:
        r"""
        Show the separation of :math:`R_i` into mean :math:`\bar R_i` and time-varying component :math:`r_i(t_k)`

        The mean coordinates are connected via a line, while the time-varying component of each suture forms its own
        line showing the movement across time.

        .. note:: Currently not connected to any window functionality, so display can not be triggered through GUI.
        """
        self.pw.clear()
        col_data = self.get_col_data(self.active_col, all_frames=True)
        means = np.mean(col_data, axis=0)  # Build mean for all column sutures across time axis
        time_comps = col_data - means  # Subtract the means to get the time-varying components

        # Plot the data
        self.pw.plot(means[:, 1]*-1, means[:, 2])  # Add line through means
        # Add separate line for time-varying component of each suture
        for row in range(self.rows):
            self.pw.plot(col_data[:, row, 1]*-1, col_data[:, row, 2])

    def toggle_mean_cov(self) -> None:
        """
        Toggle between the display of the currently active column as coronal view, and displaying the mean and
        covariance of column positions across all timesteps.

        Activating the toggle calls either :func:`plot_mean_cov` or :func:`plot_cur_col` depending on what is currently
        shown.
        """
        if self.pw.plotItem.curves[0].name() == 'column':
            self.plot_mean_cov()
        else:
            self.plot_cur_col()

    def plot_mean_cov(self):
        """
        Plot the data of the currently active column by showing mean and covariance of suture positions across time.

        For each of the sutures in the current column, mean position across time is calculated. Based on the mean, the
        covariance matrix of each sutures coordinates is calculated and shown as an ellipse in the plot.

        .. todo:: This method is meant to mimick the calculation of the empirical eigenfunctions of the sutures, but I
            don't think this is the correct way... Try to implement the eigenfunctions as described in the papers by
            Berry.
        """
        # Delete currently shown data and make sure no column data is remaining
        self.pw.clear()
        self.col_plot = None

        col_data = self.get_col_data(self.active_col, all_frames=True)

        # Plot line connecting the mean position of column sutures
        means = np.mean(col_data, axis=0)
        time_comps = col_data - means
        self.pw.plot(means[:, 1]*-1, means[:, 2], name='means')

        # For each suture in the column, calculate the covariance between its coronal coordinates and plot them as an
        # ellipse
        for row in range(self.rows):
            row_cov = np.cov(time_comps[:, row, 1:], rowvar=False)
            u, s, vt = np.linalg.svd(row_cov)
            angle = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
            width, height = 2 * np.sqrt(s)
            x = (means[row, 1] + width/2) * -1
            y = means[row, 2] - height/2
            cov_roi = CovEllipse((x, y), (width, height), angle, movable=False)
            self.pw.addItem(cov_roi)

        # Emit which column is plotted for highlighting
        self.column_shown.emit(self.active_col)


class CovEllipse(pg.EllipseROI):
    """
    Helper class for plotting covariances in pyqtgraph.

    **Bases**: :pyqtgraph:`EllipseROI`

    Only the :func:`__init__` method is modified to have covariance ellipse drawn without cale and rotate handles and
    set the correct ellipse angle without moving the whole ellipse from its set position.

    :param tuple pos: (x, y) tuple for setting the position of the bottom-left bounding box corner of the ellipse.
    :param tuple size: (width, height) tuple setting the ellipse size.
    :param float angle: Angle to rotate ellipse by.
    :param **kwargs: Any additional keywords are passed on to the constructor of :pyqtgraph:`ROI`.
    """
    def __init__(self, pos, size, angle, **kwargs):
        self.path = None
        pg.ROI.__init__(self, pos, size, **kwargs)
        self.setAngle(angle, center=(.5, .5))

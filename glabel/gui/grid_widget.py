from typing import Union, List, Tuple
from abc import ABC, ABCMeta, abstractmethod

import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QPointF, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QWidget, QRadioButton, QFrame, QSizePolicy, QGridLayout, QButtonGroup, QVBoxLayout, \
    QApplication
import pyqtgraph as pg
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent


class MetaClass(ABCMeta, type(QWidget)):
    """
    Meta-class inheriting from :pyqt:`QWidget <qwidget>` for creating custom abstract class.
    """
    pass


class GridWidget(ABC, QWidget, metaclass=MetaClass):
    """
    Abstract base class for grid widgets.

    **Bases**: :pyqt:`QWidget <qwidget>` (for defining custom signals)

    This base class dictates common methods for all grid widgets.
    """

    Free = 0  #: Integer code for free movement within grid
    LeftRight = 1  #: Integer code for row-wise movement within grid
    BottomTop = 2  #: Integer code for column-wise movement within grid

    key_signal = pyqtSignal(int)  #: Signal emitting pressed keys
    activate_signal = pyqtSignal(int)  #: Signal emitting grid position ID when set to active
    deactivate_signal = pyqtSignal(int)  #: Signal emitting grid position ID when set to inactive
    request_reassign_signal = pyqtSignal(dict)  #: Signal emitting reassignment information dictionary when requested

    def __init__(self, num_rows: int, num_cols: int, stereo: bool, mode: int = 0, snaking: bool = True,
                 begin_top: bool = False):
        """
        Abstract class! Do not instantiate this as a grid widget!

        :param num_rows: Number of rows in the grid
        :param num_cols: Number of columns in the grid
        :param stereo: If True, the grid is doubled to label landmarks in stereoscopic views.
        :param mode: Traversal mode within the grid. Valid values are [0, 1, 2] for [free, row-wise, col-wise].
        :param snaking: If True, enable snaking mode for traversal through grid.
        :param begin_top: If True, begin traversal in top-left corner of grid. Otherwise begin in bottom-left.
        """
        ABC.__init__(self)
        QWidget.__init__(self)

        self.num_rows = num_rows  #: Number of rows in the grid
        self.num_cols = num_cols  #: Number of columns in the grid
        self.stereo = stereo  #: Boolean value setting if grid is stereo or not
        self.begin_top = begin_top  #: Boolean value setting if traversal beginning is in top-left or not
        self.path = None  #: Calculated grid traversal path calculated from settings

        #: Boolean array holding which positions have been placed yet
        self.placements = np.zeros((self.num_rows * self.num_cols * (2 if self.stereo else 1)), dtype=np.bool)

        self.mode = mode  #: Grid traversal mode. Can be one of [0, 1, 2]
        self.snaking = snaking  #: Boolean value setting if traversal has snaking mode enabled or not

    @abstractmethod
    def build_grid(self): ...

    @abstractmethod
    def set_active(self, pos: Union[tuple, int]): ...

    @abstractmethod
    def mark_placed(self, grid_id: int): ...

    @abstractmethod
    def mark_missing(self, grid_id: int): ...

    @abstractmethod
    def get_cur_1d(self) -> int: ...

    def get_id(self, row: int, col: int) -> int:
        """
        Get the id of the :class:`RadioButton <PyQt5.QtWidgets.QRadioButton>` from the Grid at the specified position.

        :param row: Row number of button.
        :param col: Column number of button.
        :return: Integer id to access the button.
        """
        offset = 0
        if col >= self.num_cols:
            offset = self.num_rows * self.num_cols
            col -= self.num_cols
        return row * self.num_cols + col + offset

    def get_pos_2d(self, grid_id: int) -> tuple:
        """
        Takes a grid-id value and returns the corresponding row and column indices.

        :param grid_id: The integer id of the grid position.
        :return: The row and column indices of the given position as tuple (row, col).
        """
        # The Ids of the first (lowest) row are encoded with a 0 as their row number which is missing from the integer
        offset = 0
        if grid_id >= self.num_cols * self.num_rows:
            grid_id -= self.num_cols * self.num_rows  # Removing offset from right half in stereo case
            offset = self.num_cols
        row = int(grid_id / self.num_cols)
        col = grid_id % self.num_cols + offset

        return row, col

    def get_cur_2d(self) -> tuple:
        """
        Get the currently active grid position in (x, y)-coordinates

        :return: Tuple of (x, y) coordinates.
        """
        return self.cur_active

    def build_path(self) -> List[int]:
        """
        Path through initialized grid according to the specified progression settings (mode, snaking).

        :return: The path through the grid taken by automatic progression.
        """
        path = []

        sides = 2 if self.stereo else 1
        for s in range(sides):
            direction = 1
            offset = self.num_cols * s

            if self.mode == GridWidget.LeftRight:
                row_dir = -1 if self.begin_top else 1
                for r in range(self.num_rows)[::row_dir]:
                    path.extend(tuple(zip([r] * self.num_cols, np.arange(offset, self.num_cols + offset)[::direction])))
                    if self.snaking:
                        direction *= -1

            elif self.mode == GridWidget.BottomTop:
                for c in range(self.num_cols):
                    if self.begin_top:
                        path.extend(reversed(tuple(zip(np.arange(self.num_rows)[::direction], [c + offset] * self.num_rows))))
                    else:
                        path.extend(tuple(zip(np.arange(self.num_rows)[::direction], [c + offset] * self.num_rows)))
                    if self.snaking:
                        direction *= -1
            else:
                return None

        return path

    def right(self):
        """
        Activate the right grid position.
        """
        max_right = self.num_rows * (2 if self.stereo else 1)
        max_right = max_right - 1

        if self.cur_active[1] < max_right:
            new_active = (self.cur_active[0], self.cur_active[1] + 1)
            self.set_active(new_active)

    def left(self):
        """
        Activate the left grid position.
        """
        if self.cur_active[1] > 0:
            new_active = (self.cur_active[0], self.cur_active[1] - 1)
            self.set_active(new_active)

    def up(self):
        """
        Activate the upwards grid position.
        """
        if self.cur_active[0] < (self.num_rows - 1):
            new_active = (self.cur_active[0] + 1, self.cur_active[1])
            self.set_active(new_active)

    def down(self):
        """
        Activate the downwards grid position.
        """
        if self.cur_active[0] > 0:
            new_active = (self.cur_active[0] - 1, self.cur_active[1])
            self.set_active(new_active)

    def next(self):
        """
        Activate the next grid position. Order is controlled by the `mode` and `snaking` class variables.
        """
        if self.mode == 0:
            pass

        else:
            cur_path_idx = self.path.index(self.cur_active)
            if cur_path_idx < len(self.path) - 1:
                self.set_active(self.path[cur_path_idx + 1])

    def previous(self):
        """
        Activate the previously active grid position. Order is controlled by the `mode` class variable.
        """
        if self.mode == 0:
            pass

        else:
            cur_path_idx = self.path.index(self.cur_active)
            if cur_path_idx > 0:
                self.set_active(self.path[cur_path_idx - 1])

    def keyPressEvent(self, ev: QKeyEvent):
        """
        Handle presses of keyboard keys. Will not react to key presses itself but just emit them via the
        :attr:`key_signal` signal.

        :param ev: The QKeyEvent specifying which key is pressed.
        """
        self.key_signal.emit(ev.key())

    def set_mode(self, mode: int):
        """
        Set the mode of the automatic progression through the position grid.

        :param mode: Selected mode:
            0: Free (navigate with keys)
            1: Left to right (row-wise progression)
            2: Bottom to top (column-wise progression)
        """
        assert mode == 0 or mode == 1 or mode == 2, "Invalid value is trying to be set for automation mode!"
        if self.mode != mode:
            self.mode = mode
            self.path = self.build_path()

    def set_snaking(self, snaking: bool):
        """
        Enable or disable snaking mode for grid traversal.

        :param snaking: Boolean value enabling or disabling snaking mode
        """
        if self.snaking != snaking:
            self.snaking = snaking
            self.path = self.build_path()

    def set_begin_top(self, begin_top: bool):
        """
        Enable or disable traversal beginning in top-left.

        :param begin_top: If set to True, grid traversal path will have its beginning in the top-left.
        """
        if self.begin_top != begin_top:
            self.begin_top = begin_top
            self.path = self.build_path()


class ButtonGrid(GridWidget):
    """
    Grid widget using :pyqt:`QRadioButton <qradiobutton>` elements as indicators for grid.

    **Bases**: :class:`GridWidget`
    """
    def __init__(self, num_rows: int, num_cols: int, stereo: bool, mode: int = 0, snaking: bool = True,
                 begin_top: bool = False):
        """
        :param num_rows: Number of rows in the grid
        :param num_cols: Number of columns in the grid
        :param stereo: If True, the grid is doubled to label landmarks in stereoscopic views.
        :param mode: Traversal mode within the grid. Valid values are [0, 1, 2] for [free, row-wise, col-wise].
        :param snaking: If True, enable snaking mode for traversal through grid.
        :param begin_top: If True, begin traversal in top-left corner of grid. Otherwise begin in bottom-left.
        """
        GridWidget.__init__(self, num_rows, num_cols, stereo, mode, snaking, begin_top)

        self.grid_layout = QGridLayout()  #: :class:`QGridLayout <qgridlayout>` holding all radio buttons
        self.grid_buttons = QButtonGroup()  #: :class:`QButtonGroup <qbuttongroup>` managing the radio buttons

        self.cur_active = (num_rows - 1, 0) if begin_top else (0, 0)  #: Currently active grid position

        self.build_grid()
        self.path = self.build_path()  #: Grid traversal path
        self.set_active(self.cur_active)
        self.setLayout(self.grid_layout)

    def build_grid(self):
        """
        Fill the :class:`GridLayout <PyQt5.QtWidgets.QGridLayout>` with
        :class:`RadioButtons <PyQt5.QtWidgets.RadioButton>` to indicate and allow selection of the currently active
        click-position.
        The buttons are grouped into a :class:`ButtonGroup <PyQt5.QtWidgets.QButtonGroup>`.
        """
        self.build_stereo_radio() if self.stereo else self.build_mono_radio()

    def build_mono_radio(self, second_side: bool = False):
        """
        Build a single grid made of :pyqt:`QRadioButtons <qradiobutton>`.

        For a monoscopic grid, this method is called once building the total grid. For a stereoscopic grid this method
        is called twice (second time with `second_side=True`, building the left and right view separately.

        The built grid is automatically filled into the :attr:´grid_buttons` and :attr:`grid_layout` attributes.

        :param second_side: Boolean setting telling the method if the built grid is for the second stereoscopic view or
            not. Set this to False for building a monoscopic grid or for the first half of the stereoscopic grid.
        """
        col_shift = self.num_cols if second_side else 0
        col_bump = 1 if second_side else 0

        cur_id = 0 if not second_side else self.num_rows * self.num_cols
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                button = QRadioButton()
                button.toggled.connect(self.on_toggle)
                button.pressed.connect(self.on_press)
                self.grid_buttons.addButton(button, cur_id)
                # The buttons are added from row 1 (not 0) to allow for nicer formatting of grid
                # (Adding the separator line which is two rows longer than buttons looks nicer)
                self.grid_layout.addWidget(button, self.num_rows - row, col + col_shift + col_bump)
                self.mark_missing(cur_id)
                cur_id += 1

    def build_stereo_radio(self):
        """
        Build a stereo grid for data with stereoscopic view of landmark grid.

        The grid itself is built by calling :func:`build_mono_radio` twice. The method adds a separation indicator
        between the halves and manages some layout options for nice formatting.
        """
        # Adding first half of the stereo grid to the layout
        self.build_mono_radio(second_side=False)
        # Placing vertical separator line between stereo grid halves
        separator = QFrame()
        separator.setFrameStyle(QFrame.VLine)
        separator.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        separator.setLineWidth(0)
        self.grid_layout.addWidget(separator, 0, self.num_cols, self.num_rows + 2, 1)
        # Adding second half of the stereo grid to the layout
        self.build_mono_radio(second_side=True)

    def set_active(self, pos):
        """
        Set the grid at the specified position to active.

        :param pos: Grid-position of the button set to active. If of type `tuple`, 2 entries are expected, with the
            first being the row-coordinate and the second being the column coordinate in the grid. If of type `int`,
        """
        if type(pos) in [int, np.int32, np.int64]:
            cur_id = int(pos)
            self.cur_active = self.get_pos_2d(pos)
        elif len(pos) == 2:
            cur_id = self.get_id(pos[0], pos[1])
            self.cur_active = pos
        else:
            raise TypeError("The passed position should be either a tuple (row: int, col: int) or two separate int"
                            "values specifying the position!")

        button = self.grid_buttons.button(cur_id)
        button.setChecked(True)

        self.activate_signal.emit(cur_id)

    def on_toggle(self):
        """
        Slot for processing and emitting newly activated and deactivated grid position when one of the button states was
        changed.
        """
        clicked_id = self.grid_buttons.checkedId()
        self.cur_active = self.get_pos_2d(clicked_id)

        # Handling actions for newly activated button
        if self.sender() == self.grid_buttons.checkedButton():
            self.activate_signal.emit(clicked_id)

        # Handling actions for newly deactivated button
        else:
            deactivated_id = self.grid_buttons.id(self.sender())
            self.deactivate_signal.emit(deactivated_id)

    def on_press(self):
        """
        Slot for handling reassignment requests when the grid was clicked with the reassignment command.
        """
        # Handling reassignment command
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            from_id = self.grid_buttons.id(self.grid_buttons.checkedButton())
            to_id = self.grid_buttons.id(self.sender())
            to_row, to_col = self.get_pos_2d(to_id)
            reassign_info = {'roi_id': from_id,
                             'new_row': to_row,
                             'new_col': to_col}
            self.request_reassign_signal.emit(reassign_info)

    def mark_placed(self, grid_id):
        """
        Set styling of :class:`QRadioButton <PyQt5.QWidgets.QRadioButton>` at position `grid_id` back to its default
        settings to indicate the position as placed.

        :param grid_id: The id into the QButtonGroup element. E.g `grid_id=24` indicates the position in the 3rd row
            (from bottom) and 5th column (from left).
        :type grid_id: int
        """
        self.grid_buttons.button(grid_id).setStyleSheet('QRadioButton::indicator::unchecked { ;};')
        self.placements[grid_id] = True

    def mark_missing(self, grid_id):
        """
        Set styling of :class:`QRadioButton <PyQt5.QWidgets.QRadioButton>` at position `grid_id` to a custom image to
        indicate the position as not placed.

        :param grid_id: The id into the QButtonGroup element. E.g `grid_id=24` indicates the position in the 3rd row
            (from bottom) and 5th column (from left).
        :type grid_id: int
        """
        # Style button with with custom radio button image to indicate position as not set
        self.grid_buttons.button(grid_id).setStyleSheet('QRadioButton::indicator::unchecked { image: url('
                                                        'glabel/gui/radio_unchecked.png); };')
        self.placements[grid_id] = False

    def get_cur_1d(self) -> int:
        """
        Get the currently active grid position as an index for accessing a 1D-array representation of available
        positions.

        :return: Integer index for accessing position in 1D-array.
        """
        return self.grid_buttons.checkedId()


class PixelGrid(GridWidget):
    """
    Grid widget using interactive image displayed in a :class:`pyqtgraph.GraphicsWindow` for grid.

    **Bases**: :class:`GridWidget`
    """
    #: Color values for indicating different states of grid positions.
    PixelColors = {'active': (0, 255, 0),
                   'hover': (255, 0, 255),
                   'placed': (255, 255, 255),
                   'missing': (100, 100, 100)}

    def __init__(self, num_rows: int, num_cols: int, stereo: bool, mode: int = 0, snaking: bool = True,
                 begin_top: bool = False):
        """
        :param num_rows: Number of rows in the grid
        :param num_cols: Number of columns in the grid
        :param stereo: If True, the grid is doubled to label landmarks in stereoscopic views.
        :param mode: Traversal mode within the grid. Valid values are [0, 1, 2] for [free, row-wise, col-wise].
        :param snaking: If True, enable snaking mode for traversal through grid.
        :param begin_top: If True, begin traversal in top-left corner of grid. Otherwise begin in bottom-left.
        """
        GridWidget.__init__(self, num_rows, num_cols, stereo, mode, snaking, begin_top)

        self.layout = QVBoxLayout()  #: Vertical layout holding widget for displaying grid image
        self.grid_window = pg.GraphicsWindow()  #: :class:`pyqtgraph.GraphicsWindow` displaying the actual grid image
        self.grid_view = self.grid_window.addViewBox(lockAspect=True)  # type: pg.ViewBox
        self.grid_img = None  #: Image being interactively used as a indicator of the landmark grid

        self.cur_active = (num_rows - 1, 0) if begin_top else (0, 0)  #: Currently active grid position
        #: Value storing which grid position is currently marked as active and which is hovered along with their color
        #: values.
        self.map_markings = {'active': (None, None),
                             'hover': (None, None)}

        self.build_grid()
        self.path = self.build_path()  #: Traversal path through grid
        self.set_active(self.cur_active)

        self.setLayout(self.layout)

    def build_grid(self):
        """
        Create the actual grid representation by creating a image and adding it to the :attr:`GridWidget.grid_view` and
        :attr:`GridWidget.layout` attributes.
        """
        self.grid_view.enableAutoRange()
        self.grid_view.setMenuEnabled(True)  # Keeping menu allows user to reset view --> Find grid again if moved off-window
        self.grid_img = pg.ImageItem(self.build_map_img(), autoLevels=False)
        self.grid_view.addItem(self.grid_img)
        self.layout.addWidget(self.grid_window)

        # sigMouseMoved gives the actual position unlike sigMouseHover
        self.grid_view.scene().sigMouseMoved.connect(self.map_hover_event)
        self.grid_view.scene().sigMouseClicked.connect(self.map_click_event)

    def build_map_img(self):
        r"""
        Create the image used as the interactive grid representation.

        The grid image is created as an image of shape [:attr:`num_rows` \*2+1, :attr:`num_cols` \*4+3 [#]_, 3]. This
        creates image large enough the separate each grid position (which are single pixels) by one pixel with an
        additional boundary of a single pixel surrounding the grid pixels.

        A grid image for a stereoscopic grid with 3 rows and 3 columns per side can be illustrated as this:

        .. math::

            \begin{bmatrix}
                . & . & . & . & . & . & . & . & . & . & . & . & . & . & . \\
                . & o & . & o & . & o & . & . & . & o & . & o & . & o & . \\
                . & . & . & . & . & . & . & . & . & . & . & . & . & . & . \\
                . & o & . & o & . & o & . & . & . & o & . & o & . & o & . \\
                . & . & . & . & . & . & . & . & . & . & . & . & . & . & . \\
                . & o & . & o & . & o & . & . & . & o & . & o & . & o & . \\
                . & . & . & . & . & . & . & . & . & . & . & . & . & . & .
            \end{bmatrix}

        , with "." markings indicating empty, unmarked pixels and "o" marking pixels indicating available grid
        positions.

        .. [#] Actual number of columns depends on the setting of :attr:`stereo`. If grid is in mono, the number of
            columns is :attr:`num_cols` \*2+1 instead.
        """
        map_cols = self.num_cols * 2 * (2 if self.stereo else 1) + (3 if self.stereo else 1)
        mid_col = map_cols // 2
        map_shape = (self.num_rows * 2 + 1, map_cols, 3)
        map_img = np.zeros(map_shape, dtype=np.uint8)

        for col in range(1, map_cols, 2):
            if self.stereo and col == mid_col:
                continue
            map_img[1::2, col, :] = self.PixelColors['missing']  # Set valid pixel positions to indicate missing

        return map_img

    def map_to_pos(self, y: Union[float, int], x: Union[float, int]) -> Tuple[int, int]:
        """
        Convert x- and y-coordinates of a pixel position in the grid image to row and column identifiers of a grid
            position.

        :param y: The row/vertical index of the pixel.
        :param x: The column/horizontal index of the pixel.
        :return: Tuple (row, col) indicating the row and column identifiers necessary for accessing the correct grid
            position. Will return (-1, -1) if the pixel position is outside of a valid grid position.
        """
        row = y
        col = x
        if type(row) == float:
            row = int(y)
        if type(col) == float:
            col = int(x)
        img_shape = self.grid_img.image.shape
        # Check if position is outside of boundaries and if position is in odd row and column (odd because of 1px
        # boundary around used image.
        if (-1 < row < img_shape[0]) and row % 2 == 1 and \
                (-1 < col < img_shape[1]) and col % 2 == 1:
            # The pixels in the 'middle split line' in stereo mode are invalid
            if self.stereo and col == img_shape[1] // 2:
                pass
            else:
                row -= row // 2 + 1
                col -= col // 2 + 1
                if self.stereo and col >= self.num_cols:
                    col -= 1
                return row, col

        return -1, -1

    def pos_to_map(self, row: int, col: int) -> Tuple[int, int]:
        """
        Convert row and column identifier of a grid position to the indices necessary for accessing the pixel in the
            grid image.

        :param row: The row index of the position.
        :param col: The column index of the position.
        :return: Tuple (y, x) indicating the indices necessary for accessing the correct grid image pixel.
        """
        y = row * 2 + 1
        if self.stereo and col >= self.num_cols:
            x = col * 2 + 3
        else:
            x = col * 2 + 1

        return y, x

    def valid_map_pos(self, y: int, x: int) -> bool:
        """
        Returns true if pixel position specified by (y, x) is within pixel corresponding to a grid position.

        :param y: Vertical pixel coordinate.
        :param x: Horizontal pixel coordinate
        :return: True if (y, x) is within pixel corresponding to grid position. False if (y, x) is between grid
            positions or outside of grid.
        """
        row, col = self.map_to_pos(y, x)  # Grid positions will be -1 outside of valid positions
        if row != -1 and col != -1:
            return True
        else:
            return False

    @pyqtSlot(object)
    def map_hover_event(self, event: QPointF):
        """
        Slot for coloring a grid position in the hover color when the mouse is hovered over it.

        :param event: Position of the mouse hover event.
        """
        xy = self.grid_img.mapFromScene(event)  # type: QPointF
        # Pixel positions in image
        y = int(xy.y())
        x = int(xy.x())

        # Only if valid position is hovered
        if self.valid_map_pos(y, x):
            self.map_mark_pos(y, x, 'hover')
        else:
            # When moving off of a valid position
            self.map_mark_pos(y, x, 'reset_hover')

    @pyqtSlot(object)
    def map_click_event(self, event: MouseClickEvent):
        """
        Handle clicking of the grid image.

        If a valid grid position is clicked set it to active. Additionally, if the keyboard modifier for reassigning
        the currently active ROI was held down during the click send out the request for reassignment.

        :param event: Event for mouse click.
        """
        # MouseClickEvent.scenePos() gives the correct pixel coordinates in the map image. Calling .pos() would
        # introduce some offset to the coordinates.
        coords = self.grid_img.mapFromScene(event.scenePos())  # type: QPointF
        y = int(coords.y())
        x = int(coords.x())

        modifiers = QApplication.keyboardModifiers()  # Access any modifiers currently activated by keyboard

        if self.valid_map_pos(y, x):
            row, col = self.map_to_pos(y, x)
        else:
            return

        # If Shift was held during click -> User wants to reassign the active grid position to the clicked position
        if modifiers == Qt.ShiftModifier:
            self.request_reassignment(row, col, y, x)

        self.map_mark_pos(y, x, 'reset_hover')  # Remove the hover effect of the clicked pixel
        self.set_active((row, col))

    def request_reassignment(self, new_row, new_col):
        """
        Emit the information about requested reassignment of the active ROI via the :attr:`request_reassign_signal`.

        :param int new_row: The new row to assign the ROI
        :param int new_col: The new column to assign the ROI
        """
        reassign_info = {'roi_id': self.get_cur_1d(),
                         'new_row': new_row,
                         'new_col': new_col}
        self.request_reassign_signal.emit(reassign_info)

    def map_mark_pos(self, y: int, x: int, action: str) -> None:
        """
        Mark the grid-image at the specified row and column location according to the specified action.

        Different actions will cause different marking behavior:

        *   *'active'*: Mark with the *active* color and unmark any other active position.
        *   *'hover'*: Mark with the *hover* color and unmark any other hovered position.
        *   *'placed'*: Mark with the *placed* color and keep other marked positions.
        *   *'missing'*: Mark with the *missing* color and keep other marked positions.
        *   *'reset_hover'*: Resets any marking caused by *hover* and returns position to its pre-hover status.
        *   *'reset_active'*: Resets any marking caused by *active* and returns position to its pre-active status.

        :param y: Y/Vertical coordinate of position. (0 at left edge)
        :param x: X/Horizontal coordinate of position. (0 at bottom edge)
        :param action: The type of marking that is used. Can be one of ['active', 'hover', 'placed', 'missing',
            'reset_hover', 'reset_active']
        """
        assert action in ['active', 'hover', 'placed', 'missing', 'reset_hover', 'reset_active'], \
            f"Invalid marking action {action} encountered!"

        if action in ['active', 'hover']:
            # Reset currently active position back to its state before becoming active
            if all(self.map_markings[action]):
                (y_prev, x_prev), clr = self.map_markings[action]
                self.grid_img.image[y_prev, x_prev, :] = clr
                # Send signal that position was deactivated
                row, col = self.map_to_pos(y_prev, x_prev)
                if action == 'active':
                    deactivated_id = self.get_id(row, col)
                    self.deactivate_signal.emit(deactivated_id)

            # Saving properties
            clr = self.grid_img.image[y, x, :].copy().tolist()
            self.map_markings[action] = ((y, x), clr)

            # Mark the position according to the action
            self.grid_img.image[y, x, :] = self.PixelColors[action]
            # Send signal that position was activated
            row, col = self.map_to_pos(y, x)
            if action == 'active':
                activated_id = self.get_id(row, col)
                self.activate_signal.emit(activated_id)

            # Automatically move current view to include active pixel position
            self.refit_view(y, x)

        elif action in ['placed', 'missing']:
            # Overwrite the saved base-color of the currently active position
            if self.map_markings['active'][0] == (y, x):
                self.map_markings['active'] = ((y, x), self.PixelColors[action])
            self.grid_img.image[y, x, :] = self.PixelColors[action]

        elif 'reset_' in action:
            action = action.split('_')[1]
            # If both position and previous color are saved -> Indicates that something is actually active to be reset
            if all(self.map_markings[action]):
                (y, x), clr = self.map_markings[action]  # Get saved attributes
                self.map_markings[action] = (None, None)  # Delete saved attributes
                self.grid_img.image[y, x, :] = clr  # Set position to previous color
                # Send deactivation signal if necessary
                if action == 'active':
                    row, col = self.map_to_pos(y, x)
                    deactivated_id = self.get_id(row, col)
                    self.deactivate_signal.emit(deactivated_id)
            else:
                return

        self.grid_img.updateImage()

    def refit_view(self, y: float, x: float):
        """
        Reset the view of the grid if the currently active pixel is not in view.

        :param y: Vertical pixel position. (0 at left edge)
        :param x: Horizontal pixel position. (0 at bottom edge)
        """
        view_rect = self.grid_view.viewRect()
        if not view_rect.contains(x, y):
            self.grid_view.autoRange()

    def set_active(self, pos: Union[tuple, int]):
        """
        Set the specified grid position to active.

        :param pos: Tuple specifying grid position to set active as (row, column).
        """
        if type(pos) in [int, np.int32, np.int64]:
            cur_id = int(pos)
            self.cur_active = self.get_pos_2d(pos)
        elif len(pos) == 2:
            cur_id = self.get_id(pos[0], pos[1])
            self.cur_active = pos
        else:
            raise TypeError("The passed position should be either a tuple (row: int, col: int) or two separate int"
                            "values specifying the position!")

        y, x = self.pos_to_map(self.cur_active[0], self.cur_active[1])
        self.map_mark_pos(y, x, 'reset_active')
        self.map_mark_pos(y, x, 'active')

        self.activate_signal.emit(cur_id)

    def mark_placed(self, grid_id: int):
        """
        Mark the grid at the specified position as having an annotation placed.

        :param grid_id: 1D grid position ID to mark as labeled.
        """
        row, col = self.get_pos_2d(grid_id)
        y, x = self.pos_to_map(row, col)
        self.map_mark_pos(y, x, 'placed')

        self.placements[grid_id] = True

    def mark_missing(self, grid_id: int):
        """
        Mark the grid at the specified position as not having an annotation placed.

        :param grid_id: 1D grid position ID to mark as not labeled.
        """
        row, col = self.get_pos_2d(grid_id)
        y, x, = self.pos_to_map(row, col)
        self.map_mark_pos(y, x, 'missing')

        self.placements[grid_id] = False

    def get_cur_1d(self) -> int:
        """
        Calculate and return the current 1D grid position ID of the currently active grid position.

        :return: Integer corresponding to the 1D index of the currently active grid position.
        """
        return self.get_id(self.cur_active[0], self.cur_active[1])

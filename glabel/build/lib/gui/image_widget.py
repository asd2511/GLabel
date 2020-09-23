#!/usr/bin/env python
from typing import List, Union, Tuple, Iterable
from collections import namedtuple
from copy import deepcopy
from os import path
from PyQt5.QtWidgets import *
from PyQt5.Qt import QKeySequence
from PyQt5.QtCore import pyqtSignal, Qt, QPointF, QRect, QPropertyAnimation, QLineF, QObject, pyqtProperty, QPoint
from PyQt5.QtGui import QPen, QFont, QColor, QPainter
from skimage.filters import gaussian as sk_gaussian
import pyqtgraph as pg
import numpy as np
import cv2
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

from glabel.analysis import surface, utils
from glabel.gui.config import lu_mouse, mousebinds
from glabel.gui.grid_widget import ButtonGrid, PixelGrid
from glabel.gui.gui_utils import edit_roi_assignment, OptionDialog
from glabel.gui.surface_viewer import SurfaceViewer, SpaghettiViewer
from glabel.nn import postprocessing

# Removing need to rotate images shown by pyqtgraph
pg.setConfigOptions(imageAxisOrder='row-major')


SuturePrediction = namedtuple('SuturePrediction', ['left_map', 'right_map', 'left_bbox', 'right_bbox'])
SortingPrediction = namedtuple('SortingPrediction', ['pred_id', 'y', 'x', 'frame', 'side', 'probabilities'])


class ImageView(pg.ImageView):
    """
    Custom class based on the pyqtgraph ImageView class to add widget displaying images and handle clickable ROIs.

    **Bases:** :class:`pyqtgraph.ImageView`

    Implements custom signals:

    =====================   =================================================================
    **Signal**              **Emits**
    keysignal               The pressed key as received by :func:`keyPressEvent` as an integer value.

                            See :pyqt:`QKey <qt.html#Key-enum>` for a list of values.
    mousesignal             Integer value [0, 1] as received by :func:`mousePressEvent`.

                            0 if signal for placing ROI is received.

                            1 if signal for deleting last placed ROI is received.
    closeup_change_signal   The id of the ROI passed to :func:`update_closeup`
    =====================   =================================================================
    """
    keysignal = pyqtSignal(int)  # Connecting to keyboard signals
    mousesignal = pyqtSignal(int)  # Connecting to mouse signals
    closeup_change_signal = pyqtSignal(int)
    global_reassign_signal = pyqtSignal(dict)

    def __init__(self, image, num_frames, grid_widget, rois: List[QPointF], closeup_view, **kwargs):
        """
        Instantiate an ImageView object to display image data.

        :param image: The image (single frame) to be displayed by the ImageView object. The image can be RGB or
            gray-valued.
        :type image: np.ndarray
        :param grid_widget: The grid widget used to select the active grid point for ROI selection.
        :type grid_widget: :class:`~gui.grid_widget.GridWidget`
        :param rois: ROI positions for currently displayed frame.
        :type rois: List[:pyqt:`QPointF <qpointf>`]
        :param closeup_view: The ImageItem object displaying the closeup view of the current ROI.
        :type closeup_view: :class:`pyqtgraph.ImageItem`
        """
        super().__init__(**kwargs)

        self.view_box = self.getView()  #: :class:`pyqtgraph.ViewBox` displaying the image
        self.view_box.setMenuEnabled(False)  # Disable right-click menu to allow deletion by right-clicking

        self.num_frames = num_frames  #: Number of frames in current data
        self.setImage(image)

        self.grid_widget = grid_widget  #: Reference to currently used :class:`~gui.grid_widget.GridWidget`
        self.grid_widget.request_reassign_signal.connect(lambda d: self.reassign_roi(**d))
        #: Reference to :class:`~pyqtgraph.ImageItem` used for showing the closeup view
        self.closeup_view = closeup_view
        self.active_idx = self.grid_widget.get_cur_1d()  #: Index of currently active grid position
        self.active_history = []  #: History tracking placed ROI positions

        self.roi_positions = rois  #: List of :pyqt:`QPointF <qpointf>` positions of placed ROIs
        self.roi_objects = [None for _ in range(len(rois))]  #: :class:`ROI` objects of placed ROIs
        self.show_rois = True  #: Boolean setting if ROIs are visible
        self.color_code_rois = False  #: Current color coding mode. Will be one of [False, 'row', 'col']
        self.draw_rois()

        self.bboxes = []  #: Bouning Boxes for suture grid regions
        self.overlays = []  #: List of :class:`pyqtgraph.ImageItem` objects for left and right suture probability map
        self.show_bboxes = True  #: Boolean setting for showing suture grid bounding boxes
        self.show_overlays = True  #: Boolean setting for showing suture probability maps

        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)

    def setImage(self, img, **kwargs) -> None:
        """
        Overloaded from :class:`pyqtgraph.ImageView` to allow adding the built-in pyqtgraph timeline-graph object
        onto the ImageView, even though only a single frame was passed to initialize it.

        Calls :meth:`pyqtgraph.ImageView.setImage` to apply the passed image data to this ImageView.

        If :attr:`num_frames` is larger than 1, indicating multi-frame data, :func:`add_timeline_plot` is called to add
        the timeline plot used by :class:`pyqtgraph.ImageView` even though only a single frame is passed into the class.

        :param img: The image data to be displayed (single frame)
        :type img: np.ndarray
        :param kwargs: Keyword arguments passed to the :class:`pyqtgraph.ImageView` constructor.
        :type kwargs: dict
        """
        super().setImage(img, **kwargs)  # Initialize the ImageView with the image-frame to be displayed
        # Add the built-in timeline onto the ImageView if the opened data has more than one frame
        if self.num_frames > 1:
            self.add_timeline_plot()

    def add_timeline_plot(self) -> None:
        """
        Adding the built-in pyqtgraph timeline-graph.

        In order to add the timeline-graph onto this ImageView, even though only a single frame was passed to construct
        it, the instance variable :attr:`pyqtgraph.ImageView.tVals` and some methods have to be overloaded.

        *   :attr:`pyqtgraph.ImageView.tVals` has to be set to the number of frames of the image-data loaded by the
            user. This variable allows the ImageView class to calculate the necessary dimensions and bounds of the
            timeline-graph. It is also used to determine the frame-number when switching frames.

        *   :func:`pyqtgraph.ImageView.hasTimeAxis` needs to be overloaded to avoid actual checks if the displayed image
            data has a time-dimension. It is therefore overloaded to simply always return True.

        After performing all necessary overloads to pass checks within the ImageView class, calling
        :func:`pyqtgraph.ImageView.roiClicked` sets all necessary variables for the timeline-graph and displays it as
        part of the ImageView object.
        """
        # Copy the general initialization of the timeline-plot from ImageView
        self.tVals = np.arange(self.num_frames)  #: Fake frame count applied during :func:´add_timeline_plot`
        self.hasTimeAxis = lambda: True  #: Overwritten method from :class:`pyqtgraph.ImageView` to always return True
        self.roiClicked()  # Call to create and show the timeline-plot object now that tVals exists

    def jumpFrames(self, n, image) -> None:
        """
        Overloads :meth:`pyqtgraph.ImageView.jumpFrames` in order to avoid existence-check for actual time-dimension in
        displayed data.

        Even though everything could be handled by :meth:`pyqtgraph.ImageView.setCurrentIndex`, the overload is useful
        to allow more flexible compatibility for disabling this 'fake' method of using the built-in pyqtgraph
        timeline-graph.

        :param n: Number of frames to jump. Can be negative.
        :type n: int
        :param image: The image data (single frame) at the frame-index that is jumped to.
            This ImageView object displays only a single frame, so jumping to another frame does not work. In order to
            'fake' multi-frame image data, the frame that should be displayed is passed in from the outside.
        :type image: np.ndarray
        """
        self.setCurrentIndex(self.currentIndex + n, image)

    def setCurrentIndex(self, ind, image) -> None:
        """
        Overloads :meth:`pyqtgraph.ImageView.setCurrentIndex` to manipulate necessary variables in ImageView to display
        new image-frame even though this ImageView only holds a single frame.

        This overload allows displaying an image-frame which is passed into the ImageView from the outside.

        :param ind: New frame-index.
        :type ind: int
        :param image: Image data (single frame) to display in ImageView.
        :type imgae: np.ndarray
        """
        # Manipulate current frame index and prevent running outside of valid frame numbers
        self.currentIndex = np.clip(ind, 0, self.num_frames - 1)
        self.imageItem.updateImage(image)  # Set the new image data
        self.ignoreTimeLine = True  # Disable movement of timeLine to fire internal signals
        self.timeLine.setValue(self.tVals[self.currentIndex])  # Move timeLine
        self.ignoreTimeLine = False  # Re-activate timeLine

    def mousePressEvent(self, e) -> None:
        """
        Handle mouse pressing events.

        **Current logic handles:**

        ============================    ========================
        **Controls**                    **Action**
        Ctrl+LMB                        Place ROI
        Ctrl+RMB                        Remove last placed ROI
        ============================    ========================

        Placing ROI
            After the correct command for placing a ROI is received, the clicked position gets saved in
            :attr:`roi_positions` and a call to :func:`draw_rois` updates the image with the new ROI object.
            The :attr:`mousesignal` emits the integer 0.

        Removing ROI
            The index of the last placed ROI is retrieved from :attr:`active_history` and passed on to
            :func:`delete_roi` to remove it.
            The :attr:`mousesignal` emits the integer 1.

        :param e: The QMouseEvent specifying location and button of click.
        :type e: :pyqt:`QMouseEvent <qmouseevent>`
        """
        modifiers = QApplication.keyboardModifiers()  # Access any modifiers currently activated by keyboard

        # Update which ROI position is currently selected as active
        self.active_idx = self.grid_widget.get_cur_1d()

        # Important, map xy coordinates to scene!
        xy = self.getImageItem().mapFromScene(e.pos())

        # Setting points only when (Ctrl + ) LMB
        if lu_mouse('PlaceROI', e.button(), modifiers):
            if xy.x() < 0 or xy.x() > self.image.shape[self.axes['x']] \
                    or xy.y() < 0 or xy.y() > self.image.shape[self.axes['y']]:
                return
            if ROI.TranslateSnap:
                xy.setX(np.rint(xy.x()))
                xy.setY(np.rint(xy.y()))
            self.roi_positions[self.active_idx] = xy  # Set position of placed ROI
            self.active_history.append(self.active_idx)  # Save order of saved ROI indices for orderly deletion
            self.draw_rois()  # Update the GUI with the added ROI
            self.update_closeup(self.roi_objects[self.active_idx])
            self.mousesignal.emit(0)

        # Removing last set point when Ctrl + RMB
        elif lu_mouse('UndoROI', e.button(), modifiers):
            # Deletion can only happen if something was placed earlier
            if self.active_history:
                self.mousesignal.emit(1)
                self.delete_roi(self.active_history.pop())

    def keyPressEvent(self, ev) -> None:
        """
        Handle presses of keyboard keys by emitting them to parent object.

        :param ev: The QKeyEvent specifying which key is pressed.
        :type ev: :pyqt:`QKeyEvent <qkeyevent>`
        """
        self.keysignal.emit(ev.key())

    def draw_rois(self) -> None:
        """
        Draw placed ROIs onto image object. Each placed ROI will be an instance of :class:`ROI`.

        First, all drawn :class:`ROI` objects are removed from the view before they are re-drawn according to the
        current placements specified by :attr:`roi_positions`.

        .. todo:: Behavior should be modified to avoid deleting and re-drawing **all** ROI positions. Instead, it should
            dynamically only update changes between calls! This could reduce the amount of lag for moving through frames
            with placed ROIs.
        """
        # Delete old placed ROIs
        for i in self.roi_objects:
            if i is not None:
                self.getView().removeItem(i)

        self.roi_objects = [None for _ in range(len(self.roi_positions))]  # Remove all drawn ROIs

        if any(roi.x() != -1 or roi.y() != -1 for roi in self.roi_positions):  # If ROIs have been placed
            # Iterate through all placed ROIs and add them to the ViewBox object
            for idx, roi in enumerate(self.roi_positions):
                # Only place "real" ROIs (which are not pre-initialized ones at (-1, -1))
                if roi.x() != -1 or roi.y() != -1:
                    t = ROI(pos=(roi.x(), roi.y()), size=(10, 10), movable=True, removable=True, roi_id=idx,
                            visible=self.show_rois, color_coded=self.color_code_rois,
                            maxBounds=self.getImageItem().boundingRect())
                    t.sigRegionChanged.connect(self.update_closeup)
                    t.sigClicked.connect(self.on_roi_click)
                    t.sig_remove_request.connect(lambda roi_id: self.delete_roi(roi_id))
                    t.sig_reassign_request.connect(lambda reassign_dict: self.reassign_roi(**reassign_dict))
                    t.sigRegionChangeStarted.connect(self.on_roi_click)
                    t.sigRegionChangeFinished.connect(self.update_position)

                    self.roi_objects[idx] = t
                    self.getView().addItem(self.roi_objects[idx])
                    if not t.visible:
                        t.setPen(QPen(Qt.NoPen))
                    if not self.grid_widget.placements[idx]:
                        self.grid_widget.mark_placed(idx)
                else:
                    if self.grid_widget.placements[idx]:
                        self.grid_widget.mark_missing(idx)
        elif any(self.grid_widget.placements):
            # Mark all as missing
            [self.grid_widget.mark_missing(grid_id) for grid_id, val in enumerate(self.grid_widget.placements) if val]

    def draw_bbox(self, box_info) -> None:
        """
        Draw a bounding box on the currently displayed image.

        The bounding box is colored according to the passed confidence value.

        ==============  ===========
        **Confidence**  **Color**
        >97.5           Green
        90&-97.5%       Yellow
        <90%            Red
        ==============  ===========

        :param box_info: The bounding box(es) as :class:`pyqtgraph.RectROI` instances and their corresponding confidence
            values. The values are ordered as [box1, conf1, box2, conf2, ...]
        :type box_info: List[pg.RectROI, float]
        """
        # Do nothing when an empty prediction is passed. This happens when the current frame has no prediction yet.
        if box_info:
            if not self.bboxes:
                self.bboxes = box_info.copy()
                for idx in range(len(self.bboxes))[::2]:
                    self.getView().addItem(box_info[idx])
                    conf_label = pg.TextItem(str(box_info[idx+1]))
                    self.bboxes[idx+1] = conf_label
                    self.getView().addItem(conf_label)

            for i in range(len(box_info))[::2]:
                pos = box_info[i].pos()
                size = box_info[i].size()
                conf = box_info[i+1]

                if conf > 0.975:
                    color = 'g'
                elif 0.9 <= conf <= 0.975:
                    color = 'y'
                else:
                    color = 'r'

                self.bboxes[i].setPen(pg.mkPen(color=color, width=2))
                self.bboxes[i].setPos(pos)
                self.bboxes[i].setSize(size)

                self.bboxes[i+1].setText(str(conf))
                self.bboxes[i+1].setColor(color)
                self.bboxes[i+1].setPos(pos)

        # Remove any existing overlays if we already were on a frame with predictions previously
        else:
            if self.bboxes:
                for box in self.bboxes:
                    self.getView().removeItem(box)
                self.bboxes = []

    def draw_prediction(self, prediction) -> None:
        """
        Draw the suture probability maps as an overlay on top of the image.

        The passed probability maps are converted to two separate :class:`pyqtgraph.ImageItem` objects with reduced opacity.
        Based on the passed bounding box information, these are set to overlay the suture grid regions in the image to
        allow easy comparison of which sutures were correctly found.

        .. note:: Calling this method with `prediction` as `None` will remove any currently shown overlays from the
            image.

        :param SuturePrediction prediction: :class:`SuturePrediction` namedtuple containing the probability maps for
            left and right view side as well as bounding boxes indicating where to place the maps on the image.
        """
        # Do nothing when an empty prediction is passed. This happens when the current frame has no prediction yet.
        if prediction:
            # Unpack the data
            l_pred = prediction.left_map
            r_pred = prediction.right_map
            l_box = prediction.left_bbox
            r_box = prediction.right_bbox

            # If no overlays exist, create new ImageItem objects and set them up as overlays
            if not self.overlays:
                l_overlay = pg.ImageItem(l_pred)
                r_overlay = pg.ImageItem(r_pred)
                self.overlays = [l_overlay, r_overlay]
                self.getView().addItem(l_overlay)
                self.getView().addItem(r_overlay)
                l_overlay.setZValue(10)
                r_overlay.setZValue(10)
                l_overlay.setOpacity(0.5)
                r_overlay.setOpacity(0.5)

            # If the overlays already exist, simply switch out the image data they display
            else:
                self.overlays[0].setImage(l_pred)
                self.overlays[1].setImage(r_pred)

            # Move the overlay ImageItems to the position dictated by the bounding boxes
            l_positioning = QRect(l_box.left, l_box.top, l_box.width, l_box.height)
            r_positioning = QRect(r_box.left, r_box.top, r_box.width, r_box.height)
            self.overlays[0].setRect(l_positioning)
            self.overlays[1].setRect(r_positioning)

        # Remove any existing overlays if we already were on a frame with predictions previously
        else:
            if self.overlays:
                for overlay in self.overlays:
                    self.getView().removeItem(overlay)
            self.overlays = []

    def delete_roi(self, deletion_idx=None):
        """
        Delete a single ROI.

        If the **deletion_idx** is not specified, the currently active grid position will be chosen as the position to
        be deleted.

        :param deletion_idx: Index into array of ROIs of ROI to be deleted.
        :type deletion_idx: int
        """
        if not deletion_idx:
            deletion_idx = self.grid_widget.get_cur_1d()
        # Remove all occurrences of the deletion_idx from the history chain
        self.active_history = [i for i in self.active_history if i != deletion_idx]
        self.roi_positions[deletion_idx] = QPointF(-1, -1)  # Remove and replace placed ROI position
        self.active_idx = int(deletion_idx)
        self.draw_rois()  # Update the GUI to remove the ROI
        self.grid_widget.set_active(self.active_idx)
        self.update_closeup()

    def delete_all(self) -> None:
        """
        Delete all placed ROIs from the current frame.
        """
        self.active_history = []  # Empty the history
        for roi in self.roi_objects:
            if roi is not None:
                self.getView().removeItem(roi)
        # Empty position and object lists
        self.roi_positions = [QPointF(-1, -1)] * len(self.roi_positions)
        self.roi_objects = [None for _ in range(len(self.roi_positions))]
        # Reset grid widget
        self.active_idx = 0
        self.grid_widget.set_active(0)
        # Update view to remove all drawn ROis
        self.draw_rois()

    def toggle_visibility_all(self) -> None:
        """
        Toggle the visibility of drawn :class:`ROI` objects.

        This will toggle :attr:´show_rois`, which will act like the visibility is changed globally, independent of the
        currently displayed frame (No frame will show ROIs until visibility is toggled again).
        """
        self.show_rois = not self.show_rois
        self.draw_rois()
        # Highlight the currently active grid position if it is placed
        cur_roi = self.roi_objects[self.grid_widget.get_cur_1d()]
        if self.show_rois and cur_roi:
            cur_roi.highlight()

    def cycle_roi_color_coding(self) -> None:
        """
        Cycle between available options for color coding currently placed ROIs.

        Available modes and the cycle sequence:

        =========   ========    =======================================================================================
        **Order**   **Mode**    **Color Coding**
        1           False       No color coding of ROIs. Each ROI is colored in the default inactive color, with
                                highlighting of the currently active ROI.
        2           'row'       Row-wise color coding. All ROIs positioned on the same row will have the same color.
        3           'col'       Column-wise color coding. All ROIs positioned in the same column will have the same
                                color.
        =========   ========    =======================================================================================

        This method will only set the class attribute :attr`color_code_rois` to a new value, while the color coding
        itself happens on a call to :func:`draw_rois`.
        """
        if not self.color_code_rois:
            self.color_code_rois = 'row'
        elif self.color_code_rois == 'row':
            self.color_code_rois = 'col'
        else:
            self.color_code_rois = False

        self.draw_rois()
        cur_roi = self.roi_objects[self.grid_widget.get_cur_1d()]
        if self.show_rois and cur_roi:
            cur_roi.highlight()

    def toggle_bbox_visibility(self) -> None:
        """
        Toggle the visibility of suture grid region bounding boxes.

        Toggling visibilty off will not delete the bounding boxes, but just set their opacity to 0.
        """
        self.show_bboxes = not self.show_bboxes
        for bbox in self.bboxes:
            if self.show_bboxes:
                bbox.setOpacity(1.0)
            else:
                bbox.setOpacity(0.0)

    def toggle_overlay_visibility(self) -> None:
        """
        Toggle the visibility of suture probability map overlays.

        Toggling visibilty off will not delete the overlays, but just set their opacity to 0.
        """
        self.show_overlays = not self.show_overlays
        for overlay in self.overlays:
            if self.show_overlays:
                overlay.setOpacity(0.5)
            else:
                overlay.setOpacity(0.0)

    def get_rois(self) -> List[QPointF]:
        """
        Return ROI positions currently placed onto image object as a list containing
        :class:`QPointF <PyQt5.QtCore.QPointF>` elements.

        :return: :attr:`roi_positions`: All ROI placements for the current frame as list of :pyqt:`QPointF <qpointf>`.
        """
        return self.roi_positions

    def update_position(self, obj) -> None:
        """
        Update the saved position of the :class:`ROI` **obj** with the current position.

        Called after e.g. a drag event changed the ROI position through pyqtgraph interfacing.

        :param obj: The instantiated ROI object whose position needs to be updated.
        :type obj: :class:`ROI`
        """
        pos = obj.centered_pos()
        self.roi_positions[obj.roi_id] = QPointF(pos[0], pos[1])

    def on_roi_click(self) -> None:
        """
        Event handling when a placed :class:`ROI` is clicked by the user.

        Slot for the signals :attr:`sigClicked` and :attr:`sigRegionChangeStarted` of :class:`ROI`.
        """
        modifiers = QApplication.keyboardModifiers()
        # If CTRL is held down --> Click is because of placing ROI, not selecting one ==> Do nothing here
        if modifiers == Qt.ControlModifier:
            return

        if modifiers == Qt.ShiftModifier:
            self.swap_rois(self.sender(), self.roi_objects[self.active_idx])

        roi = self.sender()
        self.active_idx = roi.roi_id
        self.update_closeup(roi)
        self.grid_widget.set_active(roi.roi_id)

    def update_closeup(self, roi=None) -> None:
        """
        Update the :attr:_closeup_view` with new information from the passed :class:`ROI`.

        The closeup view retrieves a 10-by-10 neighborhood with its center at the center of the ROI from the shown image
        frame and displays it.

        Slot for signal :attr:`sigRegionChanged` of :class:`ROI`

        Emits :attr:`ROI.roi_id` of the passed **roi** as :attr:`closeup_change_signal`

        :param roi: ROI object for which the closeup view should show its neighborhood. If not passed, the currently
            active grid position will be used to update the closeup.
        :type roi: :class:`ROI`
        """
        if not self.closeup_view:
            return
        if not roi:
            roi = self.roi_objects[self.active_idx]
        image = self.getImageItem().image
        try:
            slice_coords, _ = roi.getArraySlice(image, self.getImageItem())
        except AttributeError:
            # No ROI object at `active_idx` found
            self.closeup_view.setImage(np.zeros((10, 10, 3)))
            return

        # Extract image data and flip it upright
        slice_data = image[slice_coords[0].start:slice_coords[0].stop-1,
                           slice_coords[1].start:slice_coords[1].stop-1]
        slice_data = np.flip(slice_data, axis=0)

        try:
            self.closeup_view.setImage(slice_data, autoRange=True)
            self.closeup_change_signal.emit(roi.roi_id)
        except ValueError:
            self.closeup_view.setImage(np.zeros((10, 10, 3)))

    def show_crosshair(self):
        """
        Displays the crosshair on the image view.
        """
        self.addItem(self.vLine, ignoreBounds=True)
        self.addItem(self.hLine, ignoreBounds=True)

        self.scene.sigMouseMoved.connect(self.mouseMoved)
        self.ui.graphicsView.setCursor(Qt.BlankCursor)

    def hide_crosshair(self):
        """
        Hides the crosshair on the image view.
        """
        self.removeItem(self.vLine)
        self.removeItem(self.hLine)

        self.scene.sigMouseMoved.disconnect(self.mouseMoved)
        self.ui.graphicsView.unsetCursor()

    def mouseMoved(self, evt):
        """
        Is called whenever the mouse is moved to update the crosshair.

        :param evt: The current mouse location
        :type evt: PyQt5.QtCore.QPointF
        """
        mousePoint = self.getImageItem().mapFromScene(evt.x(), evt.y())

        self.vLine.setPos(mousePoint.x())
        self.hLine.setPos(mousePoint.y())

    def swap_rois(self, roi_a, roi_b):
        """
        Swap the coordinates and grid correspondence of two ROI objects.

        :param roi_a: First ROI object.
        :type roi_a: ROI
        :param roi_b: Second ROI object.
        :type roi_b: ROI
        """
        # Enter the swapped positions in the list of all placed ROI positions --> This makes sure that on the next
        # `draw_rois()` call the updated information is displayed
        self.roi_positions[roi_b.roi_id] = pg.Point(roi_a.centered_pos())
        self.roi_positions[roi_a.roi_id] = pg.Point(roi_b.centered_pos())
        self.draw_rois()

        id_a = self.grid_widget.get_pos_2d(roi_a.roi_id)
        id_b = self.grid_widget.get_pos_2d(roi_b.roi_id)
        scene_p = self.getImageItem().mapToScene(roi_a.pos() + pg.Point(0, 10))
        view_p = self.getView().mapFromScene(scene_p).toPoint()
        popup_pos = self.mapToGlobal(view_p)
        QToolTip.showText(popup_pos, f"Swapped {id_a} <-> {id_b}")

    def reassign_roi(self, roi_id, new_row, new_col) -> None:
        """
        Reassign the ROI currently set to `roi_id` the new grid ID found at (`new_row`, `new_col`).

        When reassigning ROI to position that is already filled with a placed ROI, confirmation is asked before the
        existing ROI is replaced.
        Reassignment can happen on the current frame or for all frames. Decision is made by user through an
        :class:`OptionDialog` instance.

        :param int roi_id: The grid ID for the ROI to be reassigned.
        :param int new_row: New row position to assign ROI.
        :param int new_col: New column position to assign ROI.

        :raises AssertionError: if the passed `roi_id` is None (e.g. because it has not been placed yet).
        """
        assert self.roi_objects[roi_id] is not None, "Cannot reassign ROI that has not been placed!"

        roi_object = self.roi_objects[roi_id]  # Reference to the ROI that is to be reassigned
        new_id = self.grid_widget.get_id(new_row, new_col)  # ID that the ROI should be assigned to

        # Ask user what to do if the desired ROI position is already asssigned to another ROI
        if any(coord != -1.0 for coord in [self.roi_positions[new_id].x(), self.roi_positions[new_id].y()]):
            if not ConfirmDialog("The desired ROI placement is already assigned.\nDo you want to overwrite the existing"
                                 " ROI placement?").exec_():
                # If not accepted, cancel the reassignment
                return

        # Ask user if reassignment should happen only on this or on all frames
        reassign_options = ['This frame', 'All frames']
        choice_dlg = OptionDialog("Reassign ROI only on this frame or on all frames?", reassign_options)
        choice_idx = choice_dlg.exec_() - 1  # -1 bc. index 0 emitted for canceled dialog
        if choice_idx == -1:
            # If user canceled the dialog, cancel the reassignment
            return
        elif choice_idx == 1:
            # If user wants to apply reassignment to all frames -> tell it to the ImageStack before continuing with the
            # current frame
            new_id = self.grid_widget.get_id(new_row, new_col)
            reassign_info = {'roi_id': roi_id, 'new_id': new_id}
            self.global_reassign_signal.emit(reassign_info)

        # Assign the existing placement to the desired new position
        self.roi_positions[new_id] = pg.Point(roi_object.centered_pos())
        # Delete the placement from the old position
        self.roi_positions[roi_id] = QPointF(-1, -1)

        # If the request for reassignment came from a PixelGrid, the pixel re-coloring has to be handled
        # Remove the hover marking before switching to correctly update the position placement to be "placed"
        if isinstance(self.sender(), PixelGrid):
            y, x = self.sender().pos_to_map(*self.sender().get_cur_2d())
            self.sender().map_mark_pos(y, x, 'reset_hover')

        # Update the displayed ROIs
        self.draw_rois()

        old_pos = self.grid_widget.get_pos_2d(roi_id)
        scene_p = self.getImageItem().mapToScene(roi_object.pos() + pg.Point(0, -10))
        view_p = self.getView().mapFromScene(scene_p).toPoint()
        popup_pos = self.mapToGlobal(view_p)
        QToolTip.showText(popup_pos, f"Reassigned {old_pos} -> ({new_row}, {new_col})")


class ImageStack(QWidget):
    """
    **Bases** :pyqt:`QWidget <qwidget>`

    The main widget holding the logic for the GLable application.

    All references to contained widgets are stored in this class to be able to pass information between widgets.

    Implements custom signal:

    ==========  ===========
    **Signal**  **Emits**
    keysignal   Pressed key as received by :func:`keyPress`
    ==========  ===========
    """

    keysignal = pyqtSignal(int)

    def __init__(self, stack, rois=None, num_rows=5, num_cols=7, stereo_grid=True, show_closeup=True, auto_mode=0,
                 auto_snaking=True, auto_copy=True, num_rows_track=0, shortcuts=None, grid_style=0, begin_top=False,
                 gaussian=False, show_crosshair=True, active_progress=0, frame_clicking=False):
        """
        Instantiate an ImageStack object holding necessary widgets to display images, control widgets and information
        displays for the GLable application.

        :param stack: Image data. Can be a single image or a multi-frame data. Ordering of axes should be [frames, y, x,
            color]
        :type stack: np.ndarray
        :param rois: Locations (XY-coordinates) of known ROI that should be drawn onto the corresponding image frame(s).
        :type rois: np.ndarray
        :param num_rows: Number grid rows for the grid of ROIs of the current images. Defaults to 5.
        :type num_rows: int
        :param num_cols: Number of grid columns for the grid of ROIs of the current images. Defaults to 7.
        :type num_cols: int
        :param stereo_grid: Boolean setting if grid widget should display a stereo view of the ROI grid.
        :type stereo_grid: bool
        :param show_closeup: Setting if closeup view of currently active position should be displayed.
        :type show_closeup: bool
        :param auto_mode: Mode for automatic progression of the active grid position. 0=row-wise; 1=column-wise.
        :type auto_mode: int
        :param auto_snaking: Setting if pathing of active grid position should utilize snaking behavior or not
        :type auto_snaking: bool
        :param auto_copy: Setting if ROI placements from current frame should be copied when switching to next frame
            that is empty.
        :type auto_copy: bool
        :param num_rows_track: Number of rows (counted from bottom) to utilize tracking behavior when using
            copy-and-track feature.
        :type num_rows_track: int
        :param shortcuts: Dictionary defining shortcuts to be used by GUI.
        :type shortcuts: dict
        :param grid_style: Style of :class:`gui.grid_widget.GridWidget` to be used. 0=ButtonGrid; 1=PixelMap
        :type grid_style: int
        :param begin_top: Setting of pathing of active grid position should have its origin at the top-left corner
            instead of the default at the bottom-left corner.
        :type begin_top: bool
        :param active_progress: Mode for behavior of automatic progression of currently active grid position.
            0=On ROI placement and empty frame; 1=On ROI placement; 2=On empty frame; 3=Never
        :type active_progress: int
        :param frame_clicking: Setting if frame clicking mode should be activated .
        :type frame_clicking: bool

        .. todo:: Settings should be globalized and not all passed to constructor to allow for easier modularization!
        """
        super().__init__()

        # Transposing images unnecessary after global pyqtgraph options setting `imgAxisOrder='row_major'

        self.stack = stack  #: Image data as numpy.ndarray. Dimension ordering should be [frames, y, x, color]
        self.stack_cache = np.copy(stack)  #: Copy of image data useful for accessing original data if blur is displayed

        # In case of a 4D array --> Assume default that first dimension is across frames
        # Otherwise, if 3D array --> If the last dimension, which could be color for single image, or (x/y)-coordinates
        #                            for gray-valued multi-frame image, is larger than 4 (hinting at gray-value image),
        #                            then also take the first dimension as frames
        if self.stack.ndim == 4 or (self.stack.ndim == 3 and self.stack.shape[2] > 4):
            self.num_frames = self.stack.shape[0]  #: Number of frames in the image data
        else:
            self.num_frames = 1

        # Storage of the clicked coordinates for each frame. Each element is a QPointF element.
        stereo_mult = 2 if stereo_grid else 1
        if not rois:
            #: List of :pyqt:`QPointF <qpointf>` initialized as all (-1, -1) used for saving roi placements. Has same
            #: dimensionality as :attr:`stack`
            self.roi_stack = [[QPointF(-1, -1)] * num_rows * num_cols * stereo_mult for _ in range(self.num_frames)]
        else:
            self.roi_stack = rois
        self.bbox_stack = None
        self.suture_predictions = None
        self.prediction_peaks = None
        self.sorting_predictions = None
        self.processed_predictions = None
        self.pred_roi_stack = None
        self.surface_f = None
        self.surf_points = None
        self.surf_viewer = None

        self.cur_frame_id = 0  #: Index of currently displayed frame
        self.auto_copy = auto_copy
        self.num_rows_track = num_rows_track

        self.gaussian = gaussian
        self.show_crosshair = show_crosshair

        self.active_progress = active_progress
        self.frame_clicking = frame_clicking
        self.locked_view = False

        #: :pyqt:`QGridLayout <qgridlayout>` holding all the necessary widgets in place
        self.grid_layout = QGridLayout()
        # Splitting the layout into left and right sides (left holds ImageView, right holds GridWidget and CloseupView)
        splitter = QSplitter(Qt.Horizontal)
        self.right_side = QGridLayout()
        self.right_side.setObjectName('RightLayout')
        self.right_widget = QWidget(flags=Qt.Widget)
        self.right_widget.setObjectName('RightWidget')

        splitter_vt = QSplitter(Qt.Vertical)
        self.top_right = QGridLayout()
        self.top_right_w = QWidget(flags=Qt.Widget)
        self.btm_right = QGridLayout()
        self.btm_right_w = QWidget(flags=Qt.Widget)

        # Grid widget for selection of currently active ROI position
        self.grid_style = grid_style
        if self.grid_style == 0:
            self.grid_widget = ButtonGrid(num_rows, num_cols, stereo_grid, auto_mode, auto_snaking, begin_top)
        elif self.grid_style == 1:
            self.grid_widget = PixelGrid(num_rows, num_cols, stereo_grid, auto_mode, auto_snaking, begin_top)
        self.grid_widget.setParent(self.right_widget)
        self.grid_widget.activate_signal.connect(self.highlight_roi)
        self.grid_widget.deactivate_signal.connect(self.dehighlight_roi)
        self.grid_widget.key_signal.connect(self.keyPress)

        self.cur_active_label = QLabel("Active position: ")
        self.cur_active_label.setAlignment(Qt.AlignLeft)
        self.cur_active_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.cur_active_label_val = QLabel("0, 0")
        self.cur_active_label_val.setAlignment(Qt.AlignLeft)
        self.cur_active_label_val.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        active_label = QHBoxLayout()
        active_label.addWidget(self.cur_active_label, alignment=Qt.AlignRight)
        active_label.addWidget(self.cur_active_label_val, alignment=Qt.AlignLeft)

        # ImageView object holding the closeup view of selected ROis
        self.closeup_view = None
        if show_closeup:
            self.closeup_window = pg.GraphicsWindow()
            self.closeup_view = self.closeup_window.addViewBox(lockAspect=True)
            self.closeup_item = pg.ImageItem(np.zeros((10, 10)))
            self.closeup_view.addItem(self.closeup_item)
            self.closeup_roi = ROI((5, 5), (10, 10), None, movable=False, removable=False)
            cross = pg.CrosshairROI((5, 5), (3, 3), movable=False, removable=False, angle=45)
            cross.setPen(pg.mkPen(width=2, color='m'))
            self.closeup_view.addItem(self.closeup_roi, ignoreBounds=True)
            self.closeup_view.addItem(cross, ignoreBounds=True)

        self.cur_closeup_label = QLabel("Closeup of ROI at position: ")
        self.cur_closeup_label_val = QLabel("")
        closeup_label = QHBoxLayout()
        closeup_label.addWidget(self.cur_closeup_label, alignment=Qt.AlignRight)
        closeup_label.addWidget(self.cur_closeup_label_val, alignment=Qt.AlignLeft)

        # Initialize shown image with first image from stack and connect signals to handlers
        img = self.get_image()
        self.img_view = ImageView(img, self.num_frames, self.grid_widget,
                                  self.roi_stack[self.cur_frame_id], self.closeup_item)
        self.img_view.global_reassign_signal.connect(lambda d: self.reassign_roi_global(**d))
        self.update_stack()

        self.img_view.mousesignal.connect(self.mousePress)
        self.img_view.keysignal.connect(self.keyPress)
        self.img_view.closeup_change_signal.connect(self.update_closeup_label)
        self.img_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_view.ui.menuBtn.hide()
        self.img_view.ui.roiBtn.hide()
        if self.show_crosshair:
            self.img_view.show_crosshair()

        self.timeline_plotitem = self.img_view.getRoiPlot().getPlotItem()
        self.timeline_plotitem.setMenuEnabled(False)  # Disabling right-click menu on timeline
        self.timeline_plotitem.buttonsHidden = True  # Hide AutoRange button on timeline
        # Capturing frame changes due to keyboard command or when dragging the slider
        self.img_view.timeLine.setMovable(False)  # TODO: Temporary fix to prevent bug when dragging timeLine!
        self.img_view.timeLine.sigPositionChanged.connect(self.changeZ)

        self.frame_label = QLabel(parent=self.img_view)
        self.frame_label.setFont(QFont("Helvetica", 12))
        self.frame_label.setStyleSheet("background-color: rgba(0, 0, 0, 100);")
        self.frame_label.move(10, 10)
        self.update_frame_label()

        self.position_label_x = QLabel(parent=self.img_view)
        self.position_label_x.setFont(QFont("Helvetica", 12))
        self.position_label_x.setStyleSheet("background-color: rgba(0, 0, 0, 100);")
        self.position_label_x.move(160, 10)

        self.position_label_y = QLabel(parent=self.img_view)
        self.position_label_y.setFont(QFont("Helvetica", 12))
        self.position_label_y.setStyleSheet("background-color: rgba(0, 0, 0, 100);")
        self.position_label_y.move(300, 10)

        self.update_position_label()
        self.img_view.scene.sigMouseMoved.connect(self.update_position_label)

        self.top_right.addLayout(active_label, 0, 0, 1, 2, alignment=Qt.AlignLeft)
        self.top_right.addWidget(self.grid_widget, 1, 0, 1, 2)
        if self.grid_style == 0:
            self.top_right.setAlignment(self.grid_widget, Qt.AlignLeft)
        self.top_right_w.setLayout(self.top_right)

        self.btm_right.addLayout(closeup_label, 0, 0, 1, 2, alignment=Qt.AlignLeft)
        self.btm_right.addWidget(self.closeup_window, 1, 0, 1, 2)
        self.btm_right_w.setLayout(self.btm_right)

        splitter_vt.addWidget(self.top_right_w)
        splitter_vt.addWidget(self.btm_right_w)

        # Put resize handle between ImageView and the right gridlayout
        splitter.addWidget(self.img_view)
        splitter.addWidget(splitter_vt)

        # Hacky way of making the right side as small as possible by making the left stupidly big
        splitter.setSizes([10e6, self.grid_widget.sizeHint().width()])

        self.grid_layout.addWidget(splitter, 0, 1, 7, 1)

        self.setLayout(self.grid_layout)

        self.keybinds = shortcuts
        #: Dictionary holding all :pyqt:`QShortcut <qshortcut>` connections used by the GUI.
        self.shortcuts = dict.fromkeys(self.keybinds.keys())
        self.init_shortcuts()

    def init_shortcuts(self) -> None:
        """
        Initialize all shortcuts with their currently set key combinations.

        Read the :attr:`shortcuts` dictionary and create a :pyqt:`QShortcut <qshortcut>` for each entry, which connects
        to its relative slot for executing the intended behavior.

        Is automatically called during :func:`__init__` to set up the necessary connections. For triggering an update to
        the set shortcuts after they have been customized by the user, see :func:`update_shortcuts`.
        """
        self.shortcuts['DeleteActiveROI'] = QShortcut(QKeySequence(self.keybinds['DeleteActiveROI']), self)
        self.shortcuts['DeleteActiveROI'].activated.connect(
            lambda: self.img_view.delete_roi(self.grid_widget.get_cur_1d())
        )
        self.shortcuts['DeleteAllROI'] = QShortcut(QKeySequence(self.keybinds['DeleteAllROI']), self)
        self.shortcuts['DeleteAllROI'].activated.connect(
            lambda: self.img_view.delete_all() if ConfirmDialog("Delete all placed ROIs of current frame?").exec_()
            else False
        )

        self.shortcuts['ActivateRightROI'] = QShortcut(QKeySequence(self.keybinds['ActivateRightROI']), self)
        self.shortcuts['ActivateRightROI'].activated.connect(
            self.grid_widget.right
        )
        self.shortcuts['ActivateLeftROI'] = QShortcut(QKeySequence(self.keybinds['ActivateLeftROI']), self)
        self.shortcuts['ActivateLeftROI'].activated.connect(
            self.grid_widget.left
        )
        self.shortcuts['ActivateUpROI'] = QShortcut(QKeySequence(self.keybinds['ActivateUpROI']), self)
        self.shortcuts['ActivateUpROI'].activated.connect(
            self.grid_widget.up
        )
        self.shortcuts['ActivateDownROI'] = QShortcut(QKeySequence(self.keybinds['ActivateDownROI']), self)
        self.shortcuts['ActivateDownROI'].activated.connect(
            self.grid_widget.down
        )

        self.shortcuts['ActivateNextROI'] = QShortcut(QKeySequence(self.keybinds['ActivateNextROI']), self)
        self.shortcuts['ActivateNextROI'].activated.connect(
            self.grid_widget.next
        )
        self.shortcuts['ActivatePreviousROI'] = QShortcut(QKeySequence(self.keybinds['ActivatePreviousROI']), self)
        self.shortcuts['ActivatePreviousROI'].activated.connect(
            self.grid_widget.previous
        )

        self.shortcuts['MoveROIUp'] =QShortcut(QKeySequence(self.keybinds['MoveROIUp']), self)
        self.shortcuts['MoveROIUp'].activated.connect(
            lambda: self.move_roi(0, -1)
        )
        self.shortcuts['MoveROIDown'] = QShortcut(QKeySequence(self.keybinds['MoveROIDown']), self)
        self.shortcuts['MoveROIDown'].activated.connect(
            lambda: self.move_roi(0, 1)
        )
        self.shortcuts['MoveROIRight'] = QShortcut(QKeySequence(self.keybinds['MoveROIRight']), self)
        self.shortcuts['MoveROIRight'].activated.connect(
            lambda: self.move_roi(1, 0)
        )
        self.shortcuts['MoveROILeft'] = QShortcut(QKeySequence(self.keybinds['MoveROILeft']), self)
        self.shortcuts['MoveROILeft'].activated.connect(
            lambda: self.move_roi(-1, 0)
        )

        self.shortcuts['NextFrame'] = QShortcut(QKeySequence(self.keybinds['NextFrame']), self)
        self.shortcuts['NextFrame'].activated.connect(
            # If clause prevents trying to access self.stack[frame] outside of index range
            lambda: self.img_view.jumpFrames(1, self.stack[self.cur_frame_id + 1])
            if self.cur_frame_id + 1 < self.num_frames else ()
        )
        self.shortcuts['PreviousFrame'] = QShortcut(QKeySequence(self.keybinds['PreviousFrame']), self)
        self.shortcuts['PreviousFrame'].activated.connect(
            # If clause prevents trying to access self.stack[frame] outside of index range
            lambda: self.img_view.jumpFrames(-1, self.stack[self.cur_frame_id - 1])
            if self.cur_frame_id - 1 >= 0 else ()
        )

        self.shortcuts['CopyNextROI'] = QShortcut(QKeySequence(self.keybinds['CopyNextROI']), self)
        self.shortcuts['CopyNextROI'].activated.connect(
            lambda: self.copy_rois_from('next')
        )
        self.shortcuts['CopyPreviousROI'] = QShortcut(QKeySequence(self.keybinds['CopyPreviousROI']), self)
        self.shortcuts['CopyPreviousROI'].activated.connect(
            lambda: self.copy_rois_from('previous')
        )
        self.shortcuts['CopyROIsToAll'] = QShortcut(QKeySequence(self.keybinds['CopyROIsToAll']), self)
        self.shortcuts['CopyROIsToAll'].activated.connect(
            lambda: self.copy_rois_to_all() if
            ConfirmDialog("Copy this frame's ROIs to all other frames? "
                          "\nAll existing ROIs will be overwritten!").exec_()
            else False
        )

        self.shortcuts['ToggleROIVisibility'] = QShortcut(QKeySequence(self.keybinds['ToggleROIVisibility']), self)
        self.shortcuts['ToggleROIVisibility'].activated.connect(
            self.img_view.toggle_visibility_all
        )
        self.shortcuts['CycleROIColorCoding'] = QShortcut(QKeySequence(self.keybinds['CycleROIColorCoding']), self)
        self.shortcuts['CycleROIColorCoding'].activated.connect(
            self.img_view.cycle_roi_color_coding
        )

        self.shortcuts['FindNewSuturePositions'] = QShortcut(QKeySequence(self.keybinds['FindNewSuturePositions']), self)
        self.shortcuts['FindNewSuturePositions'].activated.connect(
            lambda: self.copy_and_track('matching')
        )
        self.shortcuts['FindNearestDark'] = QShortcut(QKeySequence(self.keybinds['FindNearestDark']), self)
        self.shortcuts['FindNearestDark'].activated.connect(
            lambda: self.copy_and_track('darkest')
        )

        self.shortcuts['FindNearestLight'] = QShortcut(QKeySequence(self.keybinds['FindNearestLight']), self)
        self.shortcuts['FindNearestLight'].activated.connect(
            lambda: self.copy_and_track('lightest')
        )

        self.shortcuts['LockView'] = QShortcut(QKeySequence(self.keybinds['LockView']), self)
        self.shortcuts['LockView'].activated.connect(
            self.toggle_view_lock
        )
        self.shortcuts['FirstFrame'] = QShortcut(QKeySequence(self.keybinds['FirstFrame']), self)
        self.shortcuts['FirstFrame'].activated.connect(
            self.first_frame
        )

        self.shortcuts['Test'] = QShortcut(QKeySequence(self.keybinds['Test']), self)
        self.shortcuts['Test'].activated.connect(
            lambda: print('Test')
        )

        QShortcut(QKeySequence(Qt.Key_P), self).activated.connect(
            self.img_view.toggle_bbox_visibility
        )

        QShortcut(QKeySequence(Qt.Key_O), self).activated.connect(
            self.img_view.toggle_overlay_visibility
        )

    def update_shortcuts(self) -> None:
        """
        Trigger an update to :attr:`shortcuts`.

        This is called after the user customizes any of the keybindings for shortcuts in
        :meth:`~gui.main_window.Main.set_shortcut_settings`.
        """
        for action in list(self.shortcuts.keys()):
            self.shortcuts[action].setKey(QKeySequence(self.keybinds[action]))

    def copy_rois_from(self, frame) -> None:
        """
        Trigger a copy process to copy all placed ROIs from the previous or next frame onto the current frame.

        As all ROIs currently placed on the displayed frame will be overwritten by the copy process, a
        :class:`ConfirmDialog` will be opened before execution, asking the user for confirmation.

        :param frame: One of ['previous', 'next']. Determines from which frame the ROIs will be copied to the current
            frame.
        :type frame: str
        """
        if ConfirmDialog(
                f"Copy ROIs from {frame} frame? \nAll existing ROIs on this frame will be overwritten!"
        ).exec_():
            copy_id = self.cur_frame_id - 1 if frame == 'previous' else self.cur_frame_id + 1
            self.img_view.roi_positions = self.roi_stack[copy_id].copy()
            self.img_view.draw_rois()

    def copy_and_track(self, mode) -> None:
        """
        Copy the ROIs from the previous frame and try to track the moved positions/objects on the current frame by using
        the specified *mode*.

        The specified *mode* determines the method for tracking the positions and the radius of the neighborhood used
        for searching.
        The general method is to copy all ROIs from the previous frame, extract a neighborhood around each ROI
        placement and search within it for the new position the ROI should be in.
        The specific search for the new ROI position is handled by :func:`auto_find_suture`.

        ==========  =========================   =========================
        **Mode**    **Tracking method**         **Neighborhood radius**
        'matching'  Template matching           6px
        'darkest'   Search for darkest pixel    10px
        'lightest'  Search for lightest pixel   10px
        ==========  =========================   =========================

        As all ROIs currently placed on the displayed frame will be overwritten by the copy process, a
        :class:`ConfirmDialog` will be opened before execution, asking the user for confirmation.

        :param mode: One of ['matching', 'darkest', 'lightest']. The mode used in tracking the ROI positions.
        :type mode: str
        """
        if ConfirmDialog(
            f"Copy ROIs from previous frame and track moved sutures by using `{mode}` mode? \nAll existing ROIs on this "
            f"frame will be overwritten!"
        ).exec_():
            radius = 10 if mode == 'matching' else 6
            self.img_view.roi_positions = self.roi_stack[self.cur_frame_id - 1].copy()
            self.img_view.draw_rois()  # Need to draw and create ROI objects before they can be moved
            for roi_id in range(len(self.img_view.roi_positions)):
                row, _ = self.grid_widget.get_pos_2d(roi_id)
                if row + 1 <= self.num_rows_track:
                    if self.img_view.roi_positions[roi_id].x() != -1 or self.img_view.roi_positions[roi_id].y() != -1:
                        self.auto_find_suture(roi_id, radius, mode)

    def update_grid_widget(self, num_rows, num_cols, stereo) -> None:
        """
        Update the :attr:`grid_widget` with new settings.

        .. note:: **Currently unused method**, because replacing :attr:`grid_widget` with a newly initialized
            :class:`~gui.grid_widget.GridWidget` will break all references that this :class:`ImageStack` holds to it.
            The current workaround skips this method in favor of simply re-initializing the whole :class:`ImageStack`.

        :param num_rows: Number grid rows for the grid of ROIs of the current images.
        :type num_rows: int
        :param num_cols: Number grid columns for the grid of ROIs of the current images.
        :type num_cols: int
        :param stereo: Specification determining if GridWidget is of stereo or mono type.
        :type stereo: bool
        """
        tmp = self.grid_widget  # Reference to old widget
        # Instantiate new widget with updated dimensions and add to layout
        if self.grid_style == 0:
            self.grid_widget = ButtonGrid(num_rows, num_cols, stereo, self.grid_widget.mode, self.grid_widget.snaking)
        elif self.grid_style == 1:
            self.grid_widget = PixelGrid(num_rows, num_cols, stereo, self.grid_widget.mode, self.grid_widget.snaking)
        self.grid_layout.addWidget(self.grid_widget, 1, 1, 1, 2)
        # Refresh references
        self.img_view.grid_widget = self.grid_widget
        # Delete old widget
        tmp.deleteLater()

    def update_grid_style(self, style) -> None:
        """
        Update the :attr:`grid_widget` with new style setting.

        .. note:: **Currently unused method**, because replacing :attr:`grid_widget` with a newly initialized
            :class:`~gui.grid_widget.GridWidget` will break all references that this :class:`ImageStack` holds to it.
            The current workaround skips this method in favor of simply re-initializing the whole :class:`ImageStack`.

        :param style: 0 = ButtonGrid; 1 = PixelGrid. Setting for which style should be set.
        :type style: int
        """
        tmp = self.grid_widget
        self.right_side.removeWidget(self.grid_widget)
        if style == 0:
            setattr(self, 'grid_widget', ButtonGrid(tmp.num_rows, tmp.num_cols, tmp.stereo, tmp.mode, tmp.snaking))
            # self.grid_widget = ButtonGrid(tmp.num_rows, tmp.num_cols, tmp.stereo, tmp.mode, tmp.snaking)
        elif style == 1:
            setattr(self, 'grid_widget', PixelGrid(tmp.num_rows, tmp.num_cols, tmp.stereo, tmp.mode, tmp.snaking))
            # self.grid_widget = PixelGrid(tmp.num_rows, tmp.num_cols, tmp.stereo, tmp.mode, tmp.snaking)
        self.right_side.addWidget(self.grid_widget, 1, 0, 1, 2, alignment=Qt.AlignLeft)
        self.init_shortcuts()  # Re-establish the connection between shortcuts and GridWidget
        self.img_view.grid_widget = self.grid_widget
        self.img_view.draw_rois()
        tmp.deleteLater()

    def update_closeup_label(self, roi_idx) -> None:
        """
        Update text of :attr:`cur_closeup_label_val` indicating which ROI is displayed by the :attr:`closeup_view`.

        Slot connected to *closeup_change_signal* of :class:`ImageView`. To change the text each time a different ROI
        is displayed in the :attr:`closeup_view`. The text to be used is generated by :func:`get_pos_label_text`.

        :param roi_idx: Index of the ROI.
        :type roi_idx: int
        """
        text = self.get_pos_label_text(roi_idx)
        self.cur_closeup_label_val.setText(text)

    def update_frame_label(self) -> None:
        """
        Update text displayed by :attr:`frame_label` telling user current and total frame numbers.

        The displayed text will also include a symbol hinting the direction towards the first frame of the data without
        any placed ROIs.

        =============== ===============================================
        **Hint symbol** **Meaning**
        >               First empty frame is later than current frame
        <               First empty frame is before current frame
        \|              Current frame is first frame without ROIs
        (Nothing)       All frames have at least one ROI placed
        =============== ===============================================

        Is called by :func:`changeZ` each time the currently displayed frame is switched.
        """
        # Get index of first frame without any placed ROIs --> Not last in case a frame was skipped
        empty_frames = np.array([not any(roi.x() != -1 or roi.y() != -1 for roi in frame) for frame in self.roi_stack])
        if not any(empty_frames):
            # All frames have been labeled
            first_empty = 0
        else:
            try:
                first_empty = np.argwhere(empty_frames).min() - 1
            except ValueError:
                # Failsafe
                first_empty = 0
        # Use single character to indicate direction to frame with ROIs
        if first_empty <= 0:
            pos_hint = ''
        elif self.cur_frame_id < first_empty:
            pos_hint = '>'
        elif self.cur_frame_id == first_empty:
            pos_hint = '|'
        else:
            pos_hint = '<'
        if self.num_frames > 1:
            text = f"{self.cur_frame_id}/{self.num_frames} {pos_hint}"
        else:
            text = "1/1"
        self.frame_label.setText(text)
        self.frame_label.adjustSize()

    def update_position_label(self, evt=None):
        """
        Update the x and y coordinate on which the mouse is currently located.

        :param evt: The current mouse location
        :type evt: PyQt5.QtCore.QPointF
        """
        if evt:
            x_float = self.img_view.getImageItem().mapFromScene(evt.x(), evt.y()).x()
            y_float = self.img_view.getImageItem().mapFromScene(evt.x(), evt.y()).y()

            shape = self.img_view.getImageItem().image.shape
            max_x = shape[0]
            max_y = shape[1]

            if (x_float < 0) or (x_float >= max_x):
                x = ""
            else:
                x = int(x_float)

            if (y_float < 0) or (y_float >= max_y):
                y = ""
            else:
                y = int(y_float)

        else:
            x = ""
            y = ""

        text_x = "X: {}".format(x)
        text_y = "Y: {}".format(y)

        self.position_label_x.setText(text_x)
        self.position_label_x.adjustSize()

        self.position_label_y.setText(text_y)
        self.position_label_y.adjustSize()

    def save_cur_rois(self):
        """
        Get the currently placed ROIs from :class:`ImageView` and save them for the current frame in the
        :attr:`roi_stack` by call to :func:`save_frame_rois`.

        Will be called when user saves data using :meth:`~gui.main_window.Main.save_as`
        """
        cur_rois = self.img_view.get_rois()
        self.save_frame_rois(cur_rois, self.cur_frame_id)

    def changeZ(self) -> None:
        """
        Perform necessary updates upon change of currently displayed frame.

        Slot connected to the :attr:`sigPositionChanged` signal emitted by :attr:`ImageView.timeLine`, which is
        triggered every time the line indicating the current frame moves.

        .. todo:: This method seems to cause a bug when manually dragging the timeLine object across the frame graph.
            It seems to cause calls to :func:`save_frame_rois` with wrong arguments, causing ROIs of some frames to be
            overwritten. Probably because jumping by many frames very quickly causing some lag in the saving process.
            Current solution is to have manual dragging disabled for the timeLine object!
        """
        # Save ROIs set on current frame
        cur_active = self.grid_widget.get_cur_1d()
        cur_rois = self.img_view.get_rois()
        self.save_frame_rois(cur_rois, self.cur_frame_id)  # TODO: Seems to not be necessary as reference is passed to img_view?

        self.cur_frame_id = self.img_view.currentIndex  # The ImageView class provides the currently displayed frame number
        self.update_frame_label()
        self.img_view.setCurrentIndex(self.cur_frame_id, self.stack[self.cur_frame_id])

        # Do not overwrite ROIs if any have been placed on the new frame
        if any(p.x() != -1 or p.y() != -1 for p in self.roi_stack[self.cur_frame_id]):
            self.img_view.roi_positions = self.roi_stack[self.cur_frame_id]  # Use existing ROIs
            self.grid_widget.set_active(cur_active)  # Re-activate the same ROI position as before frame jump
            self.img_view.active_idx = cur_active  # Tell ImageView which position is active
            # Clear old ROis and set new ROIs to scene
            self.img_view.draw_rois()
            # If the currently selected grid position was already placed, highlight it again and show its closeup
            if self.img_view.roi_objects[cur_active]:
                self.img_view.roi_objects[cur_active].highlight()
                self.img_view.update_closeup()
        # Otherwise, on empty frame, copy ROIs if wanted
        else:
            if self.auto_copy:
                self.img_view.roi_positions = cur_rois.copy()  # Copy current ROIs to next frame
            else:
                self.img_view.roi_positions = self.roi_stack[self.cur_frame_id]  # Set with uninitialized ROIs
            # Clear old ROIs and set new ROIs to scene
            self.img_view.draw_rois()

            # Set first ROI position as active
            first_active = (self.grid_widget.num_rows - 1, 0) if self.grid_widget.begin_top else (0, 0)

            # Decide if free mode is active or not
            # if self.grid_widget.mode == 0:
            #     self.grid_widget.set_active(cur_active)
            #
            # else:
            #     self.grid_widget.set_active(first_active)

            if self.active_progress in [0, 1]:
                # Set first ROI position as active
                first_active = (self.grid_widget.num_rows - 1, 0) if self.grid_widget.begin_top else (0, 0)
            else:
                # Set the previously active position to active again
                first_active = cur_active
            self.grid_widget.set_active(first_active)

        # Draw the bounding boxes of new frame if any have been created
        if self.bbox_stack:
            self.img_view.draw_bbox(self.bbox_stack[self.cur_frame_id])

        if self.suture_predictions:
            self.img_view.draw_prediction(self.suture_predictions[self.cur_frame_id])

        # Update the displayed 3D surface if it is currently shown
        if self.surf_viewer:
            self.show_surface()

    def first_frame(self) -> None:
        """
        Return to showing the first frame in the image data.

        Calls :func:`ImageView.jumpFrames` to jump back to the very first frame in the opened data.
        """
        self.img_view.jumpFrames(-self.cur_frame_id, self.stack[0])

    def toggle_view_lock(self) -> None:
        """
        Toggle between a locked and unlocked view. Being locked means that no mouse interaction for panning/zooming of
        the image is possible. Will also enable placing ROIs with simple Left click without needing to hold CTRL key.
        """
        if not self.locked_view:
            self.locked_view = True
            self.img_view.view_box.setMouseEnabled(x=False, y=False)  # Disabling mouse interaction
            mousebinds['PlaceROI']['modifier'] = Qt.NoModifier
            # Overlay label indicating that the view is locked
            lock_label = QLabel("Locked View", parent=self.img_view)
            lock_label.setObjectName("lock_label")
            lock_label.setFont(QFont("Helvetica", 12))
            lock_label.setFixedWidth(lock_label.width() + 25)
            lock_label.setStyleSheet("background-color: rgba(0, 0, 0, 100);")
            lock_label.move(10, 50)
            lock_label.show()
        else:
            self.locked_view = False
            self.img_view.view_box.setMouseEnabled(x=True, y=True)  # Enable mouse interaction again
            mousebinds['PlaceROI']['modifier'] = Qt.ControlModifier
            # Remove the overlayed label
            lock_label = self.img_view.findChild(QLabel, "lock_label")
            lock_label.deleteLater()

    def keyPress(self, key) -> None:
        """
        Emit coded value of pressed key in as :attr:`keysignal`.

        No processing of the pressed key happens here. For handling of various key combinations see
        :func:´init_shortcuts`.

        :param key: Coded value of pressed key as seen here: :pyqt:`QKey.key <qt.html#Key-enum>`
        :type key: int
        """
        self.keysignal.emit(key)

    def mousePress(self, sig) -> None:
        """
        Handle mouse events received from :class:`ImageView` by triggering correct action depending on automation
        settings.

        This is a slot for :attr:`ImageView.mousesignal`, which emits the integer values 0 or 1 for placing a ROI or
        removing the last ROI respectively.

        :param sig: One of [0, 1]. 0 = Signal for placing ROI; 1 = Signal for removing last placed ROI.
        :type sig: int
        """
        if sig == 0:
            if self.active_progress in [0, 2]:
                self.grid_widget.next()
            elif self.active_progress == 3 and self.frame_clicking and self.cur_frame_id + 1 < self.num_frames:
                # QTimer.singleShot(1, lambda: self.img_view.jumpFrames(1, self.stack[self.cur_frame_id + 1]))
                self.img_view.jumpFrames(1, self.stack[self.cur_frame_id + 1])

        elif sig == 1:
            if self.active_progress in [0, 2]:
                self.grid_widget.previous()

    def highlight_roi(self, idx) -> None:
        """
        Highlight the placed ROI object specified by *idx* by changing its color to the active color.

        Slot connected to :attr:`GridWidget.activate_signal <gui.grid_widget.GridWidget.activate_signal>`.

        If called for a ROI, will call :meth:`ROI.highlight()` to set its color to :attr:`ROI.ActiveColor` and
        :func:`update_closeup` to show its closeup.

        :param idx: The index of the ROI object. (Its 1D grid position)
        :type idx: int
        """
        text = self.get_pos_label_text(idx)
        self.cur_active_label_val.setText(text)

        # Highlight the ROI object for the selected idx if an object has been placed already
        try:
            roi = self.img_view.roi_objects[idx]
            roi.highlight()
            self.img_view.update_closeup(roi)
        except AttributeError:  # Will be raised if the specified ROI was not set yet
            pass
        except (ValueError, IndexError):  # Will be raised if trying to highlight an ROI after all have been deleted
            self.img_view.closeup_view.setImage(np.zeros((10, 10)))

        # Highlight the same ROI in the 3D scatterplot if available
        if self.surf_viewer:
            self.surf_viewer.color_roi(idx, np.array([0.0, 1.0, 0.0, 1.0]))

    def dehighlight_roi(self, idx):
        """
        Return a previously highlighted ROI object back to its default state.

        Slot connected to :attr:`GridWidget.deactivate_signal <gui.grid_widget.GridWidget.deactivate_signal>`.

        If called for a ROI, will call :meth:`ROI.dehighlight()` to set its color back to the default
        :attr:`ROI.ROI_color`.

        :param idx: The index of the ROI object. (Its 1D grid position)
        :type idx: int
        """
        try:
            self.img_view.roi_objects[idx].dehighlight()
        except AttributeError:
            pass

        # Remove highlight of the same ROI in the 3D scatterplot if available
        if self.surf_viewer:
            self.surf_viewer.color_roi(idx, np.array([1.0, 1.0, 1.0, 0.5]))

    def move_roi(self, delta_x, delta_y, roi_id=None) -> int:
        """
        Move an ROI by (*delta_y*, *delta_x*).

        If no *roi_id* is passed, the currently active ROI will be moved.

        :param delta_x: Pixel value to move ROI horizontally by.
        :type delta_x: int
        :param delta_y: Pixel value to move ROI vertically by.
        :type delta_y: int
        :param roi_id: Index of ROI. (Its 1D grid position)
        :type roi_id: int
        """
        if not roi_id:
            roi_id = self.grid_widget.get_cur_1d()
        try:
            roi = self.img_view.roi_objects[roi_id]
            if roi is not None:
                roi.translate(delta_x, delta_y, False)
        except AttributeError:
            pass

    def save_frame_rois(self, rois, frame) -> None:
        """
        Save the currently placed ROIs on the current frame in :attr:`roi_stack`.

        :param rois: Currently placed ROIs as list of :class:`QPointF <PyQt5.QtCore.QPointF>` elements.
        :type rois: list[PyQt5.QtCore.QPointF.QPointF]
        :param frame: Index specifying frame to save points to.
        :type frame: int

        :raises ValueError: if the number of placed ROIs does not match the number of allowed ROIs in the frame (as
            determined by the grid dimensions).
        """
        max_rois = len(self.roi_stack[0])  # Max number of ROIs allowed for current grid is size of one frame roi_stack
        # Raise error if too many ROIs are placed
        if len(rois) > max_rois:
            raise ValueError("Number of placed ROIs exceeds maximum number of allowed ROIs for current grid settings!")
        else:
            self.roi_stack[frame][:len(rois)] = rois

    def copy_rois_to_all(self) -> None:
        """
        Copy the ROIs from the current frame onto all other frames in the data.

        .. caution:: This will overwrite all existing ROI placements on all other frames.
        """
        cur_rois = self.img_view.get_rois()
        self.roi_stack = [cur_rois for _ in self.roi_stack]

    def get_pos_label_text(self, arr_idx) -> str:
        """
        Text for indicating the currently active grid position to the user in format *'{row}, {col}'*.

        If the grid is currently set to stereo, an indication for which side of the grid the position belongs to is
        added.

        :param arr_idx: Index of the grid position for which to return the text.
        :return: String representation of grid position in format '{row}, {col}'.
        """
        row, col = self.grid_widget.get_pos_2d(arr_idx)

        if self.grid_widget.stereo and col < self.grid_widget.num_cols:
            text = f"{row}, {col} (left view)"
        elif self.grid_widget.stereo and col >= self.grid_widget.num_cols:
            text = f"{row}, {col} (right view)"
        else:
            text = f"{row}, {col}"

        return text

    def auto_find_suture(self, roi_id, radius, mode, template_radius=3):
        """
        Find the suture/object in the current frame marked by a ROI in the previous frame.

        The neighborhood of size *template_radius* around the ROI center in the previous frame is considered the target
        to be searched for. The neighborhood of size *radius* of the same center position in the current frame is
        searched for that target.

        =========== ================================================================================================
        **Mode**    **Method**
        'matching'  Uses :meth:`cv2.matchTemplate` using the :attr:`cv2.TM_CCOEFF_NORMED` method to find the target
                    neighborhood in the new frame.

                    This tends to work pretty well for tracking objects that keep consistent in shape and contrast.
        'darkest'   Searches for the darkest pixel of the gaussian blurred neighborhood of the new frame.

                    Kernel size for blurring is based on *template_radius* with sigma = 0.
        'lightest'  Searches for the lightest pixel of the gaussian blurred neighborhood of the new frame.
        =========== ================================================================================================

        :param roi_id: Index of the ROI for which the new position is determined.
        :type roi_id: int
        :param radius: Radius of neighborhood in current frame that is searched for new position.
        :type radius: int
        :param mode: One of ['matching', 'darkest', 'lightest']. Mode for searching the new position.
        :type mode: str
        :param template_radius: Radius of neighborhood around ROI center in previous frame, whose contents are
            considered the object to be searched for.
        :type template_radius: int

        :raises ValueError: if mode is other than allowed modes.
        """
        prev_pos = self.roi_stack[self.cur_frame_id - 1][roi_id].toPoint()
        cur_pos = self.img_view.roi_positions[roi_id].toPoint()

        suture = self.stack_cache[self.cur_frame_id - 1][prev_pos.y() - template_radius:prev_pos.y() + template_radius,
                                             prev_pos.x()-template_radius:prev_pos.x()+template_radius].astype(np.uint8)
        cur_neighborhood = self.stack_cache[self.cur_frame_id][cur_pos.y() - radius:cur_pos.y() + radius,
                                                   cur_pos.x()-radius:cur_pos.x()+radius].astype(np.uint8)

        if mode == 'matching':
            match_map = cv2.matchTemplate(cur_neighborhood, suture, cv2.TM_CCOEFF_NORMED)
            top_left = np.argwhere(match_map == np.max(match_map))[0]
            center = [int(dim / 2) for dim in match_map.shape]
        elif mode == 'darkest' or mode == 'lightest':
            # Perform search for darkest or lightest pixel on blurred neighborhood to remove any noisy dark pixels
            if mode == 'darkest':
                cur_neighborhood = cv2.GaussianBlur(cur_neighborhood, (template_radius, template_radius), 0)
                top_left = np.argwhere(cur_neighborhood == np.min(cur_neighborhood))[0]
            else:
                cur_neighborhood = cv2.GaussianBlur(cur_neighborhood, (0, 0), 0.5)
                top_left = np.argwhere(cur_neighborhood == np.max(cur_neighborhood))[0]
            center = [int(dim / 2) for dim in cur_neighborhood.shape]
        else:
            raise ValueError(f"Encountered unknown mode `{mode}` in ImageStack.auto_find_suture()")
        offset = top_left - center
        self.move_roi(offset[1], offset[0], roi_id)

    def add_bboxes(self, bboxes, confidences) -> None:
        """
        Add suture grid region bounding boxes for saving and display.

        Processed bounding boxes are stored and boxes for the current frame are passed on to the :class:`ImageView` for
        display.

        :param List[List[Box]] bboxes: Bounding boxes for both views for each frame as :class:`Box` namedtuples.
        :param List[List[float]] confidences: Bounding box confidences.
        """
        self.bbox_stack = postprocessing.process_region_predictions(bboxes, confidences, self.num_frames)
        # Draw the bounding box(es) for the current frame
        self.img_view.draw_bbox(self.bbox_stack[self.cur_frame_id])

    def add_suture_predictions(self, predictions, bboxes) -> None:
        """
        Add suture probability maps for saving and display.

        Processed prediction maps are stored and probability maps for current frame are passed on to the
        :class:`ImageView` for display.

        :param np.ndarray predictions: Numpy array of shape (#frames*2, dim, dim), with `dim` being the dimension of the
            expanded bounding box size for suture maps. (Default dimension is 224 x 224)
        :param List[List[Box]] bboxes: List of lists of :class:`Box` namedtuples. Same concept as for the
            `region_boxes` list, just for the expanded suture identification bounding boxes. (Default boxes are of size
            224 x 224).
        """
        self.suture_predictions = postprocessing.process_suture_maps(predictions, bboxes, self.num_frames)
        # Draw the suture probability maps for the current frame
        self.img_view.draw_prediction(self.suture_predictions[self.cur_frame_id])

    def add_sorted_predictions(self, sortings):
        """
        Add suture sorting prediction for saving and display.

        .. caution:: The currently set :attr:`roi_stack` is replaced by the passed predictions.

        :param List[nn.suture_detection.SortingPrediction] sortings: List of :class:`SortingPrediction` namedtuples. The
            same peaks as stored in the `suture_peaks` variable, but with added sorting inference data from last
            automation step. The namedtuple objects contain references to frame and view side for each suture, as well
            as membership probabilities for each individual grid position.
        """
        self.sorting_predictions, self.pred_roi_stack = postprocessing.process_suture_sortings(sortings,
                                                                                               self.num_frames,
                                                                                               self.grid_widget.num_rows * self.grid_widget.num_cols * 2,
                                                                                               True)
        self.roi_stack = self.pred_roi_stack.copy()
        self.img_view.roi_positions = self.roi_stack[self.cur_frame_id]
        self.img_view.draw_rois()

    def process_predictions(self):
        """
        Process added suture sorting predictions in order to show them to the user.

        Steps to take before the received predictions can be visualized as ROIs on the data:

        Resolve duplicate IDs
            In case that the ID with maximum probability is not only ocurring once in the predictions for this frame,
            the conflict needs to be resolved.

        Convert unsorted sortings per frame to sorted suture ids
            When adding the predictions, we created a list of predictions for each frame, but appended them unsorted to
            that list. We now have to sort them based on their ID probabilities into the grid.

        .. warning:: Deprecated! Was replaced in favor of :func:`postprocessing.process_suture_sortings` to keep all
            processing functions better organized.

        .. todo:: Remove this function!
        """
        self.processed_predictions = []
        self.pred_roi_stack = [[QPointF(-1, -1)] * len(self.roi_stack[0]) for _ in range(self.num_frames)]

        rows = self.grid_widget.num_rows
        cols = self.grid_widget.num_cols
        sutures = rows * cols
        discarded_sortings = []
        sorting_preds = deepcopy(self.sorting_predictions)  # Create copy to not manipulate saved predictions
        for frame_ctr, frame_sortings in enumerate(sorting_preds):
            left_sortings = [None] * sutures
            right_sortings = [None] * sutures
            for pred in frame_sortings:
                if np.max(pred.probabilities) <= 0.0:
                    discarded_sortings.append(pred)
                    continue
                id = pred.pred_id
                sort_list = left_sortings if pred.side == 'left' else right_sortings
                # If no other pred has claimed the predicted position, simply assign it to this pred
                if not sort_list[id]:
                    sort_list[id] = pred
                # Otherwise, compare the probabilities and give the position to the pred with higher probability, the
                # lower probability pred will be added back to the list of unprocessed peaks with the probability for
                # the now known invalid pred set to its negative value to invalidate it (but allow some debugging).
                else:
                    this_prob = pred.probabilities[id]
                    that_prob = sort_list[id].probabilities[id]
                    if this_prob > that_prob:
                        that_pred = sort_list[id]
                        that_pred.probabilities[id] *= -1
                        new_id = np.argmax(that_pred.probabilities)
                        mod_pred = SortingPrediction(new_id, that_pred.y, that_pred.x, that_pred.frame, that_pred.side,
                                                     that_pred.probabilities)
                        sort_list[id] = pred
                        frame_sortings.append(mod_pred)
                    else:
                        pred.probabilities[id] *= -1
                        new_id = np.argmax(pred.probabilities)
                        mod_pred = SortingPrediction(new_id, pred.y, pred.x, pred.frame, pred.side, pred.probabilities)
                        frame_sortings.append(mod_pred)

            self.processed_predictions.append(left_sortings + right_sortings)
            qpoint_predictions = [QPointF(p.x, p.y) if p else QPointF(-1, -1) for p in left_sortings + right_sortings]
            self.pred_roi_stack[frame_ctr] = qpoint_predictions

        self.roi_stack = self.pred_roi_stack.copy()
        self.img_view.roi_positions = self.roi_stack[self.cur_frame_id]
        self.img_view.draw_rois()

    def get_image(self):
        """
        Applies image enhancement with a gaussian filter if enabled and returns the current image.
        """
        if self.gaussian:
            self.stack = sk_gaussian(self.stack)
        else:
            self.stack = np.copy(self.stack_cache)

        if self.num_frames > 1:
            img = self.stack[self.cur_frame_id]
        else:
            img = self.stack

        return img

    def update_stack(self):
        """
        Updates the stack by calling get_image() and updates the image view.
        """
        self.img_view.setImage(self.get_image())

    def reassign_roi_global(self, roi_id, new_id) -> None:
        """
        Perform ROI reassignment for all frames.

        :param int roi_id: ID of ROI to reassign to new posiiton.
        :param int new_id: ID of grid position to assign ROI to.
        """
        for frame_idx in range(self.num_frames):
            # The current frame is directly handled in the ImageView class to properly update the visuals, here only
            # the other frames need to be edited
            if frame_idx == self.cur_frame_id:
                continue
            edit_roi_assignment(self.roi_stack[frame_idx], roi_id, new_id, method='replace')

    def calculate_surface(self, calibration) -> None:
        """
        Use currently placed ROI annotations :attr:´roi_stack` and available transformation matrix :attr:´surface_f` to
        reconstruct 3D surface points stored in :attr:`surf_points`.

        :param dict calibration: Calibration data used for creating transformation matrix.
        """
        self.surf_points = surface.get_3d_points(self.roi_stack, self.surface_f, calibration,
                                                 self.grid_widget.num_rows, self.grid_widget.num_cols)

    def show_surface(self) -> None:
        """
        Show the reconstructed 3D surface in a :class:`SurfaceViewer` window.

        Showing the surface requires previous calculation of :attr:`surf_points` based on annotations and calibration
        data.
        """
        if self.surf_viewer is None:
            self.surf_viewer = SurfaceViewer(self.grid_widget.num_rows, self.grid_widget.num_cols, parent=self)
            # self.surf_viewer = SpaghettiViewer(self.grid_widget.num_rows, self.grid_widget.num_cols)
            self.surf_viewer.close_signal.connect(self._surface_viewer_closed)
        self.surf_viewer.set_data(self.surf_points)
        self.surf_viewer.show_frame(self.cur_frame_id)
        self.surf_viewer.show()

    def _surface_viewer_closed(self) -> None:
        """
        Slot called when surface viewer window is closed to also destroy its reference.

        Resetting the :attr:`surf_viewer` reference back to None is necessary to prevent re-showing the surface viewer
        window as soon as any action is performed that is connected to it (e.g. switching frame, switching active
        ROI,...)
        """
        self.surf_viewer = None


class ROI(pg.EllipseROI):
    """
    **Bases:** :class:`EllipseROI <pyqtgraph.EllipseROI>`

    Custom ROI class based on pyqtgraph's ROI classes. Used to easily place the ROI object's origin in the center of the
    ROI shape (pyqtgraph usually has the origin at the top-left).
    """
    ROI_color = 'k'  #: Color for placed ROI objects that are currently not active
    Active_color = 'g'  #: Color for placed ROI object that is currently active.
    #: Colors for encoding row or column membership of ROI.
    Coded_colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000', '#D2E23F', '#6BFBC4']
    TranslateSnap = False  #: Boolean setting forcing ROI position to have full-pixel coordinates.
    sig_remove_request = pyqtSignal(int)  #: Connecting to context option for removal, will emit the ROI ID
    sig_reassign_request = pyqtSignal(dict)  #: Connecting to context option for reassignment

    def __init__(self, pos, size, roi_id, visible=True, color_coded=False, **args):
        """
        Create ROI circle object.

        Placing the ROI's center at the specified position is implemented by calculating the top-left corner for the
        specified position and passing that as position argument to the constructor of :class:`pyqtgraph.ROI`.

        :param pos: Position coordinates as (y, x). This is the center of the ROI circle.
        :type pos: Iterable
        :param size: Diameter of ROI circle.
        :type size: int
        :param roi_id: Identifier for ROI.
        :type roi_id: int
        :param visible: Boolean setting if ROI is visible or not.
        :type visible: bool
        :param args: Further arguments passed on to constructor for :class:`pyqtgraph.ROI`.
        :type args: dict
        """
        self.path = None
        #: Top-left coordinates (y, x) of ROI calculated from passed *pos* argument.
        self.position = [pos[0] - size[0]/2., pos[1] - size[1]/2.]
        if self.TranslateSnap:
            self.position = [np.rint(coord) for coord in self.position]
        pg.ROI.__init__(self, self.position, size, translateSnap=self.TranslateSnap, **args)

        self.aspectLocked = True
        self.roi_id = roi_id  #: Index into 1D array --> Use this to access e.g. ImageView.roi_objects[roi.roi_idx]
        self.size = size  #: Diameter of ROI circle.
        self.visible = visible  #: Boolean setting if ROI is visible or not.
        self.color_coded = color_coded  #: Whether the ROI position should be color coded. One of [False, 'row', 'col']

        # self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setPen(self.get_base_color())

    def _makePen(self) -> None:
        """
        Overwritten method of :class:`pyqtgraph.ROI` to change color and width of ROI to be more visible on bright
        backgrounds when hovered.
        """
        # Generate the pen color for this ROI based on its current state.
        if self.mouseHovering:
            return pg.mkPen(width=5, color='m')
        else:
            return self.pen

    def centered_pos(self) -> list:
        """
        Return the position of the ROI center.

        The implementation gets the current position from the object (important for updating the position after a drag
        event!) and calculates the center position of the ROI for it.

        :return: The current position centered w.r.t. to the ROI shape.
        """
        pos = self.pos()
        pos = [pos[0] + self.size[0]/2., pos[1] + self.size[1]/2.]
        return pos

    def highlight(self) -> None:
        """
        Highlight the ROI by coloring it in :attr:`Active_color`. Will draw the ROI even if :attr:`visible` is set to
        False.
        """
        self.setPen(pg.mkPen(width=5, color=ROI.Active_color))

    def dehighlight(self) -> None:
        """
        Return the ROI back to its normal coloring of :attr:`ROI_color`. If the ROI is currently set to
        :attr:`visible` = False, will make it invisible instead.
        :return:
        """
        if self.visible:
            self.setPen(self.get_base_color())
        else:
            self.setPen(QPen(Qt.NoPen))

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        else:
            self.sigClicked.emit(self, ev)

    def raiseContextMenu(self, ev):
        menu = self.getContextMenus()

        pos = ev.screenPos()
        menu.popup(QPoint(pos.x(), pos.y()))
        return True

    def getContextMenus(self, event=None):
        if self.menu is None:
            self.menu = QMenu()

            # Just a label used as a header title for the menu
            title_action = QWidgetAction(self.menu)
            title = QLabel("ROI " + str(self.roi_id))
            title_action.setDefaultWidget(title)
            self.menu.addAction(title_action)
            self.menu.title = title

            # Action for deleting the selected ROI
            remove_action = QAction("Remove ROI", self.menu)
            remove_action.triggered.connect(self.remove_requested)
            self.menu.addAction(remove_action)
            self.menu.green = remove_action

            # Action taking the currently set values from `row_action` and `col_action` and reassigning the ROI using
            # these as new position.
            id_action = QAction("Reassign to position:", self.menu)
            self.menu.addAction(id_action)
            self.menu.green = id_action

            # Selection of new row for reassignment
            row_action = QWidgetAction(self.menu)
            row_box = QSpinBox()
            row_box.setPrefix('row ')
            row_box.setMinimum(0)
            row_box.setObjectName('row_box')
            row_action.setDefaultWidget(row_box)
            self.menu.addAction(row_action)
            self.menu.row_action = row_action
            self.menu.row_box = row_box

            # Selection of new column for reassignment
            col_action = QWidgetAction(self.menu)
            col_box = QSpinBox()
            col_box.setPrefix('col ')
            col_box.setMinimum(0)
            col_box.setObjectName('col_box')
            col_action.setDefaultWidget(col_box)
            self.menu.addAction(col_action)
            self.menu.col_action = col_action
            self.menu.col_box = col_box

            id_action.triggered.connect(lambda: self.reassign_requested(self.menu.row_box.value(),
                                                                        self.menu.col_box.value()))

        # Assign values outside of menu creation so they get set even after the menu has been created once.
        self.menu.findChild(QWidget, 'row_box').setValue(self.roi_id // 5)
        self.menu.findChild(QWidget, 'col_box').setValue(self.roi_id % 5)

        return self.menu

    def remove_requested(self) -> None:
        """
        Trigger signal indicating to receiver that ROI deletion is requested.

        Deletions are handled by the :class:`ImageView` object. Triggering the deletion via the ROI itself needs the
        ROI to inform the :class:`ImageView` that it needs to be deleted.
        """
        self.sig_remove_request.emit(self.roi_id)

    def reassign_requested(self, row, col) -> None:
        """
        Trigger signal indicating to receiver that ROI should be reassigned to new grid position.

        Reassignments are handled by :class:`ImageView` which needs to be informed about the wanted reassignment through
        the signal-slot system.

        :param int row: New row to assign.
        :param int col: New column to assign.
        """
        self.sig_reassign_request.emit({
            'roi_id': self.roi_id,
            'new_row': row,
            'new_col': col
        })

    def get_base_color(self) -> QPen:
        """
        Create and return the default :pyqt:`QPen <qpen>` for this ROI.

        When color coding of ROIs is not activated (see :class:`ImageView`), the base color is :attr:`ROI_color`
        (defaults to black).
        For active color coding, the base color is determined by row or column membership for 'row' and 'col' coding
        respectively.
        The colors used for encoding rows and columns are defined in :attr:`Coded_colors`.

        :return: :pyqt:`QPen <qpen>` instance for coloring this ROI.

        :raises ValueError: if :attr:`color_coded` is set to other than [False, 'row', 'col'].
        """
        if not self.color_coded:
            return pg.mkPen(cosmetic=True, width=3, color=ROI.ROI_color)

        if self.color_coded == 'row':
            offset_id = self.roi_id if self.roi_id < 35 else self.roi_id - 35
            color_id = offset_id // 5
        elif self.color_coded == 'col':
            color_id = self.roi_id % 5
        else:
            raise ValueError(f"Unknown value for color_coded {self.color_coded} encountered!")

        return pg.mkPen(cosmetic=True, width=3, color=self.Coded_colors[color_id])


class ConfirmDialog(QDialog):
    """
    **Bases:** :pyqt:`QDialog <qdialog>`

    Simple Dialog window showing a message to user and asking for confirm or cancel action.

    Window will display a single message string along 'Accept' and 'Cancel' buttons. Calling :meth:`exec_` on this
    ConfirmDialog will return True if user clicks 'Accept' and False otherwise.
    """
    def __init__(self, msg):
        """
        Create and show a Dialog window to the user displaying *msg* and asking to confirm or cancel.

        :param msg: Message to display in dialog window.
        :type msg: str
        """
        super().__init__()

        self.setWindowTitle("Confirm Action")

        qbtn = QDialogButtonBox.Yes | QDialogButtonBox.Cancel

        text = QLabel(msg)

        self.buttonBox = QDialogButtonBox(qbtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(text, alignment=Qt.AlignLeft)
        self.layout.addWidget(self.buttonBox, alignment=Qt.AlignLeft)
        self.setLayout(self.layout)

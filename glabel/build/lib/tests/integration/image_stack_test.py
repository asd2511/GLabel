import sys
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PyQt5.QtCore import QPointF, QPoint
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from PyQt5.Qt import Qt

from gui.main_window import Main, LoginWindow
from gui.image_widget import ImageStack, ImageView, ROI
from gui.grid_widget import ButtonGrid, PixelGrid
from gui.config import settings, keybinds, mousebinds
from tests.unit.main_window_test import MainWindowTest

app = QApplication(sys.argv)


class ImageStackIntegrationTest(unittest.TestCase):

    username = MainWindowTest.username
    screen_height = MainWindowTest.screen_height
    screen_width = MainWindowTest.screen_width

    test_file = list(Path(os.getcwd()).rglob('*testdata.tiff'))[0].as_posix()

    test_dims = [107, 768, 768, 3]

    place_btn = mousebinds['PlaceROI']['button']
    place_mod = mousebinds['PlaceROI']['modifier']

    @classmethod
    def setUpClass(cls) -> None:
        """
        Perform the same preparations as for testing the Main class.

        This will deal with the config.json path and temporarily remove it to force the GUI into using the default
        configurations as read from :mod:`gui.config`.
        """
        MainWindowTest.setUpClass()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Perform the same teardown actions as for testing the Main class.

        This will resotre the at setup deleted user configuration.
        """
        MainWindowTest.tearDownClass()
        super().tearDownClass()

    @patch('gui.main_window.LoginWindow')
    def setUp(self, mock_login) -> None:
        super().setUp()
        mock_screen = Mock()
        mock_screen.width.return_value = self.screen_width
        mock_screen.height.return_value = self.screen_height

        login = mock_login.return_value
        login.name_edit.text.return_value = self.username

        # Start the GUI
        self.main = Main(mock_screen)
        # Open a path and get all useful references
        self.main.open(self.test_file)
        self.image_stack: ImageStack = self.main.image_stack  # Get the reference to the initialized ImageStack
        self.image_view: ImageView = self.image_stack.img_view  # Get the reference to the initialized ImageView
        self.grid_widget: ButtonGrid = self.image_stack.grid_widget  # Get the reference to the initialized ButtonGrid
        self.grid_widget.set_mode(1)  # Set mode to progress row-wise

    def test_init(self):
        """Coming from the setUp, the GUI should be initialized according to the image data and the default settings"""
        # Image data should be loaded into ImageStack and ImageView
        self.assertCountEqual(self.image_stack.stack.shape, self.test_dims)  # Image data should have correct dimensions
        self.assertCountEqual(self.image_view.image.shape, self.test_dims[1:])  # ImageView should have single frame
        self.assertEqual(self.image_stack.num_frames, self.test_dims[0])
        np.testing.assert_array_equal(self.image_stack.stack[0], self.image_view.image)  # Should display first frame
        # Grid widget should be initialized according to config
        self.assertEqual(self.grid_widget.num_rows, settings['NumRows'])
        self.assertEqual(self.grid_widget.num_cols, settings['NumCols'])
        self.assertEqual(self.grid_widget.cur_active, (0, 0))  # First grid position should be active

    def test_place_roi(self):
        """Placing an ROI should create an ROI object at the position and mark it as placed in the grid"""
        roi_size = (10, 10)  # This might have to be made dynamic in the future (as in read from config path)
        self.grid_widget.set_snaking(False)  # Turn off snaking to make tracking of active position easier

        # Place ROI for all grid positions (*2 bc of stereo view)
        for roi_ctr in range(self.grid_widget.num_rows * self.grid_widget.num_cols * 2):
            click_y, click_x = 100 + roi_ctr, 100 + roi_ctr  # Introduce 1px shift per placement
            click_qpoint = QPoint(click_y, click_x)
            # Place the ROI
            QTest.mouseClick(self.image_view, self.place_btn, self.place_mod, click_qpoint)

            # ImageView should have saved position
            self.assertEqual(self.image_view.roi_positions[roi_ctr], click_qpoint)
            roi: ROI = self.image_view.roi_objects[roi_ctr]
            self.assertIsInstance(roi, ROI)  # ImageView should have placed ROI object
            roi_qpoint = roi.pos().toQPoint()
            # Position value of actual ROI object is shifted in relation to its size (bc. position of ROIs in pyqtgraph
            # always refers to their top-left corner)
            self.assertEqual(roi_qpoint, QPoint(click_y - roi_size[0] / 2, click_x - roi_size[1] / 2))
            if roi_ctr < self.grid_widget.num_rows * self.grid_widget.num_cols * 2 - 1:
                self.assertEqual(self.grid_widget.get_cur_1d(), roi_ctr + 1)  # Next grid position should be active
            else:
                self.assertEqual(self.grid_widget.get_cur_1d(), roi_ctr)  # No new grid position after last one

        # Continuing to place ROIs should only overwrite the last one and not create new ones
        roi_ctr += 1
        click_y, click_x = 100 + roi_ctr, 100 + roi_ctr
        click_qpoint = QPoint(click_y, click_x)
        QTest.mouseClick(self.image_view, self.place_btn, self.place_mod, click_qpoint)
        # ImageView should have overwritten the last position
        self.assertEqual(self.image_view.roi_positions[roi_ctr-1], click_qpoint)
        roi_qpoint = self.image_view.roi_objects[roi_ctr-1].pos().toQPoint()
        self.assertEqual(roi_qpoint, QPoint(click_y - roi_size[0] / 2, click_x - roi_size[1] / 2))
        self.assertEqual(self.grid_widget.get_cur_1d(), roi_ctr - 1)  # Should not have changed grid position

    def test_switch_frame_empty(self):
        """Switching frames with no placed ROIs should simply change the displayed image"""
        next_signal = self.image_stack.shortcuts['NextFrame'].activated
        prev_signal = self.image_stack.shortcuts['PreviousFrame'].activated

        # Should start with first frame displayed
        np.testing.assert_array_equal(self.image_view.imageItem.image, self.image_stack.stack[0])

        # Trying to switch to previous frame should do nothing
        prev_signal.emit()
        np.testing.assert_array_equal(self.image_view.imageItem.image, self.image_stack.stack[0])

        # Switching frames until end
        for frame_ctr in range(1, self.test_dims[0] - 1):
            next_signal.emit()
            np.testing.assert_array_equal(self.image_view.imageItem.image, self.image_stack.stack[frame_ctr])

        # Keeping on switching frames should do nothing
        next_signal.emit()
        np.testing.assert_array_equal(self.image_view.imageItem.image, self.image_stack.stack[-1])

        # Switching frames back to front
        for frame_ctr in range(self.test_dims[0] - 2, -1, -1):
            prev_signal.emit()
            np.testing.assert_array_equal(self.image_view.imageItem.image, self.image_stack.stack[frame_ctr],
                                          f"Wrong image encountered at frame_ctr {frame_ctr}:"
                                          f"\nImageView.currentIndex is {self.image_view.currentIndex}")

    def test_switch_frame_placed(self):
        """Switching frames with placed ROIs should save the placements and switch to the new frame"""
        base_y, base_x = 100, 100
        next_signal = self.image_stack.shortcuts['NextFrame'].activated
        prev_signal = self.image_stack.shortcuts['PreviousFrame'].activated

        # Go over all frames, always placing one ROI which should be saved when moving from that frame
        for frame_ctr in range(0, self.test_dims[0]):
            # Place a single ROI
            click_qpoint = QPoint(base_y + frame_ctr, base_x + frame_ctr)
            QTest.mouseClick(self.image_view, self.place_btn, self.place_mod, click_qpoint)
            # Switch to next frame
            next_signal.emit()
            # Check if ROI was saved on previous frame (will always be the first ROI bc grid resets on frame switch)
            self.assertEqual(self.image_stack.roi_stack[frame_ctr][0], click_qpoint,
                             f"frame_ctr: {frame_ctr}, ImageView.currentIndex: {self.image_view.currentIndex}")

        # Going backwards should load the saved positions and display them as ROIs
        for frame_ctr in range(self.test_dims[0] - 1, -1, -1):
            click_qpoint = QPoint(base_y + frame_ctr, base_x + frame_ctr)
            self.assertEqual(self.image_view.roi_positions[0], click_qpoint,
                             f"frame_ctr: {frame_ctr}, ImageView.currentIndex: {self.image_view.currentIndex}")
            self.assertIsInstance(self.image_view.roi_objects[0], ROI)
            prev_signal.emit()

    def test_placed_save_load(self):
        """Saving to path with placed ROIs should save all positions to a .rois path"""
        save_dir = Path(self.test_file).parent.as_posix()
        save_file = save_dir + '/testsave.rois'
        base_y, base_x = 10, 10
        next_signal = self.image_stack.shortcuts['NextFrame'].activated
        roi_range = self.grid_widget.num_rows * self.grid_widget.num_cols * 2 // 10  # 10% of all ROI placements
        self.grid_widget.set_snaking(False)  # Turn off snaking to make checks easier (linear iteration)

        # Go over all frames and place 10% of ROIs
        for frame_ctr in range(0, self.test_dims[0]):
            for roi_ctr in range(roi_range):
                click_qpoint = QPoint(base_y + roi_ctr, base_x + frame_ctr)
                QTest.mouseClick(self.image_view, self.place_btn, self.place_mod, click_qpoint)
            next_signal.emit()

        # Save the placement data
        self.main.save_as(save_file)
        self.assertTrue(os.path.isfile(save_file))  # Should have saved the path

        # Load the path again
        self.main.open(save_file)
        # Restore all references that are lost when 'open' reinitializes all widgets
        new_image_stack: ImageStack = self.main.image_stack
        new_image_view: ImageView = new_image_stack.img_view
        new_grid_widget: ButtonGrid = new_image_stack.grid_widget
        next_signal = new_image_stack.shortcuts['NextFrame'].activated

        # All placements should be loaded
        for frame_ctr in range(0, self.test_dims[0]):
            for roi_ctr in range(roi_range):
                click_qpoint = QPoint(base_y + roi_ctr, base_x + frame_ctr)
                new_grid_widget.set_active((0, 0))
                self.assertEqual(new_image_view.roi_positions[roi_ctr], click_qpoint,
                                 f"frame_ctr {frame_ctr}, roi_ctr {roi_ctr}")  # Position should be loaded
                self.assertIsInstance(new_image_view.roi_objects[roi_ctr], ROI)  # ROIs should be displayed
            next_signal.emit()  # Go to next frame

        # Delete the created save_file
        os.remove(save_file)


if __name__ == '__main__':
    unittest.main()

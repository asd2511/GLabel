import sys
import itertools as it
from typing import Generator, Tuple
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QGridLayout, QButtonGroup, QApplication, QFrame, QVBoxLayout
from PyQt5.QtCore import Qt, QPointF
from pyqtgraph import GraphicsWindow, ViewBox

from gui.grid_widget import GridWidget, ButtonGrid, PixelGrid

app = QApplication(sys.argv)


class ButtonGridTest(unittest.TestCase):

    # Default values necessary for initializing grid
    num_rows = 7
    num_cols = 5
    stereo = True

    l_rows = 31
    l_cols = 31
    l_stereo = False

    # Path to image used for styling radio button of missing placement
    missing_img = 'gui/radio_unchecked.png'

    def setUp(self) -> None:
        self.mono_grid = ButtonGrid(self.l_rows, self.l_cols, self.l_stereo)
        self.stereo_grid = ButtonGrid(self.num_rows, self.num_cols, self.stereo)

    @patch.object(ButtonGrid, 'set_active')
    @patch.object(ButtonGrid, 'build_grid')
    @patch.object(GridWidget, 'build_path')
    def test_init(self, mock_build_path, mock_build_grid, mock_set_active):
        """Instance variables should be set to defaults after instantiating"""
        # Instantiate ButtonGrid with defaults
        self.grid = ButtonGrid(self.num_rows, self.num_cols, self.stereo)
        # Necessary values as set by tester
        self.assertEqual(self.grid.num_rows, self.num_rows)
        self.assertEqual(self.grid.num_cols, self.num_cols)
        self.assertEqual(self.grid.stereo, self.stereo)
        # Default values omitted in initialization
        self.assertEqual(self.grid.mode, 0)
        self.assertEqual(self.grid.snaking, True)
        self.assertEqual(self.grid.begin_top, False)
        self.assertEqual(self.grid.cur_active, (0, 0))
        # Instantiated instance objects
        self.assertIsInstance(self.grid.grid_layout, QGridLayout)
        self.assertIsInstance(self.grid.grid_buttons, QButtonGroup)
        # Initial calls to methods
        mock_build_path.assert_called_once()
        mock_build_grid.assert_called_once()
        mock_set_active.assert_called_with((0, 0))

        mock_set_active.reset_mock()
        # Instantiate ButtonGrid with parameter begin_top set to True --> Should set top left position to active
        self.grid = ButtonGrid(self.num_rows, self.num_cols, self.stereo, begin_top=True)
        self.assertEqual(self.grid.begin_top, True)
        self.assertEqual(self.grid.cur_active, (self.num_rows-1, 0))
        mock_set_active.assert_called_with((self.num_rows-1, 0))

    @patch.object(ButtonGrid, 'set_active')
    @patch.object(ButtonGrid, 'build_stereo_radio')
    @patch.object(ButtonGrid, 'build_mono_radio')
    def test_build_grid(self, mock_build_mono, mock_build_stereo, mock_set_active):
        """Calling build_grid should in turn call build_stereo_radio/build_mono_radio depending on stereo parameter"""
        self.grid = ButtonGrid(self.num_rows, self.num_cols, stereo=True)
        mock_build_stereo.assert_called_once()  # Should build stereo
        mock_build_mono.assert_not_called()  # Should not build mono
        mock_set_active.assert_called_with((0, 0))  # Should set (0, 0) to active

        mock_build_stereo.reset_mock()
        mock_build_mono.reset_mock()
        mock_set_active.reset_mock()

        self.grid = ButtonGrid(self.num_rows, self.num_cols, stereo=False)
        mock_build_stereo.assert_not_called()  # Should not build stereo
        mock_build_mono.assert_called_once()  # Should not build mono
        mock_set_active.assert_called_with((0, 0))  # Should set (0, 0) to active

    def test_build_mono_radio(self):
        """Calling build_mono_radio should build a single grid of QRadioButtons depending on grid dimensions"""
        # Instantiating a mono ButtonGrid will call build_mono_radio to build its grid
        self.grid = ButtonGrid(self.num_rows, self.num_cols, stereo=False)
        buttons = self.grid.grid_buttons
        layout = self.grid.grid_layout

        # Grid should be size rows*cols
        self.assertEqual(len(buttons.buttons()), self.num_rows * self.num_cols)
        self.assertEqual(layout.count(), self.num_rows * self.num_cols)
        self.assertEqual(buttons.checkedId(), 0)  # First button should be checked
        # All buttons should be styled with the image for missing positions
        for btn in buttons.buttons():
            self.assertTrue(self.missing_img in btn.styleSheet())
        # Button-IDs should start at 0 and increase row-wise (without snaking), no matter what type of pathing is set
        id_counter = 0
        for row in range(self.num_rows, 0, -1):  # (Rows range from 1 to 7 because within layout)
            for col in range(self.num_cols):
                btn = layout.itemAtPosition(row, col).widget()
                btn.setChecked(True)
                checked_id = buttons.checkedId()
                self.assertEqual(checked_id, id_counter)
                id_counter += 1

    def test_build_mono_radio_2nd_side(self):
        """Calling build_mono_radio with argument 'second_side' as True should create buttons for second half of grid"""
        # Instantiating stereo ButtonGrid
        self.grid = ButtonGrid(self.num_rows, self.num_cols, stereo=True)
        # Delete the placed buttons
        self.grid.grid_layout = QGridLayout()
        self.grid.grid_buttons = QButtonGroup()

        # Call to create only the 'second side' of the grid buttons
        self.grid.build_mono_radio(second_side=True)
        buttons = self.grid.grid_buttons
        layout = self.grid.grid_layout

        # Grid should be size rows*cols
        self.assertEqual(len(buttons.buttons()), self.num_rows * self.num_cols)
        self.assertEqual(layout.count(), self.num_rows * self.num_cols)
        # All buttons should be styled as missing
        for btn in buttons.buttons():
            self.assertTrue(self.missing_img in btn.styleSheet())
        # Button-IDs should start at rows*cols and increase row-wise (without snaking), no matter what type of pathing
        id_counter = self.num_rows * self.num_cols
        for row in range(self.num_rows, 0, -1):  # Rows range from 1 to 7
            for col in range(self.num_cols+1, self.num_cols*2+1):
                btn = layout.itemAtPosition(row, col).widget()
                btn.setChecked(True)
                checked_id = buttons.checkedId()
                self.assertEqual(checked_id, id_counter)
                id_counter += 1

    @patch.object(ButtonGrid, 'set_active')
    @patch.object(ButtonGrid, 'build_mono_radio')
    def test_build_stereo_radio(self, mock_build_mono, mock_set_active):
        """Calling build_stereo_radio should build a two-sided grid of QRadioButtons depending on grid dimension"""
        self.grid = ButtonGrid(self.num_rows, self.num_cols, stereo=True)

        self.assertEqual(mock_build_mono.call_count, 2)  # Called once for each side
        self.assertEqual(self.grid.grid_layout.count(), 1)  # Build_mono is patched --> Only separator is added
        self.assertIsInstance(self.grid.grid_layout.itemAt(0).widget(), QFrame)

    @patch.object(ButtonGrid, 'activate_signal')
    def test_set_active_stereo(self, mock_signal):
        """Calling set_active on stereo grid should set the specified grid position as the currently active position"""
        # Activate all positions on left side of grid once
        for coords in list(it.product(range(self.num_rows), range(self.num_cols))):
            self.stereo_grid.set_active(coords)
            activate_id = coords[0]*self.num_cols+coords[1]
            self.assertEqual(self.stereo_grid.grid_buttons.checkedId(), activate_id)
            mock_signal.emit.assert_called_with(activate_id)  # Emitting grid_id
        # Activate all positions on right side of grid once
        offset = self.num_rows * self.num_cols
        for coords in list(it.product(range(self.num_rows), range(self.num_cols, self.num_cols*2))):
            self.stereo_grid.set_active(coords)
            col = coords[1] - self.num_cols
            activate_id = coords[0]*self.num_cols+col+offset
            self.assertEqual(self.stereo_grid.grid_buttons.checkedId(), activate_id)
            mock_signal.emit.assert_called_with(activate_id)  # Emitting grid_id

        # Grid indices are sometimes passed as np.int32 types to set_active --> Should handle np.int32 values
        self.stereo_grid.set_active(np.int32(3))
        self.assertEqual(self.stereo_grid.grid_buttons.checkedId(), 3)

        # Other types or dimension mismatches should raise error
        self.assertRaises(TypeError, self.stereo_grid.set_active, [0])
        self.assertRaises(TypeError, self.stereo_grid.set_active, [0, 0, 0])
        self.assertRaises(TypeError, self.stereo_grid.set_active, 1.5)

    @patch.object(ButtonGrid, 'activate_signal')
    def test_set_active_mono(self, mock_signal):
        """Calling set_active on mono grid should set the specified grid position as the currently active position"""
        # Activate all positions in grid once
        for coords in list(it.product(range(self.l_rows), range(self.l_cols))):
            self.mono_grid.set_active(coords)
            activate_id = coords[0]*self.l_cols+coords[1]
            self.assertEqual(self.mono_grid.grid_buttons.checkedId(), activate_id)
            mock_signal.emit.assert_called_with(activate_id)

    @patch.object(ButtonGrid, 'activate_signal')
    @patch.object(ButtonGrid, 'deactivate_signal')
    def test_on_click(self, mock_deactivate_signal, mock_activate_signal):
        """on_click being called on toggle of a QRadioButton should emit the ids of activated and deactivated button"""
        # Click once on all buttons --> Should emit the newly activated and deactivated ids in the respective signal
        # For stereo grid
        for btn_id in range(1, self.num_rows*self.num_cols*2):
            QTest.mouseClick(self.stereo_grid.grid_buttons.button(btn_id), Qt.LeftButton, Qt.NoModifier)
            mock_deactivate_signal.emit.assert_called_with(btn_id - 1)
            mock_activate_signal.emit.assert_called_with(btn_id)
        # For mono grid
        for btn_id in range(1, self.l_rows*self.l_cols):
            QTest.mouseClick(self.mono_grid.grid_buttons.button(btn_id), Qt.LeftButton, Qt.NoModifier)
            mock_deactivate_signal.emit.assert_called_with(btn_id - 1)
            mock_activate_signal.emit.assert_called_with(btn_id)

    def test_mark_placed(self):
        """Calling mark_placed for a grid position should set its styling and placement value accordingly"""
        # For stereo grid
        for btn_id in range(self.num_rows*self.num_cols*2):
            self.assertFalse(self.stereo_grid.placements[btn_id])  # All initialized as not placed
            self.stereo_grid.mark_placed(btn_id)
            self.assertFalse(self.missing_img in self.stereo_grid.grid_buttons.button(btn_id).styleSheet())
            self.assertTrue(self.stereo_grid.placements[btn_id])
        # For mono grid
        for btn_id in range(self.l_rows*self.l_cols):
            self.assertFalse(self.mono_grid.placements[btn_id])
            self.mono_grid.mark_placed(btn_id)
            self.assertFalse(self.missing_img in self.mono_grid.grid_buttons.button(btn_id).styleSheet())
            self.assertTrue(self.mono_grid.placements[btn_id])

    def test_mark_missing(self):
        """Calling mark_placed for a grid position should set its styling and placement value accordingly"""
        # For stereo grid
        for btn_id in range(self.num_rows*self.num_cols*2):
            self.stereo_grid.mark_placed(btn_id)  # Mark it as placed
            self.assertTrue(self.stereo_grid.placements[btn_id])  # Check if it is set to placed
            self.stereo_grid.mark_missing(btn_id)  # Mark it as missing again
            self.assertTrue(self.missing_img in self.stereo_grid.grid_buttons.button(btn_id).styleSheet())
            self.assertFalse(self.stereo_grid.placements[btn_id])
        # For mono grid
        for btn_id in range(self.l_rows*self.l_cols):
            self.mono_grid.mark_placed(btn_id)
            self.assertTrue(self.mono_grid.placements[btn_id])
            self.mono_grid.mark_missing(btn_id)
            self.assertTrue(self.missing_img in self.mono_grid.grid_buttons.button(btn_id).styleSheet())
            self.assertFalse(self.mono_grid.placements[btn_id])

    def test_get_cur_1d(self):
        """get_cur_id should return the integer index pointing to the currently active grid position"""
        # For stereo grid
        self.assertEqual(self.stereo_grid.get_cur_1d(), 0)  # (0, 0) should be initially active
        for btn_id in range(self.num_rows*self.num_cols*2):
            self.stereo_grid.set_active(btn_id)
            self.assertEqual(self.stereo_grid.get_cur_1d(), btn_id)
        # For mono grid
        self.assertEqual(self.mono_grid.get_cur_1d(), 0)
        for btn_id in range(self.l_rows*self.l_cols):
            self.mono_grid.set_active(btn_id)
            self.assertEqual(self.mono_grid.get_cur_1d(), btn_id)


class PixelGridTest(unittest.TestCase):

    # Default values necessary for initializing grid
    num_rows = 7
    num_cols = 5
    stereo = True

    l_rows = 31
    l_cols = 31
    l_stereo = False

    def setUp(self) -> None:
        self.stereo_grid = PixelGrid(self.num_rows, self.num_cols, self.stereo)
        self.mono_grid = PixelGrid(self.l_rows, self.l_cols, self.l_stereo)

    @patch.object(PixelGrid, 'set_active')
    @patch.object(PixelGrid, 'build_grid')
    @patch.object(GridWidget, 'build_path')
    def test_init(self, mock_build_path, mock_build_grid, mock_set_active):
        """Instance variables should be set to defaults after instantiation"""
        # Instantiate default PixelGrid
        self.grid = PixelGrid(self.num_rows, self.num_cols, self.stereo)
        # Necessary values as set by tester
        self.assertEqual(self.grid.num_rows, self.num_rows)
        self.assertEqual(self.grid.num_cols, self.num_cols)
        self.assertEqual(self.grid.stereo, self.stereo)
        # Default values omitted in initialization should be set to defaults
        self.assertEqual(self.grid.mode, 0)
        self.assertEqual(self.grid.snaking, True)
        self.assertEqual(self.grid.begin_top, False)
        self.assertEqual(self.grid.cur_active, (0, 0))
        # Instantiated instance objects
        self.assertIsInstance(self.grid.layout, QVBoxLayout)
        self.assertIsInstance(self.grid.grid_window, GraphicsWindow)
        self.assertIsInstance(self.grid.grid_view, ViewBox)
        self.assertIsNone(self.grid.grid_img)
        for i in range(2):
            self.assertIsNone(self.grid.map_markings['active'][i])
            self.assertIsNone(self.grid.map_markings['hover'][i])
        # Initial calls to methods
        mock_build_path.assert_called_once()
        mock_build_grid.assert_called_once()
        mock_set_active.assert_called_with((0, 0))

        mock_set_active.reset_mock()
        # Instantiate PixelMap with begin_top as True --> Should activate top left in __init__
        self.grid = PixelGrid(self.num_rows, self.num_cols, self.stereo, begin_top=True)
        self.assertTrue(self.grid.begin_top)
        self.assertEqual(self.grid.cur_active, (self.num_rows-1, 0))
        mock_set_active.assert_called_with((self.num_rows-1, 0))

    @patch('gui.grid_widget.pg.ImageItem')
    @patch.object(ViewBox, 'addItem')
    @patch.object(PixelGrid, 'build_map_img')
    def test_build_grid(self, mock_build_map_img, mock_add_item, mock_imageitem):
        """Calling build_grid should initialize the ViewBox with the map image and correct settings"""
        mock_imageitem.return_value = 'mocked ImageItem'

        # Delete already initialized values from setUp
        self.stereo_grid.grid_img = None
        self.mono_grid.grid_img = None
        self.stereo_grid.grid_view = self.stereo_grid.grid_window.addViewBox(lockAspect=True)
        self.mono_grid.grid_view = self.mono_grid.grid_window.addViewBox(lockAspect=True)

        self.stereo_grid.build_grid()
        # grid_view of PixelGrid should be initialized with image
        mock_build_map_img.assert_called_once()
        mock_add_item.assert_called_with('mocked ImageItem')
        self.assertTrue(self.stereo_grid.grid_view.autoRangeEnabled())  # Autorange should be enabled
        self.assertTrue(self.stereo_grid.grid_view.menuEnabled())  # Menu should be enabled to allow user to reset view

        mock_build_map_img.reset_mock()
        mock_add_item.reset_mock()

        # Test for mono grid
        self.mono_grid.build_grid()
        mock_build_map_img.assert_called_once()
        mock_add_item.assert_called_with('mocked ImageItem')
        self.assertTrue(self.mono_grid.grid_view.autoRangeEnabled())  # Autorange should be enabled
        self.assertTrue(self.mono_grid.grid_view.menuEnabled())  # Menu should be enabled to allow user to reset view

    def test_build_map_img(self):
        """Calling build_map_img should build and return a pixelmap depending on the used grid dimensionality"""
        stereo_img = self.stereo_grid.build_map_img()
        mono_img = self.mono_grid.build_map_img()
        # General assertions about type and shape
        # Images should be numpy arrays
        self.assertIsInstance(stereo_img, np.ndarray)
        self.assertIsInstance(mono_img, np.ndarray)
        # Should have three dimensions (y, x, clr)
        self.assertEqual(len(stereo_img.shape), 3)
        self.assertEqual(len(mono_img.shape), 3)
        # Should have dimensionality determined by grid dimensions
        self.assertEqual(stereo_img.shape, (self.num_rows*2+1, self.num_cols*4+3, 3))
        self.assertEqual(mono_img.shape, (self.l_rows*2+1, self.l_cols*2+1, 3))
        # See if actually enough positions are created in image
        self.assertEqual(sum(stereo_img[:, 1, 0] > 0), self.num_rows)
        self.assertEqual(sum(stereo_img[1, :, 0] > 0), self.num_cols * 2)
        self.assertEqual(sum(mono_img[:, 1, 0] > 0), self.l_rows)
        self.assertEqual(sum(mono_img[1, :, 0] > 0), self.l_cols)
        # Testing actual image content
        for row in range(stereo_img.shape[0]):
            for col in range(stereo_img.shape[1]):
                if row % 2 == 1 and col % 2 == 1:
                    if col != stereo_img.shape[1] // 2:
                        self.assertTrue(all(stereo_img[row, col, :] == PixelGrid.PixelColors['missing']))
        for row in range(mono_img.shape[0]):
            for col in range(mono_img.shape[1]):
                if row % 2 == 1 and col % 2 == 1:
                    self.assertTrue(all(mono_img[row, col, :] == PixelGrid.PixelColors['missing']))

    def test_map_to_pos_stereo(self):
        """Method map_to_pos should convert pixel coordinates of stereo PixelMap to grid positions"""
        # Valid grid coordinates and their corresponding grid positions
        valid_coords = [(1, 1), (1, 9), (13, 9), (13, 13), (1, 21)]
        valid_positions = [(0, 0), (0, 4), (6, 4), (6, 5), (0, 9)]
        # Invalid grid coordinates for which (-1, -1) should be returned
        invalid_coords = [(0, 0), (2, 9), (1, 11), (13, 20), (14, 22)]

        for coords, pos in zip(valid_coords, valid_positions):
            y, x = coords
            self.assertEqual(self.stereo_grid.map_to_pos(y, x), pos)
        for coords in invalid_coords:
            y, x = coords
            self.assertEqual(self.stereo_grid.map_to_pos(y, x), (-1, -1))

    def test_map_to_pos_mono(self):
        """Method map_to_pos should convert pixel coordinates of mono PixelMap to grid positions"""
        valid_coords = [(1, 1), (1, 9), (23, 23), (61, 61)]
        valid_positions = [(0, 0), (0, 4), (11, 11), (30, 30)]
        invalid_coords = [(0, 0), (20 , 20), (28, 56), (62, 62)]

        for coords, pos in zip(valid_coords, valid_positions):
            y, x = coords
            self.assertEqual(self.mono_grid.map_to_pos(y, x), pos)
        for coords in invalid_coords:
            y, x = coords
            self.assertEqual(self.mono_grid.map_to_pos(y, x, ), (-1, -1))

    def test_pos_to_map_stereo(self):
        """Method pos_to_map should convert grid positions of stereo PixelMap to pixel coordinates"""
        positions = [(0, 0), (3, 5), (1, 7), (6, 10)]
        coords = [(1, 1), (7, 13), (3, 17), (13, 23)]

        for pos, coord in zip(positions, coords):
            row, col = pos
            self.assertEqual(self.stereo_grid.pos_to_map(row, col), coord)

    def test_pos_to_map_mono(self):
        """Method map_to_pos should convert grid positions of mono PixelMap to pixel coordinates"""
        positions = [(0, 0), (24, 13), (7, 29), (31, 31)]
        coords = [(1, 1), (49, 27), (15, 59), (63, 63)]

        for pos, coord in zip(positions, coords):
            row, col = pos
            self.assertEqual(self.mono_grid.pos_to_map(row, col), coord)

    def test_valid_map_pos(self):
        """Method valid_map_pos should return boolean indicating if passed pixel coordinates are a valid map position"""
        # Inputs for stereo grid
        valid_coords = list(self.get_coords_pos(n=10, which='coords', valid=True, stereo=True))
        invalid_coords = list(self.get_coords_pos(n=10, which='coords', valid=False, stereo=True))
        # Testing method for stereo grid
        for coord in valid_coords:
            y, x = coord
            self.assertTrue(self.stereo_grid.valid_map_pos(y, x))
        for coord in invalid_coords:
            y, x = coord
            self.assertFalse(self.stereo_grid.valid_map_pos(y, x))

        # Inputs for mono grid
        valid_coords = list(self.get_coords_pos(n=10, which='coords', valid=True, stereo=False))
        invalid_coords = list(self.get_coords_pos(n=10, which='coords', valid=False, stereo=False))
        # Testing method for mono grid
        for coord in valid_coords:
            y, x = coord
            self.assertTrue(self.mono_grid.valid_map_pos(y, x))
        for coord in invalid_coords:
            y, x = coord
            self.assertFalse(self.mono_grid.valid_map_pos(y, x))

    @patch.object(PixelGrid, 'map_mark_pos')
    @patch.object(PixelGrid, 'valid_map_pos')
    def test_map_hover_event(self, mock_valid_pos, mock_mark_pos):
        """The slot for a hover event should check for valid positioning and mark pixel accordingly"""
        hover_pos = QPointF(0, 0)  # Dummy position that is hovered
        mock_valid_pos.return_value = True  # Make sure position is accepted as valid

        self.stereo_grid.map_hover_event(hover_pos)  # Fake triggering the hover slot

        mock_valid_pos.assert_called_with(0, 0)  # Should check for validity
        mock_mark_pos.assert_called_with(0, 0, 'hover')  # Should mark as hovered

        # Testing invalid hover position
        mock_valid_pos.reset_mock()
        mock_valid_pos.return_value = False

        self.stereo_grid.map_hover_event(hover_pos)

        mock_valid_pos.assert_called_with(0, 0)
        mock_mark_pos.assert_called_with(0, 0, 'reset_hover')

    @patch.object(PixelGrid, 'set_active')
    @patch.object(PixelGrid, 'map_mark_pos')
    @patch.object(PixelGrid, 'valid_map_pos')
    def test_map_click_event(self, mock_valid_pos, mock_mark_pos, mock_set_active):
        """The slot for clicking the pixelmap should check for valid positioning and mark pixel accordingly"""
        # Create mock for click event at position corresponding to row=0, col=0
        click_event = Mock()
        click_event.scenePos.return_value = QPointF(1.0, 1.0)
        # Make sure position is accepted as valid
        mock_valid_pos.return_value = True
        # Fake the click event
        self.stereo_grid.map_click_event(click_event)

        mock_valid_pos.assert_called_with(1, 1)  # Should check for validity of position
        mock_mark_pos.assert_called_with(1, 1, 'reset_hover')  # Should remove hover mark of position
        mock_set_active.assert_called_with((0, 0))  # Should set position (now no longer coordinates) as active

        # Test for invalid clicked position doing nothing
        mock_valid_pos.return_value = False
        mock_mark_pos.reset_mock()
        mock_set_active.reset_mock()
        # Fake click again
        self.stereo_grid.map_click_event(click_event)

        mock_valid_pos.assert_called_with(1, 1)
        mock_mark_pos.assert_not_called()
        mock_set_active.assert_not_called()

    @patch.object(PixelGrid, 'deactivate_signal')
    @patch.object(PixelGrid, 'activate_signal')
    def test_map_mark_pos_active(self, mock_activate_signal, mock_deactivate_signal):
        """Calling map_mark_pos should allow to mark a given pixel as active"""
        # Always use the same pixel location, other tests make sure that handling of locations work
        y1, x1 = 1, 1  # = position (0, 0)
        y2, x2 = 1, 3  # = position (0, 1)

        # Activating a pixel
        self.stereo_grid.map_mark_pos(y1, x1, 'active')

        # Should remember the previously used marking at the position
        self.assertEqual(self.stereo_grid.map_markings['active'], ((y1, x1), list(PixelGrid.PixelColors['missing'])))
        # Should mark the pixel with active color
        self.assertTrue(all(self.stereo_grid.grid_img.image[y1, x1, :] == PixelGrid.PixelColors['active']))
        # Activate signal should send id of activated grid position
        mock_activate_signal.emit.assert_called_with(0)

        # Activate the next pixel
        self.stereo_grid.map_mark_pos(y2, x2, 'active')

        # Should remember the value of the newly activated pixel
        self.assertEqual(self.stereo_grid.map_markings['active'], ((y2, x2), list(PixelGrid.PixelColors['missing'])))
        # Should send signal for deactivating previously active pixel
        mock_deactivate_signal.emit.assert_called_with(0)
        # Should reset first pixel to remembered value
        self.assertTrue(all(self.stereo_grid.grid_img.image[y1, x1, :] == PixelGrid.PixelColors['missing']))
        # Should mark the new pixel as active
        self.assertTrue(all(self.stereo_grid.grid_img.image[y2, x2, :] == PixelGrid.PixelColors['active']))
        # Activate signal should send id of second activated grid position
        mock_activate_signal.emit.assert_called_with(1)

    def test_map_mark_pos_hover(self):
        """Calling map_mark_pos should allow to mark a given pixel position as hovered"""
        y1, x1 = 3, 1  # = position (1, 0)
        y2, x2 = 3, 3  # = position (1, 1)

        # Hover a pixel (using mono_grid this time)
        self.mono_grid.map_mark_pos(y1, x1, 'hover')

        # Should remember previously used marking
        self.assertEqual(self.mono_grid.map_markings['hover'], ((y1, x1), list(PixelGrid.PixelColors['missing'])))
        # Should mark position as hovered
        self.assertTrue(all(self.mono_grid.grid_img.image[y1, x1, :] == PixelGrid.PixelColors['hover']))

        # Hover over the next pixel
        self.mono_grid.map_mark_pos(y2, x2, 'hover')

        # Should reset the color of the first pixel
        self.assertTrue(all(self.mono_grid.grid_img.image[y1, x1, :] == PixelGrid.PixelColors['missing']))
        # Should remember the marking of the newly hovered pixel
        self.assertEqual(self.mono_grid.map_markings['hover'], ((y2, x2), list(PixelGrid.PixelColors['missing'])))
        # Should mark newly hovered position
        self.assertTrue(all(self.mono_grid.grid_img.image[y2, x2, :] == PixelGrid.PixelColors['hover']))

    def test_map_mark_pos_placed(self):
        """Calling map_mark_pos should allow to mark a given pixel position as placed"""
        y1, x1 = 5, 1  # = position (2, 0)
        y2, x2 = 5, 3  # = position (2, 1)

        # Make pixel active
        self.stereo_grid.set_active(self.stereo_grid.map_to_pos(y1, x1))
        # Place a pixel
        self.stereo_grid.map_mark_pos(y1, x1, 'placed')

        # Should overwrite the remembered underlying pixel color with the placed color
        self.assertEqual(self.stereo_grid.map_markings['active'], ((y1, x1), PixelGrid.PixelColors['placed']))
        # Should color the pixel as placed (overwriting the active color)
        self.assertTrue(all(self.stereo_grid.grid_img.image[y1, x1, :] == PixelGrid.PixelColors['placed']))

        # Place pixel without setting it to active previously (usecase somewhat unknown)
        self.stereo_grid.map_mark_pos(y2, x2, 'placed')

        # Should not overwrite rememberd active map marking
        self.assertEqual(self.stereo_grid.map_markings['active'], ((y1, x1), PixelGrid.PixelColors['placed']))
        # Should color the second pixel as placed
        self.assertTrue(all(self.stereo_grid.grid_img.image[y2, x2, :] == PixelGrid.PixelColors['placed']))

    def test_map_mark_pos_missing(self):
        """Calling map_mark_pos should allow to mark a given pixel position as missing"""
        y1, x1 = 13, 13  # = position (6, 5)  (first column in right view)
        y2, x2 = 13, 15  # = positoin (6, 6)

        # Mark both pixels as placed
        self.stereo_grid.map_mark_pos(y1, x1, 'placed')
        self.stereo_grid.map_mark_pos(y2, x2, 'placed')

        # Make first position active
        self.stereo_grid.set_active(self.stereo_grid.map_to_pos(y1, x1))
        # Mark the pixel as missing (as if deleting the active placement)
        self.stereo_grid.map_mark_pos(y1, x1, 'missing')

        # Should overwrite the the remembered underlying color
        self.assertEqual(self.stereo_grid.map_markings['active'], ((y1, x1), PixelGrid.PixelColors['missing']))
        # Should mark the pixel as missing (overwriting active color)
        self.assertTrue(all(self.stereo_grid.grid_img.image[y1, x1, :] == PixelGrid.PixelColors['missing']))

        # Mark next pixel as missing without setting it as active
        self.stereo_grid.map_mark_pos(y2, x2, 'missing')
        # Should not overwrite remembered active map marking
        self.assertEqual(self.stereo_grid.map_markings['active'], ((y1, x1), PixelGrid.PixelColors['missing']))
        # Should color pixel as missing
        self.assertTrue(all(self.stereo_grid.grid_img.image[y2, x2, :] == PixelGrid.PixelColors['missing']))

    @patch.object(PixelGrid, 'deactivate_signal')
    def test_map_mark_pos_reset(self, mock_deactivate_signal):
        """Calling map_mark_pos should allow to reset a active or hovered pixel to its base color"""
        y1, x1 = self.mono_grid.pos_to_map(24, 16)
        y2, x2 = self.mono_grid.pos_to_map(24, 17)

        # Make first pixel being placed and make it active
        self.mono_grid.map_mark_pos(y1, x1, 'placed')
        self.mono_grid.set_active(self.mono_grid.map_to_pos(y1, x1))
        # Mark second pixel as hovered
        self.mono_grid.map_mark_pos(y2, x2, 'hover')

        # Reset both marked pixels (passed coordinates do not matter --> Coordinates to reset retrieved from saved
        # dictionary 'map_markings'
        self.mono_grid.map_mark_pos(0, 0, 'reset_active')
        self.mono_grid.map_mark_pos(1000, 1000, 'reset_hover')

        # Should delete the saved underlying colors from 'map_markings'
        self.assertEqual(self.mono_grid.map_markings['active'], (None, None))
        self.assertEqual(self.mono_grid.map_markings['hover'], (None, None))
        # Should reset the coloring of first pixel to placed and second pixel as missing
        self.assertTrue(all(self.mono_grid.grid_img.image[y1, x1, :] == PixelGrid.PixelColors['placed']))
        self.assertTrue(all(self.mono_grid.grid_img.image[y2, x2, :] == PixelGrid.PixelColors['missing']))
        # Should send id of previously active pixel as deactivated_signal
        row, col = self.mono_grid.map_to_pos(y1, x1)
        mock_deactivate_signal.emit.assert_called_with(self.mono_grid.get_id(row, col))

    def test_refit_view(self):
        self.skipTest("Test not implemented because no idea how to test it properly")

    def test_set_active_int(self):
        """Calling set active for a grid id should set the specified grid position to active"""
        ids = [0, 1, 29, 45, 69]
        ys = [1, 1, 11, 5, 13]
        xs = [1, 3, 9, 13, 21]

        for id, y, x in zip(ids, ys, xs):
            self.stereo_grid.set_active(id)
            self.assertTrue(all(self.stereo_grid.grid_img.image[y, x, :] == PixelGrid.PixelColors['active']))

    def test_set_active_tuple(self):
        """Calling set_active for a tuple(row, col) should set the specified grid position to active"""
        rows = [0, 12, 27]
        cols = [3, 18, 30]
        ys = [1, 25, 55]
        xs = [7, 37, 61]

        for row, col, y, x in zip(rows, cols, ys, xs):
            self.mono_grid.set_active((row, col))
            self.assertTrue(all(self.mono_grid.grid_img.image[y, x, :] == PixelGrid.PixelColors['active']))

    def test_mark_placed(self):
        """Calling mark_placed on a grid id should mark that position as placed and save it in placements attribute"""
        ids = [0, 45]
        ys = [1, 5]
        xs = [1, 13]

        for id, y, x in zip(ids, ys, xs):
            self.stereo_grid.mark_placed(id)
            self.assertTrue(all(self.stereo_grid.grid_img.image[y, x, :] == PixelGrid.PixelColors['placed']))
            self.assertTrue(self.stereo_grid.placements[id])

    def test_mark_missing(self):
        """Calling mark_missing on a grid id should mark that position as missing and remove it from placements"""
        ids = [0, 45]
        ys = [1, 5]
        xs = [1, 13]
        # Mark the positions as placed
        for id, y, x in zip(ids, ys, xs):
            self.stereo_grid.mark_placed(id)

        # Mark the positoins as missing again --> Marking and saving in placements should change
        for id, y, x in zip(ids, ys, xs):
            self.stereo_grid.mark_missing(id)
            self.assertTrue(all(self.stereo_grid.grid_img.image[y, x, :] == PixelGrid.PixelColors['missing']))
            self.assertFalse(self.stereo_grid.placements[id])

    def test_get_cur_1d(self):
        """Calling get_cur_1d should return the grid id of the currently activated grid position"""
        for id in range(self.num_rows * self.num_cols * 2):
            self.stereo_grid.set_active(id)
            self.assertEqual(self.stereo_grid.get_cur_1d(), id)

    def get_coords_pos(self, n, which, valid, stereo) -> Generator[Tuple[int, int], None, None]:
        """
        Generator yielding *n* coordinates or positions as specified by *which*. In case *which* is set to 'coords',
        *valid* specifies if coordinates are valid map coordinates or not (not valid meaning coordinates corresponding
        to off-grid positions).

        :param n: Number of tuples to return
        :type n: int
        :param which: One of ['coords', 'pos']. Specify if pixel coordinates or grid positions should be returned.
        :type which: str
        :param valid: Only relevant when *which* is set to 'coords'. Boolean setting if returned pixel coordinates
            correspond to valid grid positions or invalid off-grid positions.
        :type valid: bool
        :param stereo: Specify if values are for stereo or mono grid.
        :type stereo: bool
        :return: Tuple(y, x) for *which='coords'* or Tuple(row, col) for *which='pos'*.
        """
        ctr = 0
        while ctr < n:
            rows = self.num_rows if stereo else self.l_rows
            cols = self.num_cols if stereo else self.l_cols
            if which == 'coords':
                y = np.random.randint(0, rows)
                x = np.random.randint(0, cols * (2 if stereo else 1))
                # Re-draw the x-coordinate if it corresponds to the midline of the stereo grid
                while stereo and valid and x * 2 + 1 == self.num_cols * 2 + 1:
                    x = np.random.randint(1, self.num_cols)
                # Valid coordinates have to be odd, invalid even
                if valid:
                    y = y * 2 + 1
                    x = x * 2 + 1
                else:
                    y = y * 2
                    x = x * 2
                yield y, x
            else:
                y = np.random.randint(0, rows)
                x = np.random.randint(0, cols * (2 if stereo else 1))
                yield y, x

            ctr += 1


if __name__ == '__main__':
    unittest.main()

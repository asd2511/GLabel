import os
import sys
import unittest
from unittest.mock import Mock, patch
import json
from typing import List, Union
from pathlib import Path

import numpy as np

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QPointF

from gui.main_window import Main, LoginWindow
from gui.image_widget import ConfirmDialog
from gui.settings_window import GridSettings
from gui.config import settings, keybinds

app = QApplication(sys.argv)


class MainWindowTest(unittest.TestCase):
    username = 'Tester'
    screen_height = 1920
    screen_width = 1080

    config = None
    config_dir_exists = False
    config_file_exists = False
    # config_dir = '../../.suturelab'
    # config_file = '../../.suturelab/config.json'
    config_dir = Path(__file__).parents[2].as_posix() + '/.suturelab'
    config_file = config_dir + '/config.json'

    @classmethod
    def setUpClass(cls):
        # Finding the config path depends on current execution directory (difference between command-line and pycharm)
        # if 'tests' in os.listdir(os.getcwd()):
        #     cls.config_dir = cls.config_dir.split('/', 1)[1]
        #     cls.config_file = cls.config_file.split('/', 1)[1]
        # Temporarily delete all traces of any set configuration forcing Main to use default configuration
        if os.path.isdir(cls.config_dir) and os.path.isfile(cls.config_file):
            cls.config_dir_exists = True
            cls.config_file_exists = True
            # Save set configuration
            with open(cls.config_file, 'r') as f:
                cls.config = json.load(f)
            # Delete all traces of configuration
            os.remove(cls.config_file)
            os.rmdir(cls.config_dir)
        elif os.path.isdir(cls.config_dir):
            cls.config_dir_exists = True
            os.rmdir(cls.config_dir)

    @classmethod
    def tearDownClass(cls):
        # Restore deleted configuration
        if cls.config_dir_exists and cls.config_file_exists:
            os.mkdir(cls.config_dir)
            with open(cls.config_file, 'w') as f:
                json.dump(cls.config, f, indent=4)
        elif cls.config_dir_exists and not cls.config_file_exists:
            os.mkdir(cls.config_dir)

    @patch('gui.main_window.LoginWindow')
    def setUp(self, mock_login):
        mock_screen = Mock()
        mock_screen.width = Mock(return_value=self.screen_width)
        mock_screen.height = Mock(return_value=self.screen_height)

        login = mock_login.return_value
        login.name_edit.text.return_value = self.username

        self.win = Main(mock_screen)

    def test_geometry(self):
        """Window geometry should be 200 narrower and 300 shorter than screen"""
        assert self.win.geometry().width() == self.screen_width-200
        assert self.win.geometry().height() == self.screen_height-300

    def test_defaults(self):
        """Default values should be set to config.py contents"""
        # Settings
        self.assertEqual(self.win.settings, settings)
        self.assertEqual(self.win.user, self.username)
        self.assertEqual(self.win.filename, settings['Filename'])
        self.assertEqual(self.win.save_file, None)
        self.assertEqual(self.win.save_mode, settings['SaveMode'])
        self.assertEqual(self.win.save_confirm, settings['SaveConfirm'])
        self.assertEqual(self.win.stack_dimensions, settings['StackDimensions'])
        self.assertEqual(self.win.num_rows, settings['NumRows'])
        self.assertEqual(self.win.num_cols, settings['NumCols'])
        self.assertEqual(self.win.stereo_grid, settings['StereoGrid'])
        self.assertEqual(self.win.auto_mode, settings['AutoMode'])
        self.assertEqual(self.win.auto_snaking, settings['AutoSnaking'])
        self.assertEqual(self.win.auto_copy, settings['AutoCopy'])
        self.assertEqual(self.win.num_rows_track, settings['NumRowsTrack'])
        self.assertIsInstance(self.win.roi_color, QColor)
        self.assertEqual(self.win.roi_color, QColor(settings['ROIColor']))
        self.assertIsInstance(self.win.active_color, QColor)
        self.assertEqual(self.win.active_color, QColor(settings['ActiveColor']))
        self.assertEqual(self.win.show_closeup, settings['ShowCloseup'])
        self.assertEqual(self.win.roi_snap, settings['ROISnap'])

        # Keybinds
        self.assertEqual(self.win.keybinds, keybinds)

    @patch('gui.main_window.open')
    @patch('gui.main_window.os')
    @patch('gui.main_window.json')
    def test_save_config(self, mock_json, mock_os, mock_open):
        """Calling Main.save_config() should create directory '.suturelab' and write path 'config.json'"""
        mock_os.mkdir.return_value = ''
        self.win.save_config()
        mock_os.mkdir.assert_called_with('.suturelab/')
        mock_open.assert_called_with('.suturelab/config.json', 'w')
        mock_json.dump.assert_called_once()

    @patch('gui.main_window.image_widget.ConfirmDialog', autospec=True)
    @patch('gui.main_window.json')
    @patch('gui.main_window.open')
    @patch('gui.main_window.os.path')
    def test_load_config(self, mock_path, mock_open, mock_json, mock_confirm_dlg):
        """Calling Main.load_config should load currently saved configuration from '.suturelab/config.json'"""
        # No config directory or saved config path yet --> Load default configuration from config.py
        mock_path.isdir.return_value = False
        mock_path.isfile.return_value = False
        ret_settings, ret_keybinds = self.win.load_config()
        self.assertEqual(ret_settings, settings)
        self.assertEqual(ret_keybinds, keybinds)

        # Config directory exists but no path yet  --> Load default configuration from config.py
        mock_path.isdir.return_value = True
        ret_settings, ret_keybinds = self.win.load_config()
        self.assertEqual(ret_settings, settings)
        self.assertEqual(ret_keybinds, keybinds)

        # Previous configuration already exists with wrong username --> Opens dialog asking if config should be used
        mock_dlg = mock_confirm_dlg.return_value
        mock_dlg.exec_.return_value = True
        mock_json.load.return_value = {'meta_data': {'user': 'not tester',
                                                     'time (UTC)': 'test_time'},
                                       'config': {'settings': settings,
                                                  'keybinds': keybinds}}
        # Loaded user differs from currently active user
        self.assertNotEqual(self.win.user, mock_json.load.return_value['meta_data']['user'])

        # Set path from wrong user to be available and load it
        mock_path.isfile.return_value = True
        ret_settings, ret_keybinds = self.win.load_config()
        mock_open.assert_called_with('.suturelab/config.json')  # Loading the config path
        mock_dlg.exec_.assert_called_once()  # Dialog asking if other config should be used

        # Previous configuration already exists with correct username
        mock_json.load.return_value = {'meta_data': {'user': self.username},
                                       'config': {'settings': settings,
                                                  'keybinds': keybinds}}
        ret_settings, ret_keybinds = self.win.load_config()
        self.assertEqual(mock_json.load.call_count, 2)  # Twice because of previous test
        mock_dlg.exec_.assert_called_once()  # Should not have been called a second time

    @patch('gui.main_window.json')
    @patch.object(Main, 'save_as')
    @patch.object(Main, 'get_ts_savefile')
    def test_save_ts_no_image(self, mock_get_ts_savefile, mock_save_as, mock_json):
        """
        Calling Main.save using the default save mode of 0 (= new path with ts) should append a timestamp to the
        saved path
        """
        save_file = './save_as.rois'
        mock_get_ts_savefile.return_value = save_file + '_001122-334455'

        self.assertIsNone(self.win.save_file)  # No save_file at startup

        self.win.save_file = save_file
        self.win.save_confirm = False  # Prevent opening confirm dialog
        self.win.save()
        mock_get_ts_savefile.assert_called_with(save_file)  # Called to add timestamp to save_file
        mock_save_as.assert_called_with(save_file + '_001122-334455')  # Saved as timestamped path
        self.assertEqual(self.win.save_file, save_file)  # Set path w/o timestamp as save_file in memory
        mock_json.dump.assert_not_called()  # No image_stack set yet --> Nothing to save

    @patch('gui.main_window.json.dump', return_value=True)
    @patch('gui.main_window.open')
    @patch.object(Main, 'get_save_dict')
    @patch.object(Main, 'get_ts_savefile')
    def test_save_ts_image(self, mock_get_ts_savefile, mock_get_save_dict, mock_open, mock_dump):
        """
        Saving with existing image_stack and save_mode set to 'timestamped' should save timestsamped path by calling
        'get_save_dict' and 'json.dump'
        """
        save_file = './save_as.rois'  # save_file without added timestamp saved in memory
        data_file = 'data_file'
        mock_get_ts_savefile.return_value = save_file + '_001122-334455'
        mock_get_save_dict.return_value = {'1': 1, '2': 2}

        self.assertIsNone(self.win.save_file)  # MainWindow has no save_file in memory --> Creates new one

        self.win.image_stack = Mock()  # image_stack object is initialized
        self.win.save_file = save_file  # MainWindow has save_file without timestamp
        self.win.filename = data_file
        self.win.save_confirm = False  # Do not ask for confirmation before saving

        self.win.save()  # Save current image_stack object

        mock_get_ts_savefile.assert_called_with(save_file)  # Should add timestamp to save_file in memory
        self.assertEqual(self.win.save_file, save_file)  # Should have save_file without timestamp in memory
        mock_open.assert_called_with(mock_get_ts_savefile.return_value, 'w')  # Should open timestamped path to write
        mock_get_save_dict.assert_called_once()  # Should have converted data to dictionary
        mock_dump.assert_called_once()  # Should have dumped image_stack.roi_stack to path

    @patch('gui.main_window.settings_window.GridSettings', autospec=True)
    @patch.object(Main, 'set_grid_settings')
    def test_grid_settings_dlg(self, mock_set_grid_settings, grid_dlg):
        """
        Clicking on 'Settings->Grid' should display grid settings dialog with current settings and set the grid
        variables in MainWindow
        """
        # Mocking the GridSettings object along with user-entered values
        mock_dlg = grid_dlg.return_value  # type: GridSettings
        mock_dlg.exec_.return_value = True

        QTest.mouseClick(self.win.settings_m, Qt.LeftButton, Qt.NoModifier)  # Click 'Settings'
        QTest.keyClick(self.win.settings_m, Qt.Key_Down, Qt.NoModifier)  # Press 'Down', navigating on 'Grid'
        QTest.keyClick(self.win.settings_m, Qt.Key_Enter, Qt.NoModifier)  # Press 'Enter', activating 'Grid'

        # GridSettings should be initialized with current settings
        grid_dlg.assert_called_once_with(self.win.num_rows, self.win.num_cols, self.win.stereo_grid, self.win.grid_style)
        # set_grid_settings should be called with initialized GridSettings object
        mock_set_grid_settings.assert_called_once_with(mock_dlg)

    @patch('gui.main_window.image_widget.ConfirmDialog', autospec=True)
    @patch.object(Main, 'open')
    def test_set_grid_settings_changed(self, mock_open, mock_confirm_dlg):
        """Calling MainWindow.set_grid_settings should change grid variables and reopen the current path"""
        # Mock the GridSettings dialog along with user-entered values
        grid_dlg = Mock()  # type: Union[GridSettings, Mock]
        grid_dlg.rows = 100
        grid_dlg.cols = 100
        grid_dlg.stereo = True
        grid_dlg.grid_style = 0

        # Mock the ConfirmDialog to accept changing the grid values
        confirm_dlg = mock_confirm_dlg.return_value
        confirm_dlg.exec_.return_value = True

        # Mock existence of image_stack
        self.win.image_stack = 'not None'

        self.win.set_grid_settings(grid_dlg)  # Call to set new grid settings

        confirm_dlg.exec_.assert_called_once()  # User should have been asked for confirmation
        # Grid variables should have been applied
        self.assertEqual(self.win.num_rows, grid_dlg.rows)
        self.assertEqual(self.win.num_cols, grid_dlg.cols)
        self.assertEqual(self.win.stereo_grid, grid_dlg.stereo)
        self.assertEqual(self.win.grid_style, grid_dlg.grid_style)
        mock_open.assert_called_with(self.win.filename)  # Current path should have been re-opened

    @patch('gui.main_window.image_widget.ConfirmDialog', autospec=True)
    @patch.object(Main, 'open')
    def test_set_grid_settings_unchanged(self, mock_open, mock_confirm_dlg):
        """Calling MainWindow.set_grid_settings without changing grid variables should not do anything"""
        # Saving previously applied grid settings
        prev_rows = self.win.num_rows
        prev_cols = self.win.num_cols
        prev_stereo = self.win.stereo_grid
        prev_style = self.win.grid_style

        # Mocking GridSettings dialog with same settings as already set in MainWindow
        grid_dlg = Mock()  # type: GridSettings
        grid_dlg.rows = prev_rows
        grid_dlg.cols = prev_cols
        grid_dlg.stereo = prev_stereo
        grid_dlg.grid_style = prev_style

        # ConfirmDialog will pass if it should be called
        confirm_dlg = mock_confirm_dlg.return_value
        confirm_dlg.exec_.return_value = True

        # Mock existence of image_stack
        self.win.image_stack = 'not None'

        self.win.set_grid_settings(grid_dlg)  # Call to set grid settings

        confirm_dlg.exec_.assert_not_called()  # User should not be asked to accept re-opening data
        # Grid variables should not be changed
        self.assertEqual(self.win.num_rows, prev_rows)
        self.assertEqual(self.win.num_cols, prev_cols)
        self.assertEqual(self.win.stereo_grid, prev_stereo)
        self.assertEqual(self.win.grid_style, prev_style)
        mock_open.assert_not_called()  # Data should not be re-opened

    @patch('gui.main_window.os.path.normpath')
    @patch.object(Main, 'get_sha_string')
    def test_get_save_dict(self, mock_get_sha_string, mock_normpath):
        """Calling 'get_save_dict' should convert data of image_stack to dictionary"""
        # Mocking image_stack object along with some ROI data
        img_stack = Mock()
        # Create random ROI float positions for 100 frames
        num_rois = self.win.num_rows * self.win.num_cols * 2
        num_frames = 100
        img_stack.roi_stack = self.build_roi_stack(num_frames)
        self.win.image_stack = img_stack
        self.win.filename = r'C:/Tester/Daten/testdata.h5'

        # Mocking hashing and path functions called during creation of dictionary
        mock_get_sha_string.return_value = 'sha'
        mock_normpath.return_value = self.win.filename

        save_dict = self.win.get_save_dict()

        save_keys = ['meta_data', 'image_file', 'backup_path', 'sha_hash', 'num_rows', 'num_columns', 'stereo_grid',
                     'frames']
        meta_keys = ['time (UTC)', 'user', 'account']
        frame_keys = ['frame', 'roi_positions']
        pos_keys = ['id', 'pos', 'placed']

        # Dict should contain all necessary keys
        self.assertEqual(list(save_dict.keys()), save_keys)
        self.assertEqual(list(save_dict['meta_data'].keys()), meta_keys)
        self.assertEqual(list(save_dict['frames'][0].keys()), frame_keys)
        self.assertEqual(list(save_dict['frames'][0]['roi_positions'][0].keys()), pos_keys)
        # Dict should have correct meta-data
        meta_dict = save_dict['meta_data']
        self.assertIsNotNone(meta_dict['time (UTC)'])
        self.assertEqual(meta_dict['user'], self.win.user)
        self.assertIsNotNone(meta_dict['account'], os.getlogin())
        # Dict should have correct settings
        self.assertEqual(save_dict['image_file'], self.win.filename)
        self.assertEqual(save_dict['backup_path'], self.win.filename)  # Same path as normpath is patched
        self.assertEqual(save_dict['sha_hash'], 'sha')  # Patched return value
        self.assertEqual(save_dict['num_rows'], self.win.num_rows)
        self.assertEqual(save_dict['num_columns'], self.win.num_cols)
        self.assertEqual(save_dict['stereo_grid'], self.win.stereo_grid)
        # Dict should contain correct number and coordinates of ROI placements
        frames_list = save_dict['frames']
        self.assertEqual(len(frames_list), num_frames)  # Dict should contain same number of frames
        for f in range(num_frames):
            self.assertEqual(len(frames_list[f]['roi_positions']), num_rois)  # Number of ROIs for each frame
            # Compare the actual ROI coordinates
            rois = [QPointF(roi['pos']['x'], roi['pos']['y']) for roi in frames_list[f]['roi_positions']]
            self.assertListEqual(rois, self.win.image_stack.roi_stack[f])
            # Last frame should contain no 'placed: True'
            if f == num_frames - 1:
                self.assertFalse(any(roi['placed'] for roi in frames_list[f]['roi_positions']))

    @patch('gui.main_window.QMainWindow.setCentralWidget')
    @patch.object(Main, 'read_image')
    @patch.object(Main, 'get_sha_string')
    @patch('gui.main_window.image_widget.ImageStack', autospec=True)
    def test_open_save_file(self, mock_image_stack, mock_get_sha_string, mock_read_image, mock_qmain_window):
        """Opening a previously saved .rois path should load all data into the ImageStack correctly as saved"""
        # Create an empty data path
        # data_file = 'test_data.h5'
        data_file = Path(__file__).parent.as_posix() + '/test_data.h5'
        open(data_file, 'a').close()
        # save_file = 'test_data.rois'
        save_file = Path(__file__).parent.as_posix() + '/test_data.rois'

        # Mock data in the ImageStack object of the MainWindow
        mock_image_stack.return_value = Mock()
        img_stack = Mock()
        rois = self.build_roi_stack(100)
        img_stack.roi_stack = rois
        self.win.image_stack = img_stack
        self.win.filename = data_file

        # Mocking sha value of non-existent test_data path
        mock_get_sha_string.return_value = 'sha'
        # Mocking loading of .h5 image data
        mock_read_image.return_value = np.empty((100, 100, 100))

        self.win.save_as(save_file)  # Save the data
        img_stack.save_cur_rois.assert_called_once()  # Call to save currently displayed frame of image_stack

        # Reset the mocked image_stack to be empty again
        self.win.image_stack.reset_mock()

        # Load the saved data
        self.win.open(save_file)

        self.assertEqual(self.win.filename, os.path.abspath(data_file))  # New filename should be abs path to path
        self.assertEqual(self.win.rois_file, save_file)  # Savefile should be string with which 'open' was called
        mock_image_stack.assert_called_once()
        mock_qmain_window.assert_called_once()
        self.assertEqual(self.win.windowTitle(), 'GLable - ' + os.path.abspath(data_file))

        # Cleanup
        os.remove(data_file)
        os.remove(save_file)

    def build_roi_stack(self, num_frames=100) -> List[List[QPointF]]:
        """
        Create a mocked roi_stack as used by the ImageStack class.

        The ROI positions will be randomly distributed across dimensions of (768, 768) to simulate a real-world image
        size.

        :return: List of lists containing QPointF objects for each random ROI position.
        """
        # Create random ROI float positions for (786, 786) image for specified number of frames
        num_rois = self.win.num_rows * self.win.num_cols * 2
        roi_stack = [[QPointF(np.random.random() * 786, np.random.random() * 786) for _ in range(num_rois)]
                               for _ in range(num_frames - 1)]
        roi_stack.append([QPointF(-1, -1) for _ in range(num_rois)])  # Add an empty frame at the end

        return roi_stack


if __name__ == '__main__':
    unittest.main()

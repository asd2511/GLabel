"""
Default config settings for the GLabel GUI used at initial startup or when no .config file was found.
"""
#!/usr/bin/env python
from PyQt5.QtCore import Qt

settings = {

    "Filename":             None,
    "SaveMode":             0,  # 0 = New path with timestamp; 1 = Overwrite existing path
    "SaveConfirm":          True,
    "StackDimensions":      None,
    "NumRows":              7,
    "NumCols":              5,
    "StereoGrid":           True,
    "GridStyle":            1,  # 0 = Buttons; 1 = Pixelmap
    "AutoMode":             2,  # 0 = Free; 1 = Left-to-Right; 2 = Bottom-to-Top
    "AutoSnaking":          False,
    "BeginTop":             False,  # Setting pathing to begin at top left (True) or bottom left (False)
    "AutoCopy":             False,
    "ActiveProgressMode":   0,
    "FrameClicking":        False,
    "NumRowsTrack":         7,
    "Gaussian":             False,
    "ROIColor":             'black',
    "ActiveColor":          'lime',
    "ShowCloseup":          True,
    "ROISnap":              False,  # Snapping ROIs to full-pixel positions
    "ShowCrosshair":        False

}  #: Default settings applied to the GUI on startup

mousebinds = {
    "PlaceROI":                 {'button': Qt.LeftButton,       'modifier': Qt.ControlModifier},
    "UndoROI":                  {'button': Qt.RightButton,      'modifier': Qt.ControlModifier},
}  #: Default mouse bindings for placing and removing annotations

keybinds = {
    "DeleteActiveROI":          Qt.Key_Delete,
    "DeleteAllROI":             Qt.Key_Delete+Qt.ShiftModifier,

    "ActivateRightROI":         Qt.ControlModifier + Qt.Key_Right,
    "ActivateLeftROI":          Qt.ControlModifier + Qt.Key_Left,
    "ActivateUpROI":            Qt.ControlModifier + Qt.Key_Up,
    "ActivateDownROI":          Qt.ControlModifier + Qt.Key_Down,

    "ActivateNextROI":          Qt.Key_E,
    "ActivatePreviousROI":      Qt.Key_Q,

    "MoveROIUp":                Qt.Key_W,
    "MoveROIDown":              Qt.Key_S,
    "MoveROIRight":             Qt.Key_D,
    "MoveROILeft":              Qt.Key_A,

    "NextFrame":                Qt.Key_N,
    "PreviousFrame":            Qt.Key_B,

    "CopyNextROI":              Qt.Key_J,
    "CopyPreviousROI":          Qt.Key_H,
    "CopyROIsToAll":            Qt.Key_0+Qt.ControlModifier,

    "ToggleROIVisibility":      Qt.Key_V,
    "CycleROIColorCoding":      Qt.Key_C,

    "FindNewSuturePositions":   Qt.Key_T,
    "FindNearestDark":          Qt.Key_Z,
    "FindNearestLight":         Qt.Key_U,

    "LockView":                 Qt.Key_L,
    "FirstFrame":               Qt.Key_Space,

    "Test":                     "Ctrl+C"
}  #: Default keybinds for controlling the GUI


def lu_mouse(action, button, modifier) -> bool:
    """
    Helper function to check if a triggered mouse action is connected with an action as defined in :attr:`mousebinds`.

    Function is called by :func:`gui.image_widget.ImageView.mousePressEvent()` each time the
    :class:`gui.image_widget.ImageView` object receives a mouse interaction. Depending on if the action matches the
    received mouse interaction, True or False is returned.

    :param str action: Action for which to determine if mouse interaction matches mousebind.
    :param int button: Encoded mouse button received by :class:`gui.image_widget.ImageView` object.
    :param KeyboardModifiers modifier: :pyqt:`KeyboardModifiers <qt.html#KeyboardModifier-enum>` modifier as received by
        :class:`gui.image_widget.ImageView` object.
    :return: True if received mouse interaction matches mouse binding defined in :attr:`mousebinds`. False otherwise.
    """
    if mousebinds[action]['button'] == button and mousebinds[action]['modifier'] == modifier:
        return True
    else:
        return False

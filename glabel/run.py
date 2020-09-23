#!/usr/bin/env python
import sys
import qdarkstyle
from PyQt5.QtWidgets import QApplication

from glabel.gui.main_window import Main, LoginWindow

__author__ = "Tobias Zillig"
__version__ = "0.2"
__status__ = "Development"


def except_hook(cls, exception, traceback):
    """
    Enables exception traces from PyQt to be output to the console.
    """
    sys.__excepthook__(cls, exception, traceback)


def run():
    sys.excepthook = except_hook
    app = QApplication(sys.argv)

    if '--debug' in sys.argv:
        debug = True
        sys.argv.remove('--debug')
    else:
        debug = False

    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))  # Setting nice dark theme to PyQt
    screen = app.primaryScreen()  # Reference to current screen parameters

    m = Main(screen.size(), debug)
    m.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    run()

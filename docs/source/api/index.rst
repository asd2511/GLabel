GLabel API Reference
====================

.. toctree::

Guides
------

.. toctree::

    extension_guide

Classes
-------

GUI
^^^

.. autosummary::
    :toctree: generated

    gui.main_window.Main
    gui.main_window.LoginWindow
    gui.image_widget.ImageStack
    gui.image_widget.ImageView
    gui.image_widget.ROI
    gui.grid_widget.GridWidget
    gui.grid_widget.ButtonGrid
    gui.grid_widget.PixelGrid
    gui.worker.Worker
    gui.worker.WorkerSignals
    gui.calibration_window.CalibrationWindow
    gui.calibration_window.CalibrationWidget
    gui.calibration_window.CubeCalibrator
    gui.calibration_window.CalibrationEdges
    gui.settings_window.GridSettings
    gui.settings_window.ShortcutSettings
    gui.settings_window.AutomationSettings

Automation
^^^^^^^^^^

.. autosummary::
    :toctree: generated

    nn.datagenerator.DataGenerator
    nn.datagenerator.RUNetGenerator
    nn.datagenerator.SortNetGenerator

Surface visualization
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated

    gui.surface_viewer.SurfaceViewer
    gui.surface_viewer.SpaghettiViewer
    gui.surface_viewer.DisplacementPlot
    gui.surface_viewer.CovEllipse

Modules
-------

GUI
^^^

.. autosummary::
    :toctree: generated

    gui.config
    gui.gui_utils

Data analysis/Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated

    analysis.utils
    analysis.surface
    analysis.evaluation_utils

Automation
^^^^^^^^^^

.. autosummary::
    :toctree: generated

    nn.suture_detection
    nn.postprocessing
    analysis.column_clustering

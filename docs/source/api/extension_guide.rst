Extend Automation functionality
======================================

How my suture detection pipeline is integrated into the GUI
-----------------------------------------------------------

Main entry point for starting the automation process is menu button *Automated labeling* in ``gui.main_window.Main``
class.
The button is connected with method ``Main.run_suture_detection`` where everything pertaining to the automation is created and
executed.

Settings
^^^^^^^^
Settings for inference are determined by user input in a separate ``gui.settings.InferenceSettings`` object by call to
``ask_inference_settings``. **These settings should be replaced by your custom needs.**
The ``InferenceSettings`` object will return a dictionary with a list of all settings relative to the automation process.
Settings are *mostly* irrelevant for starting the automation, but are passed on to the actual function that executes
the pipeline later on.

Progress Dialog
^^^^^^^^^^^^^^^^
To give users feedback on current step in automation process a ``QProgressDialog`` is instantiated.
Based on the entered settings, the total number of steps for the automation pipeline is calculated/guessed (guessed
where necessary because not all pipeline steps have a pre-determined number of steps).

Worker class for multithreading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to not block the GUI during execution of the automation, a separate worker thread is used.
This is done through the ``gui.worker.Worker`` class based on this tutorial:
https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/

Function
::::::::
The `Worker` is a class ``QRunnable`` object which receives a function object and additional positional and keyword
arguments.
The passed function is the one to be executed - which is the function to start the automation pipeline
(``nn.suture_detection.auto_annotate`` in my case).
All positional and keyword arguments passed to the ``Worker`` are directly passed on to that function.

Signals
:::::::
In order to communicate between the automation thread and the main GUI thread, ``pyqtsignal`` objects are used.
All signals available to the worker class are defined in ``gui.worker.WorkerSignals`` which could be extended as not
all must be used by your pipeline.
A couple of basic signals are implemented by me, with the ones actually used being:

* ``progress`` (emitting a single int after every pipeline step) -> Passed as ``progress_callback``
* ``message`` (emitting a string describing current step) -> Passed as ``message_callback``
* ``result`` (emitting any object holding the pipeline results; a dictionary in my case)
* ``finished`` (emitting nothing, but being used to signal when pipeline is finished)
* ``canceled`` (**Actually a slot which was intended for cancelling the automation but not implemented yet**) -> Passed as
  ``cancel_callback``

All except for the ``finished`` signal are directly passed on to the executed worker function.
**They must be received** as a keyword argument there and can be used to emit any useful update to the main GUI thread.

Each signal is connected via a *lambda* function in the main gui method ``run_suture_detection`` before actually starting
the worker thread.
You can connect these signals using the typical signal-slot system of PyQt5.
As an example for my case, I connected the ``progress`` signal to the progress value of the ``QProgressDialog``.
Within the various executed pipeline functions in ``nn.suture_detection``, I emit that signal any time a defined
pipeline step has executed (e.g. a batch has been predicted) in order to update the displayed progress bar.

Places where you can/should/must extend
----------------------------------------
Main method
    The method ``run_suture_detection`` in the ``Main`` class is specific to my use case of detecting sutures.
    You should probably create your own separate method for your use case, but you might use it to get an idea of how
    to structure your method.

    You also need to create your own method for receiving the output generated by your pipeline. In my case, a
    dictionary with all results is emitted via the ``Worker`` class' ``result`` signal. This signal is connected to
    the method ``Main.distribute_inferences`` which must be changed for your use.

Distribution and processing methods
    The automation results emitted via the ``result`` signal of the ``Worker`` are passed on to
    ``Main.distribute_inferences`` in my case. **You need your own distribution method for your results.**

    The distribution method is meant to supply all *processing methods* with the data they need to store and display
    the automation results to the user. In my case, all of the data is passed on to the ``ImageStack`` class.
    Each pipeline result is separately sent to a processing method there where the results are stored and displayed
    within the GUI.
    **You also will need your own processing and handling methods based on your results.**

Settings
    For giving the user options relevant for your automation you could:

    * Create your own ``InferenceSettings`` type of class
    * Sub-class it and overwrite the functions for creating the settings dictionary
    * Extend the ``InferenceSettings`` class itself to dynamically decide which settings to create.

ProgressDialog
    If your pipeline does not allow for calculating the number of steps in the pipeline, it might be better to remove
    the progress dialogs percentage display and just show text indicating the current step.

    I would recommend using any type of progress indicator in case of long automation pipelines (long being >5s) just
    in order to provide a good user experience.

Worker
    The signals implemented for the ``Worker`` class might be extended to provide the messaging functionality you need
    between pipeline and GUI threads.

    If the passed signals (progress, message and callback) do not work for you and are troublesome for you, you could
    modify the behavior of the ``Worker`` itself to only optionally pass them to your function.

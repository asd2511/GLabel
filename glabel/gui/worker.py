"""
Mainly based on https://www.learnpyqt.com/courses/concurrent-execution/multithreading-pyqt-applications-qthreadpool/
"""
import traceback
import sys

from PyQt5.QtCore import QRunnable, pyqtSlot, pyqtSignal, QObject


class Worker(QRunnable):
    """
    **Bases:** :pyqt:`QRunnable <qrunnable>`

    Worker thread for multithreading in PyQt5.

    Handles worker thread setup, signals and wrap-up.

    For details to the emitted signals, see their definition in :class:`WorkerSignals`.

    :param fn: The function callback to run on this worker thread. Supplied args and kwargs will be passed through
        to this function.
    :type fn: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn  #: Function to be executed in worker thread
        self.args = args  #: Positional arguments passed to executed function
        self.kwargs = kwargs  #: Keyword arguments passed to executed function
        self.signals = WorkerSignals()  #: :class:`WorkerSignals` used to communicate to and from the worker thread
        self.callbacks = {
            'progress_callback': self.signals.progress,
            'message_callback': self.signals.message,
            'cancel_callback': self.signals.canceled
        }  #: Dictionary with :attr:`signals` that is passed on to the executed function

    @pyqtSlot()
    def run(self):
        """
        :pyqt:`PyQtSlot <threads-qobject>` to execute the runner function with the passed args and kwargs.
        """
        try:
            result = self.fn(*(self.args + (self.callbacks,)))
        except:  # Bad form, but we really want to catch any occurring exception
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class WorkerSignals(QObject):
    """
    Defines the signals available for a running worker thread.

    Available signals are:

    ==========  ==================  ====================================================================================
    **Signal**  **Emits**           **Emitted when**
    finished    None                Function execution is finished to signal completion.
    result      any python object   The function finished and returns any python object.
    progress    int                 Each time the worker's `progress_callback` signal is triggered.
    canceled    None                **Not working yet!** The User cancels the process by using the button on the
                                    ProgressDialog.
    message     str                 The current working step during inference changes. This updates the label of the
                                    QProgressDialog.
    error       tuple               When any error is encountered. Will emit tuple of `Exception type` and `Exception
                                    value` and formatted traceback.
    ==========  ==================  ====================================================================================
    """
    finished = pyqtSignal()
    result = pyqtSignal(object)
    progress = pyqtSignal(int)
    canceled = pyqtSlot()
    message = pyqtSignal(str)
    error = pyqtSignal(tuple)

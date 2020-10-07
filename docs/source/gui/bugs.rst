Known Bugs
==========
List of currently known bugs for which to look out.

Critical
--------

Saving data
^^^^^^^^^^^
Errors with :attr:`save_file` and :attr:`filename` in :class:`Main`
    When saving data there seems to occur some mixup in the way the :attr:`save_file` and :attr:`filename` attributes
    are saved in :class:`Main`. Observed issues are:

    * Saving to wrong file in when using **File->Save** or **CTRL+S**

    * Issues in splitting filename for building `file_suggest` in :func:`save_as`, where :attr:`filename` is sometimes
      encountered as ``None`` and therefore **no data can be saved at all, no matter what user does**

    .. note:: Current fix lets user decide if saving a file will overwrite existing file or always create new file with timestamp as part of filename. Needs to be fully tested!

Frame Slider
^^^^^^^^^^^^
Using the frame slider with placed ROIs in the data will cause wrong copying of ROIs to random frames.
    Observed for data with 100 frames of placed ROIs. When dragging the line of the frame slider from last frame
    back to the first frame, ROIs on some frames (~every 10th frame or so) was showing wrong ROI placements which
    were likely wrongfully copied from some other frame.

    Most likely reason is that the call to :func:`save_frame_rois` in :func:`changeZ` gets mixed up information
    from which frame the ROIs should be saved because the dragging happens too fast.

    .. note:: Currently used workaround is to disable dragging of the frame indicator line!

Less Critical
-------------

Frame Clicking
^^^^^^^^^^^^^^
When using frame clicking for sutures that overlap, the currently active position switches to the wrong suture.
	The click received during frame clicking when the **view is locked** is processed correctly for the current frame. But the click event still seems to persist after the frame has switched. If another suture is then under the current mouse position in the new frame, that suture receives a click and gets activated.

    .. note:: Current solution is to disable the selection of ROIs when the view is locked.

Segmentation Map
^^^^^^^^^^^^^^^^
Displaying segmentation map causes `ValueError` for unmarked frames.
    Opening a segmentation map in the GUI will display the data as binary images. Unmarked images will therefore be
    completely black. The histogram widget or :func:`auto_range` function of the :class:`ImageView` will look for the
    value distribution of the frame and raise a `ValueError` because of an error with :func:`np.arange`.

    Could be solved by disabling the histogram or auto_range checking for segmentation map data.
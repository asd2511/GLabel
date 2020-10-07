***************
Manual Labeling
***************
After the image data has been loaded and all widgets have been initialized, the data can be manually labeled.

ROI Placement
=============
Placing a ROI object on the image data can be done by using **CTRL + Left mouse button**.
(If :ref:`frame clicking` is enabled and the View is currently locked, ROIs are placed using only **Left mouse button** instead)

This creates a circular object with its center at the position of the received click. The central pixel coordinates
(relative to the top left corner) of placed ROI objects are also the later saved coordinates.
Each of the placed ROI objects is an instantiated :doc:`ROI <../api/generated/gui.image_widget.ROI>` object.

.. note:: All placed ROIs are created and stored via the :doc:`ImageView <../api/generated/gui.image_widget.ImageView>`
    class. As long as no switch between displayed frame happens, this is the only place where a reference to the
    annotations exist. Only when the display switches to another frame is the annotation data passed on to the
    :doc:`ImageStack <../api/generated/gui.image_widget.ImageStack>` object.

Depending on the automation settings set in :doc:`settings_menu`, the currently active grid position moves automatically
to the next position in line. How the grid is traversed can be changed to `row-wise`, `column-wise` or `off`.

.. _frame clicking:

Frame Clicking
--------------
One noteworthy method of manual data annotation is what I like to call ``frame clicking``.
This is what I found to be the fastest way for annotating the suture data, although it might not necessarily be the
most accurate.

``Frame clicking`` turns the automatic traversal of the suture grid during annotation from row- or column-wise to
"frame-wise". This means that for each placed ROI, the automation process moves the image data one frame forward. The
currently active grid position is not changed during this switch.

.. rubric:: How to activate

You can activate the ``frame clicking`` behavior by adjusting the automation settings via the :doc:`settings_menu` in
the following way:

1. Set the `automatic progression` setting to any method you like the most.
2. For the selection box setting the *Mode for automatic progression through the grid* to never.
   This prevents the automation process from ever changing the currently active grid position by itself, without the
   user giving explicit input to do so.
3. Check the now available box for *Activate frame clicking*

What makes this method fast for manual annotation of sutures is that a lot of the sutures do not move very much between
frames, making clicking very fast as no large mouse movements have to be made. This works especially well for sutures in
the lower rows of the suture grid, as not much movement happens there. The problem of inaccuracy comes in when clicking
too fast for sutures that **do** move quite a lot during the surface oscillation. You should take some care to not click
too fast for the upper rows of the suture grid.

.. rubric:: Additional tips

In order to additionally help with the method of `frame clicking`, you can lock the view of the
:doc:`ImageView <../api/generated/gui.image_widget.ImageView>` by using the assigned hotkey. Locking the view changes
two things:

1. You can no longer move the displayed image region using the mouse. This helps with fast clicking through frames as
   you do not want to accidentally move the image during fast clicks with slight mouse movements.
2. Placing a ROI is now done using only **Left click**, without holding down **CTRL**. This is possible because the GUI
   no longer needs to distinguish *click + drag to move image* from *click to place ROI* (because image movement is
   now locked).

Semi-Automated Labeling
=======================
In order to help speed up the manual labeling process some basic semi-automation functions are implemented into GLabel.
These 'automation' tools are very basic and situational. It heavily depends on your data if these are helpful or create
more work than they safe.

In concept, these automation tools are based on the idea that labeling one frame of data gives plenty of information
about how to label the next frame of data. If you data varies very heavily between frames, these methods are bound to
break.

The use of these tools can be set through the automation :doc:`settings_menu` and controlled via implemented hotkeys.

Copying ROIs
------------
Default hotkey
    **H**: Copy from previous frame

    **J**: Copy from next frame

The most basic labeling helper method is to simply copy existing annotations from the previous or the next frame onto
the currently displayed frame.
This saves some amount of work by requiring you to only move ROIs for which the annotated landmark position has changed,
without having to re-click each single ROI.
This should work well for very stationary data.

Copy ROIs globally
------------------
Default hotkey
    **Ctrl + 0**: Copy ROIs from this frame to all other frames

If you know that your data has very little movement in the landmarks to annotate you can label only one frame and copy
the labels to all other frames.
From then on you could inspect all ROIs by moving through the frames and only adjust ROI placements where needed.
This really only works well for landmarks with very little movement across all available frames.
If your landmarks move too much for this method, copying of individual frames to the next one is a better solution.

Copy & Track ROIs (Pixel Intensity)
-----------------------------------
Default hotkey
    **Z**: Copy and track darkest pixel

    **U**: Copy and track lightest pixel

When the landmarks to be labeled have a high contrast w.r.t to their background, tracking them based on their pixel
intensity might work well.
Depending on if you label dark landmarks (e.g. sutures) or light landmarks (e.g. laser/light points), two separate copy
and track functions exist.

1. Copy all ROIs from the previous frame to the exact same coordinates on the current frame.
2. For each ROI:

    a) Find the darkest/lightest pixel value in a :math:`10\times10` px neighborhood
    b) Move the ROI center to that pixel coordinate
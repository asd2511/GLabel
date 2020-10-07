Closeup View
============
The Closeup View displays a closeup of a 10x10 pixel neighborhood from the currently active ROI placement.

.. image:: ../res/closeup_elem.png

It uses a function to automatically maximize the contrast of pixel values in the shown neighborhood, which helps in
seeing the boundaries of any object to be labeled.

.. caution::
    The Closeup View will always put the center of the ROI placement at the central intersection of pixels, even if
    the actual ROI is placed at sub-pixel positions in the image data.

    This might cause a visible offset between the ROI object on the image data and the ROI in the closeup.

    You should always consider the ROI placement on the actual image data as the true placement of any ROI!
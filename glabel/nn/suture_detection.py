"""
Module with functions for utilizing neural networks to fully-automatically identify and track sutures in video data.

The written functions are utilized by the GLabel GUI through user interaction. They can also be used from outside of the
GUI by importing this module into your python script.
"""
import json
import h5py
from os import path
from typing import List, Tuple, Union
import itertools
from collections import namedtuple

import cv2
import numpy as np
from pyqtgraph import RectROI
from scipy import ndimage
from skimage.feature import peak_local_max
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
import efficientnet.tfkeras  # Necessary import for custom EfficientNet layers not included in default tensorflow

# Give Keras access to the dice loss function and IoU score
get_custom_objects().update({'dice_loss': dice_loss})
get_custom_objects().update({'iou_score': iou_score})

yolo_cfg = path.join(path.dirname(__file__), 'yolov3_custom.cfg')  #: Default path to YOLOv3 network .cfg file
yolo_weights = path.join(path.dirname(__file__), './yolov3_suture_regions.weights')  #: Default path to YOLOv3 weights
tiny_cfg = path.join(path.dirname(__file__), 'yolov3-tiny_custom.cfg')  #: Default path to YOLOv3-Tiny .cfg file
tiny_weights = path.join(path.dirname(__file__), 'yolov3-tiny_custom_last.weights')  #: Default path to YOLOv3-Tiny weights
unet_model = path.join(path.dirname(__file__), 'unet.h5')  #: Default path to U-Net model
runet_model = path.join(path.dirname(__file__), 'Runet.h5')  #: Default path to recurrent U-Net model
sortnet_model = path.join(path.dirname(__file__), 'efficient_sort.h5')  #: Default path to EfficientNetB0 model

RUNET_TIMESTEPS = 3  #: Default sequence length for recurrent U-Net inputs

#: Namedtuple for handling bounding box information in a readable way
Box = namedtuple('Box', ['top', 'left', 'width', 'height', 'frame', 'side'])
#: Namedtuple for handling Peak/Suture positions in a readable way
Peak = namedtuple('Peak', ['y', 'x', 'frame', 'side'])
#: Namedtuple for combining information about suture position and inference values in a readable way
SortingPrediction = namedtuple('SortingPrediction', ['pred_id', 'y', 'x', 'frame', 'side', 'probabilities'])


class ProgLogger(Callback):
    """
    Custom callback for communication of inference progress to the :pyqt:`QProgressDialog <qprogressdialog>` of main
    GUI thread.

    .. caution::
        Callback for canceling the inference process is not supported yet!

    :param progress_callback: PyqtSignal which is used to emit a single `1` every time inference has finished processing
        a batch.
    :type progress_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param cancel_callback: Callback used for triggering interupt of the inference process.
    :type cancel_callback: Any
    """
    def __init__(self, progress_callback, cancel_callback):
        super(ProgLogger, self).__init__()
        self.prog_clbk = progress_callback  #: Callback used for sending progress updates
        self.cancel_clbk = False  #: TODO: Not implemented yet!

    def on_predict_batch_end(self, batch, logs=None):
        """
        Called at the end of every batch of the prediction.

        Is used to send a signal to the main GUI thread that a batch has been completed, which will be represented by
        an increase in the progress bar.
        We simply emit the integer value of 1 here, which is received by the main GUI thread.

        For more info on this method see:
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback#on_predict_batch_end

        :param batch: Index of the finished batch.
        :type batch: int
        :param logs: Dictionary with metric results for this batch.
        :type logs: dict
        """
        # Receiving the signal to cancel the inference should stop the tensorflow predict loop. This is not yet
        # implemented (this condition will never be True), because I could not find a way to correctly do this yet.
        if self.cancel_clbk:
            self.model.stop_training = True
        elif self.prog_clbk:
            self.prog_clbk.emit(1)
        else:
            pass


class UNetInfGen(Sequence):
    """
    Generator object creating batches for U-Net and recurrent U-Net models for predicting individual suture positions.

    **Bases:** :tf:`Sequence <tf/keras/utils/Sequence>`

    :param img_data: Image data for which the suture locations are predicted. This image data is the complete, data as
        opened by the user in the GUI (no cropping or any other processing).
    :type img_data: np.ndarray
    :param yolo_boxes: Bounding boxes for regions containing sutures as predicted by the first inference step. List
        should have same length as image data has frames. Each list entry consists of two :class:`Box` objects for the
        left and right view in the frame respectively.
    :type yolo_boxes: List[Box, Box]
    :param batch_size: Batch size to use for inference.
    :type batch_size: int
    :param int timesteps: Number of timesteps to use when creating input data for recurrent U-Net.
    :param bool rgbinput: Boolean setting that if set to True will create 3-channel RGB input data, or if set to False
        will create single channel grayscale data.
    """
    def __init__(self, img_data, yolo_boxes, batch_size, timesteps=None, rgbinput=False):
        super().__init__()
        self.img_data = img_data  #: Image data from which to create input data
        self.yolo_boxes = yolo_boxes  #: Bounding boxes from first inference step indicating suture grid regions
        self.unet_boxes = normalize_boxes(self.yolo_boxes, 224)  # Convert irregular yolo boxes to regular 224x224 boxes
        self.batch_size = batch_size  #: Batch size
        self.timesteps = timesteps  #: Timesteps used for creating recurrent U-Net input data
        self.rgbinput = rgbinput  #: Boolean toogle for creating RGB or grayscale input data

        self.batch_calls = []  #: List keeping track of which batches were requested by tensorflow during inference

    def __len__(self) -> int:
        """
        Return the total number of input batches available from the passed data.

        For each frame in the image data, two inputs will be created, one for each stereoscopic view. The total number
        of available input samples is therefore:

        .. math::

            n_{batches} = \frac{2\cdotn_{frames}}{\textrm{batch_size}}

        The calculated value is rounded up to the nearest integer if necessary.

        :return: Number of available input batches.
        """
        return int(np.ceil((self.img_data.shape[0]*2) / self.batch_size))

    def __getitem__(self, index) -> np.ndarray:
        """
        Get a single batch of input data.

        Will be called by tensorflow during inference process for feeding input data in batches.

        .. todo:: Update batch creation for recurrent U-Net! Current implementation is not reliable!

        For each frame, the next :attr:`RUNET_TIMESTEPS` frames will be accessed to build the input sample. For details
        of the building process, see :func:`get_runet_input` and :func:`get_unet_input`.

        :param index: Index of batch to return.
        :type index: int
        :return: Input batch for RUNet inference of shape [batch_size, timesteps, 224, 224, 1].
        """
        # Index into image data indicating starting point of batch. We divide by 2 because each frame of the image data
        # contains 2 used inputs for the network.
        batch_start = (index * self.batch_size) // 2

        if self.timesteps is None or self.timesteps == 1:
            batch_inputs = self.get_batch(index * self.batch_size)
        else:
            batch_inputs = self.get_recurrent_batch(batch_start)

        return batch_inputs

    def get_batch(self, batch_start) -> np.ndarray:
        """
        Build a batch of input data for non-recurrent U-Net models.

        This method will utilize the :func:`get_unet_input` function for creating individual inputs and group them
        together into a full batch of input data.

        :param int batch_start: Index into available image data indicating where to start building the current batch
            from. From this index onward :attr:`batch_size` samples will be created from the data.
        :return: Numpy array of shape (:attr:`batch_size`, 224, 224, channels)
        """
        channels = 3 if self.rgbinput else 1
        batch_inputs = np.empty((self.batch_size, 224, 224, channels))

        # Iterate over needed frames of data. As each frame contains two regions for which inputs are generated, the
        # batch size is halved to create the correct number of samples for a batch.
        # max function ensures correct behavior for a batch_size of 1
        for idx in range(self.batch_size):
            ctr = batch_start + idx
            frame_idx = ctr // 2
            side_idx = ctr % 2

            if frame_idx >= self.img_data.shape[0]:
                continue

            box = self.unet_boxes[frame_idx][side_idx]
            img = self.img_data[frame_idx]
            input_img, input_box = get_unet_input(img, box, 224)

            if self.rgbinput:
                input_img = np.repeat(input_img, 3, axis=3)
            batch_inputs[idx] = input_img

        return batch_inputs

    def get_recurrent_batch(self, batch_start) -> np.ndarray:
        """
        Build a batch of input data for recurrent U-Net models.

        This method will utilize the :func:`get_runet_input` function for creating individual inputs and group them
        together into a full batch of input data.

        .. warning:: The current implementation is not reliable and must be updated to be in line with the non-recurrent
            batch creation!

        :param int batch_start: Index into available image data indicating where to start building the current batch
            from. From this index onward :attr:`batch_size` samples will be created from the data.
        :return: Numpy array of shape (:attr:`batch_size`, 224, 224, channels)
        """
        batch_inputs = np.empty((self.batch_size, self.timesteps, 224, 224, 1))
        # Iterate over needed frames of data. As each frame contains two regions for which inputs are generated, the
        # batch size is halved to create the correct number of samples for a batch.
        for idx in range(self.batch_size // 2):
            ctr = batch_start + idx

            # If the current batch would ask for too many image frames that do not exist anymore, skip the loop.
            if ctr >= self.img_data.shape[0]:
                continue

            # If we run out of enough imgs/boxes to form a complete time sequence, use a reversed sequence. Should
            # still be a valid data sequence because of periodicity of movement in data --> Direction should not
            # matter
            reverse_seq = True if ctr > self.img_data.shape[0] - self.timesteps else False
            used_boxes = self.unet_boxes[ctr:ctr+self.timesteps] if not reverse_seq else \
                self.unet_boxes[ctr+1-self.timesteps:ctr+1][::-1]
            used_imgs = self.img_data[ctr:ctr+self.timesteps] if not reverse_seq else \
                self.img_data[ctr+1-self.timesteps:ctr+1][::-1]
            l_boxes = [boxes[0] for boxes in used_boxes]
            r_boxes = [boxes[1] for boxes in used_boxes]
            l_input, l_box = get_runet_input(used_imgs, l_boxes, 224)
            r_input, r_box = get_runet_input(used_imgs, r_boxes, 224)

            batch_inputs[idx*2] = l_input
            batch_inputs[idx*2+1] = r_input

        return batch_inputs


class SortGen(Sequence):
    """
    Generator object creating batches for suture sorting network (EfficientNetB0).

    **Bases:** :tf:`Sequence <tf/keras/utils/Sequence>`

    :param List[Peak] peaks: Unsorted list of :class:`Peak` objects as generated by peak finding step.
    :param np.ndarray suture_maps: Probability maps for individual suture positions as generated by suture
        identification step. Array of shape [#frames * 2, patch_size, patch_size] (default patch_size = 224). Ordering
        of maps is alternating left-right view side of one frame before going to next frame.
    :param List[List[Box]]: Bounding boxes corresponding to suture maps. Required to relate peak coordinates determined
        on suture probability maps back to full image coordinate system.
    :param int batch_size: Batch size.
    :param int peak_r: Radius of peak markings when creating input data for sorting network. Created markings will be
        of size (peak_r*2+1, peak_r*2+1) to have central pixel of marking at peak coordinate.
    :param bool binary_basemap: Boolean setting for switching mode of input data creation. If True, the first (red)
        channel of the input maps is created as a binary map of range [0, 1], with each peak coordinate marked as a
        rectangle of size defined by :attr:`peak_r`. If set to False, the suture probability map as passed in
        :attr:`suture_maps` is used as the first (red) channel. Inference accuracy might differ depending on what type
        of data the sorting network was trained.
    """
    def __init__(self, peaks, suture_maps, suture_boxes, batch_size, peak_r=2, binary_basemap=False):
        super().__init__()
        self.peaks = peaks  #: Unsorted list of :class:`Peak` objects for which input data is created
        self.suture_maps = suture_maps  #: 3D numpy array containing suture probability maps
        self.suture_boxes = suture_boxes  #: List of length-2 lists of bounding boxes of suture probability maps
        self.batch_size = batch_size  #: Batch size for input batch creation
        self.peak_r = peak_r  #: Radius of peak markings in input data
        self.binary_basemap = binary_basemap  #: Boolean value telling if first channel is binary or grayscale
        self.first_frame = self.calc_frame_offset(self.peaks)  #: Value for when inference did not start at first frame

        self.num_samples_total = len(self.peaks)  #: Number of input batches

    @staticmethod
    def calc_frame_offset(peaks) -> int:
        """
        If inference was not started from the first frame, the passed peaks have an offset in their `frame` property
        which needs to be taken care of when trying to access the suture_maps list with the `frame` index. We therefore
        need to find the smalles frame number from all the passed peaks, which is then subtracted from each listed frame
        index when trying to access the correct corresponding suture map.

        :param peaks: List of peaks for which the lowest occurring frame number is returned
        :type peaks: List[Peak]
        :return: Lowest occurring frame number from the passed peaks indicating the frame inference started at.
        """
        cur_first = np.inf
        for peak in peaks:
            if peak.frame < cur_first:
                cur_first = peak.frame

        return cur_first

    def __len__(self) -> int:
        return int(np.ceil(self.num_samples_total / self.batch_size))

    def __getitem__(self, index):
        batch_peaks = self.peaks[index * self.batch_size:(index+1)*self.batch_size]
        batch_inputs = np.empty((self.batch_size, 224, 224, 3))

        for idx, peak in enumerate(batch_peaks):
            # suture_maps are of shape [#frames*2, 224, 224], with two consecutive maps corresponding to the left and
            # right side of the same frame respectively.
            side_offset = 0 if peak.side == 'left' else 1
            map_idx = (peak.frame - self.first_frame) * 2 + side_offset
            map_box = self.suture_boxes[peak.frame - self.first_frame][side_offset]

            if self.binary_basemap:
                base_map = self.get_binary_basemap(peak, map_box)
            else:
                base_map = self.suture_maps[map_idx]

            y = peak.y - map_box.top
            x = peak.x - map_box.left
            input_map = np.zeros((224, 224, 3))
            input_map[:, :, 0] = base_map  # First channel are all predicted sutures (probability map)
            input_map[y-self.peak_r-1:y+self.peak_r, x-self.peak_r-1:x+self.peak_r, 1] = 1  # Second and third channel are
            input_map[y-self.peak_r-1:y+self.peak_r, x-self.peak_r-1:x+self.peak_r, 2] = 1  # only the single suture

            batch_inputs[idx] = input_map

        return batch_inputs

    def get_binary_basemap(self, peak, map_box):
        """
        Get fully binary input map showing all suture peak positions belonging to patch of passed `peak`.

        Use this function to generate the base map if a truly binary map is wanted instead of using the actual
        probability map generated by the suture finding network (which has values in the full range [0, 1]).

        :param peak: Peak for which to generate the base map.
        :type peak: Peak
        :param map_box: Bounding box for patch to generate input base map for.
        :type map_box: Box
        :return: Truly binary input base map.
        """
        frame = peak.frame
        side = peak.side
        # Find all peaks that belong to the same patch as the passed peak
        valid_peaks = list(filter(lambda p: p.frame == frame and p.side == side, self.peaks))

        basemap = np.zeros((224, 224))
        for val_p in valid_peaks:
            y = val_p.y - map_box.top
            x = val_p.x - map_box.left
            basemap[y-self.peak_r-1:y+self.peak_r, x-self.peak_r-1:x+self.peak_r] = 1

        return basemap


class SortNetInfGen(Sequence):
    """
        **Bases:** :class:`Sequence <tf.keras.utils.Sequence>`

        Generator object creating batches for EfficientNet inference predicting suture sorting.

        :param suture_maps:
        :type suture_maps:
        :param suture_boxes:
        :type suture_boxes:
        :param batch_size: Batch size to use for inference.
        :type batch_size: int
        :param peak_r: Radius of peaks created in individual suture channels. Defaults to 2.
        :type peak_r: int
        :param debug: Optional boolean setting if variables for better debugging should be created. Defaults to False.
        :type debug: bool
        """
    def __init__(self, suture_maps, suture_boxes, batch_size, peak_r=2, filter_dist_relative=0.3, debug=False):
        super().__init__()
        # Maps are of shape [#maps, 224, 224], with the sequence of maps being left/right view alternating
        self.suture_maps = suture_maps
        self.suture_boxes = suture_boxes
        self.batch_size = batch_size
        self.peak_r = peak_r
        self.filter_dist_relative = filter_dist_relative
        self.debug = debug

        self.peaks = []  #: List of all found peaks in the passed suture_maps of type List[Peak].
        self.used_peaks = []  #: Peaks in sequence of use in creating input samples (only unique index calls)
        self.num_samples_total = None  #: Total number of peaks found in the passed suture_maps
        self.process_predictions()

        self.batch_calls = []  #: List keeping track of which batches were requested by tensorflow during inference

        if self.debug:
            self.inspect_inputs = []

    def process_predictions(self) -> None:
        """
        Process the passed suture predictions from the recurrent UNet and prepare the data for inference using
        EfficientNet.

        .. note::
            This does not cache the prepared inference inputs! It will only search for all peaks in the passed suture
            maps and cache their suture map correspondence and coordinates.

        Every single predicted suture will be converted into a single input sample for the EfficientNet, making it
        necessary to know the exact number of found peaks before starting the feed iteration, because we need to know
        the number of batches in our data. This method will extract all individual predicted sutures as peaks in the
        maps and store their info.
        """
        for map_idx, prediction in enumerate(self.suture_maps):
            frame_idx = map_idx // 2  # Two predictions per frame
            box = self.suture_boxes[frame_idx][map_idx%2]
            peaks = peak_local_max(prediction, min_distance=3, threshold_abs=0.05)
            # Filter out any peaks that are too far from the center point of mass of all peaks
            r, c = np.where((abs(peaks - peaks.mean(axis=0)) > (224 * self.filter_dist_relative,) * 2))
            peaks = np.delete(peaks, r, axis=0)

            self.peaks.extend([Peak(coords[0], coords[1], box.frame, box.side) for coords in peaks])

        self.num_samples_total = len(self.peaks)

    def __len__(self) -> int:
        """
        Return the number of batches available for the data.

        The total number of available samples is dependent on the number of predicted sutures in the passed input data.
        The preprocessing done by :func:`process_predictions` is used to extract the total number of predicted sutures,
        which divided by the specified batch size gives the total number of input batches.

        :return: Total number of available input batches.
        """
        return int(np.ceil(self.num_samples_total / self.batch_size))

    def __getitem__(self, index) -> np.ndarray:
        """
        Get a single batch of input data.

        Will be called by tensorflow during inference process for feeding input data in batches.

        The `index` parameter determines which peaks found by :func:`process_predictions` will be used for this batch.
        Each peak will be processed individually, by combining its global map containing all peaks for its corresponding
        patch (the suture map) with the individual peak on its own. The input sample itself will be constructed as a
        3-channel image, with the first channel being the global map, and the other channels being the same map created
        for the individual suture.

        :param index: Index of batch to return.
        :type index: int
        :return: Input batch for RUNet inference of shape [batch_size, 224, 224, 3].
        """
        # This method is called by tensorflow quite often in the process of running the prediction, even
        # mixing the order of the `index` calls. As we need a correct association between the used peaks and the
        # prediction outputs, we make sure to only save the used peaks on unique calls to the generator. This
        # prevents mixups and an overly large list of used peaks after prediction has finished.
        if index not in self.batch_calls:
            self.batch_calls.append(index)
            save_peaks = True
        else:
            save_peaks = False

        batch_peaks = self.peaks[index*self.batch_size:(index+1)*self.batch_size]
        batch_inputs = np.empty((self.batch_size, 224, 224, 3))

        for idx, peak in enumerate(batch_peaks):
            # suture_maps are of shape [#frames*2, 224, 224], with two consecutive maps corresponding to the left and
            # right side of the same frame respectively.
            side_offset = 0 if peak.side == 'left' else 1
            map_idx = peak.frame * 2 + side_offset
            y = peak.y
            x = peak.x

            base_map = self.suture_maps[map_idx]
            map_box = self.suture_boxes[peak.frame][side_offset]
            input_map = np.zeros((224, 224, 3))
            input_map[:, :, 0] = base_map  # First channel are all predicted sutures (probability map)
            input_map[y-self.peak_r:y+self.peak_r, x-self.peak_r:x+self.peak_r, 1] = 1  # Second and third channel are
            input_map[y-self.peak_r:y+self.peak_r, x-self.peak_r:x+self.peak_r, 2] = 1  # only the single suture

            batch_inputs[idx] = input_map

            if save_peaks:
                peak_img_rel = Peak(peak.y + map_box.top, peak.x + map_box.left, peak.frame, peak.side)
                self.used_peaks.append(peak_img_rel)

        if self.debug:
            self.inspect_inputs.append(batch_inputs)

        return batch_inputs


def fix_data_bits(img_data) -> np.ndarray:
    """
    Pre-process image data by converting datatype from np.uint16 to np.uint8 if necessary.

    Conversion is necessary if image data is data converted from CINE format, as the used conversion script parses the
    data into 16bit format (but actually using only 10bit), unsuitable for inference.

    The conversion works by rescaling the image data to the range [0, 255] and then converting to unsigned 8bit.

    :param img_data: The image data to convert.
    :type img_data: np.ndarray
    :return: Image data converted to unsigned 8bit.
    """
    if img_data.dtype == np.uint16:
        # Convert data from its default 10bit representation to 8bit
        img_data = (((img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))) * 255).astype(np.uint8)

    return img_data


def convert_to_ntboxes(list_boxes) -> List[Box]:
    """
    Convert list of float values for bounding boxes to list of namedtuple objects for improved readability.

    :param list list_boxes: Bounding box values.
    :return: List of :class:`Box` objects.
    """
    nt_boxes = []
    for frame_idx, frame_boxes in enumerate(list_boxes):
        l_box = frame_boxes[0]
        r_box = frame_boxes[1]
        fnt_boxes = []
        for box, side in zip([l_box, r_box], ['l', 'r']):
            nt_box = Box(box[1], box[0], box[2], box[3], frame_idx, side=side)
            fnt_boxes.append(nt_box)
        nt_boxes.append(fnt_boxes)

    return nt_boxes


def convert_to_ntpredictions(peaks, id_probs):
    """
    Convert individual lists of discrete suture coordinates and inferred ID probabilities to single list of
    :class:`SortingPrediction` objects.

    :param list peaks: Discrete suture coordinates.
    :param list id_probs: Grid ID probabilities.
    :return: List of SortingPrediction namedtuple objects unifying all data of sorting inference.
    """
    sorted_peaks = []
    for peak, prob in zip(peaks, id_probs):
        sorted_peaks.append(SortingPrediction(pred_id=np.argmax(prob),
                                              y=peak.y,
                                              x=peak.x,
                                              frame=peak.frame,
                                              side=peak.side,
                                              probabilities=prob))

    return sorted_peaks


def find_cfg_setting(cfg_file, setting) -> Union[str, None]:
    """
    Find the value of the first occurrence of specified `setting` in the specified `cfg_file`.

    :param cfg_file: Path to the text file to search for the setting value.
    :type cfg_file: str
    :param setting: Name of the setting to return the value of.
    :type setting: str
    :return: Value of the setting as a string as written in the cfg file or None if the setting could not be found.
    """
    with open(cfg_file, 'r') as f:
        line = f.readline().strip('\n')
        cnt = 1
        while line:
            # Skip commented lines
            if line[0] == '#':
                pass
            else:
                split_line = line.split('=')
                if split_line[0] == setting:
                    return split_line[1]

            # Read the next line each iteration
            line = f.readline().strip('\n')
        return None


def run_region_detect(img_data, net, frame_delta, progress_callback, message_callback, cancel_callback=None,
                      weights=None, start_frame=0) -> \
        Tuple[List[List[Box]], List[List[float]]]:
    """
    Detect regions containing sutures with the (Tiny) YOLO network.

    :param img_data: Image data for which to detect suture regions. Typically a sequence of image frames.
    :type img_data: np.ndarray
    :param net: Which type of YOLO network to use for detection. Available networks are YOLOv3 and Tiny YOLO. If `net`
        is specified as 'yolo', the YOLOv3 network is used, otherwise the Tiny YOLO net.
    :type net: str
    :param frame_delta: Number of frames to move before the next inference. E.g. passing a value of 25 will run
        inference only for every 25th frame and use the detected bounding box for the next 25 frames to the next
        prediction. Value must be in range (0, number of frames)!
    :type frame_delta: int
    :param progress_callback: PyQtSignal which is used to indicate progression during inference.
    :type progress_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param message_callback: PyQtSignal which is used to transmit messages back to the main GUI thread to be displayed
        on the progress dialog for inference.
    :type message_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :return: Tuple of two lists. The first listing the bounding boxes detected for the given frames. The second being
        the corresponding confidence values.
    """
    if message_callback:
        message_callback.emit("Beginning suture region detection...")

    num_frames = img_data.shape[0]
    # assert 0 < frame_delta <= num_frames, "The step between frames must be in 0 < frame_delta < #frames !"
    # Limit the frame_delta value to the maximum value if it would be too large otherwise
    if frame_delta > num_frames:
        frame_delta = num_frames

    # In case of a single image, expand the dimension as (1 frame, height, width) to allow iteration over frames
    if img_data.ndim == 2:
        img_data = np.expand_dims(img_data, axis=0)

    if img_data.dtype == np.uint16:
        # Convert data from its default 10bit representation to 8bit
        img_data = (((img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))) * 255).astype(np.uint8)

    if message_callback:
        message_callback.emit("Reading YOLO network...")
    height, width = img_data[0].shape
    if net == 'yolo':
        yolo = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        in_height = in_width = 768
    elif net == 'tiny':
        yolo = cv2.dnn.readNet(tiny_weights, tiny_cfg)
        in_height = in_width = 416
        # TODO: This is because of an error during training. I accidentally left the number of channels at 3 during
        #  training, which requires filling the color dimension with copies of the gray-valued image. Should be
        #  re-trained in the future!
        img_data = np.repeat(img_data[:, :, :, np.newaxis], 3, axis=3)
    else:
        # Assume that the passed `net` is a path to a .cfg file -> Passing `weights` is then required
        assert weights is not None, \
            f"Path to weights file is required when passing network as path but received {weights}!"
        net_path = path.abspath(net)
        weights_path = path.abspath(weights)
        yolo = cv2.dnn.readNet(weights_path, net_path)
        in_height = int(find_cfg_setting(net_path, 'height'))
        in_width = int(find_cfg_setting(net_path, 'width'))
        in_channels = int(find_cfg_setting(net_path, 'channels'))
        # If there are color channels required, but our data is only single-channeled, expand it
        if in_channels > 1 and len(img_data.shape) == 3:
            img_data = np.repeat(img_data[:, :, :, np.newaxis], in_channels, axis=3)

    out_layers = get_output_layers(yolo)

    if message_callback:
        message_callback.emit("Predicting suture regions...")

    confidences = []
    boxes = []
    for ctr, frame in enumerate(img_data[::frame_delta]):
        frame = img_data[ctr]

        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1/255.0, size=(in_height, in_width), mean=0, swapRB=False, crop=False)
        yolo.setInput(blob)
        outputs = yolo.forward(out_layers)

        # Stack the predictions from all three yolo layers
        detections = np.vstack(outputs)

        # Split all detections into belonging to left or right half of image
        l_detections = detections[np.where(detections[:, 0] < 0.5)[0]]
        r_detections = detections[np.where(detections[:, 0] > 0.5)[0]]
        # Find the prediction with the maximum confidence on both sides of the image
        max_l_detection = l_detections[np.argmax(l_detections[:, 4])]
        max_r_detection = r_detections[np.argmax(r_detections[:, 4])]

        frame_boxes = []
        frame_confidences = []
        for detection in [max_l_detection, max_r_detection]:
            # Find the confidence for the region
            box_confidence = detection[4]
            class_confidence = detection[5]
            # Get the bounding box coordinates
            xc = int(detection[0] * width)
            yc = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = xc - w // 2
            y = yc - h // 2

            frame_confidences.append(float(box_confidence))
            frame_boxes.append([x, y, w, h])

        if progress_callback:
            progress_callback.emit(1)

        # Add the confidences and boxes for the current frame to the overall list
        confidences.append(frame_confidences)
        boxes.append(frame_boxes)

    # If only every nth frame was used for inference --> Copy the inference for the next n frames to get a prediction
    # for each frame
    if frame_delta > 1:
        # Repeat the predictions #frame_delta times to fill up list that has same length as number of frames
        confidences = list(itertools.chain.from_iterable(itertools.repeat(x, frame_delta) for x in confidences))
        boxes = list(itertools.chain.from_iterable(itertools.repeat(x, frame_delta) for x in boxes))
        # Handle cases where number of frames is unevenly divided by frame_delta (remove overshoot)
        if len(confidences) > num_frames:
            confidences = confidences[:num_frames]
        if len(boxes) > num_frames:
            boxes = boxes[:num_frames]

    # Convert the list of raw boxes to a namedtuple objects for easier comprehension of contents
    conv_boxes = []
    for ctr, (l_box, r_box) in enumerate(boxes):
        l_box_nt = Box(top=l_box[1], left=l_box[0], width=l_box[2], height=l_box[3], frame=ctr+start_frame, side='left')
        r_box_nt = Box(top=r_box[1], left=r_box[0], width=r_box[2], height=r_box[3], frame=ctr+start_frame, side='right')
        conv_boxes.append([l_box_nt, r_box_nt])

    return conv_boxes, confidences


def load_regions_inference(filename, formatted=False) -> Union[List, Tuple]:
    """
    Load inference data for suture regions from a saved .json file.

    :param filename: Path to the inference file.
    :type filename: str
    :param formatted: Boolean setting that if True, will have the function return the final formatted list of regions
        which can directly be used by the ImageStack class to display bounding boxes.
    :type formatted: bool
    :return: If formatted=True: List of lists that contain the region bounding box information.
        If formatted=False: Tuple of two lists of lists, first for bounding boxes, second for confidences.
    :raises KeyError: if the file does not contain a listing of `settings`
    """
    # Read the saved json string as dictionary
    with open(filename, 'r') as f:
        inf_data = json.load(f)

    settings = inf_data.pop('settings')
    num_frames = settings['to_frame'] - settings['from_frame']
    total_frames = settings['total_frames']

    if formatted:
        # Each frame element of the bbox_stack has 4 elements in the following order:
        # Left patch bounding box (RectROI), left patch confidence, right patch bounding box (RectROI), right patch
        # confidence
        bbox_stack = [[None, None, None, None] for _ in range(total_frames)]
        for key, rect_dict in inf_data.items():
            frame_idx, side = key.split('_')  # Extract frame number and patch side from the encoded key string
            frame_idx = int(frame_idx)
            entry_idx = 0 if side == 'l' else 2
            bbox = RectROI((rect_dict['x'], rect_dict['y']), (rect_dict['w'], rect_dict['h']))
            bbox_stack[frame_idx][entry_idx] = bbox
            bbox_stack[frame_idx][entry_idx + 1] = rect_dict['conf']

        return bbox_stack

    else:
        region_boxes = [[] for _ in range(num_frames)]
        region_confs = [[] for _ in range(num_frames)]
        for key, rect_dict in inf_data.items():
            frame_idx, side = key.split('_')
            frame_idx = int(frame_idx)
            dict_idx = frame_idx - settings['from_frame']
            bbox = Box(
                top=int(rect_dict['y']),
                left=int(rect_dict['x']),
                width=int(rect_dict['w']),
                height=int(rect_dict['h']),
                frame=frame_idx,
                side='left' if side == 'l' else 'right'
            )
            region_boxes[dict_idx].append(bbox)
            region_confs[dict_idx].append(rect_dict['conf'])

        return region_boxes, region_confs


def detect_roi(img_data) -> np.ndarray:
    """
    Detect the suture region using YOLO

    :param img_data: Image data to detect suture regions in
    :type img_data: np.ndarray
    """
    # In case of a single image, expand the dimension as (1 frame, height, width) to allow iteration over frames
    if img_data.ndim == 2:
        image_data = np.expand_dims(img_data, axis=0)

    height, width = img_data[0].shape
    yolo = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    out_layers = get_output_layers(yolo)

    confidences = []
    boxes = []
    for frame in img_data:
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(height, width), mean=0, swapRB=False, crop=False)
        yolo.setInput(blob)
        outputs = yolo.forward(out_layers)

        # Stack the predictions from all three yolo layers
        detections = np.vstack(outputs)

        # Split all detections into belonging to left or right half of image
        l_detections = detections[np.where(detections[:, 0] < 0.5)[0]]
        r_detections = detections[np.where(detections[:, 0] > 0.5)[0]]
        # Find the prediction with the maximum confidence on both sides of the image
        max_l_detection = l_detections[np.argmax(np.max(l_detections[:, :5], axis=1))]
        max_r_detection = r_detections[np.argmax(np.max(r_detections[:, :5], axis=1))]

        for detection in [max_l_detection, max_r_detection]:
            frame_boxes = []
            frame_confidences = []
            # Find the confidence for the region
            scores = detection[:5]
            confidence = scores[np.argmax(scores)]
            # Get the bounding box coordinates
            xc = int(detection[0] * width)
            yc = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = xc - w // 2
            y = yc - h // 2

            frame_confidences.append(float(confidence))
            frame_boxes.append([x, y, w, h])

            # Draw the bounding box on the current frame
            draw_bbox(frame, "sutures", confidence, x, y, x+w, y+h)

        # Add the confidences and boxes for the current frame to the overall list
        confidences.append(frame_confidences)
        boxes.append(frame_boxes)

    return img_data


def get_output_layers(net) -> List[str]:
    """
    Get the YOLO layers from the YOLO net that make the actual region prediction

    :param net: The YOLO network
    :type net: cv2.dnn_Net
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_bbox(img, label, confidence, x, y, x_end, y_end) -> None:
    """
    Draw a bounding around the region predicted by YOLO.

    The bounding box coordinates are defined by the absolute pixel values specified by *x, y, x_end, y_end*.

    The color of the bounding box reflects the certainty of the detection, with the green channel reflecting certainty
    and the red channel uncertainty.

    The image is modified in-place and is not returned after drawing onto it.

    :param img: Image to draw onto
    :type img: np.ndarray
    :param label: Class name shown next to bounding box
    :type label: str
    :param confidence: Confidence of detected bounding box
    :type confidence: float
    :param x: Coordinate of left edge of bounding box
    :type x: int
    :param y: Coordinate of top edge of bounding box
    :type y: int
    :param x_end: Coordinate of right edge of bounding box
    :type x_end: int
    :param y_end: Coordinate of bottom edge of bounding box
    :type y_end: int
    """
    color = ((1-confidence)*255, confidence*255, 128)
    cv2.rectangle(img, (x, y), (x_end, y_end), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, str(confidence), (x_end + 10, y_end + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_unet_box(yolo_box, size) -> Box:
    # Find center point of suture region defined by YOLO box
    x_center = yolo_box.left + yolo_box.width // 2
    y_center = yolo_box.top + yolo_box.height // 2
    # Extend bounding box to defined size
    top = y_center - size // 2
    left = x_center - size // 2

    return Box(top=top, left=left, width=size, height=size, frame=yolo_box.frame, side=yolo_box.side)


def normalize_boxes(yolo_boxes, size) -> List[List[Box]]:
    """
    Convert the received yolo bounding boxes to evenly sized boxes usable for RUNet inputs.

    The returned list of frame boxes keeps the same ordering as the passed yolo boxes. As these should(!) be in the
    correct frame order, this order is preserved for the normalized boxes and can be used to correctly assign the boxes
    to their frames based on their list indices.

    :param yolo_boxes: Bounding boxes of suture regions. Each list entry is a list of 2 Boxes, one for each viewpoint.
    :type yolo_boxes: List[List[Box]]
    :param size: Size to enlarge all yolo boxes to.
    :type size: int
    :return: List of lists of regular sized boxes. Same data structure as before, just the boxes have been converted to
        be all of the same specified size.
    """
    unet_boxes = []
    for frame_boxes in yolo_boxes:
        l_idx = 0 if frame_boxes[0].side == 'left' else 1
        l_box = frame_boxes[l_idx]
        r_box = frame_boxes[1 - l_idx]

        unet_l_box = get_unet_box(l_box, size)
        unet_r_box = get_unet_box(r_box, size)

        unet_boxes.append([unet_l_box, unet_r_box])

    return unet_boxes


def get_unet_input(img, box, size) -> np.ndarray:
    """
    Extract the input image for prediction using the (single-frame/non-recurrent) UNet.

    :param img: Single image frame to extract the input image from.
    :type img: np.ndarray
    :param box: Irregular sized bounding box of suture region as predicted from YOLO network.
    :type box: Box
    :param size: Regular size to extend patch to. The single passed value will be used for both height and width.
    :type size: int
    :return: Image data of extended patch of shape [1, size, size, 1] ready for input into UNet.
    """
    # Extract the enlarged image patch
    xc = box.left + box.width // 2
    yc = box.top + box.height // 2
    resized_box = Box(top=yc-size//2,
                      left=xc-size//2,
                      width=size, height=size,
                      frame=box.frame, side=box.side)
    patch = img[resized_box.top:resized_box.top + size, resized_box.left:resized_box.left + size]
    if np.max(patch) > 1:
        # Normalize input to [0, 1]
        patch = patch / 255.0
    # TODO: New normalization to range [-1, 1] HAS TO BE CHECKED! (But should be ok) MUST ONLY BE USED FOR FINAL UNET!!
    patch = 2 * ((patch - patch.min()) / (patch.max() - patch.min())) - 1

    # Extend input dimensions to [1, size, size, 1] to fit into UNet
    patch = np.expand_dims(patch, axis=(0, 3))

    return patch, resized_box


def get_runet_input(img_sequence, boxes, size) -> Tuple[np.ndarray, Box]:
    sequence_length = img_sequence.shape[0]
    runet_input = np.empty((RUNET_TIMESTEPS, size, size))
    # Save only the "true" box for this input sequence, which is the box of the first frame
    box = boxes[0]

    assert sequence_length == RUNET_TIMESTEPS, f"Image sequence in RUNet input creation has different length than " \
                                               f"specified input sequence length! {sequence_length} != {RUNET_TIMESTEPS}"

    for i in range(sequence_length):
        img = img_sequence[i]
        bounding_box = boxes[i]
        patch, box = get_unet_input(img, bounding_box, size)
        # Fill the input sample in sequence with the patches
        runet_input[i] = patch[0, :, :, 0]
    # Extend input to shape [1, sequence, size, size, 1] to fit for Recurrent UNet
    runet_input = np.expand_dims(runet_input, axis=(0, 4))

    return runet_input, box


def run_suture_detect(img_data, bboxes, net, unet_batch_size, progress_callback, message_callback,
                      cancel_callback=None) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Perform segmentation of individual sutures.

    :param np.ndarray img_data: Video frames to segment sutures in.
    :param list bboxes: Bounding boxes as inferred by :func:`run_region_detect`.
    :param str net: Which neural network architecture to use. Currently only supports 'unet' and 'runet' or path to a
        saved .h5 model file.
    :param int unet_batch_size: Batch size to use for segmentation.
    :param progress_callback: PyQtSignal which is used to indicate progression during inference.
    :type progress_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param message_callback: PyQtSignal which is used to transmit messages back to the main GUI thread to be displayed
        on the progress dialog for inference.
    :type message_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param cancel_callback: Callback intended to cancel inference process. Currently not functional!
    :type cancel_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :return: Tuple of numpy array and list of lists (predictions, boxes).
        Predictions: The generated segmentation maps.
        Boxes: The expanded bounding boxes relative to full-sized images specifying where segmentation maps are located.
    """
    if message_callback:
        message_callback.emit("Reading (R)U-net...")

    if net == 'unet':
        model = load_model(unet_model)
    elif net == 'runet':
        model = load_model(runet_model)
    else:
        model = load_model(net)

    if message_callback:
        message_callback.emit("Preparing suture detection...")

    input_layer = model.get_layer(index=0)
    input_shape = input_layer.input_shape[0]
    # Find if the model expects recurrent input data
    if len(input_shape) == 5:
        timesteps = input_shape[1]
    else:
        timesteps = None

    rgbchannels = True if input_shape[-1] == 3 else False
    gen = UNetInfGen(img_data, bboxes, unet_batch_size, timesteps, rgbchannels)
    if progress_callback:
        prog_logger = ProgLogger(progress_callback, cancel_callback)
    else:
        prog_logger = Callback()

    if message_callback:
        message_callback.emit("Predicting suture locations...")

    # Returned predictions are of shape [#frames*2, 224, 224, 1]
    predictions = model.predict(gen,
                                workers=4,
                                verbose=1,
                                callbacks=[prog_logger])
    # The bounding boxes corresponding to the predictions
    boxes = gen.unet_boxes
    # If the batch_size divides the data unevenly, the last prediction batch will be filled to the full batch
    # size with empty data, which we do not care about or want.
    predictions = predictions[:len(boxes)*2]

    return np.squeeze(predictions), boxes


def load_suture_detect(filename, formatted=False) -> Tuple[np.ndarray, List]:
    """
    Load suture segmentations from file.

    :param str filename: Path to saved .h5 file.
    :param bool formatted: If loaded inferences should be returned as formatted data. Not implemented yet!
    :return: Tuple of segmentation maps and corresponding bounding boxes. Same as :func:`run_suture_detect`.
    """
    with h5py.File(filename, 'r') as f:
        settings = json.loads(f['settings'][()])

        num_frames = settings['to_frame'] - settings['from_frame']
        suture_boxes = [[None, None] for _ in range(num_frames)]
        suture_maps = np.empty((num_frames * 2, 224, 224))

        dict_combs = list(itertools.product(range(settings['from_frame'], settings['to_frame']), ['l', 'r']))
        dict_keys = ['_'.join(map(str, parts)) for parts in dict_combs]

        if formatted:
            # TODO: Implement final sorted suture map loading
            raise NotImplementedError("Loading of formatted suture maps is not implemented yet!")

        else:
            for idx, key in enumerate(dict_keys):
                frame_idx, side = key.split('_')
                frame_idx = int(frame_idx)
                side_idx = 0 if side == 'l' else 1

                grp = f[key]
                box = json.loads(grp['box'][()])
                suture_map = np.asarray(grp['map'])

                suture_boxes[idx//2][side_idx] = Box(
                    top=box['y'],
                    left=box['x'],
                    width=box['w'],
                    height=box['h'],
                    frame=frame_idx,
                    side='left' if side == 'l' else 'right'
                )
                suture_maps[idx] = suture_map

    return suture_maps, suture_boxes


def run_peak_finding(prob_maps, map_boxes, tight_boxes, distance, threshold, progress_callback, message_callback,
                     cancel_callback):
    """
    Determine discrete suture coordinates from generated segmentation maps.

    :param np.ndarray prob_maps: Segmentation maps generated in previous automation step.
    :param list map_boxes: List of lists of bounding boxes for segmentation maps.
    :param list tight_boxes: List of lists of bounding boxes of suture grid regions generated in first automation step.
    :param float distance: Distance paramter used by skimage's `peak_local_max` function.
    :param float threshold: Intensity threshold used by skimage's `peak_local_max` function.
    :param progress_callback: PyQtSignal which is used to indicate progression during inference.
    :type progress_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param message_callback: PyQtSignal which is used to transmit messages back to the main GUI thread to be displayed
        on the progress dialog for inference.
    :type message_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param cancel_callback: Callback intended to cancel inference process. Currently not functional!
    :type cancel_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :return: List of lists of unsorted discrete suture coordinates per frame.
    """
    if message_callback:
        message_callback.emit("Finding suture coordinates...")

    found_peaks = []
    for map_idx, pmap in enumerate(prob_maps):
        frame_idx = map_idx // 2  # Two predictions per frame
        side_idx = map_idx % 2
        tight_box = tight_boxes[frame_idx][side_idx]
        map_box = map_boxes[frame_idx][side_idx]
        # Distance transform on probability map to reduce any binary areas to peaks
        edt_map = ndimage.distance_transform_edt(pmap > 0.8)
        # Find the local maxima
        map_peaks = peak_local_max(edt_map, min_distance=distance, threshold_abs=threshold, indices=False)
        # Reduce groups of local maxima to single peak
        labels = ndimage.label(map_peaks)[0]
        merged_peaks = ndimage.center_of_mass(map_peaks, labels, range(1, np.max(labels) + 1))
        merged_peaks = np.array(merged_peaks)

        global_peaks = merged_peaks + [map_box.top, map_box.left]  # Add the offset of the patch
        # Filter out any peaks that are too far outside of the tight fitting suture region boxes, except for the top
        # portion where any offset is allowed because of the vocal fold movement at the top
        width_tol = tight_box.width * 0.1
        height_tol = tight_box.height * 0.1
        width_outlier_idcs = np.where(np.logical_or(global_peaks[:, 1] < tight_box.left-width_tol,
                                                     global_peaks[:, 1] > tight_box.left+tight_box.width+width_tol))[0].tolist()
        height_outlier_idcs = np.where(global_peaks[:, 0] > tight_box.top+tight_box.height+height_tol)[0].tolist()
        global_peaks = np.delete(global_peaks, width_outlier_idcs+height_outlier_idcs, axis=0).astype(int)

        found_peaks.extend([Peak(coords[0], coords[1], map_box.frame, map_box.side) for coords in global_peaks])

        if progress_callback:
            progress_callback.emit(1)

    return found_peaks


def load_peak_finding(filename) -> list:
    """
    Load peak finding results from file.

    :param str filename: Path to .json file to load data from.
    :return: List of unsorted discrete suture coordinates.
    """
    with open(filename, 'r') as f:
        inf_data = json.load(f)

    settings = inf_data.pop('settings')

    suture_peaks = []
    for key, rect_dict in inf_data.items():
        frame_idx, side = key.split('_')
        frame_idx = int(frame_idx)
        patch_peaks = inf_data[key]
        for peak_coords in patch_peaks:
            nt_peak = Peak(
                x=peak_coords[0],
                y=peak_coords[1],
                frame=frame_idx,
                side=side
            )
            suture_peaks.append(nt_peak)

    return suture_peaks


def run_peak_sort(peaks, probability_maps, bboxes, network, batch_size, binary, progress_callback, message_callback,
                  cancel_callback) -> List[SortingPrediction]:
    """
    Perform sorting of found discrete suture coordinates into suture grid structure.

    :param list peaks: List of lists of unsorted discrete suture coordinates per frame.
    :param np.ndarray probability_maps: Segmentation maps generated in second automation step.
    :param list bboxes: Bounding boxes corresponding to segmentation maps.
    :param str network: Path to network architecture .h5 file to use.
    :param int batch_size: Batch size for inference.
    :param bool binary: Boolean setting for use of binary or certainty values in creation of 'classification maps'.
    :param progress_callback: PyQtSignal which is used to indicate progression during inference.
    :type progress_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param message_callback: PyQtSignal which is used to transmit messages back to the main GUI thread to be displayed
        on the progress dialog for inference.
    :type message_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :param cancel_callback: Callback intended to cancel inference process. Currently not functional!
    :type cancel_callback: PyQt5.QtCore.pyqtBoundSignal.pyqtBoundSignal
    :return: List of :class:`SortingPrediction` specifying grid ID probabilities for each found suture coordinate.
    """
    if message_callback:
        message_callback.emit("Reading EfficientNet...")

    sortnet = load_model(network)

    if message_callback:
        message_callback.emit("Preparing suture sorting...")
    gen = SortGen(peaks, probability_maps, bboxes, batch_size, binary_basemap=binary)

    prog_logger = ProgLogger(progress_callback, cancel_callback)

    if message_callback:
        message_callback.emit("Sorting found sutures...")
    sortings = sortnet.predict(gen,
                               verbose=1,
                               callbacks=[prog_logger])

    # If the effnet_batch_size divides the data unevenly, the last prediction batch will be filled to the full batch
    # size with empty data, which we do not care about or want.
    sortings = sortings[:len(peaks)]

    sorted_peaks = convert_to_ntpredictions(peaks, sortings)

    return sorted_peaks


def load_peak_sort(filename) -> List[SortingPrediction]:
    """
    Load existing sorting inference from file.

    :param str filename: Path to .json file to load inference from.
    :return: List of :class:`SortingPrediction`.
    """
    with open(filename, 'r') as f:
        inf_data = json.load(f)

    settings = inf_data.pop('settings')

    sorted_sutures = []
    for key, rect_dict in inf_data.items():
        frame_idx, side = key.split('_')
        frame_idx = int(frame_idx)
        patch_sortings = inf_data[key]
        for sorted_peak in patch_sortings:
            sorting = SortingPrediction(
                pred_id=sorted_peak['pred_id'],
                x=sorted_peak['x'],
                y=sorted_peak['y'],
                frame=frame_idx,
                side=side,
                probabilities=sorted_peak['probabilities']
            )
            sorted_sutures.append(sorting)

    return sorted_sutures


def run_detect_and_sort(img_data, frame_delta, unet_batch_size, effnet_batch_size, progress_callback, message_callback,
                        cancel_callback):
    """
    Run full end-to-end inference for predicting suture grid positions and correspondences.

    The returned tuple contains the following data in this order:

    =================   ==============================  ================================================================
    **Var name**        **dtype/dimensions**            **Contains**
    yolo_boxes          List[#frames]->List[2]->Box     For each frame->For each side->Suture region bounding box
    yolo_confs          List[#frames]->List[2]->float   For each frame->For each frame->Bounding box confidence
    unet_predictions    ndarray[#frames*2, 224, 224]    For each side (alternating)->Suture probability map
    unet_boxes          List[#frames]->List[2]->Box     For each frame->For each side->Enlarged (224, 224) box
    sortings            ndarray[#sutures, 35]           For each found suture->Suture ID probability
    sort_peaks          List[#sutures]->Peak            For each found suture->Peak location from probability map
    =================   ==============================  ================================================================

    :param img_data: Image input data for inference. Must be of shape [#frames, 224, 224].
    :type img_data: np.ndarray
    :param frame_delta: Hop-interval for Tiny YOLO suture region inference. Only every `frame_delta` frames an inference
        will be run.
    :type frame_delta: int
    :param unet_batch_size: Batch size for input data to UNet for finding individual sutures. Was trained on batch size
        of 10.
    :type unet_batch_size: int
    :param effnet_batch_size: Batch size for input data to EfficientNetB0 for sorting found sutures into their grid
        positions. Was trained on batch size of 32.
    :param progress_callback: Callback for reporting inference progress.
    :type progress_callback: pyqtsignal
    :param message_callback: Callback for reporting current step of inference.
    :type message_callback: pyqtsignal
    :param cancel_callback: Callback for canceling the inference progress.
    :type cancel_callback: pyqtslot
    :return: Tuple of yolo_boxes, yolo_confidences, unet_predictions, unet_boxes, sortings and used_sorting_peaks.
    """
    # TODO: ONLY FOR DEBUGGING
    img_data = img_data[:10]

    img_data = fix_data_bits(img_data)

    yolo_boxes, yolo_confs, unet_predictions, unet_boxes = run_suture_detect(img_data, 'runet', frame_delta,
                                                                             unet_batch_size, progress_callback,
                                                                             message_callback, cancel_callback)
    if message_callback:
        message_callback.emit("Reading EfficientNet...")
    sortnet = load_model(sortnet_model)

    if message_callback:
        message_callback.emit("Preparing suture sorting...")
    gen = SortNetInfGen(unet_predictions, unet_boxes, effnet_batch_size)
    prog_logger = ProgLogger(progress_callback, cancel_callback)

    if message_callback:
        message_callback.emit("Sorting found sutures...")
    sortings = sortnet.predict(gen,
                               verbose=1,
                               callbacks=[prog_logger])

    # If the effnet_batch_size divides the data unevenly, the last prediction batch will be filled to the full batch
    # size with empty data, which we do not care about or want.
    sortings = sortings[:gen.num_samples_total]

    return yolo_boxes, yolo_confs, unet_predictions, unet_boxes, sortings, gen.used_peaks


def auto_annotate(img_data, settings, callbacks) -> dict:
    """
    Fully-automatic annotate suture grid positions.

    The returned results dictionary holds the predicted suture annotations and all intermediary results necessary for
    the final prediction.

    ==================  ==========================================================================
    **Dictionary Key**  **Values**
    'region_boxes'      The bounding box information for suture regions.
    'region_confs'      Confidence values for corresponding region bounding boxes.
    ------------------  --------------------------------------------------------------------------
    'suture_maps'       Suture probability maps.
    'suture_boxes'      Enlarged (224x224) bounding box for positioning of probability map.
    ------------------  --------------------------------------------------------------------------
    'suture_peaks'      Individual peak coordinates found on the suture probability maps.
    ------------------  --------------------------------------------------------------------------
    'sorted_sutures'    Final predicted suture annotations. Coordinates and position probabilities
                        for each found suture.
    ==================  ==========================================================================


    :param img_data: The image/video data to run inference on. Shape must be [#frames, 768, 768].
    :type img_data: np.ndarray
    :param settings: Settings dictionary created by asking user for inference settings. For creation and available keys
        see :meth:`ask_inference_settings <gui.main_window.Main.ask_inference_settings>`.
    :type settings: dict
    :param callbacks: Dictionary of callbacks used during inference process to relay information to user.
    :type callbacks: dict
    :return: Results dictionary holding predicted sorting of sutures and all intermediary results. Dictionary keys are:
        'region_boxes', 'region_confs', 'suture_maps, 'suture_boxes', 'suture_peaks', 'sorted_sutures'.
    """
    img_data = fix_data_bits(img_data)

    results = {}
    # --------------------------------------------------------------------
    # Finding suture regions
    if settings['run_region_detect']:
        if not settings['load_regions']:
            region_boxes, region_confs = run_region_detect(img_data,
                                                           settings['region_network_path'],
                                                           settings['region_frame_delta'],
                                                           weights=settings['region_weights_path'],
                                                           start_frame=settings['from_frame'],
                                                           **callbacks)
        else:
            region_boxes, region_confs = load_regions_inference(settings['regions_file'], False)
        results.update({'region_boxes': region_boxes,
                        'region_confs': region_confs})
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Calculating suture probability maps
    if settings['run_suture_find']:
        if not settings['load_maps']:
            suture_maps, suture_boxes = run_suture_detect(img_data,
                                                          region_boxes,
                                                          settings['suture_find_network'],
                                                          settings['suture_find_batch'],
                                                          **callbacks)
        else:
            suture_maps, suture_boxes = load_suture_detect(settings['maps_file'], False)
        results.update({'suture_maps': suture_maps,
                        'suture_boxes': suture_boxes})
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Finding individual peaks in probability maps
    if settings['run_peak_find']:
        if not settings['load_peaks']:
            suture_peaks = run_peak_finding(suture_maps,
                                            suture_boxes,
                                            region_boxes,
                                            settings['peak_find_distance'],
                                            settings['peak_find_threshold'],
                                            **callbacks)
        else:
            suture_peaks = load_peak_finding(settings['peaks_file'])
        results.update({'suture_peaks': suture_peaks})
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Predict grid positions for found sutures
    if settings['run_peak_sort']:
        if not settings['load_sortings']:
            sorted_sutures = run_peak_sort(suture_peaks,
                                           suture_maps,
                                           suture_boxes,
                                           settings['suture_sort_network'],
                                           settings['suture_sort_batch'],
                                           binary=True,
                                           **callbacks)
        else:
            sorted_sutures = load_peak_sort(settings['sorting_file'])
        results.update({'sorted_sutures': sorted_sutures})
    # --------------------------------------------------------------------

    return results

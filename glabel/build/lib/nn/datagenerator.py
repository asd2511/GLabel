import re
import os
import random
import glob
import h5py
import itertools
from typing import List

import numpy as np
import imageio as io
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import albumentations as alb
import flammkuchen as fl


def sort_alphanumeric(file_list) -> list:
    """Sort the given iterable alphanumerically."""
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(file_list, key=alphanum_key)


def real_filename(fname) -> str:
    """
    Get the "real" filename of a file of name structure "Suture-ID_Cine-ID_frame-number.h5".

    .. warning:: This is situational for my filenames and should be modified in the future! The current use of this
        function is only happening in :class:`RUNetGenerator`

    .. todo:: Modify the use of this function to be more general

    :param str fname: Raw filename for which to find the "real" filename.
    :return: "Real" filename of the specified file.
    """
    shortname = os.path.basename(fname)
    parts = shortname.split('_')

    return '_'.join(parts[:-2])


def _get_runet_augmenter(sequence_length) -> alb.Compose:
    """
    Defines augmentations used on data and returns an Albumentations Composition object.

    :param int sequence_length: Length of time series for recurrent data.
    :return: Albumentations Composition object for data augmentation
    """
    alb_dict = {}
    for i in range(1, sequence_length):
        alb_dict['image' + str(i)] = 'image'

    aug = alb.ReplayCompose([
        alb.RandomBrightnessContrast(0.1, 0.1, p=0.75, ),
        alb.RandomGamma((90, 110), p=0.75),
        alb.Blur(3, p=0.5),
        alb.GaussNoise((3., 10.), p=0.5),
        alb.Rotate(limit=20, border_mode=0, value=[0, 0, 0], p=0.75),
        alb.HorizontalFlip(p=0.5),  # Horizontal flip messes up in case of multi-class problem
        ],
        p=1,
        additional_targets=alb_dict)

    return aug


class DataGenerator(Sequence):
    """
    Tensorflow Keras datagenerator for creating input data for U-Net.

    :param str data_dir: Directory from which to pull data. Must contain suture grid images in .jpg format in
        subdirectory `images` and ground truth segmentation maps in subdirectory `maps`.
    :param int batch_size: Batch size of inputs.
    :param int classes: Number of classes for input segmentation map creation. Defaults to 1. Difference is only made
        between single class and multiple classes. When setting this to a values >1, the created segmentation maps will
        create individual pixel intensity for each grid position, allowing U-Net to directly predict suture sorting
        itself.
    :param bool augment: Use data augmentation. Defaults to True.
    :param bool shuffle: Shuffle input data. Defaults to True.
    :param bool rgbchannel: Create input segmentation maps as 3-channel RGB input. Defaults to False.
    """
    def __init__(self, data_dir, batch_size, classes=1,
                 augment=True, shuffle=True, rgbchannel=False):
        self.data_dir = data_dir  #: Directory from which data is pulled.
        self.batch_size = batch_size  #: Batch size of input batches.
        self.num_classes = classes  #: Number of classes in segmentation maps. Only options are [1, >1].
        self.augment = augment  #: If data is augmented when creating input samples.
        self.shuffle = shuffle  #: If data is shuffled when creating input samples.
        self.rgbchannel = rgbchannel  #: If created input segmentation maps are RGB images or grayscale.

        self.img_files = glob.glob(data_dir + '/images/*.jpg')  #: List of all found .jpg image files

        # For single-class data, the segmentation maps are expected to be
        # saved as .jpeg files, otherwise we expect hdf5 files.
        if self.num_classes == 1:
            self.map_files = glob.glob(data_dir + '/maps/*.jpg')  #: Ground truth segmentation maps
        else:
            # Multiclass maps are 35-channel data
            self.map_files = glob.glob(data_dir + '/maps/*.h5')

        self.aug = self._get_augmenter()  #: Albumentations Composition object for augmentations.

        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Returns number of batches per epoch.

        :return: The number of batches in one epoch as integer value.
        """
        return len(self.img_files) // self.batch_size

    def __getitem__(self, index) -> tuple:
        """
        Generate one batch of data.

        :param index: Batch index
        :type index: int
        :return: Tuple of of two numpy arrays, each of shape [batch_size, height, width, 1]. The first for the input
            images, the second for segmentation maps.
        """
        img_ids = self.img_files[(index*self.batch_size):(index+1)*self.batch_size]
        map_ids = self.map_files[(index*self.batch_size):(index+1)*self.batch_size]

        imgs = []
        maps = []

        for img_id, map_id in zip(img_ids, map_ids):
            # Load images, augment and normalize to range [-1, 1] for images and [0, 1] for maps
            img = io.imread(img_id)

            # For single-class data, the segmentation maps are expected to be
            # saved as .jpeg files, otherwise we expect hdf5 files.
            if self.num_classes == 1:
                segmap = io.imread(map_id)
            else:
                with h5py.File(map_id) as f:
                    segmap = np.asarray(f['data'])

            if self.augment:
                augmented = self.aug(image=img, mask=segmap)  # augment image
                img = augmented['image']
                segmap = augmented['mask']

            img = 2 * ((img - img.min()) / (img.max() - img.min())) - 1
            segmap = (segmap - segmap.min()) / (segmap.max() - segmap.min())
            # Additional threshold to clean up jpg/png noise (which is somehow introduced when saving+reloading the
            # binary maps?!)
            segmap = (segmap > 0.5) * 1.0

            # img = preprocess_input(img, mode='tf')

            if self.rgbchannel:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

            imgs.append(img)
            maps.append(np.round(segmap))  # Ensuring binary maps

        imgs = np.asarray(imgs)
        if not self.rgbchannel:
            imgs = np.expand_dims(imgs, axis=3)

        if self.num_classes == 1:
            maps = np.expand_dims(np.asarray(maps), axis=3)
        else:
            maps = np.asarray(maps)

        return imgs, maps

    def on_epoch_end(self):
        """
        Prepare data for next epoch by re-shuffling data files in-place.
        """
        # Shuffle images
        if self.shuffle:
            shuffled = list(zip(self.img_files, self.map_files))
            random.shuffle(shuffled)
            self.img_files, self.map_files = zip(*shuffled)

    @staticmethod
    def _get_augmenter():
        """Defines used augmentations"""
        aug = alb.Compose([
                alb.RandomBrightnessContrast(0.1, 0.1, p=0.75,),
                alb.RandomGamma((90, 110), p=0.75),
                # border_mode = cv2.BORDER_CONSTANT = 0
                alb.Rotate(limit=20, border_mode=0, value=[0, 0, 0], p=0.75),
                alb.HorizontalFlip(p=0.5),  # Horizontal flip messes up in case of multi-class problem
                alb.Blur(3, p=0.5),
                alb.GaussNoise((3., 10.), p=0.5)],
            p=1)
        return aug


class RUNetGenerator(Sequence):
    """
    Tensorflow Keras datagenerator for creating input data for recurrent U-Net.

    .. note:: The current implementation of creating the recurrent inputs relies on the duality of the left/right
        stereoscopic view. This forces the :attr:`batch_size` to be an even number in order to handle input samples
        for full frames. This method was already fixed for the non-recurrent datagenerator and the changes should be
        applied to here as well

    .. todo:: Fix the implementation to allow odd batch sizes!

    :param str data_dir: Directory from which to pull data. Must contain suture grid images in .jpg format in
        subdirectory `images` and ground truth segmentation maps in subdirectory `maps`.
    :param int batch_size: Batch size of inputs.
    :param int timesteps: Sequence length for recurrent input.
    :param int input_height: Input image height. Defaults to 224.
    :param int input_width: Input image width. Defaults to 224.
    :param bool augment: Use data augmentation. Defaults to True.
    :param bool shuffle: Shuffle input data. Defaults to True.
    :raises AssertionError: if :attr:`batch_size` is not an even number. The reason for forcing an even batch size is
        because the way the input samples are created is based on the left/right duality of the stereoscopic data. This
        was already better solved for the non-recurrent U-Net which should be applied here as well.
    """
    def __init__(self, data_dir, batch_size, timesteps, input_height=224, input_width=224, augment=True, shuffle=True):
        self.data_dir = data_dir  #: Data directory from which data is pulled
        assert batch_size % 2 == 0, "Please use a batch size that is divisible by 2 (each frame has 2 patches --> " \
                                    "Prevent using a half frame for training; just more convenient)"
        self.batch_size = batch_size  #: Batch size
        self.timesteps = timesteps  #: Timesteps for recurrent input samples
        self.height = input_height  #: Input image height
        self.width = input_width  #: Input image width
        self.augment = augment  #: If input data should be augmented
        self.shuffle = shuffle  #: If input data should be shuffled

        #: Input image files in .jpg format found in the specified :attr:`data_dir`.
        self.img_files = sort_alphanumeric(glob.glob(data_dir + '/images/*.jpg'))
        #: Target segmentation maps found in the specified :attr:`data_dir`.
        self.map_files = sort_alphanumeric(glob.glob(data_dir + '/maps/*.jpg'))
        #: List of files that are sequenced for recurrent inputs.
        self.sequences = self._get_file_sequences(self.img_files, self.map_files)
        self.on_epoch_end()

        self.aug = _get_runet_augmenter(self.timesteps)  #: Albumentations Composition object for data augmentation

    def _get_file_sequences(self, img_flist, target_flist) -> List[List[str]]:
        """
        Create lists of files that will be used together to form a single sample.

        This pre-pairing of files is necessary to perform before shuffling the list of files so that the time-sequences
        of input images are still belonging together.

        If a file is too close to the end of the total file list or too close to a switch in base filenames, the sample
        sequence for that file will be built in reverse. Such samples should still be valid as training and validation
        samples because of the periodicity of the used data, which makes the direction of movement in the data not
        matter.

        :param img_flist: Complete list of image files found in the data directory.
        :type img_flist: List[str]
        :param target_flist: Complete list of target files found in the data direcotry.
        :type target_flist: List[str]
        :return: List of lists of length :attr:`timesteps` +1 containing names of files belonging to one sample sequence
            and the corresponding target filename as the last entry.
        """
        # Sort the lists alphanumerically to ensure continuity of frames
        img_flist = sort_alphanumeric(img_flist)
        target_flist = sort_alphanumeric(target_flist)
        # Group all files in the list into separate lists with files belonging to the same base file
        unique_files = set([real_filename(f) for f in img_flist])
        unique_lists = [[f for f in img_flist if real_filename(f) == uf] for uf in unique_files]

        sequences = []
        for ul in unique_lists:
            # Split into files belonging to left and right patches
            l_files = [f for f in ul if '_l.jpg' in f]
            r_files = [f for f in ul if '_r.jpg' in f]

            for flist in [l_files, r_files]:
                for idx, file in enumerate(flist):
                    sequence = []
                    # If we run out of enough files to form a complete time sequence, use a reversed sequence. Should
                    # still be a valid data sequence because of periodicity of movement in data --> Direction should not
                    # matter
                    reverse_seq = True if idx > len(flist) - self.timesteps else False

                    file_seq = flist[idx:idx + self.timesteps] if not reverse_seq \
                        else flist[idx + 1 - self.timesteps:idx + 1][::-1]
                    t_file = file.replace('images', 'maps')
                    # Sanity check
                    assert os.path.isfile(t_file), "Generated path to segmantation map target is not correct!"
                    file_seq.append(t_file)
                    sequences.append(file_seq)

        return sequences

    def __len__(self) -> int:
        """
        Number of batches available for an epoch.

        :return: Total number of available batches.
        """
        return len(self.sequences) // self.batch_size

    def __getitem__(self, index) -> tuple:
        """
        Generate a batch of input samples.

        :param index: Index of batch to generate.
        :return: Tuple(inputs, targets)

            - inputs: (:attr:`batch_size`, :attr:`timesteps`, :attr:`height`, :attr:`width`) np.ndarray holding
              recurrent inputs for generated batch.
            - targets: (:attr:`batch_size`, :attr:`height`, :attr:`width`) np.ndarray holding segmentation map targets
              for generated batch.
        """
        samples = self.sequences[index*self.batch_size:(index+1)*self.batch_size]

        inputs = np.empty((self.batch_size, self.timesteps, self.height, self.width))
        targets = np.empty((self.batch_size, self.height, self.width))

        for batch_idx, sample in enumerate(samples):
            # The samples are constructed such that the corresponding target file is always the last entry in the list
            img_files = sample[:-1]
            target_file = sample[-1]

            # Load the data and put them into the input/target matrix
            input_imgs = []
            for seq_idx, f in enumerate(img_files):
                img = io.imread(f)
                input_imgs.append(img)
            target = io.imread(target_file)

            if self.augment:
                random.seed(random.randint(0, 9999))
                augmented = self.aug(image=input_imgs[0],
                                     image1=input_imgs[1],
                                     image2=input_imgs[2],
                                     mask=target)
                input_imgs = [augmented[key] for key in ['image', 'image1', 'image2']]
                target = augmented['mask']

            sample_input = np.transpose(np.dstack(input_imgs), axes=(2, 0, 1))  # To [timesteps, height, width]
            sample_input = sample_input.astype(np.float)

            # Normalize input to [-1, 1]
            sample_input = 2*((sample_input-np.max(sample_input)) / (np.max(sample_input)-np.min(sample_input)))-1
            # Normalize target maps to [0, 1]
            target = target / 255.0
            # Additional threshold to clean up jpg/png noise (which is somehow introduced when saving+reloading the
            # binary maps?!)
            target = (target > 0.5) * 1.0

            # Add the data to the total input and target matrix
            inputs[batch_idx] = sample_input
            targets[batch_idx] = target

        # Expand the dimensions to add indicate single channel data to tensorflow
        return np.expand_dims(inputs, axis=4), np.expand_dims(targets, axis=3), [None]

    def on_epoch_end(self):
        """
        Prepare data for next epoch by re-shuffling data files in-place.
        """
        # Shuffle the sequenced samples inplace
        if self.shuffle:
            random.shuffle(self.sequences)


class SortNetGenerator(Sequence):
    """
    Tensorflow Keras datagenerator for creating input data for EfficientNetB0 (for suture sorting).

    **Bases**: :tf:`Sequence <tf/keras/utils/Sequence>`

    :param str data_dir: Directory from which to pull data. Must contain suture segmentation maps as .h5 files in
        subdirectory `maps` and target .h5 files in subdirectory `targets`.
    :param int batch_size: Batch size of inputs.
    :param int num_classes: Number of classes in your data. This should be the number of available grid positions
        in your landmark grid. (E.g. 35 for the default data of 7x5 suture grids)
    :param int input_height: Input image height. Defaults to 224.
    :param int input_width: Input image width. Defaults to 224.
    :param bool augment: Use data augmentation. Defaults to True.
    :param bool shuffle: Shuffle input data. Defaults to True.
    :param bool cache: If True, all available data will be loaded into memory at initialization. This should only
        be activated if your dataset is small.
    """
    def __init__(self, data_dir, batch_size, num_classes, input_height=224, input_width=224, augment=True, shuffle=True,
                 cache=False):
        self.data_dir = data_dir  #: Directory from which data is pulled.
        self.batch_size = batch_size  #: Batch size for input batches
        self.num_classes = num_classes  #: Number of classes in your data. Should match the number of grid positions.
        self.input_height = input_height  #: Input image height
        self.input_width = input_width  #: Input image width
        self.augment = augment  #: If input data should be augmented
        self.shuffle = shuffle  #: If input data should be shuffled
        self.cache = cache  #: If input data should be cached into memory

        self.map_files = glob.glob(data_dir + '/maps/*.h5')  #: Found segmentation map files
        self.target_files = glob.glob(data_dir + '/targets/*.h5')  #: Found target files
        #: List of tuples that holds all available combinations of indices for map_file x channel_idx. Using one entry
        #: from this list gives as first value which map_file to load and as second value which channel from that
        #: map_file to use.
        self.file_channel_mappings = list(itertools.product(range(len(self.map_files)), range(self.num_classes)))

        if self.cache:
            #: If :attr:`cache` is True, holds all image data as one big numpy array. Numpy array of shape (len(
            #: :attr:`map_files` ), :attr:`input_height', :attr:`input_width`, :attr:`num_classes` +1).
            self.img_data = None
            #: If :attr:`cache` is True, holds all target data as one big numpy array. Numpy array of shape (len(
            #: :attr:`map_files` ), :attr:`num_classes`, :attr:`num_classes` ).
            self.target_data = None
            self._cache_data()

        self.on_epoch_end()

        self.aug = self._get_augmenter()  #: Albumentations Composition object for data augmentation

    def _cache_data(self):
        """
        Cache all found data into memory.

        The found image and target files will be loaded into the attributes :attr:`img_data` and :attr:`target_data`
        respectively.
        """
        self.img_data = np.empty((len(self.map_files), self.input_height, self.input_width, self.num_classes+1))
        self.target_data = np.empty((len(self.map_files), self.num_classes, self.num_classes))
        for f_idx, file in enumerate(self.map_files):
            target_file = file.replace('maps', 'targets')
            with h5py.File(file, 'r') as f:
                self.img_data[f_idx, :, :] = np.asarray(f['data'])
            with h5py.File(target_file, 'r') as f:
                self.target_data[f_idx] = np.asarray(f['data'])

    def __len__(self) -> int:
        """
        Get total number of available batches for epoch.

        :return: Number of baches available from the found data.
        """
        return len(self.file_channel_mappings) // self.batch_size

    def __getitem__(self, index):
        """
        Generate a batch of input data.

        Inputs are generated as an RGB image with the first/red channel showing the complete suture segmentation map.
        The other channels have one single suture location from the segmentation map copied onto them. This creates a
        classification map that has all found sutures marked in red, while the one suture to be classified is marked in
        white.

        The data for creating the input maps is stored in .h5 files as numpy arrays with depth :attr:`num_classes` + 1.
        The first channel of all these arrays contains the complete suture segmentation map. All other channels are
        empty (zero-valued) channels, with only the region of the suture having non-zero values.

        Example:

            We want to build the classification map for the suture at grid position (1, 3) on the left of frame 5. The
            suture grid is stereo with 7 rows and 5 columns.
            The steps taken to create the map are:

            1. Load the image data from the .h5 file *filename* _5_l.h5. This is an array of shape (224, 224, 36).
            2. Take the first channel of the data as the global suture map showing all found sutures.
            3. Take channel 8, which contains the individual suture at grid position (1, 3). (row \* #cols+col)
            4. Combine the global and individual map by stacking (global, individual, individual) into one RGB image.

        :param int index: Index of batch to create.
        :return: Tuple (inputs, targets, None)

            - inputs: Numpy array of shape (:attr:`batch_size`, :attr:`input_height`, :attr:`input_width`, 3) holding
              input image data.
            - targets: Numpy array of shape (:attr:`batch_size`, :attr:`num_classes`) holding class labels for each
              input image.
        """
        batch_mappings = self.file_channel_mappings[index*self.batch_size:(index+1)*self.batch_size]

        batch_inputs = np.empty((self.batch_size, self.input_height, self.input_width, 3))
        batch_targets = np.empty((self.batch_size, self.num_classes))
        for batch_idx, mapping in enumerate(batch_mappings):
            if not self.cache:
                map_file = self.map_files[mapping[0]]
                channel_idx = mapping[1]
                target_file = map_file.replace('maps', 'targets')
                # Load channel+1 because data has num_classes+1 channels (first one is "global" map)
                # input_img = fl.load(map_file, sel=fl.aslice[:, :, [0, channel_idx+1]])
                with h5py.File(map_file, 'r') as f:
                    input_img = np.asarray(f['data'])[:, :, [0, channel_idx+1]]
                input_img = np.dstack((input_img, input_img[:, :, 1]))  # Repeat the individual map to get 3-channel data
                # target = fl.load(target_file, sel=fl.aslice[channel_idx])
                with h5py.File(target_file, 'r') as f:
                    target = np.asarray(f['data'])[channel_idx]
            else:
                input_img = self.img_data[mapping[0], :, :, [0, mapping[1]+1]]
                input_img = np.transpose(input_img, axes=(1, 2, 0))
                input_img = np.dstack((input_img, input_img[:, :, 1]))
                target = self.target_data[mapping[0], mapping[1]]

            # We do not want to use a sample for an unplaced ROI position. To ensure that we nevertheless have the
            # correct batch size for our input, we repeat the previous sample instead of the empty one.
            if np.max(input_img[:, :, 1]) == 0:
                input_img = batch_inputs[batch_idx-1].copy()
                target = batch_targets[batch_idx-1].copy()

            if self.augment:
                augmented = self.aug(image=input_img)
                input_img = augmented['image']

            # Check is necessary because when we replace a sample with unplaced ROI with the previous sample, it is
            # already normalized.
            if np.max(input_img) > 1.0:
                input_img = input_img / 255.0
            if np.max(target) > 1.0:
                target = target // 255

            batch_inputs[batch_idx] = input_img
            batch_targets[batch_idx] = target

        return batch_inputs, batch_targets, [None]

    def on_epoch_end(self):
        """
        Called automatically be tensorflow after a full epoch worth of batches has been created. Shuffles the data
        inplace for the next epoch.
        """
        if self.shuffle:
            random.shuffle(self.file_channel_mappings)

    @staticmethod
    def _get_augmenter() -> alb.Compose:
        """
        Create an Albumentations Composition object for input data augmentation.

        Only affine transformations are used for augmentation.

        :return: Composition object
        """
        aug = alb.Compose([
            alb.ShiftScaleRotate(shift_limit=0.1,
                                 scale_limit=0.1,
                                 rotate_limit=20,
                                 border_mode=0, value=[0, 0, 0], p=0.75),
            ],
            p=1)

        return aug

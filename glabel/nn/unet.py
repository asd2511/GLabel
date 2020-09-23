import numpy as np
import tensorflow as tf
from tensorflow import keras


def conv_block(x, filters, batch_norm=True):
    x = keras.layers.Conv2D(filters,
                            kernel_size=(3, 3),
                            use_bias=False,
                            padding='same',
                            strides=1,
                            kernel_initializer='he_uniform')(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Activation('relu')(x)

    return x


def recurrent_conv_block(x, filters, batch_norm=True, return_sequence=True):
    """
    The input of a ConvLSTM is a set of images over time as a 5D tensor with shape
    (samples, time_steps, rows, cols, channels).
    if return_sequences = True, then it returns a sequence as a 5D tensor with shape
    (samples, time_steps, filters, rows, cols)
    """
    # x = keras.layers.Bidirectional(
    #     keras.layers.ConvLSTM2D(filters,
    #                             kernel_size=(3, 3),
    #                             data_format='channels_last',
    #                             recurrent_activation='hard_sigmoid',
    #                             use_bias=False,
    #                             padding='same',
    #                             strides=1,
    #                             kernel_initializer='he_uniform',
    #                             recurrent_initializer='orthogonal',
    #                             return_sequences=return_sequence)
    # )(x)
    x = keras.layers.ConvLSTM2D(filters,
                                kernel_size=(3, 3),
                                activation='relu',
                                data_format='channels_last',
                                recurrent_activation='hard_sigmoid',
                                use_bias=False,
                                padding='same',
                                strides=1,
                                kernel_initializer='he_uniform',
                                recurrent_initializer='orthogonal',
                                return_sequences=return_sequence)(x)

    if batch_norm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Activation('relu')(x)

    return x


class Unet(object):
    def __init__(self, filters=64, layers=4, activation='sigmoid', classes=1, input_shape=None):
        self.num_filters = filters
        self.num_layers = layers
        self.activation = activation
        self.num_classes = classes
        self.input_shape = input_shape if input_shape else (None, None, 1)

    def build(self):
        model_input = keras.Input(shape=self.input_shape)

        to_concat = []

        x = model_input
        # Contracting path
        for i in range(self.num_layers):
            # 2 3x3 convolutions with equal number of filters followed by a max pool with receptive field 2x2
            # The result of the second convolution is saved to be re-used in the decoding step
            # The number of filters is doubled at each downsampling step
            x = conv_block(x, self.num_filters * (2 ** i))
            x = conv_block(x, self.num_filters * (2 ** i))
            to_concat.append(x)

            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # First of two 3x3 convolutions at lowest resolution
        x = conv_block(x, self.num_filters * (2 ** self.num_layers))

        # Expansive path
        for i, filter_factor in enumerate(np.arange(self.num_layers)[::-1]):
            # Convolution at lower resolution, followed by upsampling to receive higher resolution data, to which the
            # result of the convolution in the encoding step is concatenated before continuing with the next convolution
            x = conv_block(x, self.num_filters * (2 ** filter_factor))

            x = keras.layers.UpSampling2D(size=(2, 2))(x)
            x = keras.layers.Concatenate()([x, to_concat[::-1][i]])

            x = conv_block(x, self.num_filters * (2 ** filter_factor))

        # Last convolution layer needed to result in even structure
        x = conv_block(x, self.num_filters)

        # Final output, 1x1 convolution
        model_output = keras.layers.Conv2D(self.num_classes,
                                           (1, 1),
                                           use_bias=False,
                                           padding='same',
                                           activation=self.activation,
                                           strides=1,
                                           kernel_initializer='glorot_uniform')(x)

        return keras.Model(model_input, model_output)


class RUnet(object):
    def __init__(self, filters=64, timesteps=5, layers=4, activation='sigmoid', classes=1, input_shape=None):
        self.num_filters = filters
        self.num_layers = layers
        self.activation = activation
        self.num_classes = classes
        self.input_shape = input_shape if input_shape else (timesteps, 224, 224, 1)

    def build(self):
        model_input = keras.Input(shape=self.input_shape)

        to_concat = []

        x = model_input
        # Contracting path
        for i in range(self.num_layers):
            # RUnet has only a single 3x3 convolution followed by a max pool with receptive field 2x2
            # The result of the ConvLSTM is saved to be re-used in the decoding step
            # The number of filters is doubled at each downsampling step
            x = recurrent_conv_block(x, self.num_filters * (2 ** i))
            # x = recurrent_conv_block(x, self.num_filters * (2 ** i))
            to_concat.append(x)

            x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), data_format='channels_last')(x)

        x = recurrent_conv_block(x, self.num_filters * (2 ** self.num_layers))

        # Expansive path
        for i, filter_factor in enumerate(np.arange(self.num_layers)[::-1]):
            # RUnet has here also only a single ConvLSTM instead of the two Conv layers as in Unet
            # Convolution at lower resolution, followed by upsampling to receive higher resolution data, to which the
            # result of the convolution in the encoding step is concatenated before continuing with the next convolution
            x = recurrent_conv_block(x, self.num_filters * (2 ** filter_factor))

            x = keras.layers.UpSampling3D(size=(1, 2, 2))(x)
            x = keras.layers.Concatenate()([x, to_concat[::-1][i]])

        # Last convolution layer needed to result in even structure
        x = recurrent_conv_block(x, self.num_filters, return_sequence=False)

        # Final output, 1x1 convolution
        model_output = keras.layers.Conv2D(self.num_classes,
                                           (1, 1),
                                           use_bias=False,
                                           padding='same',
                                           activation=self.activation,
                                           strides=1,
                                           kernel_initializer='glorot_uniform')(x)

        return keras.Model(model_input, model_output)

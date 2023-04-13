import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, UpSampling2D

from .layer import Bias, ConvBlock, PixelNorm, WSConv2d, WSTransposedConv2d


class Generator(Model):
    def __init__(self, filters, img_channels, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self._filters = filters
        self._initial_rgb = WSConv2d(img_channels, kernel_size=1, strides=1, padding='valid')
        self._activation = LeakyReLU(0.2)
        self._prog_blocks = []
        self._rgb_layers = [self._initial_rgb]

        for i in range(len(self._filters) - 1):
            self._prog_blocks.append(ConvBlock(self._filters[i + 1]))
            self._rgb_layers.append(WSConv2d(img_channels, kernel_size=1, strides=1, padding='valid'))

    def _initial_block(self, inpt, filters):
        output = WSTransposedConv2d(filters, padding='valid')(inpt)
        output = Bias([output.shape[0], 1, 1, output.shape[-1]])(output)
        output = self._activation(output)
        output = WSConv2d(filters)(output)
        output = Bias([output.shape[0], 1, 1, output.shape[-1]])(output)
        output = self._activation(output)
        output = PixelNorm()(output)
        return output

    def _fade_in(self, alpha, upscaled, generated):
        return tf.tanh(alpha * generated + (1 - alpha) * upscaled)

    def call(self, inpt, alpha, steps):
        output = self._initial_block(inpt, self._filters[0])

        if steps == 0:
            return self._initial_rgb(output)

        for step in range(steps):
            upscaled = UpSampling2D()(output)
            output = self._prog_blocks[step](upscaled)

        final_upscaled = self._rgb_layers[steps - 1](upscaled)
        final_output = self._rgb_layers[steps](output)
        return self._fade_in(alpha, final_upscaled, final_output)

    def get_config(self):
        raise NotImplementedError

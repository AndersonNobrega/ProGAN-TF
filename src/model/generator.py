import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LeakyReLU, UpSampling2D
from tensorflow.keras.optimizers import Adam

from .layer import Bias, ConvBlock, PixelNorm, WSConv2d, WSTransposedConv2d


class Generator(Model):
    def __init__(self, filters, learning_rate, img_channels, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self._filters = filters
        self._learning_rate = learning_rate
        self._optimizer = Adam(learning_rate=self._learning_rate, beta_1=0, beta_2=0.99, epsilon=1e-8)
        self._initial_rgb = WSConv2d(img_channels, kernel_size=1, strides=1, padding='valid')
        self._activation = LeakyReLU(0.2)
        self._prog_blocks = []
        self._rgb_layers = [self._initial_rgb]

        for i in range(len(self._filters) - 1):
            self._prog_blocks.append(ConvBlock(self._filters[i + 1]))
            self._rgb_layers.append(WSConv2d(img_channels, kernel_size=1, strides=1, padding='valid'))

        # Layers to be used in the initial block
        self._transposedconv = WSTransposedConv2d(filters[0], padding='valid')
        self._transposedconv_bias = Bias()
        self._conv_a = WSConv2d(filters[0])
        self._conv_a_bias = Bias()
        self._pixel_norm = PixelNorm()

    def _initial_block(self, inpt):
        output = self._transposedconv(inpt)
        output = self._transposedconv_bias(output)
        output = self._activation(output)
        output = self._conv_a(output)
        output = self._conv_a_bias(output)
        output = self._activation(output)
        output = self._pixel_norm(output)
        return output

    def _fade_in(self, alpha, upscaled, generated):
        return tf.tanh(alpha * generated + (1 - alpha) * upscaled)

    def call(self, inpt, alpha, steps):
        output = self._initial_block(inpt)

        if steps == 0:
            return self._initial_rgb(output)

        for step in range(steps):
            upscaled = UpSampling2D()(output)
            output = self._prog_blocks[step](upscaled)

        final_upscaled = self._rgb_layers[steps - 1](upscaled)
        final_output = self._rgb_layers[steps](output)
        return self._fade_in(alpha, final_upscaled, final_output)

    def loss(self, fake_output):
        return -1 * tf.reduce_mean(fake_output)

    def optimizer(self):
        return self._optimizer

    def get_config(self):
        raise NotImplementedError

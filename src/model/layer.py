import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Layer, LeakyReLU


class WSConv2d(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same', gain=2):
        super(WSConv2d, self).__init__()
        self._weight_init = RandomNormal()
        self._conv_layer = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=self._weight_init,
            use_bias=False,
            activation=None,
        )
        self._gain = gain
        self._kernel_size = kernel_size

    def call(self, inpt, *args, **kwargs):
        scale = tf.cast(tf.sqrt(self._gain / (inpt.shape[-1] * (tf.square(self._kernel_size)))), tf.float32)
        output = self._conv_layer(inpt * scale)
        self._conv_layer.get_weights()
        return output


class WSTransposedConv2d(Layer):
    def __init__(self, filters, kernel_size=4, strides=1, padding='same', gain=2):
        super(WSTransposedConv2d, self).__init__()
        self._weight_init = RandomNormal()
        self._conv_layer = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=self._weight_init,
            use_bias=False,
            activation=None,
        )
        self._gain = gain
        self._kernel_size = kernel_size

    def call(self, inpt, *args, **kwargs):
        scale = tf.cast(tf.sqrt(self._gain / (inpt.shape[-1] * (tf.square(self._kernel_size)))), tf.float32)
        output = self._conv_layer(inpt * scale)
        return output


class Bias(Layer):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.shape = shape
        self._config = {'shape': shape}
        self.b = self.add_weight('bias', shape=shape, initializer=Zeros(), trainable=True)

    def call(self, inpt, *args, **kwargs):
        return inpt + self.b


class PixelNorm(Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def call(self, inpt, *args, **kwargs):
        return inpt / tf.sqrt(tf.reduce_mean(tf.square(inpt), axis=-1, keepdims=True) + self.epsilon)


class ConvBlock(Layer):
    def __init__(self, filters, pixel_norm=True, kernel=3, strides=1, gain=2):
        super(ConvBlock, self).__init__()
        self._conv_a = WSConv2d(filters, kernel, strides, 'same', gain)
        self._conv_b = WSConv2d(filters, kernel, strides, 'same', gain)
        self._activation = LeakyReLU(0.2)
        self._pixel_norm = PixelNorm()
        self._use_pixel_norm = pixel_norm

    def call(self, inputs, *args, **kwargs):
        output = self._conv_a(inputs)
        output = self._activation(Bias([output.shape[0], 1, 1, output.shape[-1]])(output))
        if self._use_pixel_norm:
            output = self._pixel_norm(output)

        output = self._conv_b(output)
        output = self._activation(Bias([output.shape[0], 1, 1, output.shape[-1]])(output))
        if self._use_pixel_norm:
            output = self._pixel_norm(output)
        return output

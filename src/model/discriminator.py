import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomNormal, Zeros
from tensorflow.keras.layers import AveragePooling2D, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

from .layer import Bias, ConvBlock, WSConv2d


class Discriminator(Model):
    def __init__(self, filters, learning_rate, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self._filters = filters
        self._learning_rate = learning_rate
        self._optimizer = Adam(learning_rate=self._learning_rate, beta_1=0, beta_2=0.99, epsilon=1e-8)
        self._final_rgb = WSConv2d(self._filters[0], kernel_size=1, strides=1, padding='valid')
        self._activation = LeakyReLU(0.2)
        self._prog_blocks = []
        self._rgb_layers = []

        for i in range(len(self._filters) - 1, 0, -1):
            self._prog_blocks.append(ConvBlock(self._filters[i - 1], pixel_norm=False))
            self._rgb_layers.append(WSConv2d(self._filters[i], kernel_size=1, strides=1, padding='valid'))

        self._rgb_layers.append(self._final_rgb)
        self._avg_pool = AveragePooling2D(pool_size=2, strides=2)

        # Layers to be used in the final block
        self._conv_a = WSConv2d(filters[0])
        self._conv_a_bias = Bias()
        self._conv_b = WSConv2d(filters[0], kernel_size=4, padding='valid')
        self._conv_b_bias = Bias()
        self._dense = Dense(1, kernel_initializer=RandomNormal(), bias_initializer=Zeros(), activation='linear')

    def _final_block(self, inpt):
        output = self._conv_a(inpt)
        output = self._conv_a_bias(output)
        output = self._activation(output)
        output = self._conv_b(output)
        output = self._conv_b_bias(output)
        output = self._activation(output)
        output = self._dense(output)
        return output

    def _fade_in(self, alpha, downscaled, output):
        return tf.tanh(alpha * output + (1 - alpha) * downscaled)

    def mini_batch_std(self, inpt):
        batch_std = tf.math.reduce_std(inpt, axis=0, keepdims=True)
        batch_mean = tf.reduce_mean(batch_std, keepdims=True)
        output = tf.tile(batch_mean, (tf.shape(inpt)[0], tf.shape(inpt)[1], tf.shape(inpt)[2], 1))
        return tf.concat([inpt, output], axis=-1)

    def call(self, inpt, alpha, steps):
        current_step = len(self._prog_blocks) - steps
        output = self._activation(self._rgb_layers[current_step](inpt))

        if steps > 0:
            downscaled = self._rgb_layers[current_step + 1](self._avg_pool(inpt))
            downscaled = self._activation(downscaled)
            output = self._avg_pool(self._prog_blocks[current_step](output))

            output = self._fade_in(alpha, downscaled, output)

            for step in range(current_step + 1, len(self._prog_blocks)):
                output = self._prog_blocks[step](output)
                output = self._avg_pool(output)

        output = self.mini_batch_std(output)
        return self._final_block(output)

    def loss(self, real_output, fake_output, grad_penalty, lambda_grad_penalty):
        total_loss = (tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)) + \
                     (grad_penalty * lambda_grad_penalty) + (0.001 * tf.reduce_mean(real_output ** 2))
        return total_loss

    def optimizer(self):
        return self._optimizer

    def get_config(self):
        raise NotImplementedError

import os

# Remove Tensorflow log spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Check for available GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('Using GPU for model training.\n')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
else:
    print('No GPU available for model training. Using CPU instead.\n')

from model import Discriminator, Generator


def main():
    seed_generator = tf.random.normal([1, 1, 1, 512], dtype=tf.float32)
    generator = Generator(filters=[512, 512, 512, 512, 256, 128, 64, 32, 16], img_channels=3)
    print(generator(seed_generator, alpha=0.5, steps=1).shape)

    seed_discriminator = tf.random.normal([1, 8, 8, 512], dtype=tf.float32)
    discriminator = Discriminator(filters=[512, 512, 512, 512, 256, 128, 64, 32, 16], img_channels=3)
    print(discriminator(seed_discriminator, alpha=0.5, steps=1).shape)


if __name__ == '__main__':
    main()

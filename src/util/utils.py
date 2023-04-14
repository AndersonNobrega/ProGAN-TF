import io

import matplotlib.pyplot as plt
import tensorflow as tf


def save_images(predictions):
    fig = plt.figure(figsize=(4, 4))

    for i in range(tf.shape(predictions)[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.cast(tf.add(tf.multiply(predictions[i, :, :, :], 127.5), 127.5), tf.int32))
        plt.axis('off')

    # Step needed to be compatible with tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def write_tensorboard_logs(file_writer, label, content, step, content_type='scalar'):
    with file_writer.as_default():
        if content_type == 'scalar':
            tf.summary.scalar(label, content, step=step)
        elif content_type == 'image':
            tf.summary.image(label, content, step=step)

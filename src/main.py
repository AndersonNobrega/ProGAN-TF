import datetime
import json
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from tqdm import tqdm

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
from util import CheckpointManager, save_images, write_tensorboard_logs


def get_args():
    parser = ArgumentParser(allow_abbrev=False, description='', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size_per_step', type=int, help='Batch size for the training dataset.', default=32)
    parser.add_argument('--epochs', type=int, help='Amount of epochs to train model.', default=1)
    parser.add_argument('--generator_lr', type=float, help='Learning rate for both the generator models.', default=1e-3)
    parser.add_argument('--discriminator_lr', type=float, help='Learning rate for both the discriminator models.', default=1e-3)
    parser.add_argument('--noise_dim', type=int, help='Dimension for noise vector used by the generator.', default=512)
    parser.add_argument('--buffer_size', type=int, help='Buffer size for dataset shuffle.', default=60000)
    parser.add_argument('--discriminator_iterations', type=int, help='Number of discriminator iterations per generator iteration.', default=5)
    parser.add_argument('--dataset', type=str, help='Image dataset that is going to be used (mnist, anime_faces).')
    parser.add_argument('--dataset_path', type=str, help='Path for the dataset that is going to be used.', default=None)
    parser.add_argument('--checkpoint_epoch', type=int, help='Amount of epochs before creating a checkpoint. If value is 0 or negative, checkpoints '
                                                             'wont be created.', default=10)
    parser.add_argument('--load_checkpoint', type=str, help='Path to restore model from a checkpoint.', default=None)
    parser.add_argument('--filters', nargs='+', help='')
    parser.add_argument('--json_file', type=str, help='Path to json file with arguments.', default=None)

    return vars(parser.parse_args())


def get_dataset_loader(dataset_name, batch_size, image_size, dataset_path=None):
    if dataset_name == 'anime_faces':
        if dataset_path is None:
            raise IOError('Dataset path is required.')
        train_images = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_path, label_mode=None, image_size=image_size, batch_size=batch_size
        )
        return train_images.map(lambda x: (x - 127.5) / 127.5)
    else:
        raise ValueError('Invalid dataset. if you wish to use a new one, please implement a loader for it.')


def gradient_penalty(discriminator, alpha, step, real_input_batch, fake_input_batch, batch_size):
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, real_input_batch.shape[-1]], minval=-1, maxval=1)
    mixed_output = (epsilon * real_input_batch) + ((1 - epsilon) * fake_input_batch)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(mixed_output)
        mixed_predictions = discriminator(mixed_output, alpha=alpha, steps=step, training=True)

    grad = gp_tape.gradient(mixed_predictions, mixed_output)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    grad_penalty = tf.reduce_mean(tf.square(grad_norm - 1))

    return grad_penalty


def discriminator_train_step(generator, discriminator, real_input_batch, noise_dim, batch_size, alpha, step):
    noise = tf.random.normal([batch_size, 1, 1, noise_dim])

    with tf.GradientTape() as discriminator_tape:
        generator_fake_input = generator(noise, alpha=alpha, steps=step, training=True)
        grad_penalty = gradient_penalty(discriminator, alpha, step, real_input_batch, generator_fake_input, batch_size)

        discriminator_real_output = discriminator(real_input_batch, alpha=alpha, steps=step, training=True)
        discriminator_fake_output = discriminator(generator_fake_input, alpha=alpha, steps=step, training=True)

        discriminator_loss = discriminator.loss(discriminator_real_output, discriminator_fake_output, grad_penalty, 10)

    gradients_of_discriminator = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
    discriminator.optimizer().apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return discriminator_loss


def generator_train_step(generator, discriminator, noise_dim, batch_size, alpha, step):
    noise = tf.random.normal([batch_size, 1, 1, noise_dim])

    with tf.GradientTape() as gen_tape:
        generator_fake_input = generator(noise, alpha=alpha, steps=step, training=True)
        discriminator_fake_output = discriminator(generator_fake_input, alpha=alpha, steps=step, training=True)

        gen_loss = generator.loss(discriminator_fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator.optimizer().apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss


def train(generator,
          discriminator,
          epochs,
          noise_dim,
          discriminator_file_writer,
          generator_file_writer,
          checkpoint_manager,
          dataset,
          batch_size_per_step,
          dataset_path,
          total_steps):
    tqdm.write("\n---------- Starting training loop... ----------\n")

    seed = tf.random.normal([16, 1, 1, noise_dim])
    generator_loss_hist = []
    discriminator_loss_hist = []
    logging_step = 0

    for step in range(total_steps):
        batch_size = batch_size_per_step[step]
        alpha = 1e-5
        loader = get_dataset_loader(dataset, batch_size, [4 * 2 ** step] * 2, dataset_path)  # Get loader for specified dataset
        total_img_batches = len(loader)

        for epoch in range(epochs):
            tqdm.write('Epoch: {}/{}'.format(epoch + 1, epochs))

            for batch_idx, samples in enumerate(tqdm(loader)):
                discriminator_loss_hist.append(
                    discriminator_train_step(generator, discriminator, samples, noise_dim, tf.shape(samples)[0], alpha, step)
                )
                generator_loss_hist.append(
                    generator_train_step(generator, discriminator, noise_dim, tf.shape(samples)[0], alpha, step)
                )

                if (batch_idx + 1) % (total_img_batches // 5) == 0:
                    write_tensorboard_logs(discriminator_file_writer, 'Loss', tf.reduce_mean(discriminator_loss_hist), logging_step, 'scalar')
                    write_tensorboard_logs(generator_file_writer, 'Loss', tf.reduce_mean(generator_loss_hist), logging_step, 'scalar')

                    write_tensorboard_logs(discriminator_file_writer, 'Real Plot', save_images(samples[:16]), logging_step, 'image')
                    # write_tensorboard_logs(generator_file_writer, 'Generated Plot', save_images(
                    #     generator(seed, alpha=alpha, steps=step, training=False)), logging_step, 'image')

                    logging_step += 1

                alpha += tf.divide(tf.cast(tf.shape(samples)[0], tf.float32), tf.multiply(tf.multiply(epochs, tf.constant(0.5)), total_img_batches))
                alpha = tf.minimum(alpha, tf.constant(1.0))

            # Check epoch and save a checkpoint if necessary
            checkpoint_manager.save(epoch)

            tqdm.write(('discriminator Loss: {:.4f} - Generator Loss: {:.4f}'.format(tf.reduce_mean(discriminator_loss_hist),
                                                                                     tf.reduce_mean(generator_loss_hist))))

            generator_loss_hist.clear()
            discriminator_loss_hist.clear()
        step += 1

    tqdm.write("\n---------- Training loop finished. ----------\n")


def main():
    # Get CLI args
    args = get_args()

    # Check if a JSON was passed and update args value
    if args['json_file'] is not None:
        args.update(json.load(open(args['json_file'])))

    # Create directory for images
    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Tensorboard file writer for training logs
    discriminator_log_dir = "logs/{}/discriminator/".format(current_time)
    generator_log_dir = "logs/{}/generator/".format(current_time)

    discriminator_file_writer = tf.summary.create_file_writer(discriminator_log_dir)
    generator_file_writer = tf.summary.create_file_writer(generator_log_dir)

    # Create Generator and Discriminator models
    generator = Generator(filters=args['filters'], learning_rate=args['generator_lr'], img_channels=3)
    discriminator = Discriminator(filters=args['filters'], learning_rate=args['discriminator_lr'])

    # Create checkpoint object
    if args['load_checkpoint'] is not None:
        checkpoint_path = args['load_checkpoint']
    else:
        checkpoint_path = pathlib.Path(__file__).resolve().parents[1] / pathlib.Path("checkpoints") / current_time
    checkpoint_manager = CheckpointManager(generator, discriminator, checkpoint_path, args['checkpoint_epoch'])

    # Train loop
    train(
        generator,
        discriminator,
        args['epochs'],
        args['noise_dim'],
        discriminator_file_writer,
        generator_file_writer,
        checkpoint_manager,
        args['dataset'],
        args['batch_size_per_step'],
        args['dataset_path'],
        len(args['filters'])
    )


if __name__ == '__main__':
    # TODO: Check why after step 1 gradient bugs
    # WARNING:tensorflow:Gradients do not exist for variables ['discriminator/ws_conv2d_11/conv2d_11/kernel:0'] when minimizing the loss.
    # WARNING:tensorflow:Gradients do not exist for variables ['generator/ws_conv2d/conv2d/kernel:0'] when minimizing the loss.
    main()

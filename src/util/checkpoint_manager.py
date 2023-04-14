import tensorflow as tf


class CheckpointManager:
    def __init__(self, generator, discriminator, path, checkpoint_epoch):
        self._path = path
        self._checkpoint_epoch = checkpoint_epoch
        self._checkpoint = tf.train.Checkpoint(enerator=generator,
                                               discriminator=discriminator)
        self._manager = tf.train.CheckpointManager(self._checkpoint, directory=self._path, max_to_keep=5)

        self.load()

    def save(self, epoch):
        if self._checkpoint_epoch > 0 and epoch % self._checkpoint_epoch == 0:
            self._manager.save()

    def load(self):
        if self._manager.latest_checkpoint:
            self._checkpoint.restore(self._manager.latest_checkpoint).assert_consumed()
            print("Checkpoint restored from {}".format(self._manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

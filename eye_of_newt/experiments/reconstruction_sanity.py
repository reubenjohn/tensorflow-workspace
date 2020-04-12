import datetime

import tensorflow as tf

from eye_of_newt.dataset import reconstruction_sanity, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS
from utils import configure_default_gpus

K = tf.keras

EXPERIMENT = 'reconstruction-sanity'
BATCH_SIZE = 64
LOG_DIR = '../logs/eye_of_newt/%s/%s' % (EXPERIMENT, datetime.datetime.now().isoformat())

configure_default_gpus()

dataset = reconstruction_sanity(BATCH_SIZE)
model = K.models.Sequential([
    K.layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
    K.layers.Dense(10, activation=K.layers.LeakyReLU()),
    K.layers.Dense(IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS, activation=K.layers.LeakyReLU()),
    K.layers.Reshape((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
])
optimizer = tf.keras.optimizers.Adam()
loss_fn = K.losses.Huber()
metrics = [K.metrics.MeanAbsoluteError()]


@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        for metric in metrics:
            metric.update_state(targets, predictions)
        # regularization_loss = tf.math.add_n(model.losses) if len(model.losses) > 0 else tf.zeros(())
        pred_loss = loss_fn(targets, predictions)
        total_loss = pred_loss  # + regularization_loss

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, predictions


def main():
    tb_callback = K.callbacks.TensorBoard(LOG_DIR)
    tb_callback.set_model(model)
    with tf.summary.create_file_writer(LOG_DIR + '/train').as_default() as writer:
        prev_steps = 0
        for epoch in range(5):
            print('Epoch: %d' % epoch)
            for curr_epoch_step, (input_batch, target_batch) in enumerate(dataset):
                step = prev_steps + curr_epoch_step
                total_loss, predictions = train_step(input_batch, target_batch)

                tf.summary.scalar('epoch', epoch, step=epoch)
                tf.summary.scalar('step', step, step=step)
                tf.summary.scalar('loss', total_loss, step=step)
                [tf.summary.scalar(metric.name, metric.result(), step=step) for metric in metrics]
                tf.summary.image('input', input_batch, step=step)
                tf.summary.image('reconstruction', predictions, step=step)
                writer.flush()
                prev_steps += curr_epoch_step

        writer.close()


if __name__ == '__main__':
    main()

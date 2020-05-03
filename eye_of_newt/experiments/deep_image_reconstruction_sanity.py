import os
from typing import Tuple

import tensorflow as tf

from experiments import form_log_directory_path, DATASET_ROOT_DIR
from eye_of_newt.data.datasets import image_path_processor
from utils import configure_default_gpus, prefetch_to_available_gpu_device

K = tf.keras

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 512, 352, 3
BATCH_SIZE, N_SANITY_SAMPLES = 32, 3
LOG_DIR = form_log_directory_path(experiment_name='deep-image-reconstruction-sanity/v4-deep-skip-noise')

configure_default_gpus()

# Dataset
dataset = tf.data.Dataset.list_files(DATASET_ROOT_DIR + '/TopGear/still_samples/*.png') \
    .map(image_path_processor(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .take(N_SANITY_SAMPLES) \
    .cache() \
    .repeat(128 * BATCH_SIZE) \
    .batch(BATCH_SIZE)
dataset = prefetch_to_available_gpu_device(dataset, buffer_size=1, use_workaround=True)


# Model
def skip_connecting_auto_encoder_model(inputs,
                                       symmetric_conv_configs: [dict],
                                       middle_conv_configs: dict,
                                       reconstruction_layer: K.layers.Layer):
    activations = inputs
    skip_connections = []
    for conv_config in symmetric_conv_configs:
        activations = K.layers.Conv2D(**conv_config)(activations)
        skip_connections.append(activations)

    activations = K.layers.Conv2D(**middle_conv_configs)(activations)
    activations = K.layers.Conv2DTranspose(**middle_conv_configs)(activations)

    for deconv_config, skip_connection in zip(reversed(symmetric_conv_configs), reversed(skip_connections)):
        noisy_skip_connection = K.layers.GaussianNoise(stddev=.25)(skip_connection)
        concatenated_activation = K.layers.Concatenate()([activations, noisy_skip_connection])
        skip_connecting_deconv = K.layers.Conv2DTranspose(**deconv_config)
        activations = skip_connecting_deconv(concatenated_activation)

    activations = reconstruction_layer(activations)
    activations = K.layers.Reshape((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))(activations)
    activations = NormalizeColorChannels()(activations)
    return K.Model(inputs=inputs, outputs=activations)


class NormalizeColorChannels(K.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.clip_by_value(inputs / 2 + .5, 0., 1.)


model = skip_connecting_auto_encoder_model(
    inputs=K.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)),
    symmetric_conv_configs=[
        dict(filters=8, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=K.layers.LeakyReLU()),
        dict(filters=16, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=K.layers.LeakyReLU()),
        dict(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation=K.layers.LeakyReLU()),
    ],
    middle_conv_configs=dict(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                             activation=K.layers.LeakyReLU()),
    reconstruction_layer=K.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation=K.layers.LeakyReLU(),
                                         name='reconstruct_color_channels')
)

# Training

optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
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
    os.makedirs(LOG_DIR, exist_ok=True)
    K.utils.plot_model(model, LOG_DIR + '/reconstruction-image-sanity-deep-model.png', show_shapes=True,
                       expand_nested=True)
    with tf.summary.create_file_writer(LOG_DIR + '/train').as_default() as writer:
        prev_steps = 0
        for epoch in range(3):
            print('Epoch: %d' % epoch)
            for curr_epoch_step, image_batch in enumerate(dataset):
                step = prev_steps + curr_epoch_step
                total_loss, predictions = train_step(image_batch, image_batch)

                tf.summary.scalar('epoch', epoch, step=epoch)
                tf.summary.scalar('step', step, step=step)
                tf.summary.scalar('loss', total_loss, step=step)
                [tf.summary.scalar(metric.name, metric.result(), step=step) for metric in metrics]
                tf.summary.image('input', image_batch, step=step)
                tf.summary.image('reconstruction', predictions, step=step)
                writer.flush()
                prev_steps += curr_epoch_step

        writer.close()


if __name__ == '__main__':
    main()

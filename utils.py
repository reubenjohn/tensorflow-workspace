import tensorflow as tf


def configure_default_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def prefetch_to_available_gpu_device(dataset: tf.data.Dataset, buffer_size: int = None, use_workaround: bool = False):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        assert len(gpus) == 1, 'Expected to find exactly 1 GPU, but found: ' + gpus
        if use_workaround:
            dataset = dataset.apply(tf.data.experimental.copy_to_device("/GPU:0"))
            with tf.device("/GPU:0"):
                return dataset.prefetch(buffer_size)
        else:
            return dataset.apply(tf.data.experimental.prefetch_to_device('/GPU:0', buffer_size))
    else:
        return dataset

import tensorflow as tf

IMG_WIDTH = 512
IMG_HEIGHT = 352
IMG_CHANNELS = 3


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, img


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def reconstruction_sanity(batch_size: int):
    dataset = tf.data.Dataset.list_files("/scratch/datasets/TopGearSnapshots/*.png")
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.repeat(1024)
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    return dataset

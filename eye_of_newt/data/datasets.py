from typing import Callable

import tensorflow as tf


def image_path_processor(img_width: int, img_height: int, img_channels: int) -> Callable[[tf.Tensor], tf.Tensor]:
    decode_img = image_decoder(img_width, img_height, img_channels)

    def process_path(file_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img

    return process_path


def image_decoder(img_width: int, img_height: int, img_channels: int) -> Callable[[tf.Tensor], tf.Tensor]:
    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=img_channels)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [img_width, img_height])

    return decode_img

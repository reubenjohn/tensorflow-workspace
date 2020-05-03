from typing import Callable, Any

import tensorflow as tf

DATASET_ROOT_DIR = '/scratch/datasets'


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


def labelled_video_element_to_frame_dataset_mapper(video_key: str, take_n: int) -> Callable[[Any], tf.data.Dataset]:
    def labelled_video_element_to_frame_dataset(labelled_video_element) -> tf.data.Dataset:
        video_sequence = labelled_video_element[video_key][:take_n]
        float32_video_sequence = tf.cast(video_sequence, tf.float32) / 255
        return tf.data.Dataset.from_tensor_slices(float32_video_sequence)

    return labelled_video_element_to_frame_dataset


def labelled_video_dataset_to_image_dataset(video_dataset: tf.data.Dataset, take_n: int):
    dataset = video_dataset.flat_map(labelled_video_element_to_frame_dataset_mapper('video', take_n=take_n))
    return dataset

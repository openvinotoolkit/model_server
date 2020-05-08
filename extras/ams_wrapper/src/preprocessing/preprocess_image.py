#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Tuple

import tensorflow as tf
import numpy as np

from src.logger import get_logger


log = get_logger(__name__)


class ImageDecodeError(ValueError):
    pass


class ImageResizeError(ValueError):
    pass


class ImageTransformError(ValueError):
    pass


def preprocess_binary_image(image: bytes, channels: int = None,
                            target_size: Tuple[int, int] = None,
                            channels_first=True,
                            dtype=tf.dtypes.float32, scale: float = None,
                            standardization=False,
                            reverse_input_channels=True) -> np.ndarray:
    """
    Preprocess binary image in PNG, JPG or BMP format, producing numpy array as a result.

    :param image: Image bytes
    :param channels: Number of image's channels
    :param target_size: A tuple of desired height and width
    :param channels_first: If set to True, image array will be in NCHW format,
     NHWC format will be used otherwise
    :param dtype: Data type that will be used for decoding
    :param scale: If passed, decoded image array will be multiplied by this value
    :param standardization: If set to true, image array values will be standarized
    to have mean 0 and standard deviation of 1
    :param reverse_input_channels: If set to True, image channels will be reversed
    from RGB to BGR format
    :raises TypeError: if type of provided parameters are incorrect
    :raises ValueError: if values of provided parameters is incorrect
    :raises ImageDecodeError(ValueError): if image cannot be decoded
    :raises ImageResizeError(ValueError): if image cannot be resized
    :raises ImageTransformError(ValueError): if image cannot be properly transformed
    :returns: Preprocessed image as numpy array
    """

    params_to_check = {'channels': channels,
                       'scale': scale}
                       
    for param_name, value in params_to_check.items():
        try:
            if value is not None and value < 0:
                raise ValueError('Invalid value {} for parameter {}.'.format(value, param_name))
        except TypeError:
            raise TypeError('Invalid type {} for parameter {}.'.format(type(value), param_name))
    
    try:
        if target_size:
            height, width = target_size
            if height <= 0 or width <= 0:
                raise ValueError('Invalid target size parameter.')
    except TypeError:
        raise TypeError('Invalid target size type.')

    if not isinstance(dtype, tf.dtypes.DType):
        raise TypeError('Invalid type {} for parameter dtype.'.format(type(dtype)))

    try:
        decoded_image = tf.io.decode_image(image, channels=channels)
        tf.dtypes.cast(decoded_image, dtype)
    except Exception as e:
        raise ImageDecodeError('Provided image is invalid, unable to decode.') from e

    if target_size:
        try:
            decoded_image = tf.image.resize(decoded_image, target_size)
        except Exception as e:
            raise ImageResizeError('Failed to resize provided binary image from: {} '
                                   'to: {}.'.format(tf.shape(decoded_image), target_size)) from e

    try:
        if standardization:
            decoded_image = tf.image.per_image_standardization(decoded_image)
        image_array = decoded_image.numpy()
        if reverse_input_channels:
            image_array = image_array[..., ::-1]
        if channels_first:
            image_array = np.transpose(image_array, [2, 0, 1])
        if scale:
            array_type = image_array.dtype
            image_array = image_array * scale
            image_array = image_array.astype(array_type)
    except Exception as e:
        log.exception(str(e))
        raise ImageTransformError('Failed to preprocess image, '
                                   'check if provided parameters are correct.') from e

    return image_array



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Pass path to the image as the first argument')
        sys.exit(1)
    img_path = sys.argv[1]

    with open(img_path, mode='rb') as img_file:
        binary_image = img_file.read()

    preprocessed_image = preprocess_binary_image(binary_image, channels_first=False)
    print(preprocessed_image.shape)

    # Keep in mind that matplotlib will not be able to display image in NCHW format
    try:
        import matplotlib.pyplot as plt
        plt.imshow(preprocessed_image)
        plt.show()
    except ImportError:
        print('Please install matplotlib if you want to inspect preprocessed image.')


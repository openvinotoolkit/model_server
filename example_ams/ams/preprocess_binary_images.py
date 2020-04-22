#
# Copyright (c) 2018 Intel Corporation
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


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def preprocess_binary_image(image: bytes, channels: int = None,
                            dtype=tf.dtypes.uint8, scale: float = None,
                            reverse_input_channels=False) -> np.ndarray:
    try:
        decoded_image = tf.io.decode_image(image, channels=channels,
                                           dtype=dtype)
        image_array = decoded_image.numpy()
        if reverse_input_channels:
            # Convert image from RGB to BGR
            image_array = image_array[..., ::-1]
        if scale:
            image_array = image_array * scale
    except Exception as e:
        print('Failed to decode provided binary image:')
        raise ValueError from e

    return image_array



if __name__ == "__main__":
    # wget https://is4-ssl.mzstatic.com/image/thumb/Purple3/v4/a6/24/fa/a624fa65-6468-c192-1a82-d7ff02e02378/source/60x60bb.jpg
    with open('2207159142_8206ab6984.jpg', mode='rb') as img_file:
        binary_image = img_file.read()

    preprocessed_image = preprocess_binary_image(binary_image, reverse_input_channels=True)

    plt.imshow(preprocessed_image)
    plt.show()
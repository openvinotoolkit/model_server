#
# Copyright (c) 2026 Intel Corporation
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

import os
import re

import cv2
import numpy as np

from tests.functional.utils.logger import get_logger

logger = get_logger(__name__)


def load_npy_labels(path):
    if path is not None:
        labels = np.load(path, mmap_mode='r', allow_pickle=False)
        return labels


def adjust_to_batch_size(np_array, batch_size):
    if batch_size > np_array.shape[0]:
        array = np_array
        for _ in range(int(batch_size / np_array.shape[0]) - 1):
            np_array = np.append(np_array, array, axis=0)
        np_array = np.append(np_array, np_array[:batch_size % array.shape[0]], axis=0)
    else:
        np_array = np_array[:batch_size, ...]
    return np_array


def transpose_input(images, axes):
    tuple_axes = [int(ax) for ax in axes]
    return images.transpose(tuple_axes)


def crop_resize(img, cropx, cropy):
    y, x, c = img.shape
    if y < cropy:
        img = cv2.resize(img, (x, cropy))
        y = cropy
    if x < cropx:
        img = cv2.resize(img, (cropx, y))
        x = cropx
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx, :]


def load_jpeg(path: str, height, width, ids, datatype=np.float32):
    img = cv2.imread(path).astype(datatype)  # BGR color format, shape HWC
    if height > 0 and width > 0:
        img = cv2.resize(img, (width, height))
    img = img.transpose(ids)
    reshaped_img = img.reshape(1, *img.shape)
    logger.debug(f"Image  {path}  shape:  {reshaped_img.shape} ; "
                 f"data range: {np.amin(reshaped_img)} : {np.amax(reshaped_img)} ")
    return reshaped_img


def load_labels(path):
    labels_extension = path.split(sep=".")[-1]
    if labels_extension == "npy":
        return load_npy_labels(path=path)
    elif labels_extension in ["txt", "json"]:
        raise NotImplementedError()
    else:
        raise RuntimeError(f"Incorrect label data type: {labels_extension}")


def load_images(data_path, height, width, ids):
    assert os.path.exists(data_path), "Error data path for images do not exists."
    if os.path.isfile(data_path):
        file_extension = os.path.basename(data_path).split(sep=".")[-1]
        assert file_extension in ['jpg', 'jpeg']
        inputs = load_jpeg(data_path, height, width, ids)
        return inputs
    elif os.path.isdir(data_path):
        inputs = []
        images = list(filter(lambda x: re.match(r".+\.jpe?g", x.lower()), os.listdir(data_path)))
        for img in images:
            path = os.path.join(data_path, img)
            inputs.append(load_jpeg(path, height, width, ids))
            assert inputs, f"Lack of data to load with search path: {data_path}"
            inputs = np.concatenate(inputs, axis=0)
            return inputs
        else:
            raise AssertionError(
                f"incorrect input_data_type value: {images} for provided input path (dir): {data_path}"
            )
    else:
        raise AssertionError(f"Invalida data_path={data_path}")


def load_numpy(data_path):
    assert os.path.isfile(data_path)
    file_extension = os.path.basename(data_path).split(sep=".")[-1]
    # optional preprocessing depending on the model
    data = np.load(data_path, mmap_mode='r+', allow_pickle=False)
    data = data - np.min(data)  # Normalization 0-255
    data = data / np.ptp(data) * 255  # Normalization 0-255
    # images = images[:,:,:,::-1] # RGB to BGR
    logger.debug(
        f'Input {os.path.basename(data_path)} shape: {data.shape}; data range: {np.amin(data)}: {np.amax(data)}'
    )
    return data


def prepare_data(data_path, expected_shape, batch_size, transpose_axes=None, expected_layout=None, data_layout=None):
    filename, file_extension = os.path.splitext(data_path)
    if file_extension == '.npy':
        data = load_numpy(data_path)
    else:
        if data_layout is None:
            data_layout = "NHWC"
        if expected_layout is None:
            expected_layout = "NCHW"

        ids = []
        for dimension in expected_layout:
            id = data_layout.lower().find(dimension.lower())
            ids.append(id - 1)

        height_index = expected_layout.lower().find("h")
        width_index = expected_layout.lower().find("w")
        expected_height = expected_shape[height_index]
        expected_width = expected_shape[width_index]

        data = load_images(data_path, expected_height, expected_width, ids[1:])

    if batch_size == -1:
        batch_size = 1
    data = adjust_to_batch_size(np_array=data, batch_size=int(batch_size))
    if transpose_axes:
        data = transpose_input(images=data, axes=transpose_axes)

    return data

def is_dynamic_shape(shape):
    return any(map(lambda x: x == -1, shape))

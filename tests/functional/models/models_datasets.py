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

import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from random import choice
from string import ascii_lowercase

import cv2
import numpy as np
from PIL import Image

from tests.functional.utils.numpy_loader import prepare_data
from tests.functional.config import binary_io_images_path, datasets_path
from tests.functional.constants.ovms import Ovms


def load_image_data_from_path(full_path, img_format, img_mode=None, size=None):
    img_byte_arr = BytesIO()
    image_obj = Image.open(full_path, mode="r", formats=None)
    if img_mode:
        image_obj = image_obj.convert(img_mode)
    if size:
        image_obj = image_obj.resize(size)
    image_obj.save(img_byte_arr, format=img_format)
    return img_byte_arr.getvalue()


class ModelDataset:
    @staticmethod
    def create(data_str):
        result = None
        ext = os.path.splitext(data_str)[1]
        if ext == ".npy":
            result = NumPyDataset(data_str)
        return result

    def __init__(self):
        self.data_path = None
        self.name = None
        self.shape = None

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        data = prepare_data(
            data_path=self.data_path,
            expected_shape=shape,
            batch_size=batch_size,
            transpose_axes=transpose_axes,
            expected_layout=layout,
            data_layout=None,
        )
        self.shape = data.shape
        return data

    def to_str(self):
        return json.dumps(self.__dict__)


class NumPyDataset(ModelDataset):

    def __init__(self, *data_path):
        self.name = data_path[0]
        self.data_path = os.path.join(datasets_path, *data_path)


class NumPyImageData(NumPyDataset):

    def __init__(self, size=None):
        super().__init__("100_v1_imgs.npy")
        self.size = size


class DummyDataset(NumPyDataset):

    def __init__(self):
        super().__init__("dummy_input.npy")


class RandomDataset(NumPyDataset):

    def __init__(self):
        super().__init__("random_1000.npy")


class EastDataset(ModelDataset):

    def __init__(self):
        self.name = "east_and_crnn"
        self.data_path = os.path.join(datasets_path, self.name)


class BinaryDummyModelDataset(ModelDataset):
    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        return np.ones(shape, datatype)

    def get_source_data(self, shape, dtype=np.float32):
        return np.ones(shape, dtype)

    def create_data(self, tmp_file_location, shape, img_format):
        data = self.get_source_data(shape)
        fname = f'generated_ones_{"x".join([str(x) for x in shape])}.{img_format.lower()}'
        os.makedirs(tmp_file_location, exist_ok=True)
        cv2.imwrite(os.path.join(tmp_file_location, fname), data)
        return load_image_data_from_path(os.path.join(tmp_file_location, fname), img_format)


@dataclass
class DefaultBinaryDataset(ModelDataset):
    _saved_labels_to_path_mapping: str = None
    image_format: str = None
    image_mode: str = None
    max_num_of_images: int = None
    offset: int = 0

    def get_path(self):
        return os.path.join(binary_io_images_path, "input_images.txt")

    def _get_image_label_mapping(self):
        image_list_path = self.get_path()
        image_labels = {}
        with open(image_list_path, "r") as f:
            for line in f.readlines():
                path, label = line.strip().split(" ")
                image_labels[path] = label
        return image_labels

    def get_data(self, shape, batch_size, transpose_axes, layout, reshape=False, datatype=np.float32):
        labels_to_path_mapping = self._get_image_label_mapping()

        i = 0
        images = []
        size = shape[-2:] if reshape else None
        for path, label in labels_to_path_mapping.items():
            i += 1
            if i <= self.offset:
                continue

            full_path = os.path.join(binary_io_images_path, path)
            img_data = load_image_data_from_path(full_path, self.image_format, size=size)
            images.append(img_data)

            if len(images) == batch_size:
                break

        self._saved_labels_to_path_mapping = labels_to_path_mapping
        return images

    def verify_match(self, response):
        result = True

        if self._saved_labels_to_path_mapping is None:
            return result

        labels = list(self._saved_labels_to_path_mapping.values())

        nu = list(response.items())
        assert len(nu) == 1  # We expect single dimension result
        nu = nu[0][1]

        if nu.shape[-1] == 1001:
            model_offset = 1
        else:
            model_offset = 0

        for i in range(nu.shape[0]):
            label = int(labels[i + self.offset])
            single_result = nu[[i], ...]
            ma = np.argmax(single_result) - model_offset
            if label != ma:
                result = False
        return result


@dataclass
class ExactShapeBinaryDataset(DefaultBinaryDataset):
    shape: dict = field(default_factory=lambda: [])
    image_format: str = Ovms.JPG_IMAGE_FORMAT

    def get_path(self):
        file_path = os.path.join(binary_io_images_path, "images", "_".join([str(x) for x in self.shape]))
        file_names = list(filter(lambda x: re.match(r".+\.jpe?g$", x.lower()), os.listdir(file_path)))
        assert len(file_names) == 1, f"Unable to find images file with shape: {self.shape}"
        return os.path.join(file_path, file_names[0])

    def get_data(self, shape, batch_size, transpose_axes, layout=None, reshape=False, datatype=np.float32):
        path = self.get_path()
        return [load_image_data_from_path(path, self.image_format)]


class LanguageModelDataset(ModelDataset):
    str_input_data = ["Lorem ipsum dolor sit amet", "consectetur adipiscing elit", "sed do eiusmod tempor"]

    def __init__(self, data_sample=0):
        try:
            self.default_str_input_data = self.str_input_data[data_sample]
        except IndexError:
            self.default_str_input_data = self.str_input_data[-1]

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        # https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_grpc_string_infer_client.py
        str_input_data = [""] if batch_size == 0 else [s.split() for s in self.get_str_input_data()]
        return np.array([str(x).encode("utf-8") for x in str_input_data], dtype=np.object_)

    def get_str_input_data(self):
        return [self.default_str_input_data]

    def get_source_data(self, shape, dtype=np.float32):
        return {}

    def create_data(self, tmp_file_location, shape, img_format):
        return {}

    @staticmethod
    def generate_random_text_list(inputs_number, word_length=5):
        return ["".join(choice(ascii_lowercase) for _ in range(word_length)) for _ in range(inputs_number)]


class FeatureExtractionModelDataset(ModelDataset):
    input_data_1 = "That is a happy person."
    input_data_2 = "That is a very happy person."
    input_data = [input_data_1, input_data_2]

    def __init__(self, data_sample=0):
        self.default_input_data = self.input_data

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        return self.default_input_data

    def get_source_data(self, shape, dtype=np.float32):
        return {}

    def create_data(self, tmp_file_location, shape, img_format):
        return {}

    def get_string_data(self):
        return self.input_data_1

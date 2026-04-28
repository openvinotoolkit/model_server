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

import base64
import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from random import choice, randint
from string import ascii_lowercase
from typing import List, Tuple

import cv2
import numpy as np
from kaldi_python_io import ArchiveReader
from PIL import Image

from tests.functional.utils.inference.serving.openai import ChatCompletionsApi
from tests.functional.utils.logger import get_logger
from common_libs.numpy_loader import load_jpeg, prepare_data, transpose_input
from ovms.config import binary_io_images_path, ovms_c_repo_path, ovms_test_repo_path
from tests.functional.config import datasets_path
from tests.functional.constants.ovms import Ovms, MediaPipeConstants
from tests.functional.constants.paths import Paths
from ovms.constants.audio_reference import AUDIO_SAMPLES, TRANSLATION_REFERENCES


logger = get_logger(__name__)


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


class LargeLanguageModelDataset(ModelDataset):
    user_content = "What is OpenVINO?"
    system_content = "You are a helpful assistant."
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    system_data = [ChatCompletionsApi.ROLE_SYSTEM, system_content]
    input_data = [system_data, user_data]

    def __init__(self, data_sample=0):
        self.default_input_data = self.input_data

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        return self.default_input_data

    def get_source_data(self, shape, dtype=np.float32):
        return {}

    def create_data(self, tmp_file_location, shape, img_format):
        return {}


class AssistantLargeLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "What is OpenVINO?"
    assistant_content = "You are a helpful assistant."
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    assistant_data = [ChatCompletionsApi.ROLE_ASSISTANT, assistant_content]
    input_data = [user_data, assistant_data]


class FeatureExtractionModelDataset(LargeLanguageModelDataset):
    input_data_1 = "That is a happy person."
    input_data_2 = "That is a very happy person."
    input_data = [input_data_1, input_data_2]

    def __init__(self, data_sample=0):
        self.default_input_data = self.input_data

    def get_string_data(self):
        return self.input_data_1


class FeatureExtractionZhModelDataset(FeatureExtractionModelDataset):
    input_data_1 = "則西安謀大洲則伊犂視" * 51    # set maximum length of allowed content to check CVS-155463
    input_data = [input_data_1]


class RerankModelDataset(LargeLanguageModelDataset):
    query = "hello"
    document_1 = "welcome"
    document_2 = "farewell"
    input_data = {
        "query": query,
        "documents": [document_1, document_2]
    }

    def __init__(self, data_sample=0):
        self.default_input_data = self.input_data

    def get_string_data(self):
        return str(self.input_data)


class ShortResponseLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "What is the capital of France?"
    system_content = "You are an assistant who limits response to one word."
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    system_data = [ChatCompletionsApi.ROLE_SYSTEM, system_content]
    input_data = [system_data, user_data]


class ShortResponseSingleMessageLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "What is the capital of France? Answer in one word."
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class LongResponseLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "Elaborate which framework is better: Tensorflow or Pytorch?"
    system_content = "You are an assistant who gives very long answers."
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    system_data = [ChatCompletionsApi.ROLE_SYSTEM, system_content]
    input_data = [system_data, user_data]


class LongResponseSingleMessageLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "Elaborate which framework is better: Tensorflow or Pytorch? Give very long answers."
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


def get_long_prompt():
    with open(os.path.join(ovms_test_repo_path, "data", "llm", "long_prompt.txt"), 'r', encoding='utf-8') as file:
        user_content = file.read()
    return user_content


class LongPromptLanguageModelDataset(LargeLanguageModelDataset):
    user_content = get_long_prompt()
    system_content = "You are a helpful assistant giving short answers."
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    system_data = [ChatCompletionsApi.ROLE_SYSTEM, system_content]
    input_data = [system_data, user_data]


class LongPromptSingleMessageLanguageModelDataset(LargeLanguageModelDataset):
    user_content = get_long_prompt()
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class SingleMessageLanguageModelDataset(LargeLanguageModelDataset):
    user_data = [ChatCompletionsApi.ROLE_USER, LargeLanguageModelDataset.user_content]
    input_data = [user_data]


class MistralLanguageModelDataset(LargeLanguageModelDataset):
    # Conversation roles must alternate user/assistant/user/assistant/...
    user_data = [ChatCompletionsApi.ROLE_USER, LargeLanguageModelDataset.user_content]
    system_data = [ChatCompletionsApi.ROLE_ASSISTANT, LargeLanguageModelDataset.system_content]
    input_data = [user_data, system_data]


class ToolsParsingLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "What is the weather like today in Paris and what is the air pollution level in Paris?"
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class GetWeatherToolsParsingLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "What is the weather like in Paris today?"
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class GetPollutionToolsParsingLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "What is the pollution level in Paris today?"
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class ToolsCallLanguageModelDataset(LargeLanguageModelDataset):
    user_content = "What is the weather like in Paris today?"
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    assistant_content = ""
    assistant_tool_calls = [{
                    "id": "TOOL_CALLS_ID",
                    "type": "function",
                    "function": {
                        "name": "get_weather", "arguments": {"location": "Paris, France"}
                    }
                }
            ]
    assistant_reasoning_content = None
    assistant_data = [
        ChatCompletionsApi.ROLE_ASSISTANT, assistant_content, assistant_tool_calls, assistant_reasoning_content
    ]
    tool_content = "15 degrees Celsius"
    tool_tool_call_id = "TOOL_CALLS_ID"
    tool_data = [ChatCompletionsApi.ROLE_TOOL, tool_content, tool_tool_call_id]
    input_data = [user_data, assistant_data, tool_data]


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


@dataclass
class CustomBinaryDataset(ModelDataset):
    image_format: str = None
    image_dim: Tuple[int, int] = None
    channel_format: str = Ovms.IMAGE_CHANNEL_FORMAT_RGB

    def load_image(self):
        image_obj = Image.new(self.channel_format, self.image_dim)
        img_byte_arr = BytesIO()
        image_obj.save(img_byte_arr, format=self.image_format)
        image_bytes = img_byte_arr.getvalue()
        return image_bytes

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        return [self.load_image()]

    def verify_match(self, response):
        return True


class ArchiveDataset(ModelDataset):
    def __init__(self, *data_path):
        self.name = data_path[0]
        self.utterance_name_list = []
        self.data_path = os.path.join(datasets_path, *data_path)

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        result = dict()
        ark_data = self.get_reference_data()
        self.utterance_name_list.clear()
        for utternace_name, data_obj in ark_data:
            if utternace_name not in self.utterance_name_list:
                self.utterance_name_list.append(utternace_name)
                result[utternace_name] = {}
            sequence_size = data_obj.shape[0]
            data = [np.expand_dims(data_obj[x], axis=0) for x in range(sequence_size)]
            for i in range(sequence_size):
                result[utternace_name][i] = data[i]
        return result

    def get_reference_data(self):
        return ArchiveReader(self.data_path)


class NumPyDataset(ModelDataset):

    def __init__(self, *data_path):
        self.name = data_path[0]
        self.data_path = os.path.join(datasets_path, *data_path)


class NumPyPerfDataset(ModelDataset):

    def __init__(self, *data_path):
        self.name = data_path[0]
        self.data_path = os.path.join(Paths.DATASET_MAIN_PATH, *data_path)

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        size = None
        if shape is not None:
            size = shape[:-2]
        elif self.size is not None:
            size = self.size
        return prepare_data(data_path=self.data_path, size=size, batch_size=batch_size, transpose_axes=transpose_axes)


class MatmulDataset(NumPyDataset):

    def __init__(self):
        super().__init__("op_data.npy")


class DummyDataset(NumPyDataset):

    def __init__(self):
        super().__init__("dummy_input.npy")


class RandomDataset(NumPyDataset):

    def __init__(self):
        super().__init__("random_1000.npy")


class NumPyImageData(NumPyDataset):

    def __init__(self, size=None):
        super().__init__("100_v1_imgs.npy")
        self.size = size


class NumPyImagePerfData(NumPyPerfDataset):

    def __init__(self, size=None):
        super().__init__("100_v1_imgs.npy")
        self.size = size


class BrainDataset(NumPyDataset):

    def __init__(self):
        super().__init__("braTS.npy")


class InceptionResnetV2Dataset(NumPyDataset):

    def __init__(self):
        super().__init__("small_imagenet", "dataset")


class CocoDataset(ModelDataset):

    def __init__(self):
        super().__init__()
        self.name = "coco"
        self.data_path = os.path.join(datasets_path, self.name)

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        return self._load_data(self.data_path, shape, batch_size, transpose_axes, datatype)

    def _load_data(self, data_path, shape, batch_size, transpose_axes, datatype=np.float32):
        result = []
        height, width = shape[-2:]
        # reverse setting used since the first image from dataset produces invalid output
        images = [file for file in sorted(os.listdir(data_path), reverse=True) if file.endswith(".jpg")]
        for idx, img in enumerate(images):
            if idx >= batch_size:
                break
            path = os.path.join(data_path, img)
            data = load_jpeg(path, height, width, [2, 0, 1], datatype)
            result.append(data)
        result = np.concatenate(result, axis=0)
        if transpose_axes:
            result = transpose_input(images=result, axes=transpose_axes)
        return result


class EastDataset(ModelDataset):

    def __init__(self):
        self.name = "east_and_crnn"
        self.data_path = os.path.join(datasets_path, self.name)


class HorizontalDataset(ModelDataset):

    def __init__(self):
        self.name = "horizontal_text"
        self.data_path = os.path.join(datasets_path, self.name, "horizontal_text.mp4")


class FaceDetectionDataset(ModelDataset):

    def __init__(self, *data_path):
        self.name = "example_client/images/people"
        self.data_path = os.path.join(ovms_c_repo_path, self.name)


class OcrNetHrNetW48PaddleDataset(ModelDataset):
    def __init__(self, *data_path):
        self.name = r"demos/common/static/images/cars/road1.jpg"
        self.data_path = os.path.join(ovms_c_repo_path, self.name)


class SmallCocoDataset:

    class ImgData(CocoDataset):

        def __init__(self):
            super().__init__()
            self.name = "small_coco_info/dataset"
            self.data_path = os.path.join(datasets_path, self.name)

        def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
            return self._load_data(self.data_path, shape, batch_size, transpose_axes)

    class ImgInfo(ModelDataset):

        def __init__(self):
            super().__init__()
            self.name = "small_coco_info/info"
            self.data_path = os.path.join(datasets_path, self.name, "data.npy")


class Unet3dDataset(NumPyPerfDataset):

    def __init__(self, size=None):
        super().__init__("imgs_validation_3d_flair_transposed.npy")
        self.size = size

    def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
        size = None
        if shape is not None:
            size = shape[:-3]
        elif self.size is not None:
            size = self.size
        return prepare_data(data_path=self.data_path, size=size, batch_size=batch_size, transpose_axes=transpose_axes)


class BertSmallIn8Dataset:
    sentence_length = 384
    token_pad = 0
    token_unk = 100
    token_cls = 101
    token_sep = 102
    token_mask = 103

    question = [randint(1000, 30522) for _ in range(7)]
    context = [randint(1000, 30522) for _ in range(374)]

    wiki_graph_bert_datase = {
        "question": [2054, 2003, 1037, 10629, 1029],
        "context": [
            1999,
            5597,
            1010,
            1998,
            2062,
            4919,
            1999,
            10629,
            3399,
            1010,
            1037,
            10629,
            2003,
            1037,
            3252,
            3815,
            2075,
            2000,
            1037,
            2275,
            1997,
            5200,
            1999,
            2029,
            2070,
            7689,
            1997,
            1996,
            5200,
            2024,
            1999,
            2070,
            3168,
            1000,
            3141,
            1000,
            1012,
            1996,
            5200,
            17254,
            2000,
            8045,
            24504,
            2015,
            2170,
            18984,
            1006,
            2036,
            2170,
            14164,
            2030,
            2685,
            1007,
            1998,
            2169,
            1997,
            1996,
            3141,
            7689,
            1997,
            18984,
            2003,
            2170,
            2019,
            3341,
            1006,
            2036,
            2170,
            4957,
            2030,
            2240,
            1007,
            1012,
            1031,
            1015,
            1033,
            4050,
            1010,
            1037,
            10629,
            2003,
            8212,
            1999,
            16403,
            12644,
            2433,
            2004,
            1037,
            2275,
            1997,
            14981,
            2030,
            7925,
            2005,
            1996,
            18984,
            1010,
            2587,
            2011,
            3210,
            2030,
            10543,
            2005,
            1996,
            7926,
            1012,
            19287,
            2024,
            2028,
            1997,
            1996,
            5200,
            1997,
            2817,
            1999,
            16246,
            5597,
            1012,
            1996,
            7926,
            2089,
            2022,
            2856,
            2030,
            6151,
            7442,
            10985,
            1012,
            2005,
            2742,
            1010,
            2065,
            1996,
            18984,
            5050,
            2111,
            2012,
            1037,
            2283,
            1010,
            1998,
            2045,
            2003,
            2019,
            3341,
            2090,
            2048,
            2111,
            2065,
            2027,
            6073,
            2398,
            1010,
            2059,
            2023,
            10629,
            2003,
            6151,
            7442,
            10985,
            2138,
            2151,
            2711,
            1037,
            2064,
            6073,
            2398,
            2007,
            1037,
            2711,
            1038,
            2069,
            2065,
            1038,
            2036,
            10854,
            2398,
            2007,
            1037,
            1012,
            1999,
            5688,
            1010,
            2065,
            2151,
            3341,
            2013,
            1037,
            2711,
            1037,
            2000,
            1037,
            2711,
            1038,
            14788,
            2000,
            1037,
            24381,
            2769,
            2000,
            1038,
            1010,
            2059,
            2023,
            10629,
            2003,
            2856,
            1010,
            2138,
            11427,
            2769,
            2003,
            2025,
            9352,
            28667,
            11514,
            3217,
            12921,
            1012,
            1996,
            2280,
            2828,
            1997,
            10629,
            2003,
            2170,
            2019,
            6151,
            7442,
            10985,
            10629,
            2096,
            1996,
            3732,
            2828,
            1997,
            10629,
            2003,
            2170,
            1037,
            2856,
            10629,
            1012,
            19287,
            2024,
            1996,
            3937,
            3395,
            3273,
            2011,
            10629,
            3399,
            1012,
            1996,
            2773,
            1000,
            10629,
            1000,
            2001,
            2034,
            2109,
            1999,
            2023,
            3168,
            2011,
            2508,
            3312,
            20016,
            1999,
            7261,
            1012,
            1031,
            1016,
            1033,
            1031,
            1017,
            1033,
            15182,
            1999,
            10629,
            3399,
            8137,
            1012,
            1996,
            2206,
            2024,
            2070,
            1997,
            1996,
            2062,
            3937,
            3971,
            1997,
            12854,
            19287,
            1998,
            3141,
            8045,
            5090,
            1012,
        ],
    }

    wiki_chess_bert_dataset = {
        "question": [2040, 21208, 2015, 7433, 2971, 7587],
        "context": [
            7433,
            2003,
            1037,
            10517,
            1998,
            6975,
            2604,
            2208,
            2209,
            2090,
            2048,
            2867,
            1012,
            2009,
            2003,
            2823,
            2170,
            2530,
            2030,
            2248,
            7433,
            2000,
            10782,
            2009,
            2013,
            3141,
            2399,
            2107,
            2004,
            27735,
            14702,
            1012,
            1996,
            2783,
            2433,
            1997,
            1996,
            2208,
            6003,
            1999,
            2670,
            2885,
            2076,
            1996,
            2117,
            2431,
            1997,
            1996,
            6286,
            2301,
            2044,
            20607,
            2013,
            2714,
            1010,
            2172,
            3080,
            2399,
            1997,
            2796,
            1998,
            4723,
            4761,
            1012,
            2651,
            1010,
            7433,
            2003,
            2028,
            1997,
            1996,
            2088,
            1005,
            1055,
            2087,
            2759,
            2399,
            1010,
            2209,
            2011,
            8817,
            1997,
            2111,
            4969,
            2012,
            2188,
            1010,
            1999,
            4184,
            1010,
            3784,
            1010,
            2011,
            11061,
            1010,
            1998,
            1999,
            8504,
            1012,
            7433,
            2003,
            2019,
            10061,
            5656,
            2208,
            1998,
            7336,
            2053,
            5023,
            2592,
            1012,
            2009,
            2003,
            2209,
            2006,
            1037,
            2675,
            7433,
            6277,
            2007,
            4185,
            14320,
            5412,
            1999,
            2019,
            2809,
            1011,
            2011,
            1011,
            2809,
            8370,
            1012,
            2012,
            1996,
            2707,
            1010,
            2169,
            2447,
            1006,
            2028,
            9756,
            1996,
            2317,
            4109,
            1010,
            1996,
            2060,
            9756,
            1996,
            2304,
            4109,
            1007,
            7711,
            7032,
            4109,
            1024,
            2028,
            2332,
            1010,
            2028,
            3035,
            1010,
            2048,
            28620,
            2015,
            1010,
            2048,
            7307,
            1010,
            2048,
            8414,
            1010,
            1998,
            2809,
            19175,
            2015,
            1012,
            1996,
            4874,
            1997,
            1996,
            2208,
            2003,
            2000,
            4638,
            8585,
            1996,
            7116,
            1005,
            1055,
            2332,
            1010,
            13557,
            1996,
            2332,
            2003,
            2104,
            6234,
            2886,
            1006,
            1999,
            1000,
            4638,
            1000,
            1007,
            1998,
            2045,
            2003,
            2053,
            2126,
            2000,
            6366,
            2009,
            2013,
            2886,
            2006,
            1996,
            2279,
            2693,
            1012,
            2045,
            2024,
            2036,
            2195,
            3971,
            1037,
            2208,
            2064,
            2203,
            1999,
            1037,
            4009,
            1012,
            4114,
            7433,
            10375,
            1999,
            1996,
            3708,
            2301,
            1012,
            7433,
            2971,
            2651,
            2003,
            9950,
            7587,
            2011,
            26000,
            1006,
            2248,
            7433,
            4657,
            1007,
            1012,
            1996,
            2034,
            21186,
            3858,
            2088,
            7433,
            3410,
            1010,
            9070,
            14233,
            8838,
            1010,
            3555,
            2010,
            2516,
            1999,
            6929,
            1025,
            10045,
            5529,
            5054,
            2003,
            1996,
            2783,
            2088,
            3410,
            1012,
            1037,
            4121,
            2303,
            1997,
            7433,
            3399,
            2038,
            2764,
            2144,
            1996,
            2208,
            1005,
            1055,
            12149,
            1012,
            5919,
            1997,
            2396,
            2024,
            2179,
            1999,
            7433,
            5512,
            1025,
            1998,
            7433,
            1999,
            2049,
            2735,
            5105,
            2530,
            3226,
            1998,
            2396,
            1998,
            2038,
            7264,
            2007,
            2060,
            4249,
            2107,
            2004,
            5597,
            1010,
            3274,
            2671,
            1010,
            1998,
            6825,
            1012,
        ],
    }

    class InputData(ModelDataset):
        def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
            output = [
                BertSmallIn8Dataset.token_cls,
                *BertSmallIn8Dataset.question,
                BertSmallIn8Dataset.token_sep,
                *BertSmallIn8Dataset.context,
                BertSmallIn8Dataset.token_sep,
            ]
            assert len(output) <= BertSmallIn8Dataset.sentence_length
            return output + [BertSmallIn8Dataset.token_pad] * (BertSmallIn8Dataset.sentence_length - len(output))

    class DataMask(ModelDataset):
        def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
            base_length = 3 + len(BertSmallIn8Dataset.question) + len(BertSmallIn8Dataset.context)
            assert base_length <= BertSmallIn8Dataset.sentence_length
            return base_length * [1] + (BertSmallIn8Dataset.sentence_length - base_length) * [0]

    class ContextMask(ModelDataset):
        def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
            base_length = 3 + len(BertSmallIn8Dataset.question) + len(BertSmallIn8Dataset.context)
            mask = (len(BertSmallIn8Dataset.question) + 2) * [0] + (len(BertSmallIn8Dataset.context) + 1) * [1]
            return mask + (BertSmallIn8Dataset.sentence_length - base_length) * [0]

    class PositionIds(ModelDataset):
        def get_data(self, shape, batch_size, transpose_axes, layout=None, datatype=np.float32):
            return list(range(BertSmallIn8Dataset.sentence_length))


class DatasetList(List[ModelDataset]):
    @property
    def names(self):
        return [dataset.name for dataset in self]


class ZebraImageDataset(NumPyDataset):
    def __init__(self):
        super().__init__("zebra.jpeg")


class ZebraInpaintingMaskDataset(NumPyDataset):
    def __init__(self):
        super().__init__("zebra_mask_inpainting.png")


class ZebraOutpaintingMaskDataset(NumPyDataset):
    def __init__(self):
        super().__init__("zebra_mask_outpainting.png")


class VisionLanguageModelImageDataset(ZebraImageDataset):
    pass


class VariousVisionLanguageModelImageDataset(VisionLanguageModelImageDataset):

    def __init__(self, image_path=None):
        self.name = os.path.basename(image_path)
        self.data_path = image_path


class VisionLanguageModelDataset(LargeLanguageModelDataset):
    image_datasets = [VisionLanguageModelImageDataset]
    user_content_text = "Describe what is in the picture."
    user_content_image_url = {"url": f"data:image/jpeg;base64,CONVERT_IMAGE_0"}
    user_content = [
        {
            ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_TEXT,
            ChatCompletionsApi.CONTENT_TYPE_TEXT: user_content_text,
        }, {
            ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL,
            ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: user_content_image_url,
        }
    ]
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]

    def __init__(self, data_sample=0):
        input_data_str = json.dumps(self.input_data)
        for i, image_dataset in enumerate(self.image_datasets):
            convert_image_text = self.convert_image(image_dataset().data_path)
            input_data_str = input_data_str.replace(f"CONVERT_IMAGE_{i}", convert_image_text)
        self.input_data = json.loads(input_data_str)
        self.default_input_data = self.input_data

    @staticmethod
    def convert_image(image_path):
        with open(image_path, "rb") as file:
            base64_image = base64.b64encode(file.read()).decode("utf-8")
        return base64_image


class LongResponseVisionLanguageModelDataset(VisionLanguageModelDataset):
    image_datasets = [VisionLanguageModelImageDataset]
    user_content_text = "Describe what is in the picture in details. Give a very long response."
    user_content_image_url = {"url": f"data:image/jpeg;base64,CONVERT_IMAGE_0"}
    user_content = [
        {
            ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_TEXT,
            ChatCompletionsApi.CONTENT_TYPE_TEXT: user_content_text,
        }, {
            ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL,
            ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: user_content_image_url,
        }
    ]
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class ImageURLVisionLanguageModelDataset(VisionLanguageModelDataset):
    def __init__(self, image_url=None, data_sample=0):
        if image_url is None:
            image_url = "https://raw.githubusercontent.com/openvinotoolkit/model_server/" \
                        "refs/heads/main/demos/common/static/images/zebra.jpeg"
        self.user_content_image_url = {"url": image_url}
        self.user_content_text = "Describe what is in the picture in details."
        self.user_content = [
            {
                ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_TEXT,
                ChatCompletionsApi.CONTENT_TYPE_TEXT: self.user_content_text,
            }, {
                ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL,
                ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: self.user_content_image_url,
            }
        ]
        self.user_data = [ChatCompletionsApi.ROLE_USER, self.user_content]
        self.input_data = [self.user_data]
        self.default_input_data = self.input_data


class ImageFilesystemVisionLanguageModelDataset(ImageURLVisionLanguageModelDataset):
    def __init__(self, image_path=None, data_sample=0):
        image_path = image_path if image_path is not None else Paths.ZEBRA_PATH_INTERNAL
        self.user_content_text = "Describe what is in the picture in details."
        self.user_content_image_url = {"url": image_path}
        user_content = [
            {
                ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_TEXT,
                ChatCompletionsApi.CONTENT_TYPE_TEXT: self.user_content_text,
            }, {
                ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL,
                ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: self.user_content_image_url,
            }
        ]
        self.user_data = [ChatCompletionsApi.ROLE_USER, user_content]
        self.input_data = [self.user_data]
        self.default_input_data = self.input_data

    @staticmethod
    def resize_image(container_path, target_size_mb=20):
        # Use this method to resize image
        target_size_bytes = target_size_mb * 1024 * 1024
        output_path = os.path.join(container_path, Paths.IMAGES, "resized_zebra.jpeg")
        input_path = MediaPipeConstants.IMAGE_FILESYSTEM_ZEBRA_JPEG

        with Image.open(input_path) as img:
            scale = 2
            while True:
                new_size = (img.width * scale, img.height * scale)
                resized_img = img.resize(new_size)
                resized_img.save(output_path, format='JPEG', quality=100)
                current_size = os.path.getsize(output_path)
                logger.debug(f"Resolution: {new_size}, current size: {current_size / (1024 * 1024):.2f} MB")

                if current_size >= target_size_bytes:
                    logger.info(f"Image saved to '{output_path}' ({current_size / (1024 * 1024):.2f} MB)")
                    break

                scale += 1

        return output_path


class VariousVisionLanguageModelDataset(VisionLanguageModelDataset):
    def __init__(self, image_path=None):
        self.image_datasets = [VariousVisionLanguageModelImageDataset]
        self.user_content_text = "Describe what is in the picture."
        self.prepare_input_data_str(image_path)

    def prepare_input_data_str(self, image_path):
        self.user_content_image_url = {"url": f"data:image/jpeg;base64,CONVERT_IMAGE_0"}
        self.user_content = [
            {
                ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_TEXT,
                ChatCompletionsApi.CONTENT_TYPE_TEXT: self.user_content_text,
            }, {
                ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL,
                ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: self.user_content_image_url,
            }
        ]
        self.user_data = [ChatCompletionsApi.ROLE_USER, self.user_content]
        self.input_data = [self.user_data]
        input_data_str = json.dumps(self.input_data)
        for i, image_dataset in enumerate(self.image_datasets):
            convert_image_text = self.convert_image(image_dataset(image_path).data_path)
            input_data_str = input_data_str.replace(f"CONVERT_IMAGE_{i}", convert_image_text)
        self.input_data = json.loads(input_data_str)
        self.default_input_data = self.input_data


class ShortResponseVariousVisionLanguageModelDataset(VariousVisionLanguageModelDataset):
    def __init__(self, image_path=None):
        self.image_datasets = [VariousVisionLanguageModelImageDataset]
        self.user_content_text = ("What is in the picture? Answer in three sentences or less, "
                                  "providing as much details as possible. Describe background.")
        self.prepare_input_data_str(image_path)


class NegativeImageTagVisionLanguageModelDataset(VisionLanguageModelDataset):
    image_datasets = [VisionLanguageModelImageDataset]
    user_content_text = "Describe what is in the pictures: <ov_genai_image_0> <ov_genai_image_1>"
    user_content_image_url = {"url": f"data:image/jpeg;base64,CONVERT_IMAGE_0"}
    user_content = [
        {
            ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_TEXT,
            ChatCompletionsApi.CONTENT_TYPE_TEXT: user_content_text,
        }, {
            ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL,
            ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: user_content_image_url,
        }
    ]
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class VisionLanguageModelBeam1ImageDataset(NumPyDataset):

    def __init__(self):
        super().__init__("beam_1.jpeg")


class VisionLanguageModelBeam2ImageDataset(NumPyDataset):

    def __init__(self):
        super().__init__("beam_2.jpeg")


class VisionLanguageModelBeam3ImageDataset(NumPyDataset):

    def __init__(self):
        super().__init__("beam_3.jpeg")


class VisionLanguageModelBeam4ImageDataset(NumPyDataset):

    def __init__(self):
        super().__init__("beam_4.jpeg")


class MultipleImagesVisionLanguageModelDataset(VisionLanguageModelDataset):
    image_datasets = [
        VisionLanguageModelBeam1ImageDataset,
        VisionLanguageModelBeam2ImageDataset,
        VisionLanguageModelBeam3ImageDataset,
        VisionLanguageModelBeam4ImageDataset,
    ]
    user_content_text = \
        "A collimated beam containing two different frequencies of light travels through vacuum and is incident on " \
        "a piece of glass. Which of the schematics below depicts the phenomenon of dispersion within the glass in a " \
        "qualitative correct manner? Select (e) if none of the options are qualitatively correct.\nA. <image 1>\nB. " \
        "<image 2>\nC. <image 3>\nD. <image 4>\n\nAnswer with the option's letter from the given choices directly."
    user_content = [
       {
           ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_TEXT,
           ChatCompletionsApi.CONTENT_TYPE_TEXT: user_content_text,
       }] + [
       {
           ChatCompletionsApi.CONTENT_TYPE: ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL,
           ChatCompletionsApi.CONTENT_TYPE_IMAGE_URL: {
               "url": f"data:image/jpeg;base64,CONVERT_IMAGE_{i}"},
       } for i in range(0, len(image_datasets))
   ]
    user_data = [ChatCompletionsApi.ROLE_USER, user_content]
    input_data = [user_data]


class ImageGenerationDataset(LargeLanguageModelDataset):
    input_data = ["Three cats sitting on the grass"]


class ImageEditDataset(LargeLanguageModelDataset):
    input_data = ["zebra in space, outer space, lots of space stars, best quality, extremely detailed",
                  ZebraImageDataset().data_path]


class ImageInpaintingDataset(LargeLanguageModelDataset):
    input_data = ["grizzly bear on a grassy field, best quality, extremely detailed",
                  ZebraImageDataset().data_path]
    mask_path = ZebraInpaintingMaskDataset().data_path


class ImageOutpaintingDataset(LargeLanguageModelDataset):
    input_data = ["zebra on a grassy field, forest in the distance, best quality, extremely detailed",
                  ZebraImageDataset().data_path]
    mask_path = ZebraOutpaintingMaskDataset().data_path


class AudioModelDataset(ModelDataset):
    def __init__(self, source_lang: str = "en", target_lang: str = "en",
                 duration: str = "short", audio_format: str = "wav", data_sample: int = 0):
        key = (source_lang, duration)
        if key not in AUDIO_SAMPLES:
            supported = sorted(AUDIO_SAMPLES.keys())
            raise ValueError(
                f"Unsupported audio sample combination: language='{source_lang}', duration='{duration}'. "
                f"Supported combinations: {supported}"
            )
        raw_sample = AUDIO_SAMPLES[key]
        if not raw_sample:
            raise ValueError(f"No audio samples configured for language '{source_lang}' and duration '{duration}'.")
        if isinstance(raw_sample, (list, tuple)):
            index = data_sample % len(raw_sample)
            sample = raw_sample[index]
        else:
            sample = raw_sample
        self.reference_audio_file = os.path.join(
            Paths.REFERENCE_AUDIO_FILES_DIR,
            sample.get_audio_file(audio_format),
        )
        self.source_name = sample.source_name
        self.input_language = source_lang
        self.output_language = target_lang
        self.duration_category = duration
        if source_lang == target_lang:
            self.reference_text = sample.reference_text
        else:
            translation_key = (source_lang, target_lang, duration)
            if translation_key not in TRANSLATION_REFERENCES:
                supported = sorted(TRANSLATION_REFERENCES.keys())
                raise ValueError(
                    f"Unsupported translation combination: source='{source_lang}', "
                    f"target='{target_lang}', duration='{duration}'. "
                    f"Supported combinations: {supported}"
                )
            self.reference_text = TRANSLATION_REFERENCES[translation_key]

    def get_data(self, shape=None, batch_size=None, transpose_axes=None, layout=None, datatype=np.float32):
        return [(self.reference_text, self.reference_audio_file)]


class AllAudioFilesDataset(ModelDataset):
    def __init__(self):
        self.name = "audio"
        self.data_path = Paths.REFERENCE_AUDIO_FILES_DIR


ovms_various_dataset_list = DatasetList([
        AllAudioFilesDataset(),
        NumPyImageData(),
        BrainDataset(),
        InceptionResnetV2Dataset(),
        EastDataset(),
        CocoDataset(),
        HorizontalDataset(),
        SmallCocoDataset.ImgData(),
        SmallCocoDataset.ImgInfo(),
        DummyDataset(),
        RandomDataset(),
        MatmulDataset(),
        VisionLanguageModelImageDataset(),
        VisionLanguageModelBeam1ImageDataset(),
        VisionLanguageModelBeam2ImageDataset(),
        VisionLanguageModelBeam3ImageDataset(),
        VisionLanguageModelBeam4ImageDataset(),
        ArchiveDataset("rm_lstm", "test_feat_1_10.ark"),
        ArchiveDataset("rm_lstm", "test_score_1_10.ark"),
        ArchiveDataset("aspire_tdnn", "mini_feat_1_10.ark"),
        ArchiveDataset("aspire_tdnn", "mini_feat_1_10_ivector.ark"),
        ArchiveDataset("aspire_tdnn", "aspire_tdnn_mini_feat_1_10_kaldi_score.ark"),
        ArchiveDataset("basic_lstm", "partitioned_input.ark"),
        ArchiveDataset("basic_lstm", "output.ark"),
        ZebraImageDataset(),
        ZebraInpaintingMaskDataset(),
        ZebraOutpaintingMaskDataset(),
    ])

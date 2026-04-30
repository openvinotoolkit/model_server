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
from dataclasses import dataclass, field

import numpy as np

from tests.functional.config import models_path, use_legacy_models
from tests.functional.constants.paths import Paths
from tests.functional.models.models import ModelInfo, ModelType
from tests.functional.object_model.shape import Shape
from tests.functional.constants.ovms import Ovms
from tests.functional.models.models_datasets import (
    DummyDataset,
    EastDataset,
    LanguageModelDataset,
    NumPyImageData,
    RandomDataset,
)


@dataclass
class ResnetLegacy(ModelInfo):
    name: str = "resnet-50-tf"
    batch_size: int = 1
    transpose_axes: str = "0321"
    inputs: dict = field(
        default_factory=lambda: {
            "map/TensorArrayStack/TensorArrayGatherV3": {
                "shape": [1, 3, 224, 224],
                "dtype": np.float32,
                "dataset": NumPyImageData(),
            }
        }
    )
    outputs: dict = field(default_factory=lambda: {"softmax_tensor": {"shape": [1, 1001], "dtype": np.float32}})


@dataclass
class ResnetFP32(ResnetLegacy):
    name: str = "resnet-50-tf_FP32"


if use_legacy_models:
    Resnet = ResnetLegacy
else:
    Resnet = ResnetFP32


@dataclass
class ResnetWrongInputName(Resnet):
    name: str = "resnet_wrong_input_name"
    inputs: dict = field(
        default_factory=lambda: {
            "wrong_input_name": {"shape": [1, 3, 224, 224], "dtype": np.float32, "dataset": NumPyImageData()}
        }
    )


@dataclass
class ResnetWrongInputShapes(Resnet):
    name: str = "resnet_wrong_input_shapes"
    inputs: dict = field(
        default_factory=lambda: {
            "map/TensorArrayStack/TensorArrayGatherV3": {
                "shape": [1, 3, 254, 214],
                "dtype": np.float32,
                "dataset": NumPyImageData(),
            }
        }
    )


@dataclass
class ResnetWrongInputShapeDim(Resnet):
    name: str = "resnet_wrong_input_shape_dim"
    inputs: dict = field(
        default_factory=lambda: {
            "map/TensorArrayStack/TensorArrayGatherV3": {
                "shape": [
                    1,
                ],
                "dtype": np.float32,
                "dataset": NumPyImageData(),
            }
        }
    )


@dataclass
class ResnetModelNameWithSlash(Resnet):
    name: str = "res/net"

    def __post_init__(self):
        super().__post_init__()
        if self.use_relative_paths:
            self.base_path = super().name
        else:
            if self.use_subconfig:
                self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, f"{super().name}_mediapipe", self.name)
            else:
                self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, super().name)
        self.model_path_on_host = os.path.join(models_path, super().name, str(self.version))


@dataclass
class ResnetModelNameWithWhitespace(Resnet):
    name: str = "res net"

    def __post_init__(self):
        super().__post_init__()
        if self.use_relative_paths:
            self.base_path = super().name
        else:
            if self.use_subconfig:
                self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, f"{super().name}_mediapipe", self.name)
            else:
                self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, super().name)
        self.model_path_on_host = os.path.join(models_path, super().name, str(self.version))


@dataclass
class Dummy(ModelInfo):
    name: str = "dummy"
    inputs: dict = field(
        default_factory=lambda: {"b": {"shape": Shape([1, 10]), "dtype": np.float32, "dataset": DummyDataset()}}
    )
    outputs: dict = field(default_factory=lambda: {"a": {"shape": Shape([1, 10]), "dtype": np.float32}})

    def prepare_input_data(self, batch_size=None, random_data=False, dtype=None, input_key=None):
        result = dict()
        for input_name, input_data in self.inputs.items():
            _dtype = dtype if dtype is not None else input_data["dtype"]
            result[input_name] = np.ones(input_data["shape"][-2:], dtype=_dtype)
        return result


@dataclass
class DummyIncrementDecrement(ModelInfo):
    name: str = "dummy_increment_decrement"
    inputs: dict = field(
        default_factory=lambda: {
            "input1": {"shape": [1, 1000], "dtype": np.float32, "dataset": RandomDataset()},
            "input2": {"shape": [1, 1000], "dtype": np.float32, "dataset": RandomDataset()},
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "output_increment": {"shape": [1, 1000], "dtype": np.float32},
            "output_decrement/sub": {"shape": [1, 1000], "dtype": np.float32},
        }
    )

    def get_expected_output(self, input_data: dict, client_type: str = None):
        result = {}
        ones = np.ones(self.input_shapes["input1"])
        arg0, arg1 = input_data
        input1 = input_data[arg0]
        result["output_increment"] = np.add(input1, ones)
        result["output_decrement/sub"] = np.subtract(input_data[arg0], input_data[arg1])
        return result


@dataclass
class DummyAdd2Inputs(ModelInfo):
    name: str = "dummy_add_2_inputs"
    inputs: dict = field(
        default_factory=lambda: {
            "input1": {"shape": [1, 1000], "dtype": np.float32, "dataset": RandomDataset()},
            "input2": {"shape": [1, 1000], "dtype": np.float32, "dataset": RandomDataset()},
        }
    )
    outputs: dict = field(default_factory=lambda: {"output_sum": {"shape": [1, 1000], "dtype": np.float32}})

    def get_expected_output(self, input_data: dict, client_type: str = None):
        result = {}
        result["output_sum"] = np.add(input_data["input1"], input_data["input2"])
        return result


@dataclass
class DummyIncrement(ModelInfo):
    name: str = "dummy_increment"
    inputs: dict = field(
        default_factory=lambda: {"input": {"shape": [1, 1000], "dtype": np.float32, "dataset": RandomDataset()}}
    )
    outputs: dict = field(
        default_factory=lambda: {"output": {"shape": [1, 1000], "dtype": np.float32, "dataset": RandomDataset()}}
    )


@dataclass
class VehicleDetection(ModelInfo):
    name: str = "vehicle-detection-0202"
    inputs: dict = field(default_factory=lambda: {"image": {"shape": [1, 3, 512, 512], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"detection_out": {"shape": [1, 1, 200, 7], "dtype": np.float32}})
    layout: str = Ovms.LAYOUT_NHWC


@dataclass
class VehicleAttributesRecognition(ModelInfo):
    name: str = "vehicle-attributes-recognition-barrier-0042"
    inputs: dict = field(default_factory=lambda: {"input": {"shape": [1, 3, 72, 72], "dtype": np.float32}})
    outputs: dict = field(
        default_factory=lambda: {
            "color": {"shape": [1, 7], "dtype": np.float32},
            "type": {"shape": [1, 4], "dtype": np.float32},
        }
    )
    layout: str = Ovms.LAYOUT_NHWC


@dataclass
class GoogleNetV2Fp32(ModelInfo):
    name: str = "googlenet-v2-fp32"
    transpose_axes: str = "0321"
    batch_size: int = 1
    inputs: dict = field(default_factory=lambda: {"input": {"shape": [1, 3, 224, 224], "dtype": np.float32}})
    outputs: dict = field(
        default_factory=lambda: {"InceptionV2/Predictions/Softmax": {"shape": [1, 1001], "dtype": np.float32}}
    )


@dataclass
class ArgMax(ModelInfo):
    name: str = "argmax"
    inputs: dict = field(
        default_factory=lambda: {
            "input1": {"shape": [1, 1001], "dtype": np.float32},
            "input2": {"shape": [1, 1001], "dtype": np.float32},
        }
    )
    outputs: dict = field(default_factory=lambda: {"argmax/Squeeze": {"shape": [1], "dtype": np.int32}})


@dataclass
class Increment4d(ModelInfo):
    name: str = "increment_4d"
    inputs: dict = field(default_factory=lambda: {"input": {"shape": [1, 3, 4, 5], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"output": {"shape": [1, 3, 4, 5], "dtype": np.float32}})

    def update_shapes(self, new_shape):
        for in_name in self.inputs:
            self.inputs[in_name]["shape"] = new_shape if isinstance(new_shape, str) else new_shape.copy()
        for out_name in self.outputs:
            self.outputs[out_name]["shape"] = new_shape if isinstance(new_shape, str) else new_shape.copy()


@dataclass
class EastFp32(ModelInfo):
    name: str = "east_fp32"
    inputs: dict = field(
        default_factory=lambda: {
            "input_images": {
                "shape": [1, 1024, 1920, 3],
                "dtype": np.float32,
                "dataset": EastDataset(),
                "layout": "NHWC:NCHW",
            }
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "feature_fusion/Conv_7/Sigmoid": {"shape": [1, 1, 256, 480], "dtype": np.float32, "layout": "NHWC:NCHW"},
            "feature_fusion/concat_3": {"shape": [1, 5, 256, 480], "dtype": np.float32, "layout": "NHWC:NCHW"},
        }
    )


@dataclass
class CrnnTf(ModelInfo):
    name: str = "crnn_tf"
    inputs: dict = field(default_factory=lambda: {"input": {"shape": [1, 3, 32, 100], "dtype": np.float32}})
    outputs: dict = field(
        default_factory=lambda: {"shadow/LSTMLayers/transpose_time_major": {"shape": [25, 1, 37], "dtype": np.float32}}
    )


@dataclass
class AgeGender(ModelInfo):
    name: str = "age-gender-recognition"
    xml_name: str = "age-gender-recognition-retail-0013"
    inputs: dict = field(default_factory=lambda: {"data": {"shape": [1, 3, 62, 62], "dtype": np.float32}})
    outputs: dict = field(
        default_factory=lambda: {
            "age_conv3": {"shape": [1, 1, 1, 1], "dtype": np.float32},
            "prob": {"shape": [1, 2, 1, 1], "dtype": np.float32},
        }
    )


@dataclass
class Emotion(ModelInfo):
    name: str = "emotions-recognition"
    inputs: dict = field(default_factory=lambda: {"data": {"shape": [1, 3, 64, 64], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"prob_emotion": {"shape": [1, 5, 1, 1], "dtype": np.float32}})


@dataclass
class FaceDetectionRetail(ModelInfo):
    name: str = "face-detection-retail-0004"
    inputs: dict = field(default_factory=lambda: {"data": {"shape": [1, 3, 300, 300], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"detection_out": {"shape": [1, 1, 200, 7], "dtype": np.float32}})


@dataclass
class LanguageModel(ModelInfo):
    is_language: bool = True

    def validate_outputs(self, outputs, expected_output_shapes=None, provided_input=None):
        if expected_output_shapes is None and provided_input is not None:
            expected_output_shapes = list(self.output_shapes.values())[0]
            expected_output_shapes[0] = len(list(provided_input.values())[0])
        return super().validate_outputs(outputs)

    def prepare_input_data(self, batch_size=None, random_data=False, input_key=None, dataset=None):
        return self.prepare_input_data_from_model_datasets(batch_size)


@dataclass
class Muse(LanguageModel):
    name: str = "muse_minimal"
    model_type: ModelType = ModelType.TFSM

    inputs: dict = field(
        default_factory=lambda: {
            "Func/StatefulPartitionedCall/input/_0": {"shape": [-1], "dtype": str, "dataset": LanguageModelDataset()}
        }
    )
    outputs: dict = field(
        default_factory=lambda: {"Func/StatefulPartitionedCall/output/_500": {"shape": [-1, 512], "dtype": str}}
    )

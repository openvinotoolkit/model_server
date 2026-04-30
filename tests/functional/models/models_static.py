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
from tests.functional.models.model_dataset import NumPyImageData
from tests.functional.constants.paths import Paths
from tests.functional.models.models import ModelInfo


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


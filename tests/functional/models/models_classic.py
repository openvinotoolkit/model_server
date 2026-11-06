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
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path

from tests.functional.models.models import ModelInfo
from tests.functional.models.models_datasets import NumPyImageData


@dataclass
class Resnet50Int8OvHf(ModelInfo):
    name: str = "OpenVINO/resnet50-int8-ov"
    is_local: bool = True
    batch_size: int = 1
    inputs: dict = field(
        default_factory=lambda: {
            "image": {
                "shape": [1, 224, 224, 3], "dtype": np.float32, "layout": "NHWC:NCHW", "dataset": NumPyImageData()
            }
        }
    )
    outputs: dict = field(
        default_factory=lambda: {"output": {"shape": [1, 1000], "dtype": np.float32}}
    )


classic_models_to_download = [Resnet50Int8OvHf]

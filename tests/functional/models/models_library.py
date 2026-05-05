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

from collections import defaultdict

import tests.functional.models.models_static
from tests.functional.utils.core import get_children_from_module
from tests.functional.config import language_models_enabled
from tests.functional.models.models_static import (
    Brain,
    DummySavedModel,
    InceptionResnetV2,
    InstanceSegmentationSecurity,
    Muse,
    Passthrough,
    Resnet,
    ResnetModelNameWithSlash,
    ResnetModelNameWithWhitespace,
    SsdliteMobilenetV2,
    UniversalSentenceEncoder,
)
from tests.functional.constants.target_device import TargetDevice
from tests.functional.models.models import ModelInfo


class ModelsLibrary:
    def __init__(self):
        self.kpi_models = []

    def create_model(self, name):
        children = get_children_from_module(ModelInfo, tests.functional.models.models_static)
        # we don't need 'name', we require only 'class_def' (without possible duplicates)
        children = list(set(map(lambda x: x[1], children)))
        result = list(filter(lambda x: x.name == name, children))
        assert result is not None and len(result) == 1, f"Expected single result, but got {result}"
        model = result[0]()  # instantiate
        return model

    def get_default_model(self, device=None):
        return self.various_models[device][0]

    @property
    def predict_models(self):
        result = defaultdict(list)
        for device, model_types in self.various_models.items():
            for model_type in model_types:
                model = model_type()
                got_dataset = model.inputs and all(
                    map(lambda x: x.get("dataset", None) is not None, model.inputs.values())
                )
                if got_dataset:
                    # Only models with proper dataset can be used in predict flow.
                    result[device].append(model_type)
        return result

    @property
    def predict_models_and_mediapipe(self):
        from tests.functional.constants.pipelines import SimpleModelMediaPipe
        result = self.predict_models
        for device, model_types in self.various_models.items():
            result[device].append(SimpleModelMediaPipe)
            if device in [TargetDevice.CPU, TargetDevice.GPU, TargetDevice.NPU, TargetDevice.AUTO]:
                result[device].append(Brain)
        return result

    @property
    def models_with_encoded_names(self):
        return [ResnetModelNameWithSlash, ResnetModelNameWithWhitespace]

    @property
    def various_models(self):
        if self.kpi_models:
            return defaultdict(lambda: self.kpi_models)

        return defaultdict(
            list,
            {
                TargetDevice.CPU: self.various_models_cpu,
                TargetDevice.GPU: self.various_models_gpu,
                TargetDevice.NPU: self.various_models_npu,
                TargetDevice.AUTO: self.various_models_auto,
                TargetDevice.HETERO: self.various_models_hetero,
            },
        )

    @property
    def language_models(self):
        return [UniversalSentenceEncoder, Passthrough, Muse]

    @property
    def various_models_cpu(self):
        various_models_cpu = [
            Resnet,
            SsdliteMobilenetV2,
            InceptionResnetV2,
            InstanceSegmentationSecurity,
        ]
        if language_models_enabled:
            various_models_cpu.extend([DummySavedModel, UniversalSentenceEncoder, Passthrough])
        return various_models_cpu

    @property
    def various_models_gpu(self):
        return [Resnet, SsdliteMobilenetV2, InceptionResnetV2]

    @property
    def various_models_npu(self):
        return [Resnet, SsdliteMobilenetV2, InceptionResnetV2]

    @property
    def various_models_auto(self):
        return [Resnet, InceptionResnetV2, SsdliteMobilenetV2]

    @property
    def various_models_hetero(self):
        return [Resnet, InceptionResnetV2, SsdliteMobilenetV2]


ModelsLib = ModelsLibrary()

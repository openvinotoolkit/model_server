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
from dataclasses import dataclass
from typing import Any, Callable, List

from dataclasses_json import dataclass_json

from tests.functional.utils.core import ComplexEncoder
from tests.functional.utils.logger import get_logger
from tests.functional.object_model.ovms_command import OvmsCommand
from tests.functional.config import logging_level_ovms
from tests.functional.constants.metrics import MetricsPolicy
from tests.functional.models import ModelInfo, Muse
from tests.functional.models.models_library import ModelsLib
from tests.functional.object_model.cpu_extension import MuseModelExtension
from tests.functional.object_model.custom_loader import CustomLoader

logger = get_logger(__name__)


@dataclass_json
@dataclass(frozen=False)
class OvmsParams(object):
    name: str = None
    grpc_port: int = None
    log_level: str = logging_level_ovms
    models: List[ModelInfo] = None
    model_name: str = None
    model_path: str = None
    nireq: int = None
    rest_port: int = None
    target_device: str = None
    check_version: bool = False
    use_config: bool = False
    custom_config: dict = None
    create_config_method: Callable[[str], None] = None
    image: str = None
    shape: Any = None
    is_stateful: bool = None
    sequence_cleaner_poll_wait_minutes: int = None
    max_sequence_number: int = None
    low_latency_transformation: bool = None
    idle_sequence_cleanup: bool = None
    model_version_policy: Any = None
    file_system_poll_wait_seconds: int = None
    cpu_extension: str = None
    custom_loaders: List[CustomLoader] = None
    rest_workers: int = None
    grpc_workers: int = None
    use_cache: bool = False
    cache_dir_path: str = None
    custom_command: OvmsCommand = None
    metrics_enable: MetricsPolicy = MetricsPolicy.NotDefined
    metrics_list: List[str] = None
    custom_graph_paths: List[str] = None
    use_custom_graphs: bool = False
    use_subconfig: bool = False
    single_mediapipe_model_mode: bool = False
    allowed_local_media_path: str = None
    allowed_media_domains: str = None
    pull: bool = None
    source_model: str = None
    gguf_filename: str = None
    model_repository_path: str = None
    task: str = None
    task_params: str = None
    list_models: bool = None
    overwrite_models: bool = None
    add_to_config: bool = False
    remove_from_config: bool = False
    resolution: str = None
    cache_size: int = None

    def __post_init__(self):
        if self.models is not None and any(isinstance(model, Muse) for model in self.models):
            self.cpu_extension = MuseModelExtension()

    def get_shape_param(self):
        result = self.shape
        if isinstance(self.shape, list):
            result = str(tuple(self.shape))
        return result

    def get_regular_models(self):
        result = []
        if self.list_models or self.add_to_config or self.remove_from_config:
            return result
        elif self.model_name is not None:
            result.append(ModelsLib.create_model(self.model_name))
        elif self.models is None:
            result.append(ModelsLib.get_default_model(self.target_device)())

        for model in self.models or []:
            result += model.get_regular_models()
        return result

    def get_layout_from_regular_models(self, regular_models_list=None):
        result = None
        if regular_models_list is None:
            regular_models_list = self.get_regular_models()
        for model in regular_models_list:
            for input_layout in model.input_layouts.values():
                if input_layout is not None:
                    result = input_layout
                    break
        return result

    def get_plugin_config_from_regular_models(self, regular_models_list=None):
        plugin_config = None
        if regular_models_list is None:
            regular_models_list = self.get_regular_models()
        for model in regular_models_list:
            plugin_config = model.plugin_config
            if plugin_config is not None:
                break
        return plugin_config

    def get_models(self):
        result = []
        if self.list_models or self.add_to_config or self.remove_from_config:
            return result
        if self.models is not None:
            result += self.models
        else:
            if self.model_name is not None:
                result.append(ModelsLib.create_model(self.model_name))
            elif self.models is None:
                result.append(ModelsLib.get_default_model(self.target_device)())
        return result

    def is_stateful_model_present(self):
        is_stateful = False
        if self.models is not None:
            is_stateful = any([x.is_stateful for x in self.models])
        else:
            if self.models:
                model = self.models[0]
                is_stateful = model.is_stateful
        return is_stateful

    def to_str(self):
        dict_params = self.to_dict()
        if "cpu_extension" in dict_params:
            dict_params["cpu_extension_path"] = self.cpu_extension_path
            del dict_params["cpu_extension"]

        if "custom_loader" in dict_params:
            del dict_params["custom_loader"]

        for model in dict_params.get("models", []):
            if "custom_loader" in model:
                # dict_params["custom_loader_path"] = self.cpu_extension_path
                del model["custom_loader"]

        serialized_params = json.dumps(dict_params, cls=ComplexEncoder)
        to_replace = {"null": "None", "false": "False", "true": "True"}
        for key, value in to_replace.items():
            serialized_params = serialized_params.replace(key, value)
        return serialized_params

    @classmethod
    def from_str(cls, params):
        deserialized = cls.from_json(params)
        return deserialized

    def ports_enabled(self):
        ports_enabled = not any([
            self.pull,
            self.list_models,
            self.overwrite_models,
            self.add_to_config,
            self.remove_from_config,
        ])
        return ports_enabled

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
import math
import os
import shutil
import stat
from dataclasses import dataclass
from distutils.dir_util import copy_tree, remove_tree
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from tests.functional.utils.logger import get_logger
from tests.functional.constants.os_type import OsType, get_host_os
from tests.functional.constants.target_device import TargetDevice
from tests.functional.config import (
    is_nginx_mtls,
    models_path,
    xdist_workers,
)
from tests.functional.constants.ovms import Ovms
from tests.functional.constants.paths import Paths
from tests.functional.object_model.custom_loader import CustomLoader
from tests.functional.object_model.ovms_mapping_config import OvmsMappingConfig
from tests.functional.object_model.shape import Shape
from tests.functional.object_model.test_environment import TestEnvironment

logger = get_logger(__name__)


class ModelType(str, Enum):
    IR = "IR"
    ONNX = "ONNX"
    PDPD = "PDPD"
    TFSM = "TFSM"  # tensorflow savedmodel


CLOUD_HEADERS = {"azure-blob": "az://", "azure-fs": "azfs://", "google": "gs://", "s3_minio": "s3://"}

DEVICE_LOADING_SPEED = {
    TargetDevice.CPU: {
        ModelType.IR: 97.4,
        ModelType.ONNX: 58.1,
        ModelType.PDPD: 97.4,
        ModelType.TFSM: 97.4,
    },  # For now TFSM works only with CPU.
    TargetDevice.GPU: {ModelType.IR: 10, ModelType.ONNX: 25, ModelType.PDPD: 10},
    TargetDevice.NPU: {ModelType.IR: 10, ModelType.ONNX: 25, ModelType.PDPD: 10},
    TargetDevice.AUTO: {ModelType.IR: 2, ModelType.ONNX: 5, ModelType.PDPD: 2},
    TargetDevice.HETERO: {ModelType.IR: 2, ModelType.ONNX: 5, ModelType.PDPD: 2},
    TargetDevice.AUTO_CPU_GPU: {ModelType.IR: 2, ModelType.ONNX: 5, ModelType.PDPD: 2},
}


@dataclass
class ModelInfo:
    name: str = None
    version: int = 1
    inputs: dict = None
    outputs: dict = None
    batch_size: int = None
    model_version_policy: str = None
    nireq: int = None
    plugin_config: object = None
    transpose_axes: str = None
    custom_loader: CustomLoader = None
    is_stateful: bool = False
    expected_batch_size: int = None
    max_sequence_number: int = None
    idle_sequence_cleanup: bool = None
    low_latency_transformation: bool = None
    base_path: str = None
    model_path_on_host = None
    model_type: ModelType = ModelType.IR
    input_shape_for_ovms: Any = None
    _compiled_layout: str = None
    _layout_for_ovms: str = None
    allow_cache: str = None
    use_mapping: bool = None
    target_device: str = None  # Currently used target device. Set up prior test in context fixture
    base_os: str = None
    is_mediapipe: bool = False
    is_language: bool = False
    use_relative_paths: bool = False
    use_subconfig: bool = False
    xml_name: str = None
    onnx_name: str = None
    is_llm: bool = False
    is_vision_language: bool = False
    is_hf_direct_load: bool = False  # model can be loaded directly from HuggingFace without conversion with optimum-cli
    gguf_filename: str = None
    is_local: bool = False
    model_subpath: str = None
    single_mediapipe_model_mode: bool = False
    tool_parser: str = None
    jinja_template: str = None
    tools_enabled: bool = False
    apply_gorilla_patch: bool = False
    gorilla_patch_name: str = None
    is_agentic: bool = False
    enable_tool_guided_generation: bool = False
    reasoning_parser: str = None
    pooling: str = None
    extra_quantization_params: str = None
    pipeline_type: str = None
    bfcl_num_threads: int = None
    max_num_batched_tokens: int = None
    is_audio: bool = False
    is_asr_model: bool = False
    is_tts_model: bool = False

    def __post_init__(self):
        if self.use_relative_paths:
            self.base_path = self.name
        else:
            if self.use_subconfig:
                self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, f"{self.name}_mediapipe", self.name)
            else:
                self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.name)
        self.model_path_on_host = os.path.join(models_path, self.name, str(self.version))
        self.set_additional_model_params()
        if get_host_os() == OsType.Windows:
            # enable model deletion https://jira.devtools.intel.com/browse/CVS-160412
            self.plugin_config = Ovms.PLUGIN_CONFIG_WINDOWS

    def set_additional_model_params(self):
        if not self.base_os:
            self.base_os = ModelInfo.base_os
        if not self.target_device:
            self.target_device = ModelInfo.target_device

    @staticmethod
    def get_list_of_config_fields():
        return [
            "name",
            "base_path",
            "batch_size",
            "model_version_policy",
            "nireq",
            "plugin_config",
            "stateful",
            "max_sequence_number",
            "idle_sequence_cleanup",
            "low_latency_transformation",
            "allow_cache",
        ]

    def get_model_file_path(self):
        filepath = None
        if self.model_type == ModelType.IR:
            ext = "xml"
            xml_name = self.xml_name if self.xml_name is not None else self.name
            filepath = f"{models_path}/{self.name}/{self.version}/{xml_name}.{ext}"
        elif self.model_type == ModelType.ONNX:
            ext = "onnx"
            onnx_name = self.onnx_name if self.onnx_name is not None else self.name
            filepath = f"{models_path}/{self.name}/{self.version}/{onnx_name}.{ext}"
        else:
            raise NotImplementedError(self.model_type)
        return filepath

    def get_bin_path(self):
        ext = "bin"
        bin_path = None
        if self.model_type == ModelType.IR:
            bin_path = f"{models_path}/{self.name}/{self.version}/{self.name}.{ext}"
        else:
            raise NotImplementedError(self.model_type)
        return bin_path

    def get_config(self):
        config = {}

        for field_name in self.get_list_of_config_fields():
            if field_name == "name":
                if self.is_mediapipe:
                    base_path = getattr(self, "base_path", None)
                    value = os.path.basename(base_path)
                else:
                    value = getattr(self, field_name, None)
            else:
                value = getattr(self, field_name, None)
            if value is not None:
                config[field_name] = value

        if self.target_device is not None:
            config["target_device"] = self.target_device

        if self.custom_loader is not None:
            config.update(self.custom_loader.model_options)

        config_shape = self._calculate_shape_for_config()
        if config_shape:
            config["shape"] = config_shape

        config_layout = self._calculate_layout_for_config()
        if config_layout:
            config["layout"] = config_layout

        if self.is_stateful:
            config["stateful"] = True

        return {"config": config}

    def set_input_shape_for_ovms(self, input_shape: Union[str, List, Dict[str, List]] = None):
        if input_shape is None:
            input_shape = self.input_shapes

        if isinstance(input_shape, dict):
            result = {}
            for input_name, shape in input_shape.items():
                if isinstance(shape, str):
                    result[input_name] = shape
                elif isinstance(shape, list) or isinstance(shape, tuple):
                    shape_dims_str = f"{','.join([str(shape_dim) for shape_dim in shape])}"
                    result[input_name] = f"({shape_dims_str})"
            self.input_shape_for_ovms = result
        elif isinstance(input_shape, list):
            self.input_shape_for_ovms = f"({','.join([str(shape_dim) for shape_dim in input_shape])})"
        else:
            self.input_shape_for_ovms = input_shape

        return self.input_shape_for_ovms

    def set_layout_for_ovms(self, layout: Union[str, dict]):
        if isinstance(layout, str):
            self._layout_for_ovms = layout
        elif isinstance(layout, dict):
            self._layout_for_ovms = json.dumps(layout)
        else:
            raise NotImplementedError()

        return self._layout_for_ovms

    def try_to_update_batch_size(self, shape):
        if shape is not None and isinstance(shape, list) and isinstance(shape[0], int):
            self.batch_size = shape[0]

    def _calculate_shape_for_config(self):
        return self.input_shape_for_ovms

    def _calculate_layout_for_config(self):
        layout_for_config = None
        if self._layout_for_ovms is not None:
            layout_for_config = self._layout_for_ovms
        else:
            layouts = []

            inputs_outputs = {}
            if self.inputs is not None:
                inputs_outputs.update(self.inputs)

            if self.outputs is not None:
                inputs_outputs.update(self.outputs)

            for input_name, input_info in inputs_outputs.items():
                if input_info is not None:
                    layout = input_info.get("layout", None)
                    if layout is not None:
                        layouts.append(f'"{input_name}": "{layout}"')

            if layouts:
                layout_for_config = ", ".join(layouts)
            else:
                layout_for_config = None

            if layout_for_config is not None:
                layout_for_config = f"{{{layout_for_config}}}"

        return layout_for_config

    def get_expected_output(self, input_data: dict, client_type: str = None):
        return None

    @staticmethod
    def is_pipeline():
        return False

    def get_model_path(self):
        return self.base_path

    def prepare_input_data(self, batch_size=None, random_data=False, input_key=None):
        result = dict()
        for input_name, input_data in self.inputs.items():
            if batch_size is not None:
                input_data["shape"][0] = batch_size
            if input_data["shape"] and input_data["shape"][0] == -1 and input_data.get("dataset", None):
                result[input_name] = input_data["dataset"].get_data(input_data["shape"], input_data["shape"][0], False)
            else:
                if random_data:
                    result[input_name] = np.random.uniform(-100.0, 100.0, input_data["shape"]).astype(
                        input_data["dtype"]
                    )
                else:
                    result[input_name] = np.ones(input_data["shape"], dtype=input_data["dtype"])
        return result

    def prepare_input_data_from_model_datasets(self, batch_size=None):
        result = dict()
        for param_name, param_data in self.inputs.items():
            if batch_size is None:
                batch_size = self.get_expected_batch_size() if self.batch_size is None else self.batch_size
            result[param_name] = param_data["dataset"].get_data(
                shape=param_data["shape"],
                batch_size=batch_size,
                transpose_axes=self.transpose_axes,
                datatype=param_data["dtype"],
            )
        return result

    def get_model_path_with_version(self, version=None):
        if version is None:
            version = self.version
        return os.path.join(self.get_model_path(), str(version))

    def get_model_files(self):
        model_dir = self.model_path_on_host
        all_files = os.listdir(model_dir)
        return all_files

    def get_expected_batch_size(self) -> int:
        expected_batch_size = None

        def get_batch_size_form_input_shape():
            model = self
            if self.inputs is None:
                model = self.clone()
            if not any([v["shape"] for v in model.inputs.values()]):
                return Ovms.SCALAR_BATCH_SIZE

            return [v["shape"] for v in model.inputs.values()][0][0]

        if self.input_shape_for_ovms is not None or self.batch_size is None or self.batch_size == "auto":
            expected_batch_size = get_batch_size_form_input_shape()
        else:
            expected_batch_size = self.batch_size

        try:
            if expected_batch_size == Ovms.SCALAR_BATCH_SIZE:
                return expected_batch_size

            if isinstance(expected_batch_size, str) and ":" in expected_batch_size:
                expected_batch_size = expected_batch_size.split(":")[0]

            expected_batch_size = int(expected_batch_size)
        except (TypeError, ValueError) as e:
            raise e.__class__(f"Calculated expected_batch_size: `{expected_batch_size}` is not a number: {e}")

        return expected_batch_size

    def clone(self, clone_model_name=None, model_path_on_host=None):
        clone = type(self)()
        if clone_model_name is not None:
            clone.name = clone_model_name
            clone.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, clone_model_name)
        if model_path_on_host is not None:
            clone.model_path_on_host = model_path_on_host

        copy_tree(self.model_path_on_host, clone.model_path_on_host)
        return clone

    def create_new_version(self, container_folder, new_version, copy_from_host_path=False, model_name=None):
        model_name = model_name if model_name is not None else self.name
        result = type(self)()

        if copy_from_host_path:
            source = self.model_path_on_host
        else:
            source = Path(container_folder, Paths.MODELS_PATH_NAME, model_name, str(self.version))

        destination = Path(container_folder, Paths.MODELS_PATH_NAME, model_name, str(new_version))
        if source != destination:
            # This check is for negative tests from TestOnlineModification
            shutil.copytree(source, destination, dirs_exist_ok=True)
            for file in destination.glob("*"):
                # resource files from shared folder should be read only.
                # Add proper access for test container folder manipulations.
                file.chmod(file.stat().st_mode | stat.S_IWRITE)

        result.version = new_version

        # Copy inputs/outputs (be aware that inputs/outputs can be mapped)
        result.inputs = self.inputs
        result.outputs = self.outputs

        if result.is_mediapipe:
            for model in result.regular_models:
                if model_name == model.name:
                    model.version = new_version

        return result

    def delete(self, container_folder, model_name=None):
        model_name = model_name if model_name is not None else self.name
        remove_tree(os.path.join(container_folder, Paths.MODELS_PATH_NAME, model_name))

    def delete_version(self, container_folder):
        remove_tree(os.path.join(container_folder, Paths.MODELS_PATH_NAME, self.name, str(self.version)))

    def restore_input_names(self):
        model = self.clone()
        self.inputs = {}
        for input_name in model.inputs:
            self.inputs[input_name] = None

    def change_input_name(self, old_name, new_name):
        tmp = self.inputs.pop(old_name)
        self.inputs[new_name] = tmp

    def change_output_name(self, old_name, new_name):
        tmp = self.outputs.pop(old_name)
        self.outputs[new_name] = tmp

    def validate_outputs(self, outputs, expected_output_shapes=None, provided_input=None):
        assert outputs, "Prediction returned no output"
        if expected_output_shapes is None:
            expected_output_shapes = list(self.output_shapes.values())
        for i, shape in enumerate(expected_output_shapes):  # Check for dynamic shape
            for j, val in enumerate(shape):
                if val == -1:
                    expected_output_shapes[i][j] = 1

        for output_name in self.output_names:
            assert (
                    output_name in outputs
            ), f"Incorrect output name, expected: {output_name}, found: {', '.join(outputs.keys())}"
            output_shapes = [list(o.shape) for o in outputs.values()]
            assert any(
                shape in expected_output_shapes for shape in output_shapes
            ), f"Incorrect output shape, expected: {expected_output_shapes}, found: {output_shapes}."

    def get_ovms_loading_time(self):
        loading_file_speed = 0.60 if self.is_on_cloud else 300.0
        model_loading_speed = DEVICE_LOADING_SPEED[self.target_device][self.model_type]
        if self.custom_loader:
            model_loading_speed = model_loading_speed * 0.10
        size = self.size
        if size > 0.0:
            size_in_mb = math.ceil(size / (1024.0 * 1024.0))
        else:
            size_in_mb = 100  # Model files not provided yet assume 100MB model

        timeout = size_in_mb * (1 / loading_file_speed + 1 / model_loading_speed)
        timeout += 60 if self.is_on_cloud else 10  # required for very small models on cloud
        timeout += 10 if self.custom_loader else 0  # required for additional overhead of custom loader invocation
        timeout += 10 if is_nginx_mtls else 0  # required for additional overhead of nginx invocation

        timeout += timeout * 0.2 * xdist_workers

        return timeout

    def prepare_resources(self, base_location):
        result = []
        resource_destination = (
            os.path.join(base_location, Paths.MODELS_PATH_NAME)
            if not self.use_subconfig
            else os.path.join(base_location, Paths.MODELS_PATH_NAME, self.name)
        )
        if not self.is_on_cloud and self.model_path_on_host is not None:
            src_model_path = Path(self.model_path_on_host)
            # model_name/version_num
            model_subpath = src_model_path.parts[-2:] if self.model_subpath is None else Path(self.model_subpath).parts
            target_model_dir = Path(resource_destination, *model_subpath)
            if not os.path.exists(target_model_dir):
                logger.debug(f"Copying {self.name} to container: {target_model_dir}")
                shutil.copytree(src_model_path, target_model_dir, dirs_exist_ok=True)
                for file in target_model_dir.glob("*"):
                    # resource files from shared folder should be read only.
                    # Add proper access for test container folder manipulations.
                    file.chmod(file.stat().st_mode | stat.S_IWRITE)
            result = [resource_destination]
        else:
            if CLOUD_HEADERS["google"] in self.base_path:
                # WA for GoogleCloud credential folder
                result = [resource_destination, os.path.join(TestEnvironment.current.base_dir, "credentials")]

        if self.custom_loader:
            assert not self.is_on_cloud, "Test framework not ready for models on cloud with custom_loader!"
            if self.custom_loader.prepare_custom_loader_resources:
                result.append(self.custom_loader.prepare_resources(base_location))
        return result

    def get_input_shape(self, input_name):
        return Shape(self.inputs[input_name]["shape"], self._compiled_layout)

    def set_shape_for_input(self, input_name, shape):
        _layout = self._compiled_layout.split(":")[0] if self._compiled_layout else None
        self.inputs[input_name]["shape"] = shape.get_shape_by_layout(_layout)

    def change_input_layout(self, new_layout):
        new_layout = new_layout.split(":")[0] if ":" in new_layout else new_layout
        for input_name, val in self.inputs.items():
            s = val["shape"]
            new_shape = [s[0], s[2], s[3], s[1]]
            val["shape"] = new_shape

    def change_input_type(self, input, type):
        self.inputs[input]["dtype"] = type

    def get_regular_models(self):
        return [self]

    def get_demultiply_count(self):
        return None

    def get_mapping_config_path(self, container_folder):
        return OvmsMappingConfig.mapping_config_path(container_folder, self)

    def get_mapping_dict(self, container_folder):
        mapping_config_path = self.get_mapping_config_path(container_folder)
        return OvmsMappingConfig.load_config(mapping_config_path)

    @property
    def is_on_cloud(self):
        result = False
        for header in CLOUD_HEADERS.values():
            if header in self.base_path:
                result = True
                break
        return result

    @property
    def size(self):
        if self.model_path_on_host is None or not os.path.exists(self.model_path_on_host):
            return 0.0
        file_list = os.listdir(self.model_path_on_host)
        file_ext = ".bin" if self.model_type == ModelType.IR else ".onnx"
        detected = [x for x in file_list if file_ext in x]
        result = 1
        if len(detected) > 0:
            result = Path(self.model_path_on_host, detected[0]).stat().st_size

        return result

    @property
    def input_names(self):
        return list(self.inputs.keys())

    @property
    def output_names(self):
        return list(self.outputs.keys())

    @property
    def input_shapes(self):
        return {k: v["shape"] for k, v in self.inputs.items()}

    @input_shapes.setter
    def input_shapes(self, shape):
        for k, v in self.inputs.items():
            v["shape"] = shape

    @property
    def input_layouts(self):
        return {k: v.get("layout", None) for k, v in self.inputs.items()} if self.inputs else {}

    @input_layouts.setter
    def input_layouts(self, layout):
        for k, v in self.inputs.items():
            v["layout"] = layout

    @property
    def output_shapes(self):
        return {k: v["shape"] for k, v in self.outputs.items()}

    @property
    def input_types(self):
        return {k: v["dtype"] for k, v in self.inputs.items()}

    @property
    def output_types(self):
        return {k: v["dtype"] for k, v in self.outputs.items()}

    @property
    def input_datasets(self):
        return {k: v["dataset"] if "dataset" in v else None for k, v in self.inputs.items()}

    def is_dynamic(self):
        return False

    @staticmethod
    def rename_input_ouput_data(data, src_name, dst_name):
        data[dst_name] = data[src_name]
        del data[src_name]
        return data


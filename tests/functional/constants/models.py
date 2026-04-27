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

import dataclasses
import inspect
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from distutils.dir_util import copy_tree, remove_tree
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import keras
import numpy as np
from bs4 import BeautifulSoup

from tests.functional.utils.assertions import ModelAnalyzerException
from tests.functional.utils.logger import get_logger
from common_libs.numpy_loader import is_dynamic_shape
from tests.functional.constants.os_type import OsType, get_host_os
from tests.functional.utils.process import Process
from ovms.config import (
    is_nginx_mtls,
    llm_models_local_path,
    models_path,
    ov_share_llm_models_local_path,
    use_legacy_models,
    xdist_workers,
    extra_llm_models_accuracy,
    extra_llm_models_test,
    ovms_c_repo_branch,
)
from ovms.constants.model_dataset import (
    ArchiveDataset,
    AssistantLargeLanguageModelDataset,
    BertSmallIn8Dataset,
    BinaryDummyModelDataset,
    BrainDataset,
    CocoDataset,
    DummyDataset,
    EastDataset,
    FaceDetectionDataset,
    FeatureExtractionModelDataset,
    FeatureExtractionZhModelDataset,
    InceptionResnetV2Dataset,
    LanguageModelDataset,
    LargeLanguageModelDataset,
    MatmulDataset,
    ModelDataset,
    NumPyImageData,
    OcrNetHrNetW48PaddleDataset,
    RandomDataset,
    RerankModelDataset,
    SingleMessageLanguageModelDataset,
    SmallCocoDataset,
    Unet3dDataset,
    VisionLanguageModelDataset,
    ImageGenerationDataset,
    AudioModelDataset,
)
from ovms.constants.ov import OV
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.ovms import Ovms
from tests.functional.constants.paths import Paths
from ovms.object_model.custom_loader import CustomLoader
from ovms.object_model.ovms_mapping_config import OvmsMappingConfig
from ovms.object_model.shape import Shape
from tests.functional.object_model.test_environment import TestEnvironment
from ovms.remote_test_environment import copy_container_folder_to_remote_k8s_cluster, delete_remote_container_folder

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


def get_model_class_by_name(class_name):
    if len(extra_acc_models) > 0:
        for model_class in extra_acc_models:
            if model_class.__name__ == class_name:
                return model_class
    else:
        models_members = inspect.getmembers(sys.modules['ovms.constants.models'])
        for name, obj in models_members:
            if name == class_name:
                return obj
    return None


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

    # REMOTE_SERVER_ADDRESS is not pre-defined, but set "on the fly" in our tests (prepare_remote_k8s_cluster_data),
    # based on the remote_ip read from the loaded kubeconfig_file (get_ip_from_kubeconfig_file).

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

        # if remote ip -> copy new version to remote machine also
        if os.environ.get("REMOTE_SERVER_ADDRESS") is not None:
            copy_container_folder_to_remote_k8s_cluster(container_folder, os.environ.get("REMOTE_SERVER_ADDRESS"))
        return result

    def delete(self, container_folder, model_name=None):
        model_name = model_name if model_name is not None else self.name
        remove_tree(os.path.join(container_folder, Paths.MODELS_PATH_NAME, model_name))
        if os.environ.get("REMOTE_SERVER_ADDRESS") is not None:
            delete_remote_container_folder(
                os.path.join(container_folder, Paths.MODELS_PATH_NAME, self.name),
                ip=os.environ.get("REMOTE_SERVER_ADDRESS"),
            )

    def delete_version(self, container_folder):
        remove_tree(os.path.join(container_folder, Paths.MODELS_PATH_NAME, self.name, str(self.version)))
        if os.environ.get("REMOTE_SERVER_ADDRESS") is not None:
            delete_remote_container_folder(
                os.path.join(container_folder, Paths.MODELS_PATH_NAME, self.name, str(self.version)),
                ip=os.environ.get("REMOTE_SERVER_ADDRESS"),
            )

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


@dataclass
class KpiModelInfo(ModelInfo):
    def __post_init__(self):
        for key, value in self.kpi_details.items():
            setattr(self, key, value)
        self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.name)
        self.model_path_on_host = os.path.join(models_path, self.name, str(self.version))

    def prepare_resources(self, base_location):
        # dynamically generated KPI models got some limitations...
        assert self.use_mapping is None
        assert self.custom_loader is None
        assert self.version == 1

        resource_destination = os.path.join(base_location, Paths.MODELS_PATH_NAME)
        target_model_dir = Path(resource_destination, self.name, str(self.version))

        for src_file in Path(self.model_path_on_host).glob("*"):
            if src_file.name.endswith(".json"):
                continue  # Skip additional files for now

            dst_file = Path(target_model_dir, src_file.name)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dst_file)

        return [resource_destination]


@dataclass
class StatefulModel(ModelInfo):
    context_window_left: int = 0
    context_window_right: int = 0
    is_stateful: bool = True

    def prepare_input_data(self, batch_size=None, random_data=False, input_key=None):
        return self.prepare_input_data_from_model_datasets(batch_size=batch_size)

    def calculate_error(self, data_list):
        result_list = []
        for data in data_list:
            result = {}
            for param_name, param_data in self.outputs.items():
                ark_data = param_data["dataset"].get_reference_data()
                for utterance_name, data_collection in ark_data:
                    result[utterance_name] = {}
                    result[utterance_name][param_name] = 0
                    for iteration_id, test_data in enumerate(data_collection):
                        square = np.square(data[utterance_name][iteration_id][param_name] - test_data).mean(axis=None)
                        mean_error = math.sqrt(square)
                        result[utterance_name][param_name] += mean_error
                    result[utterance_name][param_name] = result[utterance_name][param_name] / len(data_collection)
            result_list.append(result)
        return result_list


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


# IR models
@dataclass
class RmLstm(StatefulModel):
    name: str = "rm_lstm4f"
    inputs: dict = field(
        default_factory=lambda: {
            "Parameter": {
                "shape": [1, 43],
                "dtype": np.float32,
                "dataset": ArchiveDataset("rm_lstm", "test_feat_1_10.ark"),
            }
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "affinetransform/Fused_Add_": {
                "shape": [1, 1494],
                "dtype": np.float32,
                "dataset": ArchiveDataset("rm_lstm", "test_score_1_10.ark"),
            }
        }
    )


@dataclass
class RmCnn4ASmbr(StatefulModel):
    name: str = "rm_cnn4a_smbr"
    inputs: dict = field(
        default_factory=lambda: {
            "Parameter": {"shape": [1, 1056], "dtype": np.float32, "dataset": BinaryDummyModelDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"affinetransform23:0": {"shape": [1, 1498], "dtype": np.float32}})


@dataclass
class TedliumDnn4Smbr(StatefulModel):
    name: str = "tedlium_dnn4_smbr"
    inputs: dict = field(
        default_factory=lambda: {
            "Parameter": {"shape": [1, 440], "dtype": np.float32, "dataset": BinaryDummyModelDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"affinetransform14:0": {"shape": [1, 3941], "dtype": np.float32}})


@dataclass
class Tedlium_Lstm4F(StatefulModel):
    name: str = "tedlium_lstm4f"
    inputs: dict = field(
        default_factory=lambda: {
            "Parameter": {"shape": [1, 43], "dtype": np.float32, "dataset": BinaryDummyModelDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"affinetransform:0": {"shape": [1, 3941], "dtype": np.float32}})


@dataclass
class WsjCnn4BSmbr(StatefulModel):
    name: str = "wsj_cnn4b_smbr"
    inputs: dict = field(
        default_factory=lambda: {
            "Parameter": {"shape": [1, 759], "dtype": np.float32, "dataset": BinaryDummyModelDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"affinetransform14:0": {"shape": [1, 3407], "dtype": np.float32}})


@dataclass
class WsjDnn5BSmbr(StatefulModel):
    name: str = "wsj_dnn5b_smbr"
    inputs: dict = field(
        default_factory=lambda: {
            "Parameter": {"shape": [1, 440], "dtype": np.float32, "dataset": BinaryDummyModelDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"affinetransform:0": {"shape": [1, 3425], "dtype": np.float32}})


@dataclass
class AspireTdnn(StatefulModel):
    name: str = "aspire_tdnn"
    context_window_left: int = 17
    context_window_right: int = 12
    inputs: dict = field(
        default_factory=lambda: {
            "input": {
                "shape": [1, 40],
                "dtype": np.float32,
                "dataset": ArchiveDataset("aspire_tdnn", "mini_feat_1_10.ark"),
            },
            "ivector": {
                "shape": [1, 100],
                "dtype": np.float32,
                "dataset": ArchiveDataset("aspire_tdnn", "mini_feat_1_10_ivector.ark"),
            },
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "Final_affine": {
                "shape": [1, 8629],
                "dtype": np.float32,
                "dataset": ArchiveDataset("aspire_tdnn", "aspire_tdnn_mini_feat_1_10_kaldi_score.ark"),
            }
        }
    )


@dataclass
class BasicLstm(StatefulModel):
    name: str = "basic_lstm"
    inputs: dict = field(
        default_factory=lambda: {
            "LSTM-Layer/lstm/transpose/placeholder_port_0": {
                "shape": [1, 1, 10],
                "dtype": np.float32,
                "dataset": ArchiveDataset("basic_lstm", "partitioned_input.ark"),
            }
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "Output-Layer/add": {
                "shape": [1, 12],
                "dtype": np.float32,
                "dataset": ArchiveDataset("basic_lstm", "output.ark"),
            }
        }
    )


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
class Resnet50Binary(ModelInfo):
    name: str = "resnet50-binary-0001"
    transpose_axes: str = "0321"
    inputs: dict = field(default_factory=lambda: {"0": {"shape": [1, 3, 224, 224], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"1463": {"shape": [1, 1000], "dtype": np.float32}})


@dataclass
class CustomReluResnet50Binary(Resnet50Binary):
    def prepare_resources(self, base_location):
        models_dir = super(CustomReluResnet50Binary, self).prepare_resources(base_location)
        self.prepare_xml_file(base_location)
        return models_dir

    def prepare_xml_file(self, container_folder):
        xml_file = Path(container_folder, "models", self.name, str(self.version), f"{self.name}.xml")
        xml_txt = xml_file.read_text().replace('type="ReLU"', 'type="CustomReLU"')
        xml_file.chmod(xml_file.stat().st_mode | stat.S_IWRITE)
        xml_file.write_text(xml_txt)


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


@dataclass
class PublicResnet(ResnetLegacy):
    # last upload date: 30 Nov 2022
    name: str = "resnet"
    inputs: dict = field(
        default_factory=lambda: {
            "map/TensorArrayStack/TensorArrayGatherV3:0": {
                "shape": [-1, -1, -1, -1],
                "dtype": np.float32,
                "dataset": NumPyImageData(),
            }
        }
    )
    outputs: dict = field(default_factory=lambda: {"softmax_tensor": {"shape": [-1, 1001], "dtype": np.float32}})

    def __post_init__(self):
        self.base_path = "gs://ovms-public-eu/resnet50-tf"


@dataclass
class Resnet50Int8(ModelInfo):
    name: str = "dl-resnet-50-tf-int8"
    inputs: dict = field(
        default_factory=lambda: {
            "input": {"shape": [1, 224, 224, 3], "dtype": np.float32, "layout": "NHWC", "dataset": NumPyImageData()}
        }
    )
    outputs: dict = field(
        default_factory=lambda: {"resnet_v1_50/SpatialSqueeze": {"shape": [1, 1000], "dtype": np.float32}}
    )


@dataclass
class Resnet50Fp32(ModelInfo):
    name: str = "resnet50-tf-fp32"
    transpose_axes: str = "0321"
    inputs: dict = field(
        default_factory=lambda: {"input": {"shape": [1, 3, 224, 224], "dtype": np.float32, "dataset": NumPyImageData()}}
    )
    outputs: dict = field(default_factory=lambda: {"predict": {"shape": [1, 1001], "dtype": np.float32}})


@dataclass
class BrainLegacy(ModelInfo):
    name: str = "brain-tumor-segmentation-0002-2"
    batch_size: int = 1  # workaround for lack of support in ovms, should be np.float16
    inputs: dict = field(
        default_factory=lambda: {"0": {"shape": [1, 4, 128, 128, 128], "dtype": np.float32, "dataset": BrainDataset()}}
    )
    outputs: dict = field(default_factory=lambda: {"304": {"shape": [1, 3, 128, 128, 128], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX
    onnx_name: str = "brain-tumor-segmentation-0002"


@dataclass
class BrainFP32(BrainLegacy):
    name: str = "brain-tumor-segmentation-0002_FP32"
    model_type: ModelType = ModelType.IR


@dataclass
class InceptionResnetV2Legacy(ModelInfo):
    name: str = "inception-resnet-v2-tf"  # workaround for lack of support in ovms, should be np.float16
    inputs: dict = field(
        default_factory=lambda: {
            "input": {"shape": [1, 3, 299, 299], "dtype": np.float32, "dataset": InceptionResnetV2Dataset()}
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "InceptionResnetV2/AuxLogits/Logits/BiasAdd/Add": {"shape": [1, 1001], "dtype": np.float32}
        }
    )


@dataclass
class InceptionResnetV2FP32(InceptionResnetV2Legacy):
    name: str = "inception-resnet-v2-tf_FP32"


@dataclass
class SsdliteMobilenetV2Legacy(ModelInfo):
    name: str = "ssdlite_mobilenet_v2_ov"  # workaround for lack of support in ovms, should be np.float16
    inputs: dict = field(
        default_factory=lambda: {
            "image_tensor": {"shape": [1, 3, 300, 300], "dtype": np.float32, "dataset": CocoDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"DetectionOutput": {"shape": [1, 1, 100, 7], "dtype": np.float32}})


@dataclass
class SsdliteMobilenetV2FP32(SsdliteMobilenetV2Legacy):
    name: str = "ssdlite_mobilenet_v2_FP32"
    transpose_axes: str = "0231"
    outputs: dict = field(default_factory=lambda: {"detection_boxes": {"shape": [1, 1, 100, 7], "dtype": np.float32}})

    def __post_init__(self):
        super().__post_init__()
        self.inputs["image_tensor"]["dtype"] = np.uint8


@dataclass
class FaceDetection(ModelInfo):
    name: str = "face-detection"
    inputs: dict = field(
        default_factory=lambda: {
            "data": {"shape": [1, 3, 300, 300], "dtype": np.float32, "dataset": FaceDetectionDataset()}
        }
    )
    outputs: dict = field(
        default_factory=lambda: {"detection_out/sink_port_0": {"shape": [1, 1, 200, 7], "dtype": np.float32}}
    )


@dataclass
class InstanceSegmentationSecurityLegacy(ModelInfo):
    name: str = "instance-segmentation-security-0010"
    inputs: dict = field(
        default_factory=lambda: {
            "im_data": {"shape": [1, 3, 800, 1344], "dtype": np.float32, "dataset": SmallCocoDataset.ImgData()},
            "im_info": {"shape": [1, 3], "dtype": np.float32, "dataset": SmallCocoDataset.ImgInfo()},
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "boxes": {"shape": [100, 4], "dtype": np.float32},
            "scores": {"shape": [100], "dtype": np.float32},
            "classes": {"shape": [100], "dtype": np.int32},
            "raw_masks": {"shape": [100, 81, 28, 28], "dtype": np.float32},
        }
    )


@dataclass
class InstanceSegmentationSecurityFP32(InstanceSegmentationSecurityLegacy):
    name: str = "instance-segmentation-security-1040_FP32"
    inputs: dict = field(
        default_factory=lambda: {
            "image": {"shape": [1, 3, 608, 608], "dtype": np.float32, "dataset": SmallCocoDataset.ImgData()}
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "boxes": {"shape": [100, 6], "dtype": np.float32},
            "labels": {"shape": [100], "dtype": np.int32},
            "masks": {"shape": [100, 28, 28], "dtype": np.float32},
        }
    )


if use_legacy_models:
    SsdliteMobilenetV2 = SsdliteMobilenetV2Legacy
    Resnet = ResnetLegacy
    InceptionResnetV2 = InceptionResnetV2Legacy
    Brain = BrainLegacy
else:
    SsdliteMobilenetV2 = SsdliteMobilenetV2FP32
    Resnet = ResnetFP32
    InceptionResnetV2 = InceptionResnetV2FP32
    Brain = BrainFP32

InstanceSegmentationSecurity = (
    InstanceSegmentationSecurityLegacy  # InstanceSegmentationSecurityFP32 needs to be corrected
)


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
class Unet3D(ModelInfo):
    name: str = "3d_unet"
    inputs: dict = field(
        default_factory=lambda: {
            "MRImages": {"shape": [1, 1, 144, 144, 144], "dtype": np.float32, "dataset": Unet3dDataset()}
        }
    )
    outputs: dict = field(
        default_factory=lambda: {"PredictionMask/Sigmoid": {"shape": [1, 1, 144, 144, 144], "dtype": np.float32}}
    )


@dataclass
class BertSmallInt8(ModelInfo):
    sentence_length = 384
    name: str = "bert-small-int8"
    inputs: dict = field(
        default_factory=lambda: {
            "attention_mask": {
                "shape": [1, BertSmallInt8.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.DataMask(),
            },
            "input_ids": {
                "shape": [1, BertSmallInt8.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.InputData(),
            },
            "position_ids": {
                "shape": [1, BertSmallInt8.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.PositionIds(),
            },
            "token_type_ids": {
                "shape": [1, BertSmallInt8.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.ContextMask(),
            },
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "output_e": {"shape": [1, BertSmallInt8.sentence_length], "dtype": np.float32},
            "output_s": {"shape": [1, BertSmallInt8.sentence_length], "dtype": np.float32},
        }
    )


@dataclass
class BertSmallFp32(BertSmallInt8):
    name: str = "bert-small-fp32"


@dataclass
class BertLargeInt8(ModelInfo):
    sentence_length = 384
    name: str = "bert-large-int8"
    inputs: dict = field(
        default_factory=lambda: {
            "result.1": {
                "shape": [1, BertLargeInt8.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.InputData(),
            },
            "result.2": {
                "shape": [1, BertLargeInt8.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.DataMask(),
            },
            "result.3": {
                "shape": [1, BertLargeInt8.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.ContextMask(),
            },
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "5211": {"shape": [1, BertLargeInt8.sentence_length], "dtype": np.float32},
            "5212": {"shape": [1, BertLargeInt8.sentence_length], "dtype": np.float32},
        }
    )


@dataclass
class BertLargeFp32(ModelInfo):
    sentence_length = 384
    name: str = "bert-large-fp32"
    inputs: dict = field(
        default_factory=lambda: {
            "0": {
                "shape": [1, BertLargeFp32.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.InputData(),
            },
            "1": {
                "shape": [1, BertLargeFp32.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.DataMask(),
            },
            "2": {
                "shape": [1, BertLargeFp32.sentence_length],
                "dtype": np.int32,
                "dataset": BertSmallIn8Dataset.ContextMask(),
            },
        }
    )
    outputs: dict = field(
        default_factory=lambda: {
            "3171": {"shape": [1, BertLargeFp32.sentence_length], "dtype": np.float32},
            "3172": {"shape": [1, BertLargeFp32.sentence_length], "dtype": np.float32},
        }
    )


# ONNX models
@dataclass
class Alexnet(ModelInfo):
    name: str = "onnx_alexnet"
    transpose_axes: str = "0321"
    inputs: dict = field(
        default_factory=lambda: {
            "data_0": {"shape": [1, 3, 224, 224], "dtype": np.float32, "dataset": NumPyImageData()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"prob_1": {"shape": [1, 1000], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class Caffenet(ModelInfo):
    name: str = "onnx_caffenet"
    inputs: dict = field(default_factory=lambda: {"data_0": {"shape": [1, 3, 224, 224], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"prob_1": {"shape": [1, 1000], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class Densenet121(ModelInfo):
    name: str = "onnx_densenet121"
    inputs: dict = field(default_factory=lambda: {"data_0": {"shape": [1, 3, 224, 224], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"fc6_1": {"shape": [1, 1000, 1, 1], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class EfficientLite4(ModelInfo):
    name: str = "onnx_efficientnet-lite4"
    inputs: dict = field(default_factory=lambda: {"images:0": {"shape": [1, 224, 224, 3], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"Softmax:0": {"shape": [1, 1000], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class GoogleNet(ModelInfo):
    name: str = "onnx_googlenet"
    transpose_axes: str = "0321"
    inputs: dict = field(
        default_factory=lambda: {
            "data_0": {"shape": [1, 3, 224, 224], "dtype": np.float32, "dataset": NumPyImageData()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"prob_1": {"shape": [1, 1000], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class Mnist(ModelInfo):
    name: str = "onnx_mnist"
    inputs: dict = field(default_factory=lambda: {"Input73": {"shape": [1, 1, 28, 28], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"Plus422_Output_0": {"shape": [1, 10], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class RcnnIlsvrc13(ModelInfo):
    name: str = "onnx_rcnn-ilsvrc13"
    inputs: dict = field(default_factory=lambda: {"data_0": {"shape": [1, 3, 224, 224], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"fc-rcnn_1": {"shape": [1, 200], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class Resnet50(ModelInfo):
    name: str = "onnx_resnet50"
    transpose_axes: str = "0321"
    inputs: dict = field(
        default_factory=lambda: {"data": {"shape": [1, 3, 224, 224], "dtype": np.float32, "dataset": NumPyImageData()}}
    )
    outputs: dict = field(default_factory=lambda: {"resnetv17_dense0_fwd": {"shape": [1, 1000], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class Vgg19(ModelInfo):
    name: str = "onnx_vgg19"
    transpose_axes: str = "0321"
    inputs: dict = field(default_factory=lambda: {"data": {"shape": [1, 3, 224, 224], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"vgg0_dense2_fwd": {"shape": [1, 1000], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class Matmul(ModelInfo):
    name: str = "onnx_matmul"
    inputs: dict = field(
        default_factory=lambda: {
            "a": {"shape": [1, 20, 20], "dtype": np.float32, "dataset": MatmulDataset()},
            "b": {"shape": [1, 20, 20], "dtype": np.float32, "dataset": MatmulDataset()},
        }
    )
    outputs: dict = field(default_factory=lambda: {"c": {"shape": [1, 20, 20], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


@dataclass
class UnsupportedModel(ModelInfo):
    name: str = "onnx_unsupported_model"
    inputs: dict = field(
        default_factory=lambda: {
            "input": {"shape": [1, 2, 3], "dtype": np.int16},
            "repeats": {"shape": [3], "dtype": np.int64},
        }
    )
    outputs: dict = field(default_factory=lambda: {"output": {"shape": [1, 2, 3], "dtype": np.float32}})
    model_type: ModelType = ModelType.ONNX


# Negative test cases
@dataclass
class InvalidContents(ModelInfo):
    name: str = "invalid_contents_model"


@dataclass
class WrongXmlEmpty(ModelInfo):
    name: str = "wrong_xml_empty"


@dataclass
class WrongXmlSubdirectory(ModelInfo):
    name: str = "wrong_xml_subdirectory"


@dataclass
class WrongXmlWithout(ModelInfo):
    name: str = "wrong_xml_without"


@dataclass
class NotExistingModelPath(Resnet):
    name: str = "not_existing_model"

    def __post_init__(self):
        super().__post_init__()
        self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.name)

    def prepare_resources(self, base_location):
        return [os.path.join(base_location, Paths.MODELS_PATH_NAME)]


@dataclass
class NoModel(Resnet):
    name: str = "no_model"


@dataclass
class NoAcceptableModel(Resnet):
    name: str = "not_acceptable_model"


@dataclass
class Yamnet(Resnet):
    name: str = "yamnet"


@dataclass
class WrongExtensionModel(Resnet):
    name: str = "wrong_extension_model"


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
class Yolo3Fp32(ModelInfo):
    name: str = "yolo-v3-tf-fp32"
    inputs: dict = field(default_factory=lambda: {"input_1": {"shape": [1, 3, 416, 416], "dtype": np.float32}})
    outputs: dict = field(
        default_factory=lambda: {
            "conv2d_74/Conv2D/YoloRegion": {"shape": [1, 255, 52, 52], "dtype": np.float32},
            "conv2d_58/Conv2D/YoloRegion": {"shape": [1, 255, 13, 13], "dtype": np.float32},
            "conv2d_66/Conv2D/YoloRegion": {"shape": [1, 255, 26, 26], "dtype": np.float32},
        }
    )


@dataclass
class Yolo3TinyFp32(ModelInfo):
    name: str = "yolo-v3-tiny-tf-fp32"
    inputs: dict = field(default_factory=lambda: {"image_input": {"shape": [1, 3, 608, 608], "dtype": np.float32}})
    outputs: dict = field(
        default_factory=lambda: {
            "conv2d_93/BiasAdd/Add": {"dtype": np.float32, "shape": [1, 255, 76, 76]},
            "conv2d_109/BiasAdd/Add": {"dtype": np.float32, "shape": [1, 255, 19, 19]},
            "conv2d_101/BiasAdd/Add": {"dtype": np.float32, "shape": [1, 255, 38, 38]},
        }
    )


@dataclass
class Yolo4Fp32(ModelInfo):
    name: str = "yolo-v4-tf-fp32"
    inputs: dict = field(default_factory=lambda: {"image_input": {"dtype": np.float32, "shape": [1, 3, 608, 608]}})
    outputs: dict = field(
        default_factory=lambda: {
            "conv2d_109/BiasAdd/Add": {"dtype": np.float32, "shape": [1, 255, 19, 19]},
            "conv2d_101/BiasAdd/Add": {"dtype": np.float32, "shape": [1, 255, 38, 38]},
            "conv2d_93/BiasAdd/Add": {"dtype": np.float32, "shape": [1, 255, 76, 76]},
        }
    )


@dataclass
class LSpeechV10(ModelInfo):
    name: str = "lspeech-s5-ext-v10"
    inputs: dict = field(default_factory=lambda: {"Parameter": {"dtype": np.float32, "shape": [1, 143]}})
    outputs: dict = field(
        default_factory=lambda: {"affinetransform10/Fused_Add_": {"dtype": np.float32, "shape": [1, 3312]}}
    )


@dataclass
class MobileNet3Small(ModelInfo):
    name: str = "mobilenet-v3-small"
    inputs: dict = field(default_factory=lambda: {"input": {"dtype": np.float32, "shape": [1, 3, 224, 224]}})
    outputs: dict = field(
        default_factory=lambda: {"MobilenetV3/Predictions/Softmax": {"dtype": np.float32, "shape": [1, 1001]}}
    )


@dataclass
class MobileNet3Large(ModelInfo):
    name: str = "mobilenet-v3-large"
    inputs: dict = field(default_factory=lambda: {"input": {"dtype": np.float32, "shape": [1, 3, 224, 224]}})
    outputs: dict = field(
        default_factory=lambda: {"MobilenetV3/Predictions/Softmax": {"dtype": np.float32, "shape": [1, 1001]}}
    )


@dataclass
class SSDMobileNet1(ModelInfo):
    name: str = "ssd-mobilenet-v1"
    inputs: dict = field(default_factory=lambda: {"image_tensor": {"dtype": np.float32, "shape": [1, 3, 300, 300]}})
    outputs: dict = field(default_factory=lambda: {"DetectionOutput": {"dtype": np.float32, "shape": [1, 1, 100, 7]}})


@dataclass
class SSDMobileNet1Coco(ModelInfo):
    name: str = "ssd_mobilenet_v1_coco"
    inputs: dict = field(default_factory=lambda: {"image_tensor": {"shape": [1, 3, 300, 300], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"DetectionOutput": {"shape": [1, 1, 100, 7], "dtype": np.float32}})


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
class FaceDetectionRetail(ModelInfo):
    name: str = "face-detection-retail-0004"
    inputs: dict = field(default_factory=lambda: {"data": {"shape": [1, 3, 300, 300], "dtype": np.float32}})
    outputs: dict = field(default_factory=lambda: {"detection_out": {"shape": [1, 1, 200, 7], "dtype": np.float32}})


@dataclass
class PublicFaceDetection(ModelInfo):
    name: str = "face-detection"
    inputs: dict = field(
        default_factory=lambda: {
            "image": {"shape": [1, 3, 512, 512], "dtype": np.float32, "dataset": FaceDetectionDataset()}
        }
    )
    outputs: dict = field(
        default_factory=lambda: {"detection_out/sink_port_0": {"shape": [1, 1, 200, 7], "dtype": np.float32}}
    )


@dataclass
class ModelInfoPaddleNative(ModelInfo):
    model_type: ModelType = ModelType.PDPD

    def validate_outputs(self, output, expected_output_shapes=None, provided_input=None):
        if expected_output_shapes is None:
            input_shapes = [x.shape for x in self.input_datasets.values()]
            # expect single channel output with height and width one of original input shape
            expected_shapes = [[1, *x[2:]] for x in input_shapes]
        assert all(
            [is_dynamic_shape(s) for s in self.output_shapes.values()]
        ), "PaddlePaddle models are expected to have dynamic shape"
        super().validate_outputs(output, expected_shapes)

    def prepare_input_data(self, batch_size=None, random_data=False, input_key=None):
        return self.prepare_input_data_from_model_datasets(batch_size=batch_size)


@dataclass
class OcrNetHrNetW48Paddle(ModelInfo):
    name: str = "ocrnet-hrnet-w48-paddle"
    inputs: dict = field(
        default_factory=lambda: {
            "x": {"shape": [1, 3, 1024, 2048], "dtype": np.float32, "dataset": OcrNetHrNetW48PaddleDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"argmax_0.tmp_0": {"shape": [1, 1024, 2048], "dtype": np.int64}})


@dataclass
class OcrNetHrNetW48PaddleNative(ModelInfoPaddleNative):
    name: str = "ocrnet-hrnet-w48-paddle-native"
    inputs: dict = field(
        default_factory=lambda: {
            "x": {"shape": [-1, 3, -1, -1], "dtype": np.float32, "dataset": OcrNetHrNetW48PaddleDataset()}
        }
    )
    outputs: dict = field(default_factory=lambda: {"argmax_0.tmp_0": {"shape": [-1, -1, -1], "dtype": np.int64}})


### Models for external benchmarking ###
### INT8 ###
@dataclass
class DLBertSmallUncasedInt8(ModelInfo):
    name: str = "dl-bert-small-uncased-whole-word-masking-squad-int8-0002"


@dataclass
class DLDeeplabV3Int8(ModelInfo):
    name: str = "dl-deeplabv3-int8"


@dataclass
class DLDensenet121Int8(ModelInfo):
    name: str = "dl-densenet-121-int8"


@dataclass
class DLEfficientnetD0Int8(ModelInfo):
    name: str = "dl-efficientdet-d0-int8"


@dataclass
class DLGoogleNetV4Int8(ModelInfo):
    name: str = "dl-googlenet-v4-int8"


@dataclass
class DLMobileNetSSDInt8(ModelInfo):
    name: str = "dl-mobilenet-ssd-int8"


@dataclass
class DLMobileNetV2Int8(ModelInfo):
    name: str = "dl-mobilenet-v2-int8"


@dataclass
class DLResnet50Int8New(ModelInfo):  # from newer cache than DLResnet50Int8
    name: str = "dl-resnet-50-int8"


@dataclass
class DLResnet18Int8(ModelInfo):
    name: str = "dl-resnet-18-int8"


@dataclass
class DLSSDResnet34Int8(ModelInfo):
    name: str = "dl-ssd-resnet34-1200-int8"


@dataclass
class DLUnetCamvidInt8(ModelInfo):
    name: str = "dl-unet-camvid-onnx-0001-int8"


@dataclass
class DLYoloV3TinyInt8(ModelInfo):
    name: str = "dl-yolo_v3_tiny-int8"


@dataclass
class DLYolo4Int8(ModelInfo):
    name: str = "dl-yolo_v4-int8"


### FP32 ###
@dataclass
class DLBertSmallUncasedFp32(ModelInfo):
    name: str = "dl-bert-small-uncased-whole-word-masking-squad-0002"


@dataclass
class DLDeeplabV3Fp32(ModelInfo):
    name: str = "dl-deeplabv3-fp32"


@dataclass
class DLDensenet121Fp32(ModelInfo):
    name: str = "dl-densenet-121-fp32"


@dataclass
class DLEfficientnetD0Fp32(ModelInfo):
    name: str = "dl-efficientdet-d0-fp32"


@dataclass
class DLGoogleNetV4Fp32(ModelInfo):
    name: str = "dl-googlenet-v4-fp32"


@dataclass
class DLMobileNetSSDFp32(ModelInfo):
    name: str = "dl-mobilenet-ssd-fp32"


@dataclass
class DLMobileNetV2Fp32(ModelInfo):
    name: str = "dl-mobilenet-v2-fp32"


@dataclass
class DLResnet50Fp32(ModelInfo):
    name: str = "dl-resnet-50-fp32"


@dataclass
class DLResnet18Fp32(ModelInfo):
    name: str = "dl-resnet-18-fp32"


@dataclass
class DLSSDResnet34Fp32(ModelInfo):
    name: str = "dl-ssd-resnet34-1200-fp32"


@dataclass
class DLUnetCamvidFp32(ModelInfo):
    name: str = "dl-unet-camvid-onnx-0001-fp32"


@dataclass
class DLYoloV3TinyFp32(ModelInfo):
    name: str = "dl-yolo_v3_tiny-fp32"


@dataclass
class DLYolo4Fp32(ModelInfo):
    name: str = "dl-yolo_v4-fp32"


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


@dataclass
class GPTModel(LanguageModel):
    def __init__(self):
        raise NotImplementedError


@dataclass
class GPTMinimalModel(GPTModel):  # Please use minimal available GPT model.
    def __init__(self):
        raise NotImplementedError


@dataclass
class LargeLanguageModel(LanguageModel):
    model_type: ModelType = ModelType.IR
    is_llm: bool = True
    is_local: bool = True
    precision: str = "INT8"
    precision_dir: str = "INT8"
    parent_name: str = None
    parent_base_dir: str = os.path.join("pytorch", "ov")
    parent_precision_dir: str = "OV_FP16-INT8_ASYM"
    max_position_embeddings: int = None
    model_path_on_parent_host: str = None
    model_subpath: str = None
    single_message_dataset: bool = False
    allows_reasoning: bool = False
    is_feature_extraction: bool = False
    is_rerank: bool = False
    is_image_generation: bool = False
    is_audio: bool = False
    is_hf_direct_load: bool = False
    is_agentic: bool = False
    gguf_filename: str = None
    pooling: str = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._own_field_defaults = {}
        for name in getattr(cls, '__annotations__', {}):
            if name in cls.__dict__:
                cls._own_field_defaults[name] = cls.__dict__[name]

    def __post_init__(self):
        self.model_base_path_on_host = llm_models_local_path
        if self.parent_name is not None and self.parent_precision_dir is not None:
            parent_model_path = os.path.join(self.parent_name, self.parent_base_dir, self.parent_precision_dir)
            self.model_path_on_parent_host = os.path.join(ov_share_llm_models_local_path, parent_model_path)
        self.model_subpath = os.path.join(self.precision_dir, Path(self.name))
        self.model_path_on_host = os.path.join(self.model_base_path_on_host, self.model_subpath)
        self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.model_subpath)

    def get_default_dataset(self):
        if self.single_message_dataset:
            return SingleMessageLanguageModelDataset
        return LargeLanguageModelDataset

    def prepare_input_data(self, batch_size=None, input_key=None, dataset=None, input_data_type=None):
        if dataset is not None:
            input_data = {input_name: dataset().get_data(None, None, None) for input_name in self.input_names}
        elif input_data_type == "string":
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_string_data()
                for input_name in self.input_names
            }
        else:
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_data(None, None, None)
                for input_name in self.input_names
            }
        return input_data


@dataclass
class FacebookOpt125Int8(LargeLanguageModel):
    name: str = "facebook/opt-125m"
    single_message_dataset: bool = True


@dataclass
class TinyLlama11BChatV10Int4SymCw(LargeLanguageModel):
    name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"


@dataclass
class Llama27BChatHfInt8(LargeLanguageModel):
    name: str = "meta-llama/Llama-2-7b-chat-hf"
    parent_name: str = "llama-2-7b-chat-hf"


@dataclass
class Llama27BChatHfInt4(Llama27BChatHfInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Llama27BChatHfInt4SymCw(Llama27BChatHfInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class MetaLlama318BInt8(LargeLanguageModel):
    name: str = "meta-llama/Llama-3.1-8B"
    parent_name: str = "llama-3.1-8b"


@dataclass
class MetaLlama318BFp16(MetaLlama318BInt8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class MetaLlama318BInt4(MetaLlama318BInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class MetaLlama318BInt4SymCw(MetaLlama318BInt8):
    name: str = "meta-llama/Llama-3.1-8B"
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"
    single_message_dataset: bool = True


@dataclass
class MetaLlama318BInstructInt8(LargeLanguageModel):
    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    parent_name: str = "llama-3.1-8b-instruct"
    tool_parser: str = "llama3"
    jinja_template: str = ("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v0.9.0/examples/"
                           "tool_chat_template_llama3.1_json.jinja")
    enable_tool_guided_generation: bool = True
    gorilla_patch_name: str = "ovms-model"
    is_agentic: bool = True


@dataclass
class MetaLlama318BInstructFp16(MetaLlama318BInstructInt8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class MetaLlama318BInstructInt4(MetaLlama318BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class MetaLlama318BInstructInt4SymCw(MetaLlama318BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class MetaLlama323BInstructInt8(LargeLanguageModel):
    name: str = "meta-llama/Llama-3.2-3B-Instruct"
    tool_parser: str = "llama3"
    jinja_template: str = ("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v0.12.0/examples/"
                           "tool_chat_template_llama3.2_json.jinja")
    enable_tool_guided_generation: bool = True
    gorilla_patch_name: str = "ovms-model"
    is_agentic: bool = True


@dataclass
class MetaLlama323BInstructInt4(MetaLlama323BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class MetaLlama323BInstructInt4SymCw(MetaLlama323BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"


@dataclass
class Mistral7BInstructv03Int8(LargeLanguageModel):
    name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    parent_name: str = "mistral-7b-instruct-v0.3"
    single_message_dataset: bool = True
    tool_parser: str = "mistral"
    jinja_template: str = ("https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.10.1.1/examples/"
                           "tool_chat_template_mistral_parallel.jinja")
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"


@dataclass
class Mistral7BInstructv03Int4(Mistral7BInstructv03Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Mistral7BInstructv03Fp16(Mistral7BInstructv03Int8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class Mistral7BInstructv03Int4SymCw(Mistral7BInstructv03Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class DeepSeekR1DistillQwen15BInt8(LargeLanguageModel):
    name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    parent_name: str = "deepseek-r1-distill-qwen-1.5b"


@dataclass
class VisionLanguageModel(LargeLanguageModel):
    is_vision_language: bool = True

    @staticmethod
    def get_default_dataset():
        return VisionLanguageModelDataset


@dataclass
class Gemma34BItInt8(VisionLanguageModel):
    name: str = "google/gemma-3-4b-it"
    parent_name: str = "gemma-3-4b-it"


@dataclass
class Gemma34BItInt4(Gemma34BItInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Gemma34BItInt4SymCw(Gemma34BItInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class Llava157bHfInt8(VisionLanguageModel):
    name: str = "llava-hf/llava-1.5-7b-hf"
    parent_name: str = "llava-1.5-7b-hf"


@dataclass
class Llava157bHfInt4(Llava157bHfInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class InternVL21BInt8(VisionLanguageModel):
    # Model is used in model_server unit_tests, so it should be generated using export_models.py script
    name: str = "OpenGVLab/InternVL2-1B"


@dataclass
class InternVL21BInt4(InternVL21BInt8):
    parent_name: str = "internvl2-1b"
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class InternVL21BFp16(InternVL21BInt4):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class MiniCPMV26Int8(VisionLanguageModel):
    name: str = "openbmb/MiniCPM-V-2_6"
    parent_name: str = "minicpm-v-2_6"


@dataclass
class MiniCPMV26Int4(MiniCPMV26Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class MiniCPMV26Fp16(MiniCPMV26Int4):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class Qwen27BInt4SymCw(LargeLanguageModel):
    name: str = "Qwen/Qwen2-7B"
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"


@dataclass
class Qwen38BInt8(LargeLanguageModel):
    name: str = "Qwen/Qwen3-8B"
    parent_name: str = "qwen3-8b"
    tool_parser: str = "hermes3"
    reasoning_parser: str = "qwen3"
    allows_reasoning: bool = True
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"
    enable_tool_guided_generation: bool = True


@dataclass
class Qwen38BInt4(Qwen38BInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Qwen38BInt4SymCw(Qwen38BInt4):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class Qwen38BFp16(Qwen38BInt8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class Qwen3VL4BInstructInt8(VisionLanguageModel):
    name: str = "Qwen/Qwen3-VL-4B-Instruct"
    tool_parser: str = "hermes3"
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"
    enable_tool_guided_generation: bool = True


@dataclass
class Qwen3VL4BInstructInt4(Qwen3VL4BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class Qwen3VL32BInstructInt4(VisionLanguageModel):
    name: str = "Qwen/Qwen3-VL-32B-Instruct"
    precision: str = "INT4"
    precision_dir: str = "INT4"
    tool_parser: str = "hermes3"
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"
    enable_tool_guided_generation: bool = True


@dataclass
class Qwen257BInstruct1MInt8(LargeLanguageModel):
    name: str = "Qwen/Qwen2.5-7B-Instruct-1M"
    parent_name: str = "qwen2.5-7b-instruct-1m"


@dataclass
class Qwen257BInstruct1MFp16(Qwen257BInstruct1MInt8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class Qwen257BInstruct1MInt4(Qwen257BInstruct1MInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Qwen257BInstruct1MInt4SymCw(Qwen257BInstruct1MInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class Qwen257BInstructInt8(LargeLanguageModel):
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    parent_name: str = "qwen2.5-7b-instruct"


@dataclass
class Qwen257BInstructFp16(Qwen257BInstructInt8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class Qwen257BInstructInt4(Qwen257BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Qwen257BInstructInt4SymCw(Qwen257BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class Qwen2VL7BInstructInt8(VisionLanguageModel):
    name: str = "Qwen/Qwen2-VL-7B-Instruct"
    parent_name: str = "qwen2-vl-7b-instruct"


@dataclass
class Qwen2VL7BInstructInt4(Qwen2VL7BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Qwen2VL7BInstructFp16(Qwen2VL7BInstructInt8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class Qwen2VL7BInstructInt4SymCw(Qwen2VL7BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class Qwen25VL7BInstructInt8(VisionLanguageModel):
    name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    parent_name: str = "qwen2.5-vl-7b-instruct"


@dataclass
class Qwen25VL7BInstructInt4(Qwen25VL7BInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Phi35VisionInstructInt8(VisionLanguageModel):
    name: str = "microsoft/Phi-3.5-vision-instruct"
    parent_name: str = "phi-3.5-vision-instruct"


@dataclass
class Phi35VisionInstructInt4SymCw(Phi35VisionInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class Phi35VisionInstructInt4(Phi35VisionInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Phi35VisionInstructFp16(Phi35VisionInstructInt8):
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class Hermes3Llama318BInt8(LargeLanguageModel):
    # Model is used in model_server unit_tests, so it should be generated using export_models.py script
    name: str = "NousResearch/Hermes-3-Llama-3.1-8B"
    tool_parser: str = "hermes3"
    jinja_template: str = ("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v0.9.0/examples/"
                           "tool_chat_template_hermes.jinja")
    enable_tool_guided_generation: bool = True
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"


@dataclass
class Hermes3Llama318BInt4(Hermes3Llama318BInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class Hermes3Llama318BInt4SymCw(Hermes3Llama318BInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"


@dataclass
class FeatureExtractionModel(LargeLanguageModel):
    use_subconfig: bool = True
    is_feature_extraction: bool = True
    pooling: str = "CLS"

    @staticmethod
    def get_default_dataset():
        return FeatureExtractionModelDataset


@dataclass
class RerankModel(LargeLanguageModel):
    use_subconfig: bool = True
    is_rerank: bool = True

    @staticmethod
    def get_default_dataset():
        return RerankModelDataset


@dataclass
class ImageGenerationModel(LargeLanguageModel):
    use_subconfig: bool = True
    is_image_generation: bool = True

    @staticmethod
    def get_default_dataset():
        return ImageGenerationDataset


@dataclass
class AlibabaNLPGteLargeEnv15(FeatureExtractionModel):
    name: str = "Alibaba-NLP/gte-large-en-v1.5"
    extra_quantization_params: str = "--library sentence_transformers"


@dataclass
class BAAIBgeLargeEnv15(FeatureExtractionModel):
    name: str = "BAAI/bge-large-en-v1.5"


@dataclass
class BAAIBgeLargeZhv15(FeatureExtractionModel):
    name: str = "BAAI/bge-large-zh-v1.5"

    @staticmethod
    def get_default_dataset():
        return FeatureExtractionZhModelDataset


@dataclass
class NomicEmbedTextv15(FeatureExtractionModel):
    name: str = "nomic-ai/nomic-embed-text-v1.5"
    pooling: str = "MEAN"
    extra_quantization_params: str = "--library sentence_transformers"


@dataclass
class SentenceTransformersAllMpnetBaseV2(FeatureExtractionModel):
    name: str = "sentence-transformers/all-mpnet-base-v2"
    pooling: str = "MEAN"


@dataclass
class ThenlperGteSmall(FeatureExtractionModel):
    # Model is used in model_server unit_tests, so it should be generated using export_models.py script
    name: str = "thenlper/gte-small"


@dataclass
class Qwen3Embedding06B(FeatureExtractionModel):
    name: str = "Qwen/Qwen3-Embedding-0.6B"
    pooling: str = "LAST"


@dataclass
class SentenceTransformersAllMiniLML12V2(FeatureExtractionModel):
    name: str = "sentence-transformers/all-MiniLM-L12-v2"
    pooling: str = "MEAN"


@dataclass
class SentenceTransformersMultiQaMpnetBaseDotV1(FeatureExtractionModel):
    name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    pooling: str = "CLS"


@dataclass
class SentenceTransformersAllDistilrobertaV1(FeatureExtractionModel):
    name: str = "sentence-transformers/all-distilroberta-v1"
    pooling: str = "MEAN"


@dataclass
class MixedBreadAIDeepsetMxbaiEmbedLargeV1(FeatureExtractionModel):
    name: str = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    pooling: str = "MEAN"


@dataclass
class IntfloatMultilingualE5LargeInstruct(FeatureExtractionModel):
    name: str = "intfloat/multilingual-e5-large-instruct"
    pooling: str = "MEAN"


@dataclass
class IntfloatMultilingualE5Large(FeatureExtractionModel):
    name: str = "intfloat/multilingual-e5-large"
    pooling: str = "MEAN"


@dataclass
class BAAIRerankerLarge(RerankModel):
    name: str = "BAAI/bge-reranker-large"


@dataclass
class BAAIRerankerV2M3(RerankModel):
    name: str = "BAAI/bge-reranker-v2-m3"


@dataclass
class BAAIRerankerBase(RerankModel):
    # Model is used in model_server unit_tests, so it should be generated using export_models.py script
    name: str = "BAAI/bge-reranker-base"


@dataclass
class CrossEncoderMsmarcoMiniLML6EnDeV1(RerankModel):
    name: str = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"


@dataclass
class Qwen3Reranker06BSeqCls(RerankModel):
    name: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"


@dataclass
class StableDiffusion35LargeTurboInt8(ImageGenerationModel):
    name: str = "stabilityai/stable-diffusion-3.5-large-turbo"
    parent_name: str = "stable-diffusion-3.5-large-turbo"


@dataclass
class StableDiffusion35LargeTurboInt4(StableDiffusion35LargeTurboInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class StableDiffusion35LargeTurboInt4SymCw(StableDiffusion35LargeTurboInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"
    model_timeout: int = 1500


@dataclass
class StableDiffusionXlBase10Int8(ImageGenerationModel):
    name: str = "stabilityai/stable-diffusion-xl-base-1.0"
    parent_precision_dir: str = None
    model_timeout: int = 900


@dataclass
class StableDiffusionXlBase10Int4(StableDiffusionXlBase10Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class StableDiffusionXlBase10Int4SymCw(StableDiffusionXlBase10Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    model_timeout: int = 1500


@dataclass
class DreamlikeAnime10Int8(ImageGenerationModel):
    name: str = "dreamlike-art/dreamlike-anime-1.0"
    parent_precision_dir: str = None


@dataclass
class DreamlikeAnime10Int4(DreamlikeAnime10Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class DreamlikeAnime10Int4SymCw(DreamlikeAnime10Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    model_timeout: int = 900


@dataclass
class SavedModel(ModelInfo):
    model_type: ModelType = ModelType.TFSM


@dataclass
class DummySavedModel(SavedModel):
    name: str = "dummy_saved_model"
    inputs: dict = field(
        default_factory=lambda: {
            "in": {
                "shape": [-1, 10],
                "dtype": np.float32,
                "dataset": DummyDataset(),
            }
        }
    )
    outputs: dict = field(default_factory=lambda: {"out": {"shape": [-1, 10], "dtype": np.float32}})


@dataclass
class UniversalSentenceEncoder(Muse):
    name: str = "usem"
    inputs: dict = field(
        default_factory=lambda: {"inputs": {"shape": [-1], "dtype": str, "dataset": LanguageModelDataset()}}
    )
    outputs: dict = field(default_factory=lambda: {"outputs": {"shape": [-1, 512], "dtype": np.float32}})

    def is_dynamic(self):
        return True


@dataclass
class Passthrough(LanguageModel):
    name: str = "passthrough"
    inputs: dict = field(
        default_factory=lambda: {"input": {"shape": [-1, -1], "dtype": np.uint8, "dataset": LanguageModelDataset()}}
    )
    outputs: dict = field(default_factory=lambda: {"copy:0": {"shape": [-1, -1], "dtype": np.uint8}})

    def validate_outputs(self, outputs, expected_output_shapes=None, provided_input=None):
        assert outputs, "Prediction returned no output"
        if expected_output_shapes is None and provided_input is not None:
            expected_output_shapes = list(self.output_shapes.values())[0]
            expected_output_shapes[0] = len(list(provided_input.values())[0])
            for output_name in self.output_names:
                assert (
                        output_name in outputs
                ), f"Incorrect output name, expected: {output_name}, found: {', '.join(outputs.keys())}"
                output_shapes = [list(o.shape) for o in outputs.values()]
                assert all(
                    shape[1] >= len(list(provided_input.values())[0]) in expected_output_shapes
                    for shape in output_shapes
                ), f"Incorrect output shape, expected: {expected_output_shapes}, found: {output_shapes}."
        else:
            return super().validate_outputs(outputs)

    def is_dynamic(self):
        return True


@dataclass
class ScalarDummy(ModelInfo):
    name: str = "scalar"
    inputs: dict = field(default_factory=lambda: {"model_scalar_input": {"dtype": np.float32, "shape": []}})
    outputs: dict = field(default_factory=lambda: {"model_scalar_output": {"dtype": np.float32, "shape": []}})


# NOTE: it is not @dataclass  (sic!)
class DynamicallyGeneratedModel(ModelInfo):
    def __init__(self, base_os, name, original_model=None, version=None, inputs=None, outputs=None, base_path=None):
        self.base_os = base_os
        self.name = name
        self.model_path_on_host = None
        if original_model is not None:
            self._original_model = original_model
            self.version = original_model.version
            self.inputs = original_model.inputs
            self.outputs = original_model.outputs
            self.base_path = original_model.base_path
        else:
            self.inputs = inputs
            self.outputs = outputs
            self.base_path = base_path
            self.version = version

        assert self.inputs, "Please provide inputs or source model"
        assert self.version, "Please provide version or source model"
        assert self.outputs, "Please provide outputs or source model"
        assert self.base_path, "Please provide base_path or source model"

    def __post_init__(self):
        pass

    def get_str_dtype(self, dtype):
        return dtype.__name__

    def generate_and_convert_model(self, container_folder):
        tmp_dir = tempfile.mkdtemp(dir=container_folder, suffix=self.name)
        self.model_path_on_host = os.path.join(container_folder, Paths.MODELS_PATH_NAME, self.name, str(self.version))

        # Create a simple model.
        # https://www.tensorflow.org/api_docs/python/tf/keras/Input
        # The data type expected by the input, as a string (float32, float64, int32...)

        inputs = [
            keras.Input(
                batch_size=details["shape"][0],
                shape=details["shape"][1:],
                dtype=self.get_str_dtype(details["dtype"]),
                name=name,
            )
            for name, details in self.inputs.items()
        ]

        _output_name, _output_details = list(self.outputs.items())[0]
        # NOTE: Only single output model is supported yet
        outputs = keras.layers.Reshape(
            target_shape=_output_details["shape"][1:],
            dtype=self.get_str_dtype(_output_details["dtype"]),
            name=_output_name,
        )(inputs[0])

        model = keras.Model(name=self.name, inputs=inputs, outputs=outputs)
        model.compile(loss="mean_squared_error")
        logger.info(f"Model dtype: {model.dtype}")
        model.export(tmp_dir)  # https://www.tensorflow.org/api_docs/python/tf/keras/Model#export

        proc = Process()
        proc.policy["log-run"]["verbose"] = False
        proc.policy["log-async-run"]["verbose"] = False
        proc.policy["log-check-output"]["stderr"] = False

        activate_path = os.path.join(sys.exec_prefix, "Scripts", "activate.bat") \
            if self.base_os == OsType.Windows else os.path.join(sys.exec_prefix, "bin", "activate")
        activate_venv = activate_path if self.base_os == OsType.Windows else f". {activate_path}"

        # https://docs.openvino.ai/2025/openvino-workflow/model-preparation/convert-model-tensorflow.html
        # #converting-non-frozen-model-formats
        cmd = " ".join([
            activate_venv,
            "&&",
            "ovc",
            os.path.basename(tmp_dir),
            "--output_model",
            os.path.join(self.model_path_on_host, "dynamic_model"),
            "--input",
            ",".join([f"{name}[{','.join(map(str, details['shape']))}]" for name, details in self.inputs.items()]),
        ])
        r = proc.run_and_check(cmd, cwd=container_folder)


@dataclass
class KpiModelAnalyzer(ModelInfo):
    xml_path: str = None
    xml_data: BeautifulSoup = None
    model: KpiModelInfo = None
    inputs: dict = None
    outputs: dict = None
    batch_size: int = 1
    model_version: str = "1"
    model_type: ModelType = ModelType.IR

    @classmethod
    def generate_kpi_model(cls, model_type: ModelInfo):
        model = model_type()
        xml_path = model.get_model_file_path()
        xml_data = cls.get_xml_data(xml_path)
        inputs, outputs = cls.get_inputs_outputs_from_xml(xml_data)
        name = cls.get_model_name(xml_data)
        precision = cls.get_model_precision(xml_data)
        kpi_model_class_name = name.capitalize().replace("-", "_") + f"_{precision}"
        kpi_model = type(
            kpi_model_class_name, (KpiModelInfo,), {"kpi_details": {"name": name, "inputs": inputs, "outputs": outputs}}
        )()
        kpi_model.base_os = ModelInfo.base_os
        kpi_model.target_device = ModelInfo.target_device
        return kpi_model

    @staticmethod
    def get_model_name(xml_data: BeautifulSoup):
        conversion_parameters = xml_data.find_all("cli_parameters")
        model_name = conversion_parameters[0].model_name["value"]
        return model_name

    @staticmethod
    def get_xml_data(xml_path: str):
        bs_data = None
        if Path(xml_path).exists():
            with open(xml_path, "r") as f:
                f_data = f.read()
            bs_data = BeautifulSoup(f_data, "xml")
        else:
            raise FileNotFoundError(xml_path)

        return bs_data

    @staticmethod
    def get_list(value: str, delimiter: str = ","):
        value = value.strip("'[]'")
        value = value.replace("', '", ", ")
        if delimiter in value:
            values = value.split(delimiter)
        else:
            values = [value]
        values = [value.strip() for value in values]
        return values

    @staticmethod
    def get_model_precision(xml_data: BeautifulSoup):
        layers = xml_data.find_all("layer")
        precision = layers[0].find_all("port")[0]["precision"]
        return precision

    @classmethod
    def get_inputs_outputs_names(cls, xml_data: BeautifulSoup):
        """
        Finds input (<input>) and output (<output>) names in conversion parameters.
        Parameters:
            xml_data: BeautifulSoup XML content
        Returns:
            input_names: list of str
            output_names: list of str
        Example of conversion_parameters section:
            <conversion_parameters>
                <compress_to_fp16 value="False"/>
                <framework value="tf"/>
                <input value="map/TensorArrayStack/TensorArrayGatherV3"/>
                <input_model value="DIR/resnet_v1-50.pb"/>
                <input_shape value="[1,224,224,3]"/>
                <layout value="map/TensorArrayStack/TensorArrayGatherV3(nhwc)"/>
                <mean_values value="map/TensorArrayStack/TensorArrayGatherV3[123.68,116.78,103.94]"/>
                <model_name value="resnet-50-tf"/>
                <output value="softmax_tensor"/>
                <output_dir value="DIR"/>
                <reverse_input_channels value="True"/>
            </conversion_parameters>
        """

        conversion_parameters = xml_data.find_all("cli_parameters")
        input_tag = conversion_parameters[0].input
        if input_tag is None:
            raise ModelAnalyzerException(f"No <input> tag found in conversion_parameters: \n{conversion_parameters}")
        input_names = cls.get_list(input_tag.attrs["value"])

        output_tag = conversion_parameters[0].output
        if output_tag is None:
            raise ModelAnalyzerException(f"No <output> tag found in conversion_parameters: \n{conversion_parameters}")
        output_names = cls.get_list(output_tag.attrs["value"])

        return input_names, output_names

    @classmethod
    def find_inputs_outputs_in_layers(cls, xml_data: BeautifulSoup, input_names: list, output_names: list):
        """
        Finds inputs and outputs in layers.
        Parameters:
            xml_data: BeautifulSoup XML content
            input_names: list of str
            output_names: list of str
        Returns:
            inputs: dict
            outputs: dict
        Example of input layer:
            <layer id="0" name="map/TensorArrayStack/TensorArrayGatherV3" type="Parameter" version="opset1">
                <data shape="1,224,224,3" element_type="f32"/>
                <output>
                    <port id="0" precision="FP32"
                    names="map/TensorArrayStack/TensorArrayGatherV3,map/TensorArrayStack/TensorArrayGatherV3:0">
                        <dim>1</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                        <dim>3</dim>
                        <rt_info>
                            <attribute name="layout" version="0" layout="[N,H,W,C]"/>
                        </rt_info>
                    </port>
                </output>
            </layer>
        """
        inputs = {}
        outputs = {}
        layers = xml_data.find_all("layer")
        for layer in layers:
            if layer.get("name") in input_names:
                input_shape = [int(dim) for dim in layer.data["shape"].split(",")]
                dtype = cls.get_data_format(layer)
                inputs.update({layer.get("name"): {"shape": input_shape, "dtype": dtype, "dataset": None}})
            # If layer name is the same as output_name
            if layer.get("name") in output_names:
                output_shape = [int(dim.text) for dim in layer.output.find_all("dim")]
                dtype = cls.get_data_format(layer)
                outputs.update({layer.get("name"): {"shape": output_shape, "dtype": dtype, "dataset": None}})

            if len(inputs) == len(input_names) and len(outputs) == len(output_names):
                break

        # Get outputs: all layers with type='Result'
        # We get the name from output_names (with actual_name_index) in case of different layer name, e.g.:
        # output name: logits
        # layer output name: logits:0
        if len(outputs) != len(output_names):
            logger.info(f"Get outputs: all layers with type='Result'")
            result_layers = [layer for layer in layers if layer.get("type") == "Result"]
            logger.info(f"result layers: {result_layers}")
            for result in result_layers:
                output_shape = [int(dim.text) for dim in result.input.find_all("dim")]
                logger.info(f"output_shape: {output_shape}")
                dtype = cls.get_data_format(result)
                actual_name_index = [i for i, name in enumerate(output_names) if name in result.get("name")][0]
                outputs.update(
                    {output_names[actual_name_index]: {"shape": output_shape, "dtype": dtype, "dataset": None}}
                )
                # Example for Efficient model - 4 output names in layers[-2].output.port['names']
                # efficientnet-b0/model/head/dense/BiasAdd,efficientnet-b0/model/head/dense/BiasAdd:0,logits,logits:0
                # Expected from layer[-1]: logits
                # Actual in docker: efficientnet-b0/model/head/dense/BiasAdd

        assert len(inputs) != 0, "Empty inputs!"
        assert len(outputs) != 0, "Empty outputs!"
        return inputs, outputs

    @classmethod
    def get_data_format(cls, layer):
        precision = None
        for elem in layer.find_all("port"):
            precision = elem.get("precision", None)
        if precision is None:
            raise ModelAnalyzerException(f"Precision not found: {elem}")
        dtype = cls.get_np_type_from_str(precision)
        return dtype

    @staticmethod
    def get_np_type_from_str(datatype: str):
        if datatype == "FP32":
            return np.float32
        elif datatype == "I64":
            return np.int64
        elif datatype == "I32":
            return np.int32
        else:
            raise NotImplementedError(f"Not implemented datatype: {datatype}")

    @staticmethod
    def get_str_type_from_np(datatype):
        if datatype == np.float32:
            return "FP32"
        elif datatype == np.int64:
            return "I64"
        elif datatype == np.int32:
            return "I32"
        else:
            raise NotImplementedError(f"Not implemented datatype: {datatype}")

    @classmethod
    def get_inputs_outputs_from_xml(cls, xml_data: BeautifulSoup):
        input_names, output_names = cls.get_inputs_outputs_names(xml_data)
        inputs, outputs = cls.find_inputs_outputs_in_layers(xml_data, input_names, output_names)
        return inputs, outputs


@dataclass
class LargeLanguageModelHuggingFace(LargeLanguageModel):
    is_local: bool = False
    is_hf_direct_load: bool = True
    input_name: str = "input"
    precision: str = "INT4"
    model_timeout: int = 900

    def _apply_diamond_defaults(self):
        """Fix field defaults for diamond inheritance.

        When a class inherits from both LargeLanguageModelHuggingFace and a specialized
        type (e.g. ImageGenerationModel), LargeLanguageModelHuggingFace's inherited field
        defaults override the specialized type's directly-defined defaults. This method
        restores the correct defaults from specialized parent classes.
        """
        cls = type(self)
        seen_fields = set()
        seen_fields.update(getattr(cls, '_own_field_defaults', {}).keys())
        seen_fields.update(getattr(LargeLanguageModelHuggingFace, '_own_field_defaults', {}).keys())
        for base in cls.__mro__:
            if base in (cls, object, LargeLanguageModelHuggingFace, LargeLanguageModel, LanguageModel, ModelInfo):
                continue
            own_defaults = getattr(base, '_own_field_defaults', {})
            for field_name, default_value in own_defaults.items():
                if field_name not in seen_fields:
                    setattr(self, field_name, default_value)
                    seen_fields.add(field_name)

    def __post_init__(self):
        self._apply_diamond_defaults()
        if self.is_local:
            self.model_base_path_on_host = llm_models_local_path
            self.model_path_on_host = os.path.join(self.model_base_path_on_host, Path(self.name))
            self.model_subpath = os.path.join("models_ov_hf", Path(self.name))
        self.base_path = os.path.join(Paths.MODELS_PATH_INTERNAL, self.name)
        self.set_additional_model_params()

    def prepare_resources(self, base_location):
        models_dir = Path(base_location, Paths.MODELS_PATH_NAME)
        models_dir.mkdir(exist_ok=True, parents=True)
        if self.is_local:
            models_sub_dir = Path(models_dir, self.name)
            shutil.copytree(self.model_path_on_host, models_sub_dir)
        return [str(models_dir)]

    def prepare_input_data(self, batch_size=None, input_key=None, dataset=None, input_data_type=None):
        if dataset is not None:
            dataset_obj = dataset if not isinstance(dataset, type) else dataset()
        else:
            dataset_obj = self.get_default_dataset()()
        if input_data_type == "string":
            input_data = {self.input_name: dataset_obj.get_string_data()}
        else:
            input_data = {self.input_name: dataset_obj.get_data(None, None, None)}
        return input_data


@dataclass
class GptOss20BInt8OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/gpt-oss-20b-int8-ov"
    precision: str = "INT8"
    tool_parser: str = "gptoss"
    reasoning_parser: str = "gptoss"
    allows_reasoning: bool = True
    is_agentic: bool = True
    enable_tool_guided_generation: bool = True
    model_timeout: int = 900
    gorilla_patch_name: str = "ovms-model"
    bfcl_num_threads: int = 10
    jinja_template: str = (f"https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/"
                           f"{ovms_c_repo_branch}/extras/chat_template_examples/chat_template_gpt_oss.jinja")
    is_local: bool = True


@dataclass
class GptOss20BInt4OvHf(GptOss20BInt8OvHf):
    name: str = "OpenVINO/gpt-oss-20b-int4-ov"
    precision: str = "INT4"
    is_local: bool = True


@dataclass
class GptOss120BInt4(LargeLanguageModel):
    name: str = "openai/gpt-oss-120b"
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_name: str = "gpt-oss-120b"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"
    tool_parser: str = "gptoss"
    reasoning_parser: str = "gptoss"
    allows_reasoning: bool = True
    is_agentic: bool = True
    enable_tool_guided_generation: bool = True
    model_timeout: int = 900
    gorilla_patch_name: str = "ovms-model"
    bfcl_num_threads: int = 10
    jinja_template: str = (f"https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/"
                           f"{ovms_c_repo_branch}/extras/chat_template_examples/chat_template_gpt_oss.jinja")


@dataclass
class Qwen3Coder30BA3BInt8OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int8-ov"
    precision: str = "INT8"
    tool_parser: str = "qwen3coder"
    is_agentic: bool = True
    enable_tool_guided_generation: bool = True
    model_timeout: int = 900
    gorilla_patch_name: str = "ovms-model"
    jinja_template: str = (f"https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/"
                           f"{ovms_c_repo_branch}/extras/chat_template_examples/"
                           f"chat_template_qwen3coder_instruct.jinja")
    bfcl_num_threads: int = 10
    max_num_batched_tokens: int = 99999999
    is_local: bool = True


@dataclass
class Qwen3Coder30BA3BInt4OvHf(Qwen3Coder30BA3BInt8OvHf):
    name: str = "OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov"
    precision: str = "INT4"
    is_local: bool = True


@dataclass
class Qwen3VL8BInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "Junrui2021/Qwen3-VL-8B-Instruct-int4"
    tool_parser: str = "hermes3"
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"
    enable_tool_guided_generation: bool = True
    is_local: bool = True
    pipeline_type: str = "VLM"


@dataclass
class Gemma34bItInt4CwOvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/gemma-3-4b-it-int4-cw-ov"
    is_local: bool = True


@dataclass
class Gemma34bItInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/gemma-3-4b-it-int4-ov"
    is_local: bool = True


@dataclass
class Gemma34bItInt8OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/gemma-3-4b-it-int8-ov"
    is_local: bool = True


@dataclass
class TinyLlama11BChatV10Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Phi35MiniInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Phi-3.5-mini-instruct-int4-ov"


@dataclass
class Phi4MiniInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Phi-4-mini-instruct-int4-ov"


@dataclass
class Phi4ReasoningInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Phi-4-reasoning-int4-ov"


@dataclass
class DeepSeekR1DistillQwen15BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov"
    single_message_dataset: bool = True


@dataclass
class DeepSeekR1DistillQwen15BInt4CwOvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-cw-ov"
    single_message_dataset: bool = True


@dataclass
class Qwen38BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen3-8B-int4-ov"
    tool_parser: str = "hermes3"
    reasoning_parser: str = "qwen3"
    allows_reasoning: bool = True
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"
    enable_tool_guided_generation: bool = True
    is_local: bool = True


@dataclass
class Qwen38BInt4CwOvHf(Qwen38BInt4OvHf):
    name: str = "OpenVINO/Qwen3-8B-int4-cw-ov"


@dataclass
class Mistral7BInstructv03Int8OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Mistral-7B-Instruct-v0.3-int8-ov"
    precision: str = "INT8"
    tool_parser: str = "mistral"
    jinja_template: str = ("https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.10.1.1/examples/"
                           "tool_chat_template_mistral_parallel.jinja")
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"
    is_local: bool = True


@dataclass
class Mistral7BInstructv03Fp16OvHf(Mistral7BInstructv03Int8OvHf):
    name: str = "OpenVINO/Mistral-7B-Instruct-v0.3-fp16-ov"
    precision: str = "FP16"
    is_local: bool = False


@dataclass
class Mistral7BInstructv03Int4OvHf(Mistral7BInstructv03Int8OvHf):
    name: str = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov"
    precision: str = "INT4"


@dataclass
class Mistral7BInstructv03Int4CwOvHf(Mistral7BInstructv03Int8OvHf):
    name: str = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"
    precision: str = "INT4"


@dataclass
class Qwen25Coder14BInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen2.5-Coder-14B-Instruct-int4-ov"
    precision: str = "INT4"


@dataclass
class StableDiffusionv15Int8OvHf(LargeLanguageModelHuggingFace, ImageGenerationModel):
    name: str = "OpenVINO/stable-diffusion-v1-5-int8-ov"
    precision: str = "INT8"


@dataclass
class LCMDreamshaperv7Int8OvHf(LargeLanguageModelHuggingFace, ImageGenerationModel):
    name: str = "OpenVINO/LCM_Dreamshaper_v7-int8-ov"
    precision: str = "INT8"


@dataclass
class BAAIBgeBaseEnv15Int8OvHf(LargeLanguageModelHuggingFace, FeatureExtractionModel):
    name: str = "OpenVINO/bge-base-en-v1.5-int8-ov"
    precision: str = "INT8"
    pooling: str = "CLS"
    is_local: bool = True
    is_feature_extraction: bool = True


@dataclass
class BAAIBgeBaseEnv15Fp16OvHf(LargeLanguageModelHuggingFace, FeatureExtractionModel):
    name: str = "OpenVINO/bge-base-en-v1.5-fp16-ov"
    precision: str = "FP16"
    pooling: str = "CLS"


@dataclass
class Qwen3Embedding06BInt8OvHf(LargeLanguageModelHuggingFace, FeatureExtractionModel):
    name: str = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov"
    precision: str = "INT8"
    pooling: str = "LAST"
    is_local: bool = True
    is_feature_extraction: bool = True


@dataclass
class Qwen3Embedding06BFp16OvHf(LargeLanguageModelHuggingFace, FeatureExtractionModel):
    name: str = "OpenVINO/Qwen3-Embedding-0.6B-fp16-ov"
    precision: str = "FP16"
    pooling: str = "LAST"


@dataclass
class BAAIRBgeRerankerBaseInt8OvHf(LargeLanguageModelHuggingFace, RerankModel):
    name: str = "OpenVINO/bge-reranker-base-int8-ov"
    precision: str = "INT8"


@dataclass
class BAAIRBgeRerankerBaseFp16OvHf(LargeLanguageModelHuggingFace, RerankModel):
    name: str = "OpenVINO/bge-reranker-base-fp16-ov"
    precision: str = "FP16"


@dataclass
class Qwen3Reranker06BSeqClsFp16OvHf(LargeLanguageModelHuggingFace, RerankModel):
    name: str = "OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov"
    precision: str = "FP16"


@dataclass
class Flux1SchnellInt8OvHf(LargeLanguageModelHuggingFace, ImageGenerationModel):
    name: str = "OpenVINO/FLUX.1-schnell-int8-ov"
    precision: str = "INT8"
    is_local: bool = True


@dataclass
class Flux1SchnellInt4OvHf(Flux1SchnellInt8OvHf):
    name: str = "OpenVINO/FLUX.1-schnell-int4-ov"
    precision: str = "INT4"


@dataclass
class Phi4MiniInstructInt8(LargeLanguageModel):
    name: str = "microsoft/Phi-4-mini-instruct"
    parent_name: str = "phi-4-mini-instruct"
    parent_precision_dir: str = "OV_FP16-INT8_ASYM"
    tool_parser: str = "phi4"
    jinja_template: str = ("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v0.9.0/examples/"
                           "tool_chat_template_phi4_mini.jinja")
    is_agentic: bool = True
    gorilla_patch_name: str = "ovms-model"
    enable_tool_guided_generation: bool = True


@dataclass
class Phi4MiniInstructInt4(Phi4MiniInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"


@dataclass
class Phi4MiniInstructInt4SymCw(Phi4MiniInstructInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


def create_dynamic_classes_from_env(models_list):
    classes = []
    for model_definition in models_list:
        model_attributes = model_definition.split(":")
        if len(model_attributes) != 4:
            raise ValueError(
                f"Invalid model definition: {model_definition}. "
                f"Expected format: class_name:name:precision:precision_dir")

        class_name, name, precision, precision_dir = map(str.strip, model_attributes)

        if class_name in globals():
            # if class already exists, only add it to the test
            classes.append(globals()[class_name])
        else:
            # if class does not already, create a new one
            base_class = VisionLanguageModel if "VL" in class_name else LargeLanguageModel
            classes.append(dataclasses.make_dataclass(
                class_name,
                [
                    ("name", str, dataclasses.field(default=name)),
                    ("precision", str, dataclasses.field(default=precision)),
                    ("precision_dir", str, dataclasses.field(default=precision_dir))
                ],
                bases=(base_class,)
            ))
    return classes


extra_acc_models = create_dynamic_classes_from_env(extra_llm_models_accuracy)
extra_test_models = create_dynamic_classes_from_env(extra_llm_models_test)


@dataclass
class Qwen205BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen2-0.5B-int4-ov"


@dataclass
class Qwen205BInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen2-0.5B-Instruct-int4-ov"


@dataclass
class Qwen2515BInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"


@dataclass
class Gemma29BItInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/gemma-2-9b-it-int4-ov"

    def get_default_dataset(self):
        return AssistantLargeLanguageModelDataset


@dataclass
class Gemma7BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/gemma-7b-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Gemma2BItInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/gemma-2b-it-int4-ov"

    def get_default_dataset(self):
        return AssistantLargeLanguageModelDataset


@dataclass
class Codegen6BMultiInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/codegen-6B-multi-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Codegen237BPInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/codegen2-3_7B_P-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Bloomz3BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/bloomz-3b-int4-ov"
    single_message_dataset: bool = True


@dataclass
class RedPajamaInciteInstruct3BV1Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/RedPajama-INCITE-Instruct-3B-v1-int4-ov"
    single_message_dataset: bool = True


@dataclass
class RedPajamaInciteChat3BV1Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/RedPajama-INCITE-Chat-3B-v1-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Phi3Mini4kInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"


@dataclass
class Phi3Medium4kInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Phi-3-medium-4k-instruct-int4-ov"


@dataclass
class Phi2Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/phi-2-int4-ov"
    single_message_dataset: bool = True


@dataclass
class OpenLlama3BV2Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/open_llama_3b_v2-int4-ov"
    single_message_dataset: bool = True


@dataclass
class StarCoder27BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/starcoder2-7b-int4-ov"
    single_message_dataset: bool = True


@dataclass
class NeuralChat7BV33Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/neural-chat-7b-v3-3-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Pythia1BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/pythia-1b-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Persimmon8BChatInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/persimmon-8b-chat-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Mistral7BInstructV02Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Mistral-7B-Instruct-v0.2-int4-ov"

    def get_default_dataset(self):
        return AssistantLargeLanguageModelDataset


@dataclass
class DollyV23BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/dolly-v2-3b-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Falcon7BInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/falcon-7b-instruct-int4-ov"
    single_message_dataset: bool = True


@dataclass
class GptNeox20BInt8OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/gpt-neox-20b-int8-ov"
    single_message_dataset: bool = True


@dataclass
class Zephyr7BBetaInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/zephyr-7b-beta-int4-ov"


@dataclass
class Mixtral8x7BInstructV01Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/mixtral-8x7b-instruct-v0.1-int4-ov"
    single_message_dataset: bool = True


@dataclass
class Mistral7BInstructV01Int4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/mistral-7b-instruct-v0.1-int4-ov"


@dataclass
class GptJ6BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/gpt-j-6b-int4-ov"
    single_message_dataset: bool = True


@dataclass
class InternVL21BInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/InternVL2-1B-int4-ov"


@dataclass
class InternVL22BInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/InternVL2-2B-int4-ov"


@dataclass
class Phi35VisionInstructInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/Phi-3.5-vision-instruct-int4-ov"
    is_local: bool = True


@dataclass
class Phi35VisionInstructInt8OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/Phi-3.5-vision-instruct-int8-ov"
    precision: str = "INT8"
    is_local: bool = True


@dataclass
class InternVL22BInt8OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/InternVL2-2B-int8-ov"
    precision: str = "INT8"
    is_local: bool = True


@dataclass
class Qwen2VL7BInstructInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/Qwen2-VL-7B-Instruct-int4-ov"


@dataclass
class Qwen25VL7BInstructInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov"


@dataclass
class Qwen25VL7BInstructInt8OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/Qwen2.5-VL-7B-Instruct-int8-ov"
    precision: str = "INT8"
    is_local: bool = True


# not enabled for GenAI https://jira.devtools.intel.com/browse/CVS-172592
# @dataclass
# class Pixtral12BInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
#     name: str = "OpenVINO/pixtral-12b-int4-ov"


@dataclass
class Llavav16Mistral7BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/llava-v1.6-mistral-7b-hf-int4-ov"


@dataclass
class InternVL24BInt4OvHf(LargeLanguageModelHuggingFace, VisionLanguageModel):
    name: str = "OpenVINO/InternVL2-4B-int4-ov"


@dataclass
class Falcon7BInstructInt4CwOvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/falcon-7b-instruct-int4-cw-ov"
    single_message_dataset: bool = True


@dataclass
class GptJ6BInt4CwOvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/gpt-j-6b-int4-cw-ov"
    single_message_dataset: bool = True


@dataclass
class Mistral7BInstructV02Int4CwOvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov"


@dataclass
class Phi3Mini4KInstructInt4CwOvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov"


@dataclass
class Phi35MiniInstructInt4CwOvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Phi-3.5-mini-instruct-int4-cw-ov"


@dataclass
class Qwen25Coder05BInstructInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov"


@dataclass
class Llama318BInstructQ4KMGGUFInt4Hf(LargeLanguageModelHuggingFace):
    name: str = "modularai/Llama-3.1-8B-Instruct-GGUF"
    gguf_filename: str = "llama-3.1-8b-instruct-q4_k_m.gguf"


@dataclass
class MetaLlama318BInstructQ4KMGGUFInt4Hf(LargeLanguageModelHuggingFace):
    name: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    gguf_filename: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"


@dataclass
class MetaLlama318BInstructGGUFInt8Hf(LargeLanguageModelHuggingFace):
    name: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    gguf_filename: str = "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    precision: str = "INT8"


@dataclass
class Llama323BInstructQ4KMGGUFInt4Hf(LargeLanguageModelHuggingFace):
    name: str = "unsloth/Llama-3.2-3B-Instruct-GGUF"
    gguf_filename: str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"


@dataclass
class Qwen257BInstructQ4KMGGUFInt4Hf(LargeLanguageModelHuggingFace):
    name: str = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    gguf_filename: str = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"


@dataclass
class Qwen257BInstructGGUFInt8Hf(LargeLanguageModelHuggingFace):
    name: str = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    gguf_filename: str = "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
    precision: str = "INT8"


@dataclass
class Qwen257BInstructGGUFFp16Hf(LargeLanguageModelHuggingFace):
    name: str = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    gguf_filename: str = "qwen2.5-7b-instruct-fp16-00001-of-00004.gguf"
    precision: str = "FP16"


@dataclass
class Qwen34BInstructQ4KMGGUFInt4Hf(LargeLanguageModelHuggingFace):
    name: str = "Qwen/Qwen3-4B-GGUF"
    gguf_filename: str = "Qwen3-4B-Q4_K_M.gguf"


@dataclass
class Qwen34BInstructGGUFInt8Hf(LargeLanguageModelHuggingFace):
    name: str = "Qwen/Qwen3-4B-GGUF"
    gguf_filename: str = "Qwen3-4B-Q8_0.gguf"
    precision: str = "INT8"


@dataclass
class Qwen306BInstructQ40GGUFInt4Hf(LargeLanguageModelHuggingFace):
    name: str = "ggml-org/Qwen3-0.6B-GGUF"
    gguf_filename: str = "Qwen3-0.6B-Q4_0.gguf"


@dataclass
class Qwen306BInstructGGUFInt8Hf(LargeLanguageModelHuggingFace):
    name: str = "Qwen/Qwen3-0.6B-GGUF"
    gguf_filename: str = "Qwen3-0.6B-Q8_0.gguf"
    precision: str = "INT8"


@dataclass
class Qwen306BInstructGGUFFp16Hf(LargeLanguageModelHuggingFace):
    name: str = "ggml-org/Qwen3-0.6B-GGUF"
    gguf_filename: str = "Qwen3-0.6B-f16.gguf"
    precision: str = "FP16"


@dataclass
class Qwen306BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen3-0.6B-int4-ov"


@dataclass
class Qwen34BInt4OvHf(LargeLanguageModelHuggingFace):
    name: str = "OpenVINO/Qwen3-4B-int4-ov"


@dataclass
class SmolLM2135InstructGGUFFp16Hf(LargeLanguageModelHuggingFace):
    name: str = "bartowski/SmolLM2-135M-Instruct-GGUF"
    gguf_filename: str = "SmolLM2-135M-Instruct-f16.gguf"
    precision: str = "FP16"


@dataclass
class DevstralSmall2507Int4(LargeLanguageModel):
    # used in demos/code_local_assistant/README.md (must be exported)
    name: str = "unsloth/Devstral-Small-2507"
    precision: str = "INT4"
    precision_dir: str = "INT4"
    tool_parser: str = "devstral"
    is_agentic: bool = True
    enable_tool_guided_generation: bool = True
    model_timeout: int = 900
    gorilla_patch_name: str = "ovms-model"
    bfcl_num_threads: int = 10
    jinja_template: str = (f"https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/"
                           f"{ovms_c_repo_branch}/extras/chat_template_examples/chat_template_devstral.jinja")


@dataclass
class Codestral22Bv01(LargeLanguageModel):
    # used in demos/code_local_assistant/README.md (must be exported)
    name: str = "mistralai/Codestral-22B-v0.1"
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class AudioModel(LargeLanguageModel):
    is_audio: bool = True
    is_asr_model: bool = False
    is_tts_model: bool = False
    tested_input_languages: tuple[str, ...] = ("en",)
    tested_output_languages: tuple[str, ...] = ("en",)
    cos_sim_threshold: float = 0.7

    def get_default_dataset(self):
        return AudioModelDataset

    def prepare_input_data(self, batch_size=None, input_key=None, dataset=None, input_data_type=None):
        if dataset is not None:
            # Support both class and instance: if dataset is already an instance, use it directly
            ds = dataset if isinstance(dataset, ModelDataset) else self.get_default_dataset()
            input_data = {input_name: ds.get_data(None, None, None) for input_name in self.input_names}
        elif input_data_type == "string":
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_string_data()
                for input_name in self.input_names
            }
        else:
            input_data = {
                input_name: self.inputs[input_name]["dataset"].get_data(None, None, None)
                for input_name in self.input_names
            }
        return input_data


@dataclass
class WhisperLargeV3Fp16(AudioModel):
    name: str = "openai/whisper-large-v3"
    is_asr_model: bool = True
    tested_input_languages: tuple[str, ...] = ("es", "zh")
    parent_name: str = "whisper-large-v3"
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class WhisperLargeV3Int8(WhisperLargeV3Fp16):
    precision: str = "INT8"
    precision_dir: str = "INT8"
    parent_precision_dir: str = "OV_FP16-INT8_ASYM"


@dataclass
class WhisperLargeV3Int4(WhisperLargeV3Fp16):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"
    model_timeout: int = 900


@dataclass
class WhisperLargeV3Fp16OvHf(LargeLanguageModelHuggingFace, AudioModel):
    name: str = "OpenVINO/whisper-large-v3-fp16-ov"
    is_asr_model: bool = True
    tested_input_languages: tuple[str, ...] = ("en", "es", "zh")
    precision: str = "FP16"
    precision_dir: str = "FP16"


@dataclass
class WhisperLargeV3Int8OvHf(LargeLanguageModelHuggingFace, AudioModel):
    name: str = "OpenVINO/whisper-large-v3-int8-ov"
    is_asr_model: bool = True
    tested_input_languages: tuple[str, ...] = ("en", "es", "zh")
    precision: str = "INT8"
    precision_dir: str = "INT8"


@dataclass
class WhisperLargeV3Int4OvHf(LargeLanguageModelHuggingFace, AudioModel):
    name: str = "OpenVINO/whisper-large-v3-int4-ov"
    is_asr_model: bool = True
    tested_input_languages: tuple[str, ...] = ("en", "es", "zh")
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class DistilWhisperLargeV3Int8OvHf(LargeLanguageModelHuggingFace, AudioModel):
    name: str = "OpenVINO/distil-whisper-large-v3-int8-ov"
    is_asr_model: bool = True
    tested_input_languages: tuple[str, ...] = ("en",)
    precision: str = "INT8"
    precision_dir: str = "INT8"


@dataclass
class WhisperLargeV3TurboFp16(AudioModel):
    name: str = "openai/whisper-large-v3-turbo"
    is_asr_model: bool = True
    parent_name: str = "whisper-large-v3-turbo"
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class WhisperLargeV3TurboInt8(WhisperLargeV3TurboFp16):
    precision: str = "INT8"
    precision_dir: str = "INT8"
    parent_precision_dir: str = "OV_FP16-INT8_ASYM"


@dataclass
class WhisperLargeV3TurboInt4(WhisperLargeV3TurboFp16):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_precision_dir: str = "OV_FP16-4BIT_DEFAULT"
    model_timeout: int = 900


@dataclass
class WhisperSmallFp16(AudioModel):
    name: str = "openai/whisper-small"
    is_asr_model: bool = True
    tested_input_languages: tuple[str, ...] = ("en", "es", "zh")
    cos_sim_threshold: float = 0.55
    parent_name: str = "whisper-small"
    precision: str = "FP16"
    precision_dir: str = "FP16"
    parent_precision_dir: str = "FP16"


@dataclass
class WhisperSmallInt8(WhisperSmallFp16):
    precision: str = "INT8"
    precision_dir: str = "INT8"
    parent_precision_dir: str = "OV_FP16-INT8_ASYM"


@dataclass
class MicrosoftSpeech5TtsFp16(AudioModel):
    name: str = "microsoft/speecht5_tts"
    is_tts_model: bool = True
    vocoder: str = "microsoft/speecht5_hifigan"
    precision: str = "FP16"
    precision_dir: str = "FP16"


@dataclass
class MicrosoftSpeech5TtsInt8(MicrosoftSpeech5TtsFp16):
    precision: str = "INT8"
    precision_dir: str = "INT8"


@dataclass
class Sandiago21Speech5TtsSpanishFp16(MicrosoftSpeech5TtsFp16):
    name: str = "Sandiago21/speecht5_finetuned_facebook_voxpopuli_spanish"
    tested_input_languages: tuple[str, ...] = ("es",)
    tested_output_languages: tuple[str, ...] = ("es",)


@dataclass
class StableDiffusionXl10Inpainting01Int8(ImageGenerationModel):
    name: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    parent_name: str = "stable-diffusion-xl-1.0-inpainting-0.1"
    parent_precision_dir: str = "OV_FP16-INT8_ASYM"


@dataclass
class StableDiffusionXl10Inpainting01Int4(StableDiffusionXl10Inpainting01Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4"
    parent_name: str = None
    parent_precision_dir: str = None


@dataclass
class StableDiffusionXl10Inpainting01Int4SymCw(StableDiffusionXl10Inpainting01Int8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_name: str = None
    parent_precision_dir: str = None


@dataclass
class DreamlikeDiffusion10InpaintingInt8(ImageGenerationModel):
    name: str = "parlance/dreamlike-diffusion-1.0-inpainting"


@dataclass
class DreamlikeDiffusion10InpaintingInt4(DreamlikeDiffusion10InpaintingInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class DreamlikeDiffusion10InpaintingInt4SymCw(DreamlikeDiffusion10InpaintingInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class StableDiffusionInpaintingInt8(ImageGenerationModel):
    name: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"


@dataclass
class StableDiffusionInpaintingInt4(StableDiffusionInpaintingInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4"


@dataclass
class StableDiffusionInpaintingInt4SymCw(StableDiffusionInpaintingInt8):
    precision: str = "INT4"
    precision_dir: str = "INT4_SYM_CW"
    parent_precision_dir: str = "OV_FP16-INT4_SYM_CW"


@dataclass
class Dreamshaper8InpaintingInt8OvHf(LargeLanguageModelHuggingFace, ImageGenerationModel):
    name: str = "OpenVINO/dreamshaper-8-inpainting-int8-ov"
    is_local: bool = True


models_llm_vlm_ov_hf = [
    obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if obj.__module__ is __name__ and
       hasattr(obj, "is_hf_direct_load") and obj.is_hf_direct_load and
       hasattr(obj, "is_feature_extraction") and not obj.is_feature_extraction and
       hasattr(obj, "is_rerank") and not obj.is_rerank and
       hasattr(obj, "is_image_generation") and not obj.is_image_generation and
       hasattr(obj, "is_audio") and not obj.is_audio and
       hasattr(obj, "is_local") and not obj.is_local and    # skip models used in accuracy
       "Int4CwOvHf" not in name and name != "LargeLanguageModelHuggingFace"
]

models_llm_vlm_npu_ov_hf = [
    obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if obj.__module__ is __name__ and
       hasattr(obj, "is_hf_direct_load") and obj.is_hf_direct_load and
       hasattr(obj, "is_feature_extraction") and not obj.is_feature_extraction and
       hasattr(obj, "is_rerank") and not obj.is_rerank and
       hasattr(obj, "is_image_generation") and not obj.is_image_generation and
       hasattr(obj, "is_audio") and not obj.is_audio and
       hasattr(obj, "is_local") and not obj.is_local and    # skip models used in accuracy
       "Int4CwOvHf" in name
]

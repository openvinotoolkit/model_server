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
import shutil
from pathlib import Path

from tests.functional.utils.assertions import OvmsTestException
from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional.utils.test_framework import skip_if_runtime
from ovms.config import disable_custom_loader, ovms_test_repo_path
from ovms.constants.custom_loader import CustomLoaderConsts
from tests.functional.constants.ovms import Config, CurrentOvmsType
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths
from tests.functional.object_model.test_environment import TestEnvironment
from ovms.remote_test_environment import copy_container_folder_to_remote_k8s_cluster, delete_remote_container_folder

logger = get_logger(__name__)


class CustomLoader:
    PARENT_KEY = Config.CUSTOM_LOADER_CONFIG_LIST
    ENABLE_FILE_DISABLE_PHRASE = "DISABLED"

    DEFAULT_LOADER_NAME = CustomLoaderConsts.DEFAULT_LOADER_NAME

    def __init__(
        self,
        model,
        name: str = DEFAULT_LOADER_NAME,
        loader_config_file: str = None,
        enable_file: str = None,
        prepare_custom_loader_resources: bool = False,
    ):
        skip_if_runtime(disable_custom_loader, "Custom loader disabled")

        file_name = CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_LIB_NAME
        self.ovms_type = CurrentOvmsType.ovms_type
        if self.ovms_type == OvmsType.KUBERNETES:
            internal_path = os.path.join("/config", file_name)
        else:
            internal_path = os.path.join(
                Paths.CUSTOM_LOADER_LIBRARIES_PATH_INTERNAL, CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_NAME, file_name
            )
        self._loader_config = self.LoaderConfig(name, internal_path, loader_config_file)

        self._model = model
        self._model_options = self.ModelOptions(name, enable_file)
        self._model_options.update_from_model(model)
        self.prepare_custom_loader_resources = prepare_custom_loader_resources

    @staticmethod
    def get_custom_loader_path(image):
        cmd = f"docker cp $(docker create --rm {image}):{CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_DOCKER_LIB} ."
        proc = Process()
        cwd = os.path.join(ovms_test_repo_path, "data", "ovms_testing_image")
        proc.run_and_check(cmd, cwd=cwd)
        dst_file_path = os.path.join(cwd, CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_LIB_NAME)
        return dst_file_path

    def prepare_resources(self, base_location):
        resource_path = Path(base_location + Paths.MODELS_PATH_INTERNAL, Paths.CUSTOM_LOADER_PATH_NAME)
        resource_path.mkdir(parents=True, exist_ok=True)
        src_file_path = CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_LIB
        dst_file_path = os.path.join(resource_path, os.path.basename(src_file_path))
        if not os.path.exists(dst_file_path):
            shutil.copy(src_file_path, dst_file_path)
        return str(resource_path)

    @property
    def loader_config(self):
        return self._loader_config

    @property
    def model_options(self):
        return self._model_options

    def get_volume_mount(self):
        loader_container_path = self.loader_config.get_loader_container_path()
        return {self.loader_host_path: {"bind": loader_container_path, "mode": "ro"}}

    def get_enable_file_name(self):
        return self._model_options["custom_loader_options"].get(CustomLoader.ModelOptions.ENABLE_FILE_KEY, "")

    @property
    def name(self):
        return self.loader_config.name

    @staticmethod
    def create_model_with_custom_loader(model_type, use_enable_file=False):
        model = model_type()
        enable_file = f"{model.name}.status" if use_enable_file else None
        custom_loader = CustomLoader(model, enable_file=enable_file)
        model.custom_loader = custom_loader
        return model

    @staticmethod
    def attach_custom_loader_to_models(
        models, name: str = DEFAULT_LOADER_NAME, loader_host_path: str = None, loader_config_file: str = None
    ):
        if loader_host_path is None:
            loader_host_path = CustomLoaderConsts.SAMPLE_CUSTOM_LOADER_LIB

        for model in models:
            model.custom_loader = CustomLoader(model, name, loader_host_path, loader_config_file)

    def get_model_enable_file_path(self, container_name):
        model_path_on_host = os.path.join(
            TestEnvironment.current.base_dir, container_name, "models", self._model.name, str(self._model.version)
        )
        model_enable_file_path = os.path.join(model_path_on_host, self.get_enable_file_name())
        return model_enable_file_path

    def enable_model(self, container_name, delete_enable_file=False, ovms_run=None):
        logger.debug(f"Enable model {self.name} in container {container_name}")
        model_enable_file_path = self.get_model_enable_file_path(container_name)
        if delete_enable_file:
            Path(model_enable_file_path).unlink("")
            if self.ovms_type == OvmsType.KUBERNETES and ovms_run is not None:
                delete_remote_container_folder(model_enable_file_path, ovms_run.ovms.remote_ip)
        else:
            Path(model_enable_file_path).write_text("")
            if self.ovms_type == OvmsType.KUBERNETES and ovms_run is not None:
                copy_container_folder_to_remote_k8s_cluster(model_enable_file_path, ovms_run.ovms.remote_ip)

    def disable_model(self, container_name, ovms_run=None):
        logger.debug(f"Disable model {self.name} in container {container_name}")
        model_enable_file_path = self.get_model_enable_file_path(container_name)
        Path(model_enable_file_path).write_text(CustomLoader.ENABLE_FILE_DISABLE_PHRASE)
        if self.ovms_type == OvmsType.KUBERNETES and ovms_run is not None:
            copy_container_folder_to_remote_k8s_cluster(model_enable_file_path, ovms_run.ovms.remote_ip)

    def add_enable_file_entry(self, enable_file_value):
        self.model_options["custom_loader_options"]["enable_file"] = enable_file_value

    def remove_enable_file_entry(self):
        del self.model_options["custom_loader_options"]["enable_file"]

    class LoaderConfig(dict):
        PARENT_KEY = "config"
        LOADER_NAME_KEY = "loader_name"
        LIBRARY_PATH_KEY = "library_path"
        LOADER_CONFIG_FILE_KEY = "loader_config_file"

        def __init__(self, name: str, loader_container_path: str, loader_config_file: str = None):
            super().__init__()
            config = dict()
            config.update({self.LOADER_NAME_KEY: name, self.LIBRARY_PATH_KEY: loader_container_path})
            if loader_config_file:
                config.update({self.LOADER_CONFIG_FILE_KEY: loader_config_file})
            self.update({self.PARENT_KEY: config})

        def __hash__(self):
            return f"{self}".__hash__()

        def get_loader_container_path(self):
            return self[self.PARENT_KEY][self.LIBRARY_PATH_KEY]

        @property
        def name(self):
            return self[self.PARENT_KEY][self.LOADER_NAME_KEY]

    class ModelOptions(dict):
        PARENT_KEY = "custom_loader_options"
        LOADER_NAME_KEY = "loader_name"
        MODEL_FILE_KEY = "model_file"
        BIN_FILE_KEY = "bin_file"
        ENABLE_FILE_KEY = "enable_file"

        def __init__(self, loader_name: str = None, enable_file: str = None):
            super().__init__()
            config = dict()
            config.update({self.LOADER_NAME_KEY: loader_name})
            if enable_file:
                config.update({self.ENABLE_FILE_KEY: enable_file})

            self.update({self.PARENT_KEY: config})

        @classmethod
        def create_from_model(cls, model):
            model_options = cls(f"{model.name}_custom_loader")
            model_options.update_from_model(model)
            return model_options

        def update_from_model(self, model):
            file_list = model.get_model_files()

            if model.model_type.value == "ONNX":  # ModelType.ONNX: # Unable to use symbol: circular import
                onnx_model_file = list(filter(lambda x: x.endswith(".onnx"), file_list))
                assert len(onnx_model_file) == 1, f"Expected single .onnx file, got: {len(onnx_model_file)}"

                self[self.PARENT_KEY].update({
                    self.MODEL_FILE_KEY: onnx_model_file[0],
                })
            elif model.model_type.value == "IR":  # ModelType.IR:  # Unable to use symbol: circular import
                xml_file = list(filter(lambda x: x.endswith(".xml"), file_list))
                bin_file = list(filter(lambda x: x.endswith(".bin"), file_list))
                assert len(xml_file) == 1, f"Expected single .xml file, got: {len(xml_file)}"
                assert len(bin_file) == 1, f"Expected single .bin file, got: {len(bin_file)}"

                self[self.PARENT_KEY].update({
                    self.MODEL_FILE_KEY: xml_file[0],
                    self.BIN_FILE_KEY: bin_file[0],
                })
            elif model.model_type.value == "PDPD":  # ModelType.PDPD  # Unable to use symbol: circular import
                pdiparams_file = list(filter(lambda x: x.endswith(".pdiparams"), file_list))
                pdmodel_file = list(filter(lambda x: x.endswith(".pdmodel"), file_list))

                assert len(pdmodel_file) == 1, f"Expected single .pdmodel file, got: {len(pdmodel_file)}"
                assert len(pdiparams_file) == 1, f"Expected single .pdiparams file, got: {len(pdiparams_file)}"

                self[self.PARENT_KEY].update({
                    self.MODEL_FILE_KEY: pdmodel_file[0],
                    self.BIN_FILE_KEY: pdiparams_file[0],
                })
            elif model.model_type.value == "TFSM":
                pb_file = list(filter(lambda x: x.endswith(".pb"), file_list))
                assert len(pb_file) == 1, f"Expected single .pb file, got: {len(pb_file)}"

                self[self.PARENT_KEY].update({
                    self.MODEL_FILE_KEY: pb_file[0],
                })
            else:
                raise OvmsTestException(f"Unexpected model type=={model.model_type.value}")

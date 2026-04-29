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

from tests.functional.constants.os_type import OsType
from tests.functional import config
from tests.functional.constants.target_device import TargetDevice


class Paths:

    MODELS_PATH_NAME = "models"
    CUSTOM_NODE_PATH_NAME = "custom_nodes"
    CUSTOM_LOADER_PATH_NAME = "custom_loader"
    CPU_EXTENSIONS = "cpu_extensions"
    CONFIG_FILE_NAME = "config.json"
    SUBCONFIG_FILE_NAME = "subconfig.json"
    GRAPH_NAME = "graph.pbtxt"
    IMAGES = "images"
    OVMS_PATH_INTERNAL = os.path.join("/") if OsType.Windows not in config.base_os else os.path.join("\\")
    MODELS_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, MODELS_PATH_NAME)
    CONFIG_PATH_INTERNAL = os.path.join(MODELS_PATH_INTERNAL, CONFIG_FILE_NAME)
    CUSTOM_NODE_LIBRARIES_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, CUSTOM_NODE_PATH_NAME)
    CUSTOM_LOADER_LIBRARIES_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, CUSTOM_LOADER_PATH_NAME)
    ROOT_PATH_CPU_EXTENSIONS = os.path.join(OVMS_PATH_INTERNAL, CPU_EXTENSIONS)
    IMAGES_PATH_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, IMAGES)
    ZEBRA_PATH_INTERNAL = os.path.join(IMAGES_PATH_INTERNAL, "zebra.jpeg")
    CACHE_INTERNAL = os.path.join(OVMS_PATH_INTERNAL, "opt", "cache")

    # DATASET
    DATASET_MAIN_PATH = os.path.join("/", "opt", "test_data")

    @staticmethod
    def CAPI_WRAPPER_PACKAGE_CONTENT_PATH(base_os):
        return os.path.join(config.c_api_wrapper_dir, base_os, "ovms")

    @staticmethod
    def get_target_device_lock_file(target_device, i):
        if isinstance(target_device, str):
            assert not all(x in target_device for x in [TargetDevice.GPU, TargetDevice.NPU])

            # generalize HETERO/AUTO/MUTLI:X => `X`
            if TargetDevice.GPU in target_device:
                return os.path.join(config.ovms_file_locks_dir, f"target_device_{TargetDevice.GPU}_{i}.lock")
            if TargetDevice.NPU in target_device:
                return os.path.join(config.ovms_file_locks_dir, f"target_device_{TargetDevice.NPU}_{i}.lock")

        return os.path.join(config.ovms_file_locks_dir, f"target_device_{target_device}_{i}.lock")


def any_is_relative_to(paths, subpath):
    return any([_path in subpath for _path in paths])

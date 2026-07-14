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

from tests.functional.config import ovms_c_repo_path
from tests.functional.constants.ovms_messages import OvmsMessages
from tests.functional.constants.paths import Paths


class CustomLoaderConsts:
    SAMPLE_CUSTOM_LOADER_NAME = "SampleCustomLoader"
    SAMPLE_CUSTOM_LOADER_LIB_NAME = "libsampleloader.so"
    SAMPLE_CUSTOM_LOADER_SRC_DIR = os.path.join(ovms_c_repo_path, "src", "example", SAMPLE_CUSTOM_LOADER_NAME)
    SAMPLE_CUSTOM_LOADER_DOCKER_LIB = os.path.join(
        Paths.CUSTOM_LOADER_LIBRARIES_PATH_INTERNAL, SAMPLE_CUSTOM_LOADER_NAME, SAMPLE_CUSTOM_LOADER_LIB_NAME
    )
    SAMPLE_CUSTOM_LOADER_LIB = os.path.join(SAMPLE_CUSTOM_LOADER_SRC_DIR, "lib", SAMPLE_CUSTOM_LOADER_LIB_NAME)
    SAMPLE_LOADER_HEADER = os.path.join(ovms_c_repo_path, "src", "customloaderinterface.hpp")
    DEFAULT_LOADER_NAME = "sampleloader"

    @staticmethod
    def get_debug_logs_for_loaded_loader(loader_name):
        return [
            OvmsMessages.CUSTOM_LOADER_LOOKING_FOR_LOADER.format(loader_name),
            OvmsMessages.CUSTOM_LOADER_TO_BE_USED.format(loader_name),
        ]

    @staticmethod
    def get_logs_for_loaded_loader(loader_name):
        return [OvmsMessages.CUSTOM_LOADER_READING_CONFIGURATION.format(loader_name)]

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

from tests.functional.constants.os_type import OsType

REDHAT_MINIMAL_BASE_IMAGE = "registry.access.redhat.com/ubi9/ubi-minimal:9.7"
REDHAT_COMMON_BASE_IMAGE = "registry.access.redhat.com/ubi9/ubi:9.7"
UBUNTU_22_BASE_IMAGE = "ubuntu:22.04"
UBUNTU_24_BASE_IMAGE = "ubuntu:24.04"

os_type_to_base_image = {
    OsType.Redhat: REDHAT_COMMON_BASE_IMAGE,
    OsType.Ubuntu22: UBUNTU_22_BASE_IMAGE,
    OsType.Ubuntu24: UBUNTU_24_BASE_IMAGE,
}

os_type_to_base_image_binary_docker = {
    OsType.Redhat: REDHAT_COMMON_BASE_IMAGE,
    OsType.Ubuntu22: UBUNTU_22_BASE_IMAGE,
    OsType.Ubuntu24: UBUNTU_24_BASE_IMAGE
}

OPENVINO_UBUNTU_20_DEV_IMAGE = "openvino/ubuntu20_dev:2024.6.0"
OPENVINO_MODEL_SERVER_LATEST = "openvino/model_server:latest"
OPENVINO_MODEL_SERVER_LATEST_GPU = "openvino/model_server:latest-gpu"
OPENVINO_MODEL_SERVER_LATEST_PY = "openvino/model_server:latest-py"
OPENVINO_MODEL_SERVER_WEEKLY = "openvino/model_server:weekly"
NO_DOC_UPDATE_IMAGES = [OPENVINO_MODEL_SERVER_LATEST, OPENVINO_MODEL_SERVER_LATEST_GPU,
                        OPENVINO_MODEL_SERVER_LATEST_PY, OPENVINO_MODEL_SERVER_WEEKLY]

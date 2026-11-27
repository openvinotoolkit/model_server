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

REDHAT9_MINIMAL_BASE_IMAGE = "registry.access.redhat.com/ubi9/ubi-minimal:9.9"
REDHAT9_COMMON_BASE_IMAGE = "registry.access.redhat.com/ubi9/ubi:9.9"
REDHAT10_MINIMAL_BASE_IMAGE = "registry.access.redhat.com/ubi10/ubi-minimal:10.3"
REDHAT10_COMMON_BASE_IMAGE = "registry.access.redhat.com/ubi10/ubi:10.3"
# Backwards compatible aliases - "redhat" means the default redhat9 image
REDHAT_MINIMAL_BASE_IMAGE = REDHAT9_MINIMAL_BASE_IMAGE
REDHAT_COMMON_BASE_IMAGE = REDHAT9_COMMON_BASE_IMAGE
UBUNTU_22_BASE_IMAGE = "ubuntu:22.04"
UBUNTU_24_BASE_IMAGE = "ubuntu:24.04"
UBUNTU_26_BASE_IMAGE = "ubuntu:26.04"

os_type_to_base_image = {
    OsType.Redhat: REDHAT9_COMMON_BASE_IMAGE,
    OsType.Redhat10: REDHAT10_COMMON_BASE_IMAGE,
    OsType.Ubuntu22: UBUNTU_22_BASE_IMAGE,
    OsType.Ubuntu24: UBUNTU_24_BASE_IMAGE,
    OsType.Ubuntu26: UBUNTU_26_BASE_IMAGE,
}

os_type_to_base_image_binary_docker = {
    OsType.Redhat: REDHAT9_COMMON_BASE_IMAGE,
    OsType.Redhat10: REDHAT10_COMMON_BASE_IMAGE,
    OsType.Ubuntu22: UBUNTU_22_BASE_IMAGE,
    OsType.Ubuntu24: UBUNTU_24_BASE_IMAGE,
    OsType.Ubuntu26: UBUNTU_26_BASE_IMAGE,
}

OPENVINO_MODEL_SERVER = "openvino/model_server"
OPENVINO_MODEL_SERVER_LATEST = f"{OPENVINO_MODEL_SERVER}:latest"
OPENVINO_MODEL_SERVER_LATEST_GPU = f"{OPENVINO_MODEL_SERVER}:latest-gpu"
OPENVINO_MODEL_SERVER_LATEST_PY = f"{OPENVINO_MODEL_SERVER}:latest-py"
OPENVINO_MODEL_SERVER_WEEKLY = f"{OPENVINO_MODEL_SERVER}:weekly"
NO_DOC_UPDATE_IMAGES = [OPENVINO_MODEL_SERVER_LATEST, OPENVINO_MODEL_SERVER_LATEST_GPU,
                        OPENVINO_MODEL_SERVER_LATEST_PY, OPENVINO_MODEL_SERVER_WEEKLY]

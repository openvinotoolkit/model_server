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
from ovms.constants.ov import OV
from tests.functional.constants.target_device import TargetDevice


class OvmsType:
    NONE = "NONE"
    DOCKER = "DOCKER"
    DOCKER_CMD_LINE = "DOCKER_CMD_LINE"
    KUBERNETES = "KUBERNETES"
    BINARY = "BINARY"
    BINARY_DOCKER = "BINARY_DOCKER"
    CAPI = "CAPI"
    CAPI_DOCKER = "CAPI_DOCKER"


# https://github.com/openvinotoolkit/model_server/blob/main/docs/deploying_server.md#deploying-model-server-on-baremetal-without-container
OVMS_BINARY_DEPENDENCIES = {
    OsType.Ubuntu22: "libcurl4-openssl-dev libpugixml1v5 libtbb12 libxml2",
    OsType.Ubuntu24: "libcurl4-openssl-dev libpugixml1v5 libtbb12 libxml2",
    OsType.Redhat: "https://mirror.stream.centos.org/9-stream/AppStream/x86_64/os/Packages/tbb-2020.3-8.el9.x86_64.rpm",
}


OVMS_BINARY_PACKAGE_NAME = "ovms"
OVMS_BINARY_PACKAGE_EXTENSIONS = (".tar.gz", ".zip")

OVMS_CAPI_DEPENDENCIES = OVMS_BINARY_DEPENDENCIES

OVMS_CAPI_UBUNTU_CPU_TOOLS = "build-essential"
OVMS_CAPI_UBUNTU_GPU_TOOLS = "clinfo curl"
OVMS_CAPI_UBUNTU_OPENCL_TOOLS = "opencl-clhpp-headers opencl-c-headers intel-opencl-icd"
OVMS_CAPI_UBUNTU_VA_TOOLS = "gpg"

OVMS_CAPI_TOOLS_DEPENDENCIES = {
    TargetDevice.CPU: {
        OsType.Ubuntu24: OVMS_CAPI_UBUNTU_CPU_TOOLS,
        OsType.Ubuntu22: OVMS_CAPI_UBUNTU_CPU_TOOLS,
        OsType.Redhat: "",
    },
    TargetDevice.GPU: {
        OsType.Ubuntu24: " ".join([
            OVMS_CAPI_UBUNTU_CPU_TOOLS,
            OVMS_CAPI_UBUNTU_GPU_TOOLS,
            OVMS_CAPI_UBUNTU_OPENCL_TOOLS,
            OVMS_CAPI_UBUNTU_VA_TOOLS,
        ]),
        OsType.Ubuntu22: " ".join([
            OVMS_CAPI_UBUNTU_CPU_TOOLS,
            OVMS_CAPI_UBUNTU_GPU_TOOLS,
            OVMS_CAPI_UBUNTU_OPENCL_TOOLS,
            OVMS_CAPI_UBUNTU_VA_TOOLS,
        ]),
        OsType.Redhat: "",
    }
}

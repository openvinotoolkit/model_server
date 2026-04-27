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

import re

from tests.functional.utils.environment_info import EnvironmentInfo
from tests.functional.constants.os_type import OsType, UBUNTU
from ovms.config import (
    docker_registry,
    is_nginx_mtls,
    ovms_cpp_docker_image,
    ovms_image,
    ovms_image_tag,
    ovms_test_image_name,
    force_use_ovms_image
)
from ovms.constants.ov import OV
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.ovms import CurrentOvmsType
from tests.functional.constants.ovms import CurrentTarget as ct
from ovms.constants.repositories import Repositories


# should be checked periodically with OVMS Dockerfiles
GPU_INSTALL_DRIVER_VERSION = {
    "redhat": "24.52.32224",
    "ubuntu22": "24.39.31294",
    "ubuntu24": "26.09.37435",
}

GPU_INSTALL_SCRIPTS = {
    OsType.Ubuntu22: ["install_ubuntu_gpu_drivers.sh", "install_va.sh"],
    OsType.Ubuntu24: ["install_ubuntu_gpu_drivers.sh", "install_va.sh"],
    OsType.Redhat: ["install_redhat_gpu_drivers.sh"],
}


class OvmsImages:
    _os_type = None
    _os_version = None

    @classmethod
    def get_os_type(cls):
        if cls._os_type is None:
            if CurrentOvmsType.is_none_type():
                cls._os_type, _ = "<NONE>", None
            else:
                cls._os_type, _ = cls._get_os_type_and_version()
        return cls._os_type

    @classmethod
    def get_os_version(cls):
        _, os_verion = cls._get_os_type_and_version()
        return os_verion

    @classmethod
    def _get_os_type_and_version(cls):
        if cls._os_type and cls._os_version:
            return cls._os_type, cls._os_version

        # 'Ubuntu 20.04.2 LTS', 'CentOS Linux 7 (Core)', 'Red Hat Enterprise Linux 8.4 (Ootpa)'
        os_name = EnvironmentInfo().get_os_distname()
        os_name_lower = os_name.lower()

        for os_name_part in os_name_lower.split():
            if str(os_name_part[0]).isdigit():
                cls._os_version = os_name_part
                break

        if "red hat" in os_name_lower:
            cls._os_type = OsType.Redhat
        elif "ubuntu 22" in os_name_lower:
            cls._os_type = OsType.Ubuntu22
        elif "ubuntu 24" in os_name_lower:
            cls._os_type = OsType.Ubuntu24
        else:
            raise NotImplementedError()

        return cls._os_type, cls._os_version


NGINX = "nginx"
DEFAULT_OVMS_IMAGE_NAME = "model_server"
DEFAULT_OVMS_IMAGE_SUFFIXES = {
    NGINX: "-nginx-mtls",
    TargetDevice.GPU: "-gpu",
    TargetDevice.NPU: "-gpu",
}

DEFAULT_OVMS_IMAGE_TAG = {
    OsType.Ubuntu22: "ubuntu22_main",
    OsType.Ubuntu24: "ubuntu24_main",
    OsType.Redhat: "redhat_main",
}


def calculate_ovms_image_suffix(target_device):
    if is_nginx_mtls:
        return DEFAULT_OVMS_IMAGE_SUFFIXES[NGINX]
    elif ct.is_gpu_based_target(target_device) or ct.is_npu_target():
        return DEFAULT_OVMS_IMAGE_SUFFIXES[TargetDevice.GPU]
    return ""


def prepare_general_os_list(base_os_list):
    os_types = set(base_os_list)
    if any(elem == OsType.Ubuntu22 or elem == OsType.Ubuntu24 for elem in base_os_list):
        os_types.add(UBUNTU)
    return os_types


def calculate_ovms_image_tag(image_tag_to_check, base_os, base_os_list):
    os_types = prepare_general_os_list(base_os_list)
    image_tag_base_os = [i for i in image_tag_to_check.split("_") if i in os_types]
    if image_tag_base_os and image_tag_base_os[0] not in base_os:
        return image_tag_to_check.replace(image_tag_base_os[0], base_os)
    return image_tag_to_check


def calculate_ovms_image_name(target_device=None, base_os=OsType.Ubuntu22):
    assert target_device, "Wrong target_device specified."
    base_os_list = [val for key, val in vars(OsType).items() if not key.startswith("__")]
    assert base_os in base_os_list, f"Wrong os specified: {base_os}"
    assert not (
        is_nginx_mtls and target_device != TargetDevice.CPU
    ), "nginx_mtls only available for CPU target_device"

    ct.target_device = target_device

    if force_use_ovms_image and ovms_image:
        return ovms_image
    elif ovms_image:
        image_name = re.sub("|".join(DEFAULT_OVMS_IMAGE_SUFFIXES.values()), "", ovms_image.split(":")[0])
        image_tag = ovms_image.split(":")[1]
        image_name = f"{image_name}{calculate_ovms_image_suffix(target_device)}"
        image_tag = calculate_ovms_image_tag(image_tag, base_os, base_os_list)
    else:
        image_name = (
            ovms_cpp_docker_image
            if ovms_cpp_docker_image
            else f"{docker_registry}/{Repositories.OV_REPO_BASENAME}/{DEFAULT_OVMS_IMAGE_NAME}"
        )
        image_name = f"{image_name}{calculate_ovms_image_suffix(target_device)}"
        image_tag = ovms_image_tag if ovms_image_tag else DEFAULT_OVMS_IMAGE_TAG[base_os]
        image_tag = calculate_ovms_image_tag(image_tag, base_os, base_os_list)

    return f"{image_name}:{image_tag}"


def calculate_ovms_binary_image_name(ovms_image_name):
    binary_image_name = f"{ovms_image_name}-binary"
    return binary_image_name


def calculate_ovms_capi_image_name(ovms_image_name):
    capi_image_name = f"{ovms_image_name}-capi"
    return capi_image_name


def calculate_ovms_test_image_name(ovms_image_name):
    test_image_name = ovms_test_image_name if ovms_test_image_name is not None else f"{ovms_image_name}-test"
    return test_image_name

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
import re
import sys
from importlib import import_module

from tests.functional.config import is_nginx_mtls, ovms_c_repo_path
from tests.functional.constants.os_type import OsType
from tests.functional.constants.ovms import CurrentTarget
from tests.functional.utils.assertions import OvmsTestException, SdlException

# enable import from ovms-c repository (tests/sdl)
sys.path.append(os.path.join(os.path.abspath(ovms_c_repo_path), "tests", "sdl"))


def get_base_os(context):
    # matching regex: test_unnecessary_packages_in_container[DOCKER-UBUNTU_20_04-CPU]
    base_os_match = re.search(rf"-({context.base_os}.*)-", context.name.lower())
    if base_os_match:
        base_os = base_os_match.group(1)
    else:
        base_os = context.base_os
    return base_os


def get_ovms_type(target_device, base_os):
    ovms_type = getattr(import_module("whitelists"), "OvmsType")

    if base_os == OsType.Windows:
        return ovms_type.WINDOWS

    if CurrentTarget.is_gpu_based_target(target_device):
        if OsType.Ubuntu24 in base_os:
            return ovms_type.UBUNTU24_GPU
        elif OsType.Ubuntu22 in base_os:
            return ovms_type.UBUNTU22_GPU
        elif OsType.Redhat in base_os:
            return ovms_type.REDHAT_GPU

    if OsType.Ubuntu24 in base_os:
        if is_nginx_mtls:
            return ovms_type.UBUNTU24_NGINX
        return ovms_type.UBUNTU24
    elif OsType.Ubuntu22 in base_os:
        if is_nginx_mtls:
            return ovms_type.UBUNTU22_NGINX
        return ovms_type.UBUNTU22
    elif OsType.Redhat in base_os:
        return ovms_type.REDHAT

    raise OvmsTestException(f"Cannot deduce image device={target_device}, base_os={base_os}")


def get_whitelist_python_option(context, whitelist_dict):
    ovms_type = context.ovms_image if context.ovms_image is not None else context.ovms_binary
    context.env_info.env_info.get_ovms_version(ovms_type)
    ovms_build_flags = context.env_info.env_info._ovms_build_flags
    whitelist = set()
    if whitelist_dict.get("python", None) is not None:
        whitelist = whitelist_dict["python"]
    else:
        whitelist = whitelist_dict["no_python"]
    return whitelist


def get_whitelist(context, whitelist_name):
    ovms_type = get_ovms_type(context.target_device, get_base_os(context))
    whitelist_dict = getattr(import_module("whitelists"), whitelist_name)[ovms_type]
    whitelist = whitelist_dict["default"] | get_whitelist_python_option(context, whitelist_dict)
    if not whitelist:
        raise OvmsTestException(f"Empty whitelist: {whitelist_name}")
    return whitelist


def validate_whitelists(context, whitelist_name, detected_objects, check_missing_objects=True):
    whitelisted_objects = get_whitelist(context, whitelist_name=whitelist_name)
    objects_name = " ".join(whitelist_name.split("_")[1:])

    unexpected_objects = set(x for x in detected_objects if x not in whitelisted_objects)
    missing_objects = set(x for x in whitelisted_objects if x not in detected_objects)

    unexpected_objects_msg = f"Detected unexpected {objects_name}: {unexpected_objects}\n"
    missing_objects_msg = f"Unable to find expected {objects_name}: {missing_objects}\n"

    if unexpected_objects and missing_objects and check_missing_objects:
        raise SdlException(unexpected_objects_msg + missing_objects_msg)
    elif unexpected_objects:
        raise SdlException(unexpected_objects_msg)
    elif missing_objects and check_missing_objects:
        raise SdlException(missing_objects_msg)

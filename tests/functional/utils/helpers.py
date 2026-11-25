#
# Copyright (c) 2020 Intel Corporation
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
import random
import re

from collections import defaultdict
from time import strftime

from tests.functional.config import ovms_c_repo_path
from tests.functional.constants.target_device import TargetDevice


ALL_AVAILABLE_OPTIONS = "*"


def get_int(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError(f"Value '{value}' of {key_name} env variable cannot be cast to int.") from exc
    return value


def get_float(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"Value '{value}' of {key_name} env variable cannot be cast to float.") from exc
    return value


def get_bool(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        value = value.lower()
        if value == "true":
            value = True
        elif value == "false":
            value = False
        else:
            raise ValueError(f"Value of {key_name} env variable is '{value}'. Should be 'True' or 'False'.")
    return value


def get_path(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value:
        value = os.path.expanduser(value)
        value = os.path.realpath(value)
    return value


def get_list(key_name, delimiter=",", fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        value = value.split(delimiter)
        for index, item in enumerate(value):
            normalized = item.strip().lower()
            if normalized == "true":
                value[index] = True
            elif normalized == "false":
                value[index] = False
    elif not value:
        value = []
    return value


MULTI_DEVICE_PREFIXES = ("AUTO", "HETERO", "MULTI")
DEVICE_INDEX_RE = re.compile(r"\b([A-Z]+)[.:](\d+)\b")


def _is_multi_device(device_str):
    prefix, separator, _ = device_str.partition(":")
    return bool(separator) and prefix in MULTI_DEVICE_PREFIXES


def get_multi_target_devices(target_devices_list):
    result = []
    td_list_copy = target_devices_list.copy()
    while td_list_copy:
        first_td = td_list_copy.pop(0)
        if _is_multi_device(first_td):  # ie: 'AUTO:GPU'
            second_td = td_list_copy.pop(0)
            assert not _is_multi_device(second_td), f"Incorrect target_device_list={target_devices_list}"
            result.append(f"{first_td},{second_td}")  # ie: 'AUTO:GPU,CPU'
        else:
            result.append(first_td)
    return result


def normalize_device(device_str):
    """ OpenVINO accepts GPU:1 and GPU.1, while OVMS model export accepts only GPU.1. """
    return DEVICE_INDEX_RE.sub(r"\1.\2", device_str)


def get_base_device(device_str):
    """ Return device without indexes, e.g. 'GPU.1' -> 'GPU', 'AUTO:GPU.1,CPU' -> 'AUTO:GPU,CPU'. """
    if not isinstance(device_str, str):
        return device_str
    return DEVICE_INDEX_RE.sub(r"\1", device_str)


class DeviceAwareDefaultDict(defaultdict):
    """
    Transparently resolves indexed devices (e.g. GPU:0 -> GPU).
    When a key like 'GPU:0' is not found, it falls back to the base device key 'GPU'.
    """

    def __missing__(self, key):
        base = get_base_device(key)
        if base != key and base in self:
            return self[base]
        # Fall back to default_factory behavior
        return super().__missing__(key)


def validate_supported_values(detected_list, supported_list):
    supported_list += ALL_AVAILABLE_OPTIONS  # 'starred expression' will be evaluated during pytest_configure
    check = all(get_base_device(_elem) in supported_list for _elem in detected_list)
    assert check, f"Not supported target devices in {detected_list}"
    return detected_list


def get_target_devices():
    """ Convert comma separated string of devices into list """
    target_devices_list = get_list("TT_TARGET_DEVICE", fallback=[TargetDevice.CPU])
    target_devices_list = [normalize_device(_td) for _td in get_multi_target_devices(target_devices_list)]
    ov_target_devices = [value for key, value in vars(TargetDevice).items() if not key.startswith("__")]
    target_devices_list = validate_supported_values(detected_list=target_devices_list, supported_list=ov_target_devices)
    return target_devices_list


def get_xdist_worker_count():
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))


def get_xdist_worker_nr():
    xdist_current_worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    if xdist_current_worker == "master":
        xdist_current_worker = 0
    else:
        xdist_current_worker = int(xdist_current_worker.lstrip("gw"))
    return xdist_current_worker


def get_short_date_string():
    date_str = strftime("%Y%m%d%H%M%S")
    return date_str


def generate_test_object_name(separator="_", prefix=""):
    date_str = get_short_date_string()
    random_sha = hex(random.getrandbits(128))[2:8]
    name = separator.join([item for item in (prefix, date_str, random_sha) if item])
    return name


def get_data_from_versions_mk():
    filepath = os.path.join(ovms_c_repo_path, "versions.mk")
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        data = f.read()

    return data

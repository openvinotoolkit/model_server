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
from typing import Any

from tests.functional.constants.target_device import TargetDevice


class SingletonMeta(type):
    """
    Metaclass for defining Singleton Classes

    src:
    https://www.datacamp.com/community/tutorials/python-metaclasses

    Singleton Design using a Metaclass

    This is a design pattern that restricts the instantiation of a class to only one object.
    This could prove useful for example when designing a class to connect to the database.
    One might want to have just one instance of the connection class.
    """
    _instances = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


ALL_AVAILABLE_OPTIONS = "*"


def get_int(key_name, fallback=None, environ=os.environ):
    value = environ.get(key_name, fallback)
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
            raise ValueError("Value of {} env variable is '{}'. Should be 'True' or 'False'.".format(key_name, value))
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
    elif not value:
        value = []
    return value


def get_multi_target_devices(target_devices_list, separator):
    result = []
    td_list_copy = target_devices_list.copy()
    while td_list_copy:
        first_td = td_list_copy.pop(0)
        if separator in first_td:  # ie: 'AUTO:GPU'
            second_td = td_list_copy.pop(0)
            assert separator not in second_td, f"Incorrect target_device_list={target_devices_list}"
            result.append(f"{first_td},{second_td}")  # ie: 'AUTO:GPU,CPU'
        else:
            result.append(first_td)
    return result


def validate_supported_values(detected_list, supported_list):
    supported_list += ALL_AVAILABLE_OPTIONS  # 'starred expression' will be evaluated during pytest_configure
    check = all(_elem in supported_list for _elem in detected_list)
    assert check, f"Not supported target devices in {detected_list}"
    return detected_list


def get_target_devices():
    """ Convert comma separated string of devices into list """
    target_devices_list = get_list("TT_TARGET_DEVICE", fallback=[TargetDevice.CPU])
    separator_multi = ":"
    if any(separator_multi in _target_device for _target_device in target_devices_list):
        target_devices_list = get_multi_target_devices(target_devices_list, separator_multi)
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

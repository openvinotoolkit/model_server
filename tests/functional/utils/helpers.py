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
import pytest
from typing import Any
from config import target_device

from functional.constants import NOT_TO_BE_REPORTED_IF_SKIPPED


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


def get_int(key_name, fallback=None):
    value = os.environ.get(key_name, fallback)
    if value != fallback:
        try:
            value = int(value)
        except ValueError:
            raise ValueError("Value '{}' of {} env variable cannot be cast to int.".format(value, key_name))
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


def get_xdist_worker_count():
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))


def get_xdist_worker_nr():
    xdist_current_worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    if xdist_current_worker == "master":
        xdist_current_worker = 0
    else:
        xdist_current_worker = int(xdist_current_worker.lstrip("gw"))
    return xdist_current_worker

def devices_not_supported_for_test(*not_supported_devices_list):
    """
    Comma separated list of devices not supported for test.
    Use as a test decorator.
    Example use:
    @devices_not_supported_for_test("CPU", "GPU")
    def test_example():
        # test implementation
    """
    return pytest.mark.skipif(target_device in not_supported_devices_list,
                              reason=NOT_TO_BE_REPORTED_IF_SKIPPED)

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

import inspect
import json
import os
import time

from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from filelock import UnixFileLock, WindowsFileLock
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tests.functional.constants.os_type import get_host_os, OsType


def is_ancestor(obj, ancestor):
    if not (inspect.isclass(obj) and obj.__bases__):
        return False
    if ancestor in obj.__bases__:   # child
        return True
    return any(filter(lambda x: is_ancestor(x, ancestor), obj.__bases__))    # check if grandchild ...


def get_children_from_module(parent, module):
    members = inspect.getmembers(module)  # [(name, class_def), ...]
    children = list(filter(lambda x: is_ancestor(x[1], parent), members))
    return children


def get_token_value(token_file_path, fallback_value=None):
    if os.path.exists(token_file_path):
        token_value = Path(token_file_path).read_text().strip()
        return token_value
    return fallback_value


def get_username():
    try:
        user_name = os.getlogin()
    except OSError as e:
        user = os.environ.get("USER", "not_known_user")
        logname = os.environ.get("LOGNAME", user)
        user_name = os.environ.get("USERNAME", logname)
    return user_name


def wait_until_file_exists(path, timeout=60):
    deadline = datetime.now() + timedelta(seconds=timeout)
    path = Path(path)
    while not path.exists() and datetime.now() < deadline:
        time.sleep(1)
    assert path.exists(), f"File do not exist {str(path)}"


class SelfDeletingCommonFileLock:

    def __init__(self, lock_file, self_delete=False, **kwargs):
        super().__init__(lock_file, **kwargs)
        self.self_delete = self_delete

    def acquire(self, **kwargs):
        Path(self.lock_file).parent.mkdir(parents=True, exist_ok=True)  # Create dir if not existing
        super().acquire(**kwargs)

    def acquire_no_raise(self, timeout):
        try:
            self.acquire(timeout=timeout)
        except TimeoutError as e:
            return False
        return True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        super().__exit__(exc_type, exc_value, exc_traceback)
        if self.self_delete:
            Path(self.lock_file).unlink(missing_ok=True)


class SelfDeletingWindowsFileLock(SelfDeletingCommonFileLock, WindowsFileLock):
    pass


class SelfDeletingUnixFileLock(SelfDeletingCommonFileLock, UnixFileLock):
    pass

    def acquire(self, **kwargs):
        Path(self.lock_file).parent.mkdir(parents=True, exist_ok=True)  # Create dir if not existing
        super().acquire(**kwargs)

    def acquire_no_raise(self, timeout):
        try:
            self.acquire(timeout=timeout)
        except TimeoutError as e:
            return False
        return True


SelfDeletingFileLock = SelfDeletingWindowsFileLock if get_host_os() == OsType.Windows else SelfDeletingUnixFileLock


class TmpDir(str):
    """Creates safe temp dir"""
    TMP_DIR = None
    SEPARATOR = "_"

    def __new__(cls, temp_dir: str = None) -> str:
        if cls.TMP_DIR is None:
            cls.TMP_DIR = TemporaryDirectory(
                prefix=datetime.now().strftime('%Y%m%d_%H%M%S_%f') + cls.SEPARATOR,
                suffix=cls.SEPARATOR + get_username(),
                dir=temp_dir)
        return cls.TMP_DIR.name


class NamedSingletonMeta(type):
    """
    Metaclass for defining Named Singleton Classes (extension for Singleton Classes)

    src:
    https://www.datacamp.com/community/tutorials/python-metaclasses

    Singleton Design using a Metaclass

    This is a design pattern that restricts the instantiation of a class to only one object.
    This could prove useful for example when designing a class to connect to the database.
    One might want to have just one instance of the connection class.
    """
    _instances = defaultdict(dict)

    def __call__(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        name = name.lower()
        if name not in cls._instances[cls]:
            cls._instances[cls][name] = super().__call__(name, *args, **kwargs)
        return cls._instances[cls][name]


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_str'):
            return obj.to_str()
        elif isinstance(obj, type):
            return str(obj)
        else:
            return json.JSONEncoder.default(self, obj)

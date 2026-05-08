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

import platform
import re

import psutil

from tests.functional.utils.logger import get_logger
from tests.functional.utils.process import Process
from tests.functional import config

logger = get_logger(__name__)

DEFAULT_BUILD_NUMBER = 0
DEFAULT_SHORT_VERSION_NUMBER = "0.0.0"
DEFAULT_FULL_VERSION_NUMBER = f"{DEFAULT_SHORT_VERSION_NUMBER}-{config.product_version_suffix}-{DEFAULT_BUILD_NUMBER}"


class BaseInfo:
    """Retrieves environment info"""
    glob_version = None
    glob_os_distname = None

    def __init__(self, image=None):
        self.image = image

    @property
    def version(self):
        """Retrieves version, but only once.

        If retrieval doesn't work, default version is returned.
        """
        if self.glob_version is None:

            self.glob_version = self.get()
            self.glob_version = \
                self.glob_version["version"]

        return self.glob_version

    @property
    def os_distname(self):
        """Retrieves os distname, but only once."""
        if self.glob_os_distname is None:
            self.glob_os_distname = CurrentOsInfo.get_os_distname()

        return self.glob_os_distname

    @classmethod
    def get(cls):
        """
        Returns constant environment info.
        """
        logger.info("BASIC INFO WITHOUT ANY API CALL")
        return {"version": DEFAULT_FULL_VERSION_NUMBER}


class EnvironmentInfo(object):
    _instances = {}

    @classmethod
    def get_instance(cls, class_info, image):
        idx = class_info, image
        if idx not in cls._instances:
            cls._instances[idx] = cls(class_info, image)
        return cls._instances[idx]

    def __init__(self, class_info=BaseInfo, image=None):
        """Stores details about environment such as build number, version number
        and allows their retrieval"""
        self.env_info = class_info(image)

    def get_build_number(self):
        """Retrieves build number (OVMS/OV/OV GenAI versions) from the environment info"""
        if config.product_build_number_from_env:
            return self._retrieve_build_number_from_environment()
        if config.product_build_number:
            return config.product_build_number
        return DEFAULT_BUILD_NUMBER

    def get_version_number(self):
        """Retrieves version number (OVMS version) from the environment info"""
        if config.product_version_number_from_env:
            return self._retrieve_version_number_from_environment()
        if config.product_version:
            return config.product_version
        return DEFAULT_FULL_VERSION_NUMBER

    @classmethod
    def get_environment_name(cls):
        """Retrieves the environment name that will be reported for a test run"""
        return config.environment_name

    def get_os_distname(self):
        """Retrieves the operating system distribution name"""
        return self.env_info.os_distname

    def _retrieve_version_number_from_environment(self):
        return self._version_number_from_environment_version(self.env_info.version)

    def _retrieve_build_number_from_environment(self):
        return self._build_number_from_environment_version(self.env_info.version)

    @classmethod
    def _build_number_from_environment_version(cls, environment_version):
        return environment_version

    @classmethod
    def _version_number_from_environment_version(cls, environment_version):
        return environment_version.split("_")[0]


class CurrentOsInfo:
    LINUX = "Linux"

    UBUNTU = "Ubuntu"
    REDHAT = "RedHat"

    """Returns the current system/OS name """
    @staticmethod
    def get_os_name():
        return platform.system()

    @staticmethod
    def get_os_distname():
        if CurrentOsInfo.LINUX == platform.system():
            if CurrentOsInfo.UBUNTU.lower() in platform.version().lower():
                return CurrentOsInfo.UBUNTU
            elif CurrentOsInfo.REDHAT.lower() in platform.version().lower():
                return CurrentOsInfo.REDHAT
        return platform.system()

    @staticmethod
    def get_os_distversion():
        if CurrentOsInfo.LINUX == platform.system():
            proc = Process()
            proc.disable_check_stderr()
            _, stdout, _ = proc.run_and_check_return_all("cat /etc/os-release")
            os_version_match = re.search(r"VERSION_ID=\"(.+)\"", stdout)
            assert os_version_match, "Unable to detect OS version"
            return os_version_match.group(1)
        return platform.version()

    @staticmethod
    def get_cpu_amount():
        return psutil.cpu_count()

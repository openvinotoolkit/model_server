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
import distro


class OsType:
    Redhat = "redhat"
    Ubuntu22 = "ubuntu22"
    Ubuntu24 = "ubuntu24"
    Windows = "windows"


UBUNTU = "ubuntu"
UBUNTU_22_OS_VERSION = "22.04"
UBUNTU_24_OS_VERSION = "24.04"
WINDOWS_MSYS_NT = "msys_nt"


def get_host_os():
    # pylint: disable=no-else-return
    result = distro.id() if distro.id() else platform.system()
    if result == UBUNTU:
        ubuntu_version = get_host_os_version()
        if ubuntu_version == UBUNTU_22_OS_VERSION:
            return OsType.Ubuntu22
        elif ubuntu_version == UBUNTU_24_OS_VERSION:
            return OsType.Ubuntu24
    elif result == "rhel":
        return OsType.Redhat
    elif result.lower() == OsType.Windows:
        return OsType.Windows
    elif WINDOWS_MSYS_NT in result.lower():
        return OsType.Windows
    raise NotImplementedError(f"OS not supported: {result}")


def get_host_os_version():
    result = distro.version()
    return result


def get_host_os_details():
    if distro.lsb_release_info():
        return distro.lsb_release_info().get("description", None)
    return platform.platform()

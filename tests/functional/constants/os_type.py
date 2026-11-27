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
    # "redhat" is the canonical value for the default Red Hat UBI 9.x image.
    # "redhat9" is accepted as a transparent input alias (normalized to "redhat" via normalize_redhat_os()).
    Redhat = "redhat"
    Redhat9 = "redhat9"
    Redhat10 = "redhat10"
    Ubuntu22 = "ubuntu22"
    Ubuntu24 = "ubuntu24"
    Ubuntu26 = "ubuntu26"
    Windows = "windows"


UBUNTU = "ubuntu"
UBUNTU_22_OS_VERSION = "22.04"
UBUNTU_24_OS_VERSION = "24.04"
UBUNTU_26_OS_VERSION = "26.04"
REDHAT_10_OS_MAJOR_VERSION = "10"
WINDOWS_MSYS_NT = "msys_nt"


def normalize_redhat_os(base_os):
    """Transparently map the 'redhat9' input alias to the canonical 'redhat' value.

    'redhat10' is a distinct, genuinely different OS and is left untouched.
    Should be applied as early as possible (e.g. right after reading TT_BASE_OS/BASE_OS),
    so that no other code needs to be aware of the 'redhat9' spelling at all.
    """
    return OsType.Redhat if base_os == OsType.Redhat9 else base_os


def get_host_os():
    # pylint: disable=no-else-return
    result = distro.id() if distro.id() else platform.system()
    if result == UBUNTU:
        ubuntu_version = get_host_os_version()
        if ubuntu_version == UBUNTU_22_OS_VERSION:
            return OsType.Ubuntu22
        elif ubuntu_version == UBUNTU_24_OS_VERSION:
            return OsType.Ubuntu24
        elif ubuntu_version == UBUNTU_26_OS_VERSION:
            return OsType.Ubuntu26
    elif result == "rhel":
        rhel_version = get_host_os_version()
        if rhel_version.split(".")[0] == REDHAT_10_OS_MAJOR_VERSION:
            return OsType.Redhat10
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

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

try:
    from grp import getgrnam
except ImportError:
    getgrnam = None

try:
    from os import getuid
except ImportError:
    getuid = None

from tests.functional.constants.os_type import get_host_os
from tests.functional.utils.test_framework import FrameworkMessages, get_parameter_from_item, skip_if

from tests.functional.config import skip_nginx_test
from tests.functional.constants.target_device import TargetDevice
from tests.functional.constants.ovms import BASE_OS_PARAM_NAME

NETWORK = "network"
PRIVILEGED = "privileged"
VOLUMES = "volumes"
DEVICES = "devices"
HOST = "host"
DOCKER_PARAMS = "docker_params"
""" TARGET_DEVICE_CONFIGURATION: VOLUMES - this map stores a list of devices that should be
                           mounted for given target device.
                           String representing device should be in form of:
                           - <path_on_host>:<path_in_container>:<cgroup_permissions>

                           Another representations are possible
                           - <path_on_host>:<path_in_container>
                             cgroup_permissions will be set to `mrw` by default

                           - <path_on_host>
                             path_in_container will be the same as path_on_host
                             cgroup_permissions will be set to `mrw`
                           """
""" TARGET_DEVICE_CONFIGURATION: NETWORK - Name of the network this container will be connected to at creation time. 
                             :type str
                           """
""" TARGET_DEVICE_CONFIGURATION: USER - Username or UID to run commands as inside the container.
                          :type int or str
                           """
""" TARGET_DEVICE_CONFIGURATION: PRIVILEGED - Give extended privileges to this container.
                                :type bool
                           """
# Currently docker imports are mandatory (even for non-docker types) and this enforce getuid() & getgrnam(...) syscalls
# for non-docker testruns.
TARGET_DEVICE_CONFIGURATION = {
    TargetDevice.CPU: lambda: {
        VOLUMES: [],
        DEVICES: [],
        NETWORK: None,
        PRIVILEGED: False,
    },
    TargetDevice.GPU: lambda: {
        VOLUMES: [],
        DEVICES: ["/dev/dri:/dev/dri:mrw"],
        NETWORK: None,
        PRIVILEGED: False,
        DOCKER_PARAMS: {"group_add": [getgrnam("render").gr_gid]},
    },
    TargetDevice.NPU: lambda: {
        VOLUMES: [],
        DEVICES: ["/dev/accel:/dev/accel:mrw", "/dev/dri:/dev/dri:mrw"],
        NETWORK: None,
        PRIVILEGED: False,
        DOCKER_PARAMS: {"group_add": [getgrnam("render").gr_gid]},
    },
    TargetDevice.AUTO: lambda: {
        VOLUMES: [],
        DEVICES: ["/dev/dri"],
        NETWORK: None,
        PRIVILEGED: False,
        DOCKER_PARAMS: {"group_add": [getgrnam("render").gr_gid]},
    },
    TargetDevice.HETERO: lambda: {
        VOLUMES: [],
        DEVICES: ["/dev/dri:/dev/dri:mrw"],
        NETWORK: None,
        PRIVILEGED: False,
        DOCKER_PARAMS: {"group_add": [getgrnam("render").gr_gid]},
    },
}


def nginx_mtls_not_supported_for_test():
    """
    Test or test class should be skipped from execution since is not supported by
    nginx_mtls images.
    """
    return skip_if(skip_nginx_test, msg=FrameworkMessages.NGINX_IMAGE_NOT_SUPPORTED)


def deselect_if_host_os_not_match_docker_base_image_runtime(item):
    should_skip = get_host_os() != get_parameter_from_item(item, BASE_OS_PARAM_NAME)
    return should_skip

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
import shutil

from pathlib import Path

from tests.functional.utils.logger import get_logger
from tests.functional.utils.test_framework import generate_test_object_name
from tests.functional.constants.os_type import OsType
from tests.functional.utils.process import Process

from tests.functional.config import ovms_c_release_artifacts_path
from tests.functional.constants.ovms import Ovms
from tests.functional.constants.ovms_type import OvmsType
from tests.functional.constants.paths import Paths


logger = get_logger(__name__)


def calculate_ovms_binary_name(base_os=OsType.Ubuntu22):
    if len(ovms_c_release_artifacts_path) == 1:
        return ovms_c_release_artifacts_path[0]
    for binary_name in ovms_c_release_artifacts_path:
        if base_os in binary_name:
            return binary_name
    return None


def get_binaries(base_os, test_object_name, tmp_dir):
    """
    Run dummy ovms instance just for copying docker resources to temporary directory
    """
    proc = Process()
    proc.disable_check_stderr()
    package_content_path = Path(Paths.CAPI_WRAPPER_PACKAGE_CONTENT_PATH(base_os))
    test_object_name = test_object_name if test_object_name is not None else generate_test_object_name()
    resource_dir = Path(tmp_dir, test_object_name)
    resource_dir.mkdir(exist_ok=True, parents=True)
    ovms_binary_path, _ = Ovms.get_ovms_binary_paths(OvmsType.BINARY, base_os)
    resource_dir_ovms = Path(resource_dir, "ovms")
    if not os.path.exists(resource_dir_ovms):
        if base_os == OsType.Windows:
            shutil.copytree(package_content_path, resource_dir_ovms)
        else:
            os.symlink(package_content_path, resource_dir_ovms)
    return os.path.join(resource_dir, ovms_binary_path), test_object_name


def get_ovms_binary_cmd_setup(base_os=None, resources_dir_path=None, environment=None, venv_activate_path=None):
    env = {} if environment is None else environment
    if base_os == OsType.Windows:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        pre_cmd = f"{resources_dir_path}\\setupvars.bat && "
        if venv_activate_path:
            pre_cmd += f"{venv_activate_path} && "
    else:
        # required for GPU
        env_neo_read_debug_keys = os.environ.get("NEOReadDebugKeys", None)
        if env_neo_read_debug_keys is not None:
            env["NEOReadDebugKeys"] = env_neo_read_debug_keys
        env_override_gpu_adress_space = os.environ.get("OverrideGpuAddressSpace", None)
        if env_override_gpu_adress_space is not None:
            env["OverrideGpuAddressSpace"] = env_override_gpu_adress_space
        # required for git/git-lfs
        env["PATH"] = f"{os.environ['PATH']}:{os.path.join(resources_dir_path, 'bin')}"
        pre_cmd = "LD_LIBRARY_PATH=${PWD}/ovms/lib PYTHONPATH=${PWD}/ovms/lib/python/ "
    logger.debug(f"Binary environment: {env}")
    return pre_cmd, env

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
from dataclasses import dataclass
from pathlib import Path

from ovms.config import ovms_c_repo_path
from tests.functional.config import tmp_dir
from tests.functional.constants.paths import Paths


@dataclass
class CpuExtension:
    base_os: str = ""
    host_lib_path: str = ""
    lib_path: str = ""
    HELLO_WORLD_MESSAGE: str = ""

    def __post_init__(self):
        if not self.base_os:
            self.base_os = CpuExtension.base_os


@dataclass
class MuseModelExtension(CpuExtension):
    lib_name: str = "libopenvino_tokenizers.so"
    lib_path: str = f"/ovms/lib/{lib_name}"


@dataclass
class InvalidCpuExtension(CpuExtension):
    lib_name: str = "cpu_extension.lib"
    lib_path: str = "/path/to/non/existent/cpu_extension.lib"


@dataclass
class BuildableCpuExtension(CpuExtension):
    make_dir: str = ""
    host_lib_path: str = ""


@dataclass
class OvmsCBuildableCpuExtension(BuildableCpuExtension):
    make_dir: str = Path(ovms_c_repo_path)


@dataclass
class SimpleReluCpuExtension(OvmsCBuildableCpuExtension):
    extension_name: str = "SampleCpuExtension"
    lib_name: str = "libcustom_relu_cpu_extension.so"
    HELLO_WORLD_MESSAGE: str = "Running Relu custom kernel for the first time (next messages won't be printed)"

    def __post_init__(self):
        super().__post_init__()
        self.cpu_extension_src_dir = Path(self.make_dir, "src", "example", self.extension_name)
        self.cpu_extension_dst_host_path = os.path.join(tmp_dir, self.extension_name)
        self.lib_path = os.path.join(Paths.ROOT_PATH_CPU_EXTENSIONS, self.extension_name, self.lib_name)
        self.host_lib_path = os.path.join(tmp_dir, self.extension_name, self.lib_name)


@dataclass
class OvmsTestBuildableCpuExtension(BuildableCpuExtension):
    make_dir: str = ""
    lib_name: str = ""
    extension_name: str = ""
    cpp_file: str = ""

    def __post_init__(self):
        self.ovms_c_cpu_ext = SimpleReluCpuExtension()
        super().__post_init__()
        self.make_dir = Path(tmp_dir, self.extension_name)


@dataclass
class CorruptedLibCpuExtension(OvmsTestBuildableCpuExtension):
    extension_name: str = "corrupted_lib"
    cpp_file: str = "CorruptedLib.cpp"
    lib_name: str = "libcorrupted_lib_cpu_extension.so"

    def __post_init__(self):
        super().__post_init__()
        self.host_lib_path = os.path.join(tmp_dir, self.extension_name, self.lib_name)
        self.lib_path = os.path.join(Paths.ROOT_PATH_CPU_EXTENSIONS, self.extension_name, self.lib_name)


@dataclass
class ThrowExceptionCpuExtension(OvmsTestBuildableCpuExtension):
    extension_name: str = "throw_exceptions"
    cpp_file: str = "ThrowExceptions.cpp"
    lib_name: str = "libthrow_exception_cpu_extension.so"
    HELLO_WORLD_MESSAGE: str = "Executing ThrowExceptions evaluate()"

    def __post_init__(self):
        super().__post_init__()
        self.host_lib_path = os.path.join(tmp_dir, self.extension_name, self.lib_name)
        self.lib_path = os.path.join(Paths.ROOT_PATH_CPU_EXTENSIONS, self.extension_name, self.lib_name)

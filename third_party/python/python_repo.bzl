#
# Copyright (c) 2023 Intel Corporation
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

def _python_repository_impl(repository_ctx):
    base_os_image = repository_ctx.os.environ.get("BASE_IMAGE", "")
    version = None
    lib_path = None

    if "ubuntu" in base_os_image:
        lib_path = "lib/x86_64-linux-gnu"
        if base_os_image == "ubuntu:20.04" or base_os_image == "docker.io/nvidia/cuda:11.8.0-runtime-ubuntu20.04":
            version = "3.8"
        else:
            version = "3.10"
    else: # for redhat
        lib_path = "lib64"
        version = "3.9"

    build_file_content = """
cc_library(
    name = "python3-lib",
    srcs = ["{lib_path}/libpython{version}.so"],
    hdrs = glob(["include/python{version}/*.h"]),
    includes = ["include/python{version}"],
    visibility = ["//visibility:public"]
)"""
    repository_ctx.file("BUILD", build_file_content.format(version=version, lib_path=lib_path))

python_repository = repository_rule(
    implementation = _python_repository_impl,
    environ = ["BASE_IMAGE"],
    local=False,
)

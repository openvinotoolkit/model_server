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

load("@rules_python//python:py_runtime.bzl", "py_runtime")
load("@rules_python//python:py_runtime_pair.bzl", "py_runtime_pair")

# System interpreter, matching the ABI of pyovms.so and @pip_deps wheels.
py_runtime(
    name = "system_python3_runtime",
    interpreter_path = "/usr/bin/python3",
    python_version = "PY3",
)

py_runtime_pair(
    name = "system_python_runtime_pair",
    py3_runtime = ":system_python3_runtime",
)

toolchain(
    name = "system_python_toolchain",
    toolchain = ":system_python_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)

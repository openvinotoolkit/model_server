#
# Copyright (c) 2024 Intel Corporation
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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")
load("//:common_settings.bzl",
     "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS", "COMMON_LOCAL_DEFINES", "PYBIND_DEPS")

COPTS_ADJUSTED = COMMON_STATIC_LIBS_COPTS + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_COPTS,
}) + select({
        "//conditions:default": ["-DPYTHON_DISABLE=1"],
        "//:not_disable_python" : ["-DPYTHON_DISABLE=0"],
}) + select({
        "//conditions:default": ["-DMEDIAPIPE_DISABLE=1"],
        "//:not_disable_mediapipe" : ["-DMEDIAPIPE_DISABLE=0"],
})

LINKOPTS_ADJUSTED = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})

load("@pip_deps//:requirements.bzl", "all_requirements")

pybind_extension(
    name = "pyovms",
    srcs = ["ovms_py_tensor.cpp",], # TODO consider naming convention if exist
    deps = ["//src/python:ovmspytensor",],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

load("@aspect_bazel_lib//:e2e/copy_action/copy.bzl", "simple_copy_file")
simple_copy_file(
    name = "copy_pyovms",
    src = "pyovms.so",
    out = "pyovms.pyd",
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_python_binding",
    srcs = glob(["tests/*.py"]),
    main = "tests/run.py",
    imports=["."],
    data = [":pyovms.so"],
    deps = all_requirements
)


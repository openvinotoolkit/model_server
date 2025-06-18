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
load("//:common_settings.bzl", "PYBIND_DEPS", "ovms_cc_library")

mediapipe_proto_library(
    name = "pythonexecutorcalculator_proto", # pythonexecutorcalculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["python_executor_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "pytensorovtensorconvertercalculator_proto", # pytensorovtensorconvertercalculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["pytensor_ovtensor_converter_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
            "@mediapipe//mediapipe/framework:calculator_options_proto",
            "@mediapipe//mediapipe/framework:calculator_proto",
    ]
)

ovms_cc_library(
    name = "utils",
    hdrs = ["utils.hpp",],
    srcs = [],
    deps = PYBIND_DEPS + [
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "pythonbackend",
    hdrs = ["python_backend.hpp",],
    srcs = ["python_backend.cpp",],
    deps = PYBIND_DEPS + [
        "//src:libovmslogging",
        "utils",
    ],
    visibility = ["//visibility:private"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "pythonnoderesources",
    hdrs = ["pythonnoderesources.hpp",],
    srcs = ["pythonnoderesources.cpp",],
    deps = PYBIND_DEPS + [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovmsmediapipe_utils",
        "pythonexecutorcalculator_cc_proto",
        "utils",
    ],
    visibility = ["//visibility:private"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "pytensorovtensorconvertercalculator",
    srcs = ["pytensor_ovtensor_converter_calculator.cc",],
    deps = PYBIND_DEPS + [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src:libovmsprecision",
        "ovmspytensor",
        "pytensorovtensorconvertercalculator_cc_proto",
        "pythonbackend",
    ],
    visibility = ["//visibility:private"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "pythonexecutorcalculator",
    srcs = ["python_executor_calculator.cc",],
    deps = PYBIND_DEPS + [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "ovmspytensor",
        "pythonexecutorcalculator_cc_proto",
        "pythonbackend",
        "pythonnoderesources",
    ],
    visibility = ["//visibility:private"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "ovmspytensor",
    hdrs = ["ovms_py_tensor.hpp",],
    srcs = ["ovms_py_tensor.cpp",],
    deps = PYBIND_DEPS,
    include_prefix = "binding/python",
    visibility = ["//visibility:public"],
    alwayslink = 1,
    #data = [":pyovms.so"], #todo split to header?
)

ovms_cc_library(
    name = "libovmspythonmodule",
    hdrs = ["pythoninterpretermodule.hpp",],
    srcs = ["pythoninterpretermodule.cpp",],
    deps = PYBIND_DEPS + [
        "//src:cpp_headers", # TODO make ovmslib depend on that
        "//src:libovmslogging",
        "//src:libovms_module",
        "//src:libovmsmediapipe_utils",
        "ovmspytensor",
        "pytensorovtensorconvertercalculator",
        "pythonbackend",
        "pythonexecutorcalculator",
        "utils",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

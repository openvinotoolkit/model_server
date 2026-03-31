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
     "COMMON_STATIC_LIBS_LINKOPTS",
     "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS",
     "COMMON_LOCAL_DEFINES", "PYBIND_DEPS", "ovms_cc_library")

# Copts for the python shared library.  Same hardening flags as
# LINUX_COMMON_STATIC_LIBS_COPTS but WITHOUT -fvisibility=hidden.
# Omitting -fvisibility=hidden lets entry-point symbols
# (PythonInterpreterModule, etc.) remain visible in the ELF dynamic
# symbol table, and avoids a -Werror=attributes conflict that arises
# when OvmsPyTensor (default visibility) has a pybind11::object field
# (hidden visibility).
_SHARED_LIB_COPTS_LINUX = [
    "-Wall",
    "-Wno-unknown-pragmas",
    "-Wno-sign-compare",
    # -fvisibility=hidden intentionally omitted
    "-Werror",
    "-Wno-deprecated-declarations",
    "-Wimplicit-fallthrough",
    "-fcf-protection=full",
    "-Wformat",
    "-Wformat-security",
    "-Werror=format-security",
    "-Wl,-z,noexecstack",
    "-fPIC",
    "-Wl,-z,relro",
    "-Wl,-z,relro,-z,now",
    "-Wl,-z,nodlopen",
    "-fstack-protector-strong",
    # pybind11 declares its entire namespace with
    # __attribute__((visibility("hidden"))), so any struct that contains a
    # pybind11 type (e.g. py::object) will trigger -Wattributes when the
    # containing struct has default visibility.  This is expected and safe
    # for code compiled into a single DSO — suppress the diagnostic here.
    "-Wno-attributes",
]

COPTS_SO = select({
    "//conditions:default": _SHARED_LIB_COPTS_LINUX,
    "//src:windows": [],  # Windows builds do not produce this .so
}) + select({
    "//conditions:default": ["-DPYTHON_DISABLE=1"],
    "//:not_disable_python": ["-DPYTHON_DISABLE=0"],
}) + select({
    "//conditions:default": ["-DMEDIAPIPE_DISABLE=1"],
    "//:not_disable_mediapipe": ["-DMEDIAPIPE_DISABLE=0"],
}) + select({
    "//conditions:default": [],
    "//:fuzzer_build": COMMON_FUZZER_COPTS,
})

LINKOPTS_SO = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build": COMMON_FUZZER_LINKOPTS,
})

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

# Shared library built from all Python binding sources.
# Deps whose symbols are compiled with -fvisibility=hidden (the project default)
# are linked into the .so with hidden visibility; only the python module
# interface symbols compiled via srcs/COPTS_SO are exported.
cc_binary(
    name = "libovmspython.so",
    linkshared = True,
    srcs = [
        "ovms_py_tensor.cpp",
        "ovms_py_tensor.hpp",
        "python_backend.cpp",
        "python_backend.hpp",
        "pythonnoderesources.cpp",
        "pythonnoderesources.hpp",
        "pytensor_ovtensor_converter_calculator.cc",
        "python_executor_calculator.cc",
        "pythoninterpretermodule.cpp",
        "pythoninterpretermodule.hpp",
        "utils.hpp",
    ],
    deps = PYBIND_DEPS + [
        ":pythonexecutorcalculator_cc_proto",
        ":pytensorovtensorconvertercalculator_cc_proto",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//third_party:openvino",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovms_module",
        "//src:libovmsmediapipe_utils",
        "//src:libovmsprecision",
        "//src:cpp_headers",
    ],
    copts = COPTS_SO,
    linkopts = LINKOPTS_SO,
    local_defines = COMMON_LOCAL_DEFINES,
    visibility = ["//visibility:public"],
)

# cc_import wrapper so that other targets can depend on the shared library
# the same way they previously depended on :libovmspythonmodule.
cc_import(
    name = "libovmspython",
    shared_library = ":libovmspython.so",
    hdrs = [
        "ovms_py_tensor.hpp",
        "python_backend.hpp",
        "pythonnoderesources.hpp",
        "pythoninterpretermodule.hpp",
        "utils.hpp",
    ],
    visibility = ["//visibility:public"],
)

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
load("@aspect_bazel_lib//:e2e/copy_action/copy.bzl", "simple_copy_file")
load("//:common_settings.bzl",
     "COMMON_STATIC_LIBS_LINKOPTS",
     "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS",
    "COMMON_LOCAL_DEFINES", "PYBIND_HEADERS_ONLY", "PYBIND_DEPS", "ovms_cc_library")

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

_SHARED_LIB_COPTS_WINDOWS = [
    "/guard:cf",
    "/W4",
    "/WX",
    "/external:anglebrackets",
    "/external:W0",
    "/sdl",
    "/analyze",
    "/Gy",
    "/GS",
    "/DYNAMICBASE",
    "/Qspectre",
    "/wd4305",
    "/wd4324",
    "/wd4068",
    "/wd4458",
    "/wd4100",
    "/wd4389",
    "/wd4127",
    "/wd4673",
    "/wd4670",
    "/wd4244",
    "/wd4297",
    "/wd4702",
    "/wd4267",
    "/wd4996",
    "/wd6240",
    "/wd6326",
    "/wd6385",
    "/wd6294",
    "/wd4005",  # Suppress macro redefinition warning (GLOG_NO_ABBREVIATED_SEVERITIES from mediapipe/glog)
    "/wd4309",  # Suppress truncation of constant value warning (mediapipe circular_buffer.h)
    "/wd4456",  # Suppress variable shadowing warning (mediapipe graph_profiler.h)
    "/wd6001",  # Suppress uninitialized memory warnings (absl/strings and mediapipe)
    "/wd6246",  # Suppress local declaration hides previous declaration (mediapipe graph_profiler.h)
    "/wd6386",  # Suppress buffer overrun warnings (absl/strings/internal/cord_rep_flat.h)
    "/wd6011",  # Suppress NULL pointer dereference warnings (mediapipe packet.h)
    "/guard:cf",
    "/utf-8",
]

COPTS_SO = select({
    "//conditions:default": _SHARED_LIB_COPTS_LINUX,
    "//src:windows": _SHARED_LIB_COPTS_WINDOWS,
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
    deps = PYBIND_HEADERS_ONLY + [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src/mediapipe_internal:mediapipe_utils",
        "pythonexecutorcalculator_cc_proto",
        "utils",
    ],
    visibility = ["//src:__pkg__"],
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "pytensorovtensorconvertercalculator",
    srcs = ["pytensor_ovtensor_converter_calculator.cc",],
    deps = PYBIND_HEADERS_ONLY + [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src:libovmsprecision",
        "ovmspytensor",
        "pytensorovtensorconvertercalculator_cc_proto",
        "pythonbackend",
    ],
    visibility = ["//src:__pkg__"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "pythonexecutorcalculator",
    srcs = ["python_executor_calculator.cc",
            "python_node_initializer.cpp",],
    deps = PYBIND_HEADERS_ONLY + [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "ovmspytensor",
        "pythonexecutorcalculator_cc_proto",
        "pythonbackend",
        "pythonnoderesources",
        "//src/mediapipe_internal:node_initializer",
        "//src:libovmslogging",
    ],
    visibility = ["//src:__pkg__"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

ovms_cc_library(
    name = "ovmspytensor",
    hdrs = ["ovms_py_tensor.hpp",],
    srcs = ["ovms_py_tensor.cpp",],
    deps = PYBIND_HEADERS_ONLY,
    include_prefix = "binding/python",
    visibility = ["//visibility:public"],
    alwayslink = 1,
    #data = [":pyovms.so"], #todo split to header?
)

ovms_cc_library(
    name = "python_headers_only",
    hdrs = [
        "ovms_py_tensor.hpp",
        "python_backend.hpp",
        "pythonnoderesources.hpp",
        "utils.hpp",
    ],
    deps = PYBIND_HEADERS_ONLY + [
        "//src:cpp_headers",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src/mediapipe_internal:mediapipe_utils",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "pythonexecutorcalculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "python_runtime_module_api_hdrs",
    hdrs = ["python_runtime_module_api.hpp"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "python_runtime_env",
    hdrs = ["python_runtime_env.hpp"],
    srcs = ["python_runtime_env.cpp"],
    deps = [
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovmspythonmodule",
    hdrs = [
        "pythoninterpretermodule.hpp",
        "python_runtime_module_api.hpp",
    ],
    srcs = ["pythoninterpretermodule.cpp",],
    deps = PYBIND_DEPS + [
        "//src:cpp_headers", # TODO make ovmslib depend on that
        "//src:libovmslogging",
        "//src:libovms_module",
        "//src:python_calculators_plugin_loader",
        "//src/mediapipe_internal:mediapipe_utils",
        "pythonbackend",
        "utils",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

# Runtime C entry symbols as a static library.
# This allows binaries like ovms_test to provide
# OVMS_createPythonInterpreterModule/OVMS_validatePythonEnvironment
# in-process, avoiding a second libovmspython.so image in the same process.
ovms_cc_library(
    name = "python_runtime_entry_static",
    srcs = ["python_runtime_entry.cpp"],
    deps = PYBIND_DEPS + [
        "libovmspythonmodule",
        "python_runtime_env",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovms_module",
        "//src:cpp_headers",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    data = ["//src/python/binding:pyovms.so"],
)

# Lightweight target for tests that need direct PythonInterpreterModule usage
# without pulling MediaPipe calculator registration code.
ovms_cc_library(
    name = "pythoninterpretermodule_runtime",
    hdrs = [
        "pythoninterpretermodule.hpp",
        "python_runtime_module_api.hpp",
    ],
    srcs = ["pythoninterpretermodule.cpp",],
    deps = PYBIND_DEPS + [
        "//src:cpp_headers",
        "//src:libovmslogging",
        "//src:libovms_module",
        "pythonbackend",
    ],
    visibility = ["//src:__pkg__"],
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
        "pythoninterpretermodule.cpp",
        "pythoninterpretermodule.hpp",
        "python_runtime_module_api.hpp",
        "python_runtime_entry.cpp",
        "utils.hpp",
    ],
    deps = PYBIND_DEPS + [
        "python_runtime_env",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovms_module",
        "//src:cpp_headers",
        "//src:python_calculators_plugin_loader",
    ],
    copts = COPTS_SO,
    linkopts = LINKOPTS_SO,
    local_defines = COMMON_LOCAL_DEFINES,
    visibility = ["//visibility:public"],
)

simple_copy_file(
    name = "copy_libovmspython",
    src = "libovmspython.so",
    out = "libovmspython.dll",
    visibility = ["//visibility:public"],
)

# cc_import wrapper so that other targets can depend on the shared library
# the same way they previously depended on :libovmspythonmodule.
cc_import(
    name = "libovmspython_unix",
    shared_library = ":libovmspython.so",
    hdrs = [
        "ovms_py_tensor.hpp",
        "python_backend.hpp",
        "pythoninterpretermodule.hpp",
        "python_runtime_module_api.hpp",
        "utils.hpp",
    ],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "libovmspython_windows",
    interface_library = ":libovmspython.so.if.lib",
    shared_library = ":copy_libovmspython",
    hdrs = [
        "ovms_py_tensor.hpp",
        "python_backend.hpp",
        "pythoninterpretermodule.hpp",
        "python_runtime_module_api.hpp",
        "utils.hpp",
    ],
    visibility = ["//visibility:private"],
)

alias(
    name = "libovmspython_select",
    actual = select({
        "//src:windows": ":libovmspython_windows",
        "//conditions:default": ":libovmspython_unix",
    }),
    visibility = ["//visibility:private"],
)

ovms_cc_library(
    name = "libovmspython",
    deps = select({
        "//src:windows": [":libovmspython_windows"],
        "//conditions:default": [":libovmspython_unix"],
    }),
    visibility = ["//visibility:public"],
)

# Minimal implementation subset needed by KFS graph executor for Python tensor
# conversion. This target avoids PYBIND_DEPS and keeps libpython linkage out of
# the main ovms binary.
ovms_cc_library(
    name = "python_tensor_runtime",
    srcs = [
        "python_backend_tensor_runtime.cpp",
        "python_backend.hpp",
        "utils.hpp",
    ],
    deps = PYBIND_HEADERS_ONLY + [
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

# Runtime implementation of the KFS OVMS_PY_TENSOR bridge vtable.
# Linking this into ovms_test allows Windows tests to resolve
# OVMS_getKfsPyTensorBridgeVTable from the current process, mirroring
# Linux RTLD_DEFAULT behavior and avoiding plugin DLL loading in unit tests.
ovms_cc_library(
    name = "kfs_python_tensor_bridge_runtime",
    srcs = [
        "kfs_python_tensor_bridge_impl.cpp",
    ],
    deps = PYBIND_HEADERS_ONLY + [
        "//src:kfs_python_tensor_bridge_hdrs",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "pythonbackend",
        "utils",
    ],
    copts = COPTS_SO,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

# cc_import wrapper for libmediapipe_framework.so to force proper linking
cc_import(
    name = "libmediapipe_framework_import",
    shared_library = "//src:libmediapipe_framework.so",
    visibility = ["//visibility:private"],
)

ovms_cc_library(
    name = "python_calculators_runtime_objects",
    srcs = [
        "kfs_python_tensor_bridge_impl.cpp",
        "python_calculators_plugin.cpp",
        "python_executor_calculator.cc",
        "python_node_initializer.cpp",
        "pytensor_ovtensor_converter_calculator.cc",
        "pythonnoderesources.cpp",
    ],
    hdrs = [
        "pythonnoderesources.hpp",
        "python_backend.hpp",
        "utils.hpp",
    ],
    deps = PYBIND_HEADERS_ONLY + [
        "//src:kfs_python_tensor_bridge_hdrs",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovmsprecision",
        "//src/mediapipe_internal:mediapipe_utils",
        "//src/mediapipe_internal:node_initializer",
        "ovmspytensor",
        "pythonexecutorcalculator_cc_proto",
        "pytensorovtensorconvertercalculator_cc_proto",
        "pythonbackend",
        "utils",
    ],
    copts = COPTS_SO,
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

# Python runtime loader library
# Encapsulates platform-specific logic for loading libovmspython.so/dll
ovms_cc_library(
    name = "python_runtime_loader",
    hdrs = ["python_runtime_loader.hpp"],
    srcs = ["python_runtime_loader.cpp"],
    deps = [
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovms_module",
        ":libovmspythonmodule",
    ],
    visibility = ["//visibility:public"],
)

# Plugin shared library containing MediaPipe Python calculators
# This library is loaded dynamically at runtime, making Python calculators
# optional and keeping the main ovms binary free from direct Python dependencies
cc_binary(
    name = "libpython_calculators.so",
    linkshared = True,
    srcs = [],
    deps = [
        ":python_calculators_runtime_objects",
    ],
    linkopts = LINKOPTS_SO + select({
        "//src:windows": [
            "/FORCE:MULTIPLE",  # Allow multiple symbol definitions for glog (linked in both this DLL and main binary)
        ],
        "//conditions:default": [
            "-Wl,--as-needed",  # Only link needed symbols
            "-Wl,--allow-shlib-undefined",  # Allow undefined symbols in plugin (resolved at runtime from main binary)
            "-Wl,-rpath,/ovms/lib:/opt/intel/openvino/runtime/lib/intel64",  # Search for dependencies in deployment and OV directories
        ],
    }),
    local_defines = COMMON_LOCAL_DEFINES,
    visibility = ["//visibility:public"],
)

simple_copy_file(
    name = "copy_libpython_calculators",
    src = "libpython_calculators.so",
    out = "libpython_calculators.dll",
    visibility = ["//visibility:public"],
)

# cc_import wrapper for libpython_calculators.so (Unix/Linux)
cc_import(
    name = "libpython_calculators_unix",
    shared_library = ":libpython_calculators.so",
    hdrs = [],  # No public headers needed; loaded via dlopen
    visibility = ["//visibility:private"],
)

# cc_import wrapper for libpython_calculators.so (Windows)
cc_import(
    name = "libpython_calculators_windows",
    interface_library = ":libpython_calculators.so.if.lib",
    shared_library = ":copy_libpython_calculators",
    hdrs = [],
    visibility = ["//visibility:private"],
)

alias(
    name = "libpython_calculators_select",
    actual = select({
        "//src:windows": ":libpython_calculators_windows",
        "//conditions:default": ":libpython_calculators_unix",
    }),
    visibility = ["//visibility:private"],
)

ovms_cc_library(
    name = "libpython_calculators",
    deps = select({
        "//src:windows": [":libpython_calculators_windows"],
        "//conditions:default": [":libpython_calculators_unix"],
    }),
    visibility = ["//visibility:public"],
)


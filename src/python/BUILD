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

load("//src:common_settings.bzl", "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS")

cc_library(
    name = "libovmspythonmodule",
    hdrs = ["pythonnoderesources.hpp",
            "ovms_py_tensor.hpp",
            "python_backend.hpp",
            "pythonnoderesources.hpp",
            "pythoninterpretermodule.hpp",
            ],
    srcs = ["python_backend.cpp",
            "ovms_py_tensor.cpp",
            "pythonnoderesources.cpp",
            "utils.hpp",
            "pythoninterpretermodule.cpp",
            "python_executor_calculator.cc",
            ],
    deps = [
        "//src:logging",
        "@com_github_glog_glog//:glog", # used to manage Mediapipe logging
    ],
    copts = COMMON_STATIC_LIBS_COPTS + select({
            "//conditions:default": [],
            "//src:fuzzer_build" : COMMON_FUZZER_COPTS,
        }),
    visibility = [
        "//visibility:public",
    ],
    linkopts = COMMON_STATIC_LIBS_LINKOPTS + select({
        "//conditions:default": [],
        "//src:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
    }),
    alwayslink = 1,
)

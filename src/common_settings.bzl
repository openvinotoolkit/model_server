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
###############################
# bazel config settings
###############################
#To build without mediapipe use flags - bazel build --define MEDIAPIPE_DISABLE=1 --cxxopt=-DMEDIAPIPE_DISABLE=1 //src:ovms
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@mediapipe//mediapipe/framework:more_selects.bzl", "more_selects")
def create_config_settings():
    native.config_setting(
        name = "disable_mediapipe",
        define_values = {
            "MEDIAPIPE_DISABLE": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_disable_mediapipe",
        negate = ":disable_mediapipe",
    )
    #To build without python use flags - bazel build --define PYTHON_DISABLE=1 //src:ovms
    native.config_setting(
        name = "disable_python",
        define_values = {
            "PYTHON_DISABLE": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_disable_python",
        negate = ":disable_python",
    )
    native.config_setting(
        name = "fuzzer_build",
        define_values = {
            "FUZZER_BUILD": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_fuzzer_build",
        negate = ":fuzzer_build",
    )
###############################
# compilation settings
###############################
COMMON_STATIC_LIBS_COPTS = [
    "-Wall",
    "-Wno-unknown-pragmas",
    "-Wno-sign-compare",
    "-fvisibility=hidden", # Needed for pybind targets
    "-Werror",
]
COMMON_STATIC_LIBS_LINKOPTS = [
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
]
COMMON_FUZZER_COPTS = [
    "-fsanitize=address",
    "-fprofile-generate",
    "-ftest-coverage",
]
COMMON_FUZZER_LINKOPTS = [
    "-fprofile-generate",
    "-fsanitize=address",
    "-fsanitize-coverage=trace-pc",
    "-static-libasan",
]
COMMON_LOCAL_DEFINES = ["SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE",]

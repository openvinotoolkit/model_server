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
#To build without mediapipe use flags - bazel build --config=linux --define MEDIAPIPE_DISABLE=1 --cxxopt=-DMEDIAPIPE_DISABLE=1 //src:ovms
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@mediapipe//mediapipe/framework:more_selects.bzl", "more_selects")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//:distro.bzl", "distro_flag")
def create_config_settings():
    distro_flag()
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
    native.config_setting(
        name = "disable_cloud",
        define_values = {
            "CLOUD_DISABLE": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_disable_cloud",
        negate = ":disable_cloud",
    )
    #To build without python use flags - bazel build --config=linux --define PYTHON_DISABLE=1 //src:ovms
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
        name = "disable_ov_trace",
        define_values = {
            "OV_TRACE": "0",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_disable_ov_trace",
        negate = ":disable_ov_trace",
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

    # is windows or mediapipe is disabled (no_http dependency)
    selects.config_setting_group(
        name = "is_windows_or_mediapipe_is_disabled_no_http",
        match_any = ["//src:windows", "//:disable_mediapipe"]
    )

    # is windows or python is disabled"(no llm dependency)
    selects.config_setting_group(
        name = "is_windows_or_python_is_disabled_no_llm",
        match_any = ["//src:windows", "//:disable_python"]
    )

  
###############################
# compilation settings
###############################
COMMON_STATIC_LIBS_COPTS = select({
                "//conditions:default": [
                    "-Wall",
                    # TODO: was in ovms bin "-Wconversion",
                    "-Wno-unknown-pragmas", 
                    "-Wno-sign-compare",
                    "-fvisibility=hidden", # Needed for pybind targets
                    "-Werror",
                ],
                "//src:windows" : [
                    "-Wall",
                    ],
                })

COMMON_STATIC_TEST_COPTS = select({
                "//conditions:default": [
                    "-Wall",
                    "-Wno-unknown-pragmas",
                    "-Werror",
                    "-Isrc",
                    "-fconcepts", # for gmock related utils
                    "-fvisibility=hidden",# Needed for pybind targets
                ],
                "//src:windows" : [
                    "-W0",
                    "-Isrc",
                    ],
                })

COMMON_STATIC_LIBS_COPTS_VISIBLE = select({
                "//conditions:default": [
                    "-Wall",
                    # TODO: was in ovms bin "-Wconversion",
                    "-Wno-unknown-pragmas", 
                    "-Wno-sign-compare",
                    "-Werror",
                ],
                "//src:windows" : [
                    "-Wall",
                    ],
                }) 

COMMON_STATIC_LIBS_LINKOPTS = select({
                "//conditions:default": [
                    "-lxml2",
                    "-luuid",
                    "-lstdc++fs",
                    "-lssl",
                    "-lcrypto",
                    # "-lovms_shared",  # Use for dynamic linking when necessary
                ],
                "//src:windows" : [
                    "",
                    ],
                })
COPTS_PYTHON = select({
    "//conditions:default": ["-DPYTHON_DISABLE=1"],
    "//:not_disable_python" : ["-DPYTHON_DISABLE=0"],
})
COPTS_MEDIAPIPE = select({
    "//conditions:default": ["-DMEDIAPIPE_DISABLE=1"],
    "//:not_disable_mediapipe" : ["-DMEDIAPIPE_DISABLE=0"],
})
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
COMMON_LOCAL_DEFINES = ["SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE"]
PYBIND_DEPS = [
    "//third_party:python3",
    "@pybind11//:pybind11_embed",
]

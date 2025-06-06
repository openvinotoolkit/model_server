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

# cc_library rule wrapper that will accept the same arguments but if user will not provide
# copts, linkopts, local_defines it will set them to the defaults
# COMMON_STATIC_LIBS_COPTS & COMMON_STATIC_LIBS_LINKOPTS
def ovms_cc_library(**kwargs):
    """
    Wrapper for cc_library that sets default copts and linkopts if not provided.
    """
    coptsProvided = kwargs.get("copts", [])
    linkoptsProvided = kwargs.get("linkopts", [])
    if "copts" not in kwargs:
        kwargs["copts"] = COMMON_STATIC_LIBS_COPTS + select({
            "//conditions:default": [],
            "//:fuzzer_build": COMMON_FUZZER_COPTS,
        })
    if "linkopts" not in kwargs:
        kwargs["linkopts"] = COMMON_STATIC_LIBS_LINKOPTS + select({
            "//conditions:default": [],
            "//:fuzzer_build": COMMON_FUZZER_LINKOPTS,
        })
    if "local_defines" not in kwargs:
        kwargs["local_defines"] = COMMON_LOCAL_DEFINES
    if "additional_copts" in kwargs:
        kwargs["copts"] += kwargs.pop("additional_copts")
    if "additional_linkopts" in kwargs:
        kwargs["linkopts"] += kwargs.pop("additional_linkopts")

    native.cc_library(
        **kwargs
    )

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
        name = "genai_bin",
        define_values = {
            "GENAI_USE_BINARY": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "not_genai_bin",
        negate = ":genai_bin",
    )
    native.config_setting(
        name = "enable_drogon",
        define_values = {
            "USE_DROGON": "1",
        },
        visibility = ["//visibility:public"],
    )
    more_selects.config_setting_negation(
        name = "enable_net_http",
        negate = ":enable_drogon",
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

    # is windows and python is enabled"
    selects.config_setting_group(
        name = "is_windows_and_python_is_enabled",
        match_all = ["//src:windows", "//:not_disable_python"]
    )

###############################
# compilation settings
###############################
LINUX_COMMON_STATIC_LIBS_COPTS = [
                    "-Wall",
                    # TODO: was in ovms bin "-Wconversion",
                    "-Wno-unknown-pragmas", 
                    "-Wno-sign-compare",
                    "-fvisibility=hidden", # Needed for pybind targets
                    "-Werror", 
                    # ov::Tensor::data method call results in deprecated warning and we use it in multiple places
                    "-Wno-deprecated-declarations",
]

WINDOWS_COMMON_STATIC_LIBS_COPTS = [
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
]

COMMON_STATIC_LIBS_COPTS = select({
                "//conditions:default": LINUX_COMMON_STATIC_LIBS_COPTS,
                "//src:windows" : WINDOWS_COMMON_STATIC_LIBS_COPTS,
                })

COMMON_STATIC_TEST_COPTS = select({
                "//conditions:default": [
                    "-Wall",
                    "-Wno-unknown-pragmas",
                    "-Werror",
                    # ov::Tensor::data method call results in deprecated warning and we use it in multiple places
                    "-Wno-deprecated-declarations",
                    "-Isrc",
                    "-fconcepts", # for gmock related utils
                    "-fvisibility=hidden",# Needed for pybind targets
                ],
                "//src:windows" : [
                        "-W0",
                        "-Isrc",
                        "/wd4996",
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
                        "-W0",
                        "-Isrc",
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
                    "/NXCOMPAT",
                    "/LTCG",
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
COPTS_DROGON = select({
    "//conditions:default": ["-DUSE_DROGON=0"],
    "//:enable_drogon" : ["-DUSE_DROGON=1"],
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

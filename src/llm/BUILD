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

cc_library(
    name = "llmcalculator",
    srcs = ["llm_calculator.cc",],
    deps = [
        "@linux_openvino//:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src/kfserving_api:kfserving_api_cpp",
        "@llm_engine//:llm_engine"

    ],
    visibility = ["//visibility:public"],
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)


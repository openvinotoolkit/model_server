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
})

LINKOPTS_ADJUSTED = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})

mediapipe_proto_library(
    name = "embeddings_calculator_proto", # embeddings_calculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["embeddings_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "embeddingscalculator",
    hdrs = [],
    srcs = ["embeddings_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@model_api//:model_api",
        "//src:httppayload",
        "//src:libhttpclientconnection",
        "//src:libovmslogging",
        "//src:libovmsprofiler",
        "embeddings_calculator_cc_proto",
        ":embeddings_api",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "embeddings_api",
    hdrs = ["embeddings_api.hpp"],
    srcs = ["embeddings_api.cpp"],
    deps = ["//src:libovmslogging",
            "@mediapipe//mediapipe/framework:calculator_framework",
            "//third_party:openvino",
            "@com_github_tencent_rapidjson//:rapidjson",],
    visibility = ["//visibility:public"],
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)
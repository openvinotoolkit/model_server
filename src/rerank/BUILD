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
load("//:common_settings.bzl", "ovms_cc_library")

mediapipe_proto_library(
    name = "rerank_calculator_proto", # rerank_calculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["rerank_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "rerank_calculator_ov_proto", # rerank_calculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["rerank_calculator_ov.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

ovms_cc_library(
    name = "rerankcalculator",
    srcs = ["rerank_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@model_api//:model_api",
        "//src:httppayload",
        "//src:libhttpclientconnection",
        "//src:libovmslogging",
        "//src:libovmsprofiler",
	    "rerank_calculator_cc_proto",
        ":rerank_api_handler",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

ovms_cc_library(
    name = "rerankcalculator_ov",
    srcs = ["rerank_calculator_ov.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@model_api//:model_api",
        "//src:httppayload",
        "//src:libhttpclientconnection",
        "//src:libovmslogging",
        "//src:libovmsprofiler",
        "rerank_calculator_ov_cc_proto",
        ":rerank_api_handler",
        "//src:sidepacket_servable",
        "//src:model_metric_reporter",
        "//src:executingstreamidguard",
        "//src:libovms_execution_context",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

ovms_cc_library(
    name = "rerank_api_handler",
    hdrs = ["rerank_utils.hpp"],
    srcs = ["rerank_utils.cpp"],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
        "@mediapipe//mediapipe/framework:calculator_framework", # required for absl status
        "//src:libovmsprofiler",
        "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

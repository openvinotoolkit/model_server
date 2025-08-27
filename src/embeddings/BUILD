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

ovms_cc_library(
    name = "embeddings_api",
    hdrs = ["embeddings_api.hpp"],
    srcs = ["embeddings_api.cpp"],
    deps = ["//src:libovmslogging",
            "@mediapipe//mediapipe/framework:calculator_framework",
            "//third_party:openvino",
            "@com_github_tencent_rapidjson//:rapidjson",],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

ovms_cc_library(
    name = "embeddings_servable",
    hdrs = ["embeddings_servable.hpp"],
    deps = ["//src:sidepacket_servable",],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "embeddings_calculator_proto", # embeddings_calculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["embeddings_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

ovms_cc_library(
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
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "embeddings_calculator_ov_proto", # embeddings_calculator_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["embeddings_calculator_ov.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

ovms_cc_library(
    name = "embeddingscalculator_ov",
    hdrs = [],
    srcs = ["embeddings_calculator_ov.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@model_api//:model_api",
        "//src:httppayload",
        "//src:libhttpclientconnection",
        "//src:libovmslogging",
        "//src:libovmsprofiler",
        "embeddings_calculator_ov_cc_proto",
        ":embeddings_servable",
        "//src:sidepacket_servable",
        "//src:model_metric_reporter",
        "//src:executingstreamidguard",
        "//src:libovms_execution_context",
        ":embeddings_api",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

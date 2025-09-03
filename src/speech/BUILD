#
# Copyright (c) 2025 Intel Corporation
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
    name = "llm_engine",  # in fact this is genai library
    srcs = [],
    deps =  ["@llm_engine//:llm_engine"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

ovms_cc_library(
    name = "speech_calculator",
    srcs = ["http_speech_calculator.cc"],
    hdrs = ["dr_wav.h"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src:httppayload",
        "//src:libovmslogging",
        "speech_calculator_cc_proto",
    ]+ select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "speech_calculator_proto",
    srcs = ["speech_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

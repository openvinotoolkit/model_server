#
# Copyright (c) 2026 Intel Corporation
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
load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")
load("//:common_settings.bzl", "ovms_cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "classification",
    hdrs = ["classification.hpp"],
)

mediapipe_proto_library(
    name = "movinet_calculator_proto",
    srcs = ["movinet_calculator.proto"],
    deps = ["@mediapipe//mediapipe/framework:calculator_proto"],
)

ovms_cc_library(
    name = "movinet_calculator",
    srcs = ["movinet_calculator.cpp"],
    deps = [
        ":classification",
        ":movinet_calculator_cc_proto",
        "//src:libovmslogging",
        "//src/mediapipe_internal:graph_side_packets",
        "//src/mediapipe_internal:live_graph_config",
        "//src/mediapipe_internal:node_initializer",
        "//third_party:opencv",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
    ],
    alwayslink = 1,
)

# Copyright 2019 The MediaPipe Authors.
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

load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bytetrack_calculators",
    deps = [
        "//src/bytetrack/utils:detection_color_by_id_calculator",
        ":kalman_filter",
        ":kalman_matrices",
        ":strack",
        ":bytetrack_calculator",
        ":render_data_passthrough",
    ],
    alwayslink = 1
)

mediapipe_proto_library(
    name = "bytetrack_calculator_proto",
    visibility = ["//visibility:public"],
    srcs = ["bytetrack_calculator.proto"],
    deps=[
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
    alwayslink = 1
)

cc_library(
    name = "kalman_matrices",
    hdrs = ["kalman_matrices.h"],
    deps = ["@eigen_archive//:eigen3"],
    alwayslink = 1
)


cc_library(
    name = "matching_utils",
    hdrs = ["matching_utils.h"],
    deps = [
        "@mediapipe//mediapipe/util/tracking:box_tracker",
    ],
    alwayslink = 1
)

cc_library(
    name = "kalman_filter",
    srcs = ["kalman_filter.cc"],
    hdrs = ["kalman_filter.h"],
    deps = [
        ":kalman_matrices",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1
)

cc_library(
    name = "basetrack",
    srcs = ["basetrack.cc"],
    hdrs = ["basetrack.h"],
    alwayslink = 1
)

cc_library(
    name = "strack",
    srcs = ["strack.cc"],
    hdrs = ["strack.h"],
    deps = [
        ":kalman_filter",
        ":basetrack",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@eigen_archive//:eigen3",
    ],
    alwayslink = 1
)

cc_library(
    name = "bytetrack_calculator",
    srcs = ["bytetrack_calculator.cc"],
    hdrs = ["bytetrack_calculator.h"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        ":bytetrack_calculator_cc_proto",
        "@mediapipe//mediapipe/framework/port:status",
        ":matching_utils",
        ":strack",
        ":kalman_filter",
        ":kalman_matrices",
         "@eigen_archive//:eigen3",
    ],
    alwayslink = 1,
)

# This calculator is for debugging to check whether render data is correctly passed through the graph.
cc_library(
    name = "render_data_passthrough",
    srcs = ["render_data_passthrough_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1
)

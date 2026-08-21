#
# Copyright (c) 2020 Intel Corporation
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

mediapipe_proto_library(
    name = "ov_yolox_tensors_to_detections_calculator_proto",
    srcs = [
        "ov_yolox_tensors_to_detections_calculator.proto",
    ],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto"
    ],
    visibility = [
        "//src:__pkg__",
    ],
)


cc_library(
    name = "ov_yolox_tensors_to_detections_calculator",
    srcs = [
        "ov_yolox_tensors_to_detections_calculator.cc",
    ],
    deps = [
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@mediapipe//mediapipe/framework/formats:location_data_cc_proto",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework:calculator_cc_proto",
        ":ov_yolox_tensors_to_detections_calculator_cc_proto",
    ],
    visibility = [
        "//src:__pkg__",
    ],
    alwayslink = True
)
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
load("//:common_settings.bzl", "LINUX_COMMON_STATIC_LIBS_COPTS", "COMMON_FUZZER_LINKOPTS", "ovms_cc_library")

COPTS_ADJUSTED = select({
    "//conditions:default": LINUX_COMMON_STATIC_LIBS_COPTS,
    "//src:windows": [],
})

LINKOPTS_ADJUSTED = select({
    "//conditions:default": [],
    "//:fuzzer_build": COMMON_FUZZER_LINKOPTS,
})

ovms_cc_library(
    name = "gst_video_source_calculator",
    srcs = ["gst_video_source_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp"],
    additional_copts = COPTS_ADJUSTED,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//src/mpi:intel_mpi",
        "//third_party:openvino",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED,
)

ovms_cc_library(
    name = "gst_capi_inference_calculator",
    srcs = ["gst_capi_inference_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp", "gst_detection_box.hpp"],
    additional_copts = COPTS_ADJUSTED,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//src:ovms_lib",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED,
)

ovms_cc_library(
    name = "gst_detection_overlay_calculator",
    srcs = ["gst_detection_overlay_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp", "gst_detection_box.hpp", "gst_nv12_overlay_draw.hpp"],
    additional_copts = COPTS_ADJUSTED,
    local_defines = select({
        "//:redhat_build": [],
        "//conditions:default": ["OVMS_ENABLE_VA_OVERLAY"],
    }),
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//src/mpi:intel_mpi",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED + select({
        "//:redhat_build": [],
        "//conditions:default": ["-lva"],
    }),
)

ovms_cc_library(
    name = "gst_video_sink_calculator",
    srcs = ["gst_video_sink_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp"],
    additional_copts = COPTS_ADJUSTED,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//src/mpi:intel_mpi",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED,
)
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
load("//:common_settings.bzl", "LINUX_COMMON_STATIC_LIBS_COPTS", "COMMON_FUZZER_LINKOPTS", "ovms_cc_library")

# MSVC warnings promoted to errors inside mediapipe/glog/absl headers. Mirrors
# the suppressions in src/python/BUILD so these calculators compile on Windows.
MEDIAPIPE_WIN_COPTS = [
    "/wd4005",  # macro redefinition (GLOG_NO_ABBREVIATED_SEVERITIES from mediapipe/glog)
    "/wd4309",  # truncation of constant value (mediapipe circular_buffer.h)
    "/wd4456",  # declaration hides previous local declaration (mediapipe graph_profiler.h)
    "/wd6001",  # uninitialized memory (absl/strings and mediapipe)
    "/wd6246",  # local declaration hides previous declaration (mediapipe graph_profiler.h)
    "/wd6386",  # buffer overrun (absl cord_rep_flat.h)
    "/wd6011",  # NULL pointer dereference (mediapipe packet.h)
]

COPTS_ADJUSTED = select({
    "//conditions:default": LINUX_COMMON_STATIC_LIBS_COPTS,
    "//src:windows": MEDIAPIPE_WIN_COPTS,
})

LINKOPTS_ADJUSTED = select({
    "//conditions:default": [],
    "//:fuzzer_build": COMMON_FUZZER_LINKOPTS,
})

mediapipe_proto_library(
    name = "gst_openvino_inference_calculator_proto",
    srcs = ["gst_openvino_inference_calculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

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
    name = "gst_video_capi_inference_calculator",
    srcs = ["gst_video_capi_inference_calculator.cc"],
    additional_copts = COPTS_ADJUSTED,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//src/mpi:intel_mpi",
        "//src:ovms_lib",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED,
)

ovms_cc_library(
    name = "gst_openvino_inference_calculator",
    srcs = ["gst_openvino_inference_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp"],
    additional_copts = COPTS_ADJUSTED,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":gst_openvino_inference_calculator_cc_proto",
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED + select({
        "//src:windows": ["d3d11.lib", "dxgi.lib"],
        "//conditions:default": [],
    }),
)

ovms_cc_library(
    name = "gst_video_openvino_inference_calculator",
    srcs = ["gst_video_openvino_inference_calculator.cc"],
    additional_copts = COPTS_ADJUSTED,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":gst_openvino_inference_calculator_cc_proto",
        "//src/mpi:intel_mpi",
        "//third_party:openvino",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED + select({
        "//src:windows": ["d3d11.lib", "dxgi.lib"],
        "//conditions:default": [],
    }),
)

ovms_cc_library(
    name = "gst_detection_overlay_calculator",
    srcs = ["gst_detection_overlay_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp", "gst_detection_box.hpp", "gst_nv12_overlay_draw.hpp"],
    additional_copts = COPTS_ADJUSTED,
    local_defines = select({
        "//:redhat_build": [],
        "//src:windows": [],
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
        "//src:windows": [],
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

ovms_cc_library(
    name = "gst_video_openvino_pipeline_inference_calculator",
    srcs = ["gst_video_openvino_pipeline_inference_calculator.cc"],
    additional_copts = COPTS_ADJUSTED,
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":gst_openvino_inference_calculator_cc_proto",
        "//src/mpi:intel_mpi",
        "//third_party:openvino",
        "//third_party:gstreamer",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    additional_linkopts = LINKOPTS_ADJUSTED + select({
        "//src:windows": ["d3d11.lib", "dxgi.lib"],
        "//conditions:default": [],
    }),
)
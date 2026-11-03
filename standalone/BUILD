#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

load("//:common_settings.bzl", "LINUX_COMMON_STATIC_LIBS_COPTS")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "mpi_openvino_video",
    srcs = ["mpi_openvino_video.cpp"],
    copts = select({
        "//conditions:default": LINUX_COMMON_STATIC_LIBS_COPTS,
        "//src:windows": [],
    }),
    deps = [
        "//src/mediapipe_gst_calculators:gst_openvino_inference_calculator_cc_proto",
        "//src/mediapipe_gst_calculators:gst_detection_overlay_calculator",
        "//src/mediapipe_gst_calculators:gst_openvino_inference_calculator",
        "//src/mediapipe_gst_calculators:gst_video_source_calculator",
        "//src/mediapipe_gst_calculators:gst_video_openvino_inference_calculator",
        "//src/mediapipe_gst_calculators:gst_video_sink_calculator",
        "//src/mpi:intel_mpi",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe_calculators//:mediapipe_calculators",
    ],
    linkopts = select({
        "//src:windows": ["d3d11.lib", "dxgi.lib"],
        "//conditions:default": ["-lOpenCL", "-lva", "-lva-drm"],
    }),
)
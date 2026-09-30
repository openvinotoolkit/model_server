#
# Copyright (c) 2023 Intel Corporation
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
#config_setting(
#    name = "linux_distribution_family",
#    constraint_values = [
#        ":debian", # like Ubuntu
#        ":fedora", # like RHEL/CentOS
#    ],
#)
load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")
load("//:common_settings.bzl",
     "LINUX_COMMON_STATIC_LIBS_COPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS")

COPTS_ADJUSTED = select({
                "//conditions:default": LINUX_COMMON_STATIC_LIBS_COPTS,
                "//src:windows" : [],
                })

LINKOPTS_ADJUSTED = select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})

mediapipe_proto_library(
    name = "ovmscalculatoroptions_proto", # ovmscalculatoroptions_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["ovmscalculator.proto",],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "dependency_free_http_test_calculators",
    linkstatic = 1,
    alwayslink = 1,
    srcs = [
        "multipart_accepting_calculator.cpp",
        "openai_chat_completions_mock_calculator.cpp",
    ],
    copts = COPTS_ADJUSTED,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//src:httppayload",
        "@mediapipe_calculators//:mediapipe_calculators", # Need this dependencies here because we use ovms/src - cannot add in ovms_dependencies because we copy src directory later in Dockerfile
    ],
    linkopts = LINKOPTS_ADJUSTED,
)


cc_library(
    name = "mediapipe_test_calculators",
    linkstatic = 1,
    alwayslink = 1,
    srcs = [
        "nooutputstreamsproducedcalculator.cpp",
        "exceptionduringprocesscalculator.cpp",
        "exceptionduringopencalculator.cpp",
        "exceptionduringclosecalculator.cpp",
        "exceptionduringgetcontractcalculator.cpp",
        "negative_calculator.cc",
        "ovms_calculator.cc",
        "ovms_image_input_calculator.cc",
        "ovms_kfs_calculator.cc",
        "error_on_negative_test_calculator.cpp",
        "streaming_test_calculator.cpp",
        "two_input_calculator.cpp",
    ],
    copts = COPTS_ADJUSTED,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//src:ovms_lib", # stringutils.hpp
        "ovmscalculatoroptions_cc_proto", # ovmscalculatoroptions_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
        "@mediapipe_calculators//:mediapipe_calculators", # Need this dependencies here because we use ovms/src - cannot add in ovms_dependencies because we copy src directory later in Dockerfile
        "//third_party:openvino",
        "//src:ovms_header",
    ],
    linkopts = LINKOPTS_ADJUSTED,
)

# Standalone smoke test: decode video with GStreamer + infer with OV directly
# Run with:
#   bazel run //src/test/mediapipe/calculators:video_decode_infer_test \
#     -- /path/to/video.avi /path/to/model.xml
cc_binary(
    name = "video_decode_infer_test",
    srcs = ["video_decode_infer_test.cpp"],
    copts = COPTS_ADJUSTED,
    deps = [
        "//src/mpi:intel_mpi",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED + select({
        "//src:windows": [],
        "//conditions:default": ["-lOpenCL"],
    }),
)

cc_library(
    name = "video_decode_infer_calculator",
    linkstatic = 1,
    alwayslink = 1,
    srcs = ["video_decode_infer_calculator.cc"],
    copts = COPTS_ADJUSTED,
    visibility = ["//visibility:public"],
    deps = [
        "//src/mpi:intel_mpi",
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED,
)

# P0: monolithic decode+infer calculator using the GPU VA zero-copy path.
cc_library(
    name = "video_decode_infer_va_calculator",
    linkstatic = 1,
    alwayslink = 1,
    srcs = ["video_decode_infer_va_calculator.cc"],
    copts = COPTS_ADJUSTED,
    visibility = ["//visibility:public"],
    deps = [
        "//src/mpi:intel_mpi",
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED,
)

# P0 graph test. Needs GPU + GStreamer + test assets; skips otherwise.
# Run with:
#   bazel run //src/test/mediapipe/calculators:video_decode_infer_va_test
cc_test(
    name = "video_decode_infer_va_test",
    srcs = ["video_decode_infer_va_test.cpp"],
    data = ["video_decode_infer_va.pbtxt"],
    copts = COPTS_ADJUSTED,
    linkstatic = 1,
    deps = [
        ":video_decode_infer_va_calculator",
        "//src/mpi:intel_mpi",
        "@com_google_googletest//:gtest_main",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED + select({
        "//src:windows": [],
        "//conditions:default": ["-lOpenCL"],
    }),
)

# P2: standalone GStreamer/VA video source calculator emitting GstVideoFramePacket.
cc_library(
    name = "gst_video_source_calculator",
    linkstatic = 1,
    alwayslink = 1,
    srcs = ["gst_video_source_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp"],
    copts = COPTS_ADJUSTED,
    visibility = ["//visibility:public"],
    deps = [
        "//src/mpi:intel_mpi",
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED,
)

# P2 graph test: proves per-packet surface ownership (N in-flight distinct
# surfaces). Needs GPU + GStreamer + test video; skips otherwise.
cc_test(
    name = "gst_video_source_test",
    srcs = ["gst_video_source_test.cpp"],
    data = ["gst_video_source.pbtxt"],
    copts = COPTS_ADJUSTED,
    linkstatic = 1,
    deps = [
        ":gst_video_source_calculator",
        "//src/mpi:intel_mpi",
        "@com_google_googletest//:gtest_main",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED + select({
        "//src:windows": [],
        "//conditions:default": ["-lOpenCL"],
    }),
)

# P3: in-tree NV12 two-plane GPU inference calculator (temporary stand-in for the
# external OpenVINOInferenceCalculator). Binds Y/UV on the shared VA context.
cc_library(
    name = "gst_nv12_inference_calculator",
    linkstatic = 1,
    alwayslink = 1,
    srcs = ["gst_nv12_inference_calculator.cc"],
    hdrs = ["gst_video_frame_packet.hpp"],
    copts = COPTS_ADJUSTED,
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED,
)

# P3/P4 split-path graph test: source -> inference, zero-copy on GPU.
cc_test(
    name = "gst_video_infer_test",
    srcs = ["gst_video_infer_test.cpp"],
    data = ["gst_video_infer.pbtxt"],
    copts = COPTS_ADJUSTED,
    linkstatic = 1,
    deps = [
        ":gst_video_source_calculator",
        ":gst_nv12_inference_calculator",
        "//src/mpi:intel_mpi",
        "@com_google_googletest//:gtest_main",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe_calculators//:mediapipe_calculators",
        "//third_party:openvino",
    ],
    linkopts = LINKOPTS_ADJUSTED + select({
        "//src:windows": [],
        "//conditions:default": ["-lOpenCL"],
    }),
)


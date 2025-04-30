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


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
    name = "mediapipe_test_calculators",
    linkstatic = 1,
    alwayslink = 1,
    srcs = [
        "nooutputstreamsproducedcalculator.cpp",
        "exceptionduringprocesscalculator.cpp",
        "exceptionduringopencalculator.cpp",
        "exceptionduringclosecalculator.cpp",
        "exceptionduringgetcontractcalculator.cpp",
        "ovms_calculator.cc",
        "ovms_image_input_calculator.cc",
        "ovms_kfs_calculator.cc",
        "streaming_test_calculator.cpp",
        "two_input_calculator.cpp",
        "openai_chat_completions_mock_calculator.cpp",
    ],
    copts = select({
            "//conditions:default": [
                "-Wall",
                "-Wno-unknown-pragmas",
                "-Werror",
                "-Wno-sign-compare",
            ],
            "//:fuzzer_build" : [
                "-fsanitize=address",
                "-fprofile-generate",
                "-ftest-coverage",
            ],
            "//src:windows" : [
                "-W0",
            ],
        }),
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
    linkopts = select({
        "//conditions:default": [
        ],
        "//:fuzzer_build" : [
            "-fprofile-generate",
            "-fsanitize=address",
            "-fsanitize-coverage=trace-pc",
            "-static-libasan",
        ],
    }),
)


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
        "streaming_test_calculator.cpp",
    ],
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror",
        "-Wno-sign-compare",
        ] + select({
            "//conditions:default": [
            ],
            "//src:fuzzer_build" : [
                "-fsanitize=address",
                "-fprofile-generate",
                "-ftest-coverage",
            ],
        }),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//src:ovmscalculatoroptions_cc_proto", # ovmscalculatoroptions_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
        "@mediapipe_calculators//:mediapipe_calculators", # Need this dependencies here because we use ovms/src - cannot add in ovms_dependencies because we copy src directory later in Dockerfile
        "@linux_openvino//:openvino",
    ],
    linkopts = select({
        "//conditions:default": [
        ],
        "//src:fuzzer_build" : [
            "-fprofile-generate",
            "-fsanitize=address",
            "-fsanitize-coverage=trace-pc",
            "-static-libasan",
        ],
    }),
)


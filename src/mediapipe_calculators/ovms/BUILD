#
# Copyright (c) 2023-2026 Intel Corporation
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

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_proto_library(
    name = "openvinoinferencecalculator_proto",
    srcs = ["openvinoinferencecalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

mediapipe_proto_library(
    name = "openvinomodelserversessioncalculator_proto",
    srcs = ["openvinomodelserversessioncalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "openvinoinferencedumputils",
    srcs = ["openvinoinferencedumputils.cc"],
    hdrs = ["openvinoinferencedumputils.h"],
    copts = ["-Isrc"],
    deps = [
        "//third_party:openvino",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvinoinferenceutils",
    srcs = ["openvinoinferenceutils.cc"],
    hdrs = ["openvinoinferenceutils.h"],
    copts = ["-Isrc"],
    deps = [
        "//src:ovms_header",
    ],
    alwayslink = 1,
)

cc_library(
    name = "modelapiovmsadapter",
    srcs = ["modelapiovmsadapter.cc"],
    hdrs = ["modelapiovmsadapter.hpp"],
    copts = ["-Isrc"],
    deps = [
        ":openvinoinferencedumputils",
        "//src:ovms_header",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework/port:logging",
        "@model_api//:model_api",
    ],
)

cc_library(
    name = "openvinoinferencecalculatoroptions",
    srcs = ["openvinoinferencecalculatoroptions.cc"],
    hdrs = ["openvinoinferencecalculatoroptions.h"],
    copts = ["-Isrc"],
    deps = [
        ":openvinoinferencecalculator_cc_proto",
        ":openvinoinferenceutils",
        "//src:ovms_header",
        "@mediapipe//mediapipe/framework:calculator_framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvinoinferencecalculator",
    srcs = ["openvinoinferencecalculator.cc"],
    hdrs = ["openvinoinferencecalculator.h"],
    copts = ["-Isrc"],
    deps = [
        ":modelapiovmsadapter",
        ":openvinoinferencecalculator_cc_proto",
        ":openvinoinferencecalculatoroptions",
        ":openvinoinferenceutils",
        "//src:ovms_header",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:tensor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "openvinomodelserversessioncalculator",
    srcs = ["openvinomodelserversessioncalculator.cc"],
    hdrs = ["openvinomodelserversessioncalculator.h"],
    copts = ["-Isrc"],
    deps = [
        ":modelapiovmsadapter",
        ":openvinoinferenceutils",
        ":openvinomodelserversessioncalculator_cc_proto",
        "//src:ovms_header",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ovms_calculator",
    deps = [
        ":modelapiovmsadapter",
        ":openvinoinferencecalculator",
        ":openvinomodelserversessioncalculator",
    ],
    alwayslink = 1,
)

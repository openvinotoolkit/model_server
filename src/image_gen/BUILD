#
# Copyright (c) 2025 Intel Corporation
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
load("//:common_settings.bzl",
     "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS", "COMMON_LOCAL_DEFINES", "PYBIND_DEPS")

COPTS_ADJUSTED = COMMON_STATIC_LIBS_COPTS + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_COPTS,
})

LINKOPTS_ADJUSTED = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})

cc_library(
    name = "llm_engine",  # in fact this is genai library, includes image gen, TODO: Change name?
    srcs = [],
    deps =  ["@llm_engine//:llm_engine"],
    visibility = ["//visibility:public"],
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "pipelines",
    hdrs = ["pipelines.hpp"],
    srcs = ["pipelines.cpp"],
    deps = [
        "imagegenpipelineargs",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "imagegen_init",
    hdrs = ["imagegen_init.hpp"],
    srcs = ["imagegen_init.cpp"],
    deps = [
        "//src:libovmslogging",
        "//src:libmodelconfigjsonparser",
        "//src:libovmsfilesystem",
        "imagegenutils",
        "imagegenpipelineargs",
        "image_gen_calculator_cc_proto",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "imagegenpipelineargs",
    hdrs = ["imagegenpipelineargs.hpp"],
    deps = [
        "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "imagegenutils",
    srcs = ["imagegenutils.cpp"],
    hdrs = ["imagegenutils.hpp"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
        "//src:httppayload",
        "//src:libovmslogging",
        "//src:libimage_conversion",
        "//src:libovmsstring_utils",
        "imagegenpipelineargs",
    ] + select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "image_gen_calculator",
    srcs = ["http_image_gen_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "//src:httppayload",
        "//src:libovmslogging",
        "image_gen_calculator_cc_proto",
        ":pipelines",
        "//src:libimage_conversion",
        ":imagegenutils",
    ]+ select({
        "//conditions:default": ["//third_party:genai", ":llm_engine"],
        "//:not_genai_bin" : [":llm_engine"],
    }),
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "image_gen_calculator_proto",
    srcs = ["image_gen_calculator.proto"],
    visibility = ["//visibility:private"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

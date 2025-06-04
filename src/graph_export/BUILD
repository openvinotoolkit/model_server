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
     "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COPTS_MEDIAPIPE", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS", "COMMON_LOCAL_DEFINES", "PYBIND_DEPS")

COPTS_ADJUSTED = COMMON_STATIC_LIBS_COPTS + COPTS_MEDIAPIPE + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_COPTS,
})

LINKOPTS_ADJUSTED = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})

cc_library(
    name = "graph_export",
    srcs = ["graph_export.cpp"],
    hdrs = ["graph_export.hpp"],
    deps =  [
        "@ovms//src/config_export_module:config_export_types",
        "@ovms//src/graph_export:graph_export_types",
        "@ovms//src:cpp_headers",
        "@ovms//src:libovmsstring_utils",
        "@ovms//src:libovmslogging",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:libovms_module",
        "@ovms//src:libovmsfilesystem",
        "@ovms//src:libovmslocalfilesystem",
        "@com_github_tencent_rapidjson//:rapidjson",
        #"@mediapipe//mediapipe/framework/port:parse_text_proto",
        #"@mediapipe//mediapipe/framework:calculator_graph",
        "@ovms//src:libovmsschema",
    ] + select({
            "//:not_disable_mediapipe": [
                "@mediapipe//mediapipe/framework/port:parse_text_proto",
                "@mediapipe//mediapipe/framework:calculator_graph",
                ],
            "//:disable_mediapipe": []
            }),
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "graph_cli_parser",
    srcs = ["graph_cli_parser.cpp"],
    hdrs = ["graph_cli_parser.hpp"],
    deps =  [
        "@ovms//src/graph_export:graph_export_types",
        "@ovms//src/config_export_module:config_export_types",
        "@ovms//src:cpp_headers",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:ovms_exit_codes",
        "@com_github_jarro2783_cxxopts//:cxxopts",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "rerank_graph_cli_parser",
    srcs = ["rerank_graph_cli_parser.cpp"],
    hdrs = ["rerank_graph_cli_parser.hpp"],
    deps =  [
        "@ovms//src/graph_export:graph_export_types",
        "@ovms//src/graph_export:graph_cli_parser",
        "@ovms//src:cpp_headers",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:ovms_exit_codes",
        "@com_github_jarro2783_cxxopts//:cxxopts",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "image_generation_graph_cli_parser",
    srcs = ["image_generation_graph_cli_parser.cpp"],
    hdrs = ["image_generation_graph_cli_parser.hpp"],
    deps =  [
        "@ovms//src/graph_export:graph_export_types",
        "@ovms//src/graph_export:graph_cli_parser",
        "@ovms//src:cpp_headers",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:ovms_exit_codes",
        "@com_github_jarro2783_cxxopts//:cxxopts",
        "@com_github_tencent_rapidjson//:rapidjson",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "embeddings_graph_cli_parser",
    srcs = ["embeddings_graph_cli_parser.cpp"],
    hdrs = ["embeddings_graph_cli_parser.hpp"],
    deps =  [
        "@ovms//src/graph_export:graph_export_types",
        "@ovms//src/graph_export:graph_cli_parser",
        "@ovms//src:cpp_headers",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:ovms_exit_codes",
        "@com_github_jarro2783_cxxopts//:cxxopts",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "graph_export_types",
    srcs = ["graph_export_types.cpp"],
    hdrs = ["graph_export_types.hpp"],
    deps =  [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

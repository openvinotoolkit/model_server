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

cc_library(
    name = "graph_export",
    srcs = ["graph_export.cpp"],
    hdrs = ["graph_export.hpp"],
    deps =  [
        "@ovms//src:cpp_headers",
        "@ovms//src:libovmsstring_utils",
        "@ovms//src:libovmslogging",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:libovms_module",
        "@ovms//src:libovmsfilesystem",
        "@com_github_tencent_rapidjson//:rapidjson",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "graph_cli_parser",
    srcs = ["graph_cli_parser.cpp"],
    hdrs = ["graph_cli_parser.hpp"],
    deps =  [
        "@ovms//src:cpp_headers",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:ovms_exit_codes",
        "@com_github_jarro2783_cxxopts//:cxxopts",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rerank_graph_cli_parser",
    srcs = ["rerank_graph_cli_parser.cpp"],
    hdrs = ["rerank_graph_cli_parser.hpp"],
    deps =  [
        "@ovms//src/graph_export:graph_cli_parser",
        "@ovms//src:cpp_headers",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:ovms_exit_codes",
        "@com_github_jarro2783_cxxopts//:cxxopts",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "embeddings_graph_cli_parser",
    srcs = ["embeddings_graph_cli_parser.cpp"],
    hdrs = ["embeddings_graph_cli_parser.hpp"],
    deps =  [
        "@ovms//src/graph_export:graph_cli_parser",
        "@ovms//src:cpp_headers",
        "@ovms//src:libovms_server_settings",
        "@ovms//src:ovms_exit_codes",
        "@com_github_jarro2783_cxxopts//:cxxopts",
    ],
    visibility = ["//visibility:public"],
)
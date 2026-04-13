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
load("//:common_settings.bzl", "COPTS_PYTHON", "COPTS_MEDIAPIPE", "ovms_cc_library")

ovms_cc_library(
    name = "node_initializer",
    srcs = [
        "node_initializer.cpp",
    ],
    hdrs = [
        "node_initializer.hpp",
    ],
    deps = [
        ":graph_side_packets",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "graph_side_packets",
    hdrs = [
        "graph_side_packets.hpp",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "graph_executor_constants",
    hdrs = [
        "graph_executor_constants.hpp",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "outputstreamobserver",
    hdrs = [
        "outputstreamobserver.hpp",
    ],
    deps = [
        ":mediapipe_utils",
        "//src:libovms_execution_context",
        "//src:model_metric_reporter",
        "//src:libovmsprofiler",
        "//src:libovmsstatus",
        "//src:libovmstimer",
        "@mediapipe//mediapipe/framework:calculator_graph",
    ],
    visibility = ["//visibility:public"],
    additional_copts = COPTS_MEDIAPIPE,
)

ovms_cc_library(
    name = "graphqueue",
    hdrs = [
        "graphqueue.hpp",
    ],
    srcs = [
        "graphqueue.cpp",
    ],
    deps = select({
            "//:not_disable_python": [
                "//src/python:libovmspythonmodule",
            ],
            "//:disable_python": []
        }) + [
        ":graph_executor_constants",
        ":graph_side_packets",
        ":mediapipe_utils",
        ":outputstreamobserver",
        ":side_packet_builder",
        "//src:libovms_queue",
        "//src:libovmslogging",
        "//src:libovms_execution_context",
        "//src:libovmstimer",
        "//src/metrics:libovmsmetrics",
        "//src:model_metric_reporter",
        "//src:libovmsprofiler",
        "//src/llm:genai_servables",
        "//third_party:openvino",
        "@mediapipe//mediapipe/framework:calculator_graph",
    ],
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON + COPTS_MEDIAPIPE,
)

ovms_cc_library(
    name = "mediapipe_utils",
    hdrs = [
        "mediapipe_utils.hpp",
        "packettypes.hpp",
    ],
    srcs = [
        "mediapipe_utils.cpp",
    ],
    deps = [
        "//src:libovmsstring_utils",
        "//src:libovmsstatus",
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "side_packet_builder",
    srcs = [
        "side_packet_builder.cpp",
    ],
    hdrs = [
        "side_packet_builder.hpp",
    ],
    deps = [
        ":graph_executor_constants",
        ":graph_side_packets",
        "@mediapipe//mediapipe/framework:calculator_graph",
    ],
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON + COPTS_MEDIAPIPE,
)

ovms_cc_library(
    name = "mediapipegraphconfig",
    srcs = [
        "mediapipegraphconfig.cpp",
    ],
    hdrs = [
        "mediapipegraphconfig.hpp",
    ],
    deps = [
        "//src/filesystem:libovmsfilesystem",
        "//src:libovmsstatus",
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src/port:rapidjson_stringbuffer",
        "//src/port:rapidjson_writer",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovms_mediapipe",
    srcs = [
        "mediapipefactory.cpp",
        "mediapipefactory.hpp",
        "mediapipegraphdefinition.cpp",
        "mediapipegraphdefinition.hpp",
        "mediapipegraphexecutor.cpp",
        "mediapipegraphexecutor.hpp",
    ],
    deps = select({
            "//:not_disable_python": [
                "//src/python:libovmspythonmodule",
            ],
            "//:disable_python": []
        }) + [
        ":graphqueue",
        ":mediapipegraphconfig",
        ":node_initializer",
        ":mediapipe_utils",
        ":side_packet_builder",
        "//src/dags:pipelinedefinitionstatus",
        "//src:libovms_single_version_servable_definition",
        "//src:libovms_tensorinfo",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:model_metric_reporter",
        "//src/metrics:libovmsmetrics",
        "//src:libovms_ov_utils",
        "//src:libovmsstring_utils",
        "//src:libovms_servable_name_checker",
        "//src/metrics:libovms_metric_provider",
        "//src/filesystem:libovmsfilesystem",
        "//src:libovms_version",
        "//src:libovms_execution_context",
        "//src:libovmstimer",
        "//src:libovmsprofiler",
        "//src/llm:genai_servables",
        "@mediapipe//mediapipe/framework:calculator_graph",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON + COPTS_MEDIAPIPE,
    alwayslink = 1,
)

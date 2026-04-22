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
        "graph_side_packets.hpp",
    ],
    deps = [
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
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
    name = "libovms_mediapipe",
    srcs = [
        "mediapipefactory.cpp",
        "mediapipefactory.hpp",
        "mediapipegraphconfig.hpp",
        "mediapipegraphconfig.cpp",
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
        ":node_initializer",
        ":mediapipe_utils",
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
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src/port:rapidjson_stringbuffer",
        "//src/port:rapidjson_writer",
        "@mediapipe//mediapipe/framework:calculator_graph",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
    ],
    visibility = ["//visibility:public"],
    additional_copts = COPTS_PYTHON + COPTS_MEDIAPIPE,
    alwayslink = 1,
)

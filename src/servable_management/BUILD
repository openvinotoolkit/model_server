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
load("//:common_settings.bzl", "ovms_cc_library")

ovms_cc_library(
    name = "servable_loading_task",
    hdrs = ["servable_loading_task.hpp"],
    deps = select({
        "//conditions:default": [],
        "//:not_disable_mediapipe": [
            "//src/mediapipe_internal:mediapipegraphconfig",
        ],
    }) + [
        "//src:modelconfig",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "servable_loading_queue",
    hdrs = ["servable_loading_queue.hpp"],
    srcs = ["servable_loading_queue.cpp"],
    deps = [
        "servable_loading_task",
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "modelmanager",
    hdrs = ["modelmanager.hpp", "servable_group_manager.hpp"],
    srcs = ["modelmanager.cpp", "servable_group_manager.cpp"],
    deps = select({
        "//conditions:default": [],
        "//:not_disable_mediapipe" : [
            "//src/mediapipe_internal:libovms_mediapipe",
        ],
    }) + [
        "servable_loading_queue",
        "//src:cleaner_utils",
        "//src:customloaders",
        "//src:libovms_config",
        "//src:libovms_model_instance_provider",
        "//src:libovms_ov_utils",
        "//src:libovms_servable_definition",
        "//src:libovms_servable_name_checker",
        "//src:libovmslogging",
        "//src:libovmsschema",
        "//src:libovmsstring_utils",
        "//src:libovmsstatus",
        "//src:model",
        "//src:modelconfig",
        "//src:modelinstance",
        "//src:modelinstanceunloadguard",
        "//src:resources_cleaner",
        "//src/dags:custom_node_library_manager",
        "//src/dags:dag_resource_manager",
        "//src/dags:pipeline_config_parser",
        "//src/dags:pipeline_factory",
        "//src/dags:pipelinedefinition",
        "//src/filesystem:libovmsfilesystem",
        "//src/filesystem:libovmsfilesystemfactory",
        "//src/graph_export:graph_export",
        "//src/metrics:libovms_metric_provider",
        "//src/metrics:libovmsmetrics",
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src/port:rapidjson_stringbuffer",
        "//src/port:rapidjson_writer",
    ],
    visibility = ["//visibility:public"],
)
ovms_cc_library(
    name = "servablemanagermodule",
    hdrs = ["servablemanagermodule.hpp"],
    srcs = ["servablemanagermodule.cpp"],
    deps = select({
        "//:not_disable_python": [
            "//src/python:libovmspythonmodule",
        ],
        "//:disable_python": []
    }) + [
        "//src:cpp_headers",
        "//src:libovms_module",
        "//src:libovmslogging",
        "modelmanager",
        "//src/metrics:libovmsmetrics",
    ],
    visibility = ["//visibility:public"],
)

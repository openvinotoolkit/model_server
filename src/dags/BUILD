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
load("//:common_settings.bzl", "COMMON_STATIC_LIBS_COPTS", "ovms_cc_library")

ovms_cc_library(
    name = "pipeline_config_parser",
    hdrs = ["pipeline_config_parser.hpp"],
    srcs = ["pipeline_config_parser.cpp"],
    deps = [
        "custom_node_library_manager",
        "entry_node",
        "exit_node",
        "nodeinfo",
        "pipeline_factory",
        "dag_resource_manager",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovms_model_instance_provider",
        "//src:libovms_servable_name_checker",
        "//src/port:rapidjson_document",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "pipelinedefinition",
    hdrs = [
        "pipelinedefinition.hpp",
        "pipelinedefinition_create_impl.hpp",
    ],
    srcs = ["pipelinedefinition.cpp"],
    deps = [
        "custom_node",
        "custom_node_library_internal_manager_wrapper",
        "dl_node",
        "node_library_utils",
        "nodeinfo",
        "nodestreamidguard",
        "pipeline",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovms_ov_utils",
        "//src:libovms_model_instance_provider",
        "//src:libovms_servable_name_checker",
        "//src:model_metric_reporter",
        "dag_resource_manager",
        "//src:libovms_single_version_servable_definition",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "pipeline_factory",
    hdrs = [
        "pipeline_factory.hpp",
        "pipeline_factory_create_impl.hpp",
    ],
    srcs = ["pipeline_factory.cpp"],
    deps = [
        "nodeinfo",
        "pipeline",
        "pipelinedefinition",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:model_metric_reporter",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "dag_capi",
    srcs = ["dag_capi.cpp"],
    deps = [
        "entry_node",
        "exit_node",
        "exitnodesession",
        "pipelinedefinition",
        "pipeline_factory",
        "//src:capi_dag_utils",
        "//src:capi_deserialization",
        "//src:capi_request_utils",
        "//src:capi_serialization",
        "//src:libovmscapi_utils",
        "//src:inferencerequest",
        "//src:inferenceresponse",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "dag_kfs",
    srcs = ["dag_kfs.cpp"],
    deps = [
        "entry_node",
        "exit_node",
        "exitnodesession",
        "pipelinedefinition",
        "pipeline_factory",
        "//src:kfs_deserialization",
        "//src:kfs_request_utils",
        "//src:kfs_serialization",
        "//src:libovms_kfs_utils",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "dag_tfs",
    srcs = ["dag_tfs.cpp"],
    deps = [
        "entry_node",
        "exit_node",
        "exitnodesession",
        "pipelinedefinition",
        "pipeline_factory",
        "//src:tfs_deserialization",
        "//src:tfs_request_utils",
        "//src:tfs_serialization",
        "//src:tfs_utils",
        "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "aliases",
    hdrs = ["aliases.hpp"],
    deps = [],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "dag_resource_manager",
    hdrs = ["dag_resource_manager.hpp"],
    deps = [],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "session_id",
    hdrs = ["session_id.hpp"],
    deps = [],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "tensormap",
    hdrs = ["tensormap.hpp"],
    deps = [
        "//third_party:openvino",
        "//src:libovmstensor_utils",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "nodesessionresult",
    hdrs = ["nodesessionresult.hpp"],
    deps = [],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "nodesessionmetadata",
    hdrs = ["nodesessionmetadata.hpp"],
    srcs = ["nodesessionmetadata.cpp"],
    deps = [
        "//src:libovmslogging",
        "//src:libovms_execution_context",
        "session_id",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "pipelineeventqueue",
    hdrs = ["pipelineeventqueue.hpp"],
    deps = [
        "//src:libovms_threadsafequeue",
        "session_id",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "nodeinputhandler",
    hdrs = ["nodeinputhandler.hpp"],
    srcs = ["nodeinputhandler.cpp"],
    deps = [
        "//src:libovmslogging",
        "tensormap",
        "session_id",
        "//src:libovmstensor_utils",
        "//src:libovmstimer",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "gathernodeinputhandler",
    hdrs = ["gathernodeinputhandler.hpp"],
    srcs = ["gathernodeinputhandler.cpp"],
    deps = [
        "//third_party:openvino",
        "//src:libovmslogging",
        "tensormap",
        "nodeinputhandler",
        "nodesessionmetadata",
        "session_id",
        "//src:libovmsstatus",
        "//src:libovms_tensorinfo",
        "//src:libovmsprofiler",
        "//src:libovms_ov_utils",
        "//src:libovmsshape",
        "//src:libovmsprecision",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "nodesession",
    hdrs = ["nodesession.hpp"],
    srcs = ["nodesession.cpp"],
    deps = [
        "//src:libovmslogging",
        "tensormap",
        "nodesessionmetadata",
        "gathernodeinputhandler",
        "session_id",
        "nodeinputhandler",
        "//src:libovmstensor_utils",
        "//src:libovmsstatus",
        "//src:libovmsshape",
        "//src:libovmsprecision",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "node",
    hdrs = ["node.hpp"],
    srcs = ["node.cpp"],
    deps = [
        "//src:libovmslogging",
        "//src:libovmsmodelversionstatus",
        "//src:libovms_execution_context",
        "aliases",
        "pipelineeventqueue",
        "session_id",
        "tensormap",
        "nodesession",
        "nodesessionresult",
        "nodesessionmetadata",
        "//src:libovms_ov_utils",
        "//src:libovmsprofiler",
        "//src:libovmsshape",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "pipelinedefinitionstatus",
    hdrs = ["pipelinedefinitionstatus.hpp"],
    srcs = ["pipelinedefinitionstatus.cpp"],
    deps = [
        "//src:libovmslogging",
        "//src:libovmsmodelversionstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "pipeline",
    hdrs = ["pipeline.hpp"],
    srcs = ["pipeline.cpp"],
    deps = [
        "aliases",
        "node",
        "//src:libovms_execution_context",
        "//src:libovmsprofiler",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "pipelineeventqueue",
        "nodesession",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "nodestreamidguard",
    hdrs = ["nodestreamidguard.hpp"],
    srcs = ["nodestreamidguard.cpp"],
    deps = [
        "//src:libovmsprofiler",
        "//src:libovms_ovinferrequestsqueue",
        "//src:model_metric_reporter",
        "//src:libovmsmodelversion",
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "dlnodesession",
    hdrs = ["dlnodesession.hpp"],
    srcs = ["dlnodesession.cpp"],
    deps = [
        "//third_party:openvino",
        "//src:libovms_model_instance_provider",
        "//src:modelinstance_h",
        "//src:modelinstanceunloadguard",
        "//src:libovms_ov_utils",
        "//src:libovmslogging",
        "//src:libovmsprofiler",
        "//src:libovmsmodelversion",
        "//src:libovmsshape",
        "//src:libovmsstatus",
        "//src:libovms_tensorinfo",
        "//src:libovmstimer",
        "node",
        "nodesession",
        "nodeinputhandler",
        "pipelineeventqueue",
        "nodestreamidguard",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "dl_node",
    hdrs = ["dl_node.hpp"],
    srcs = ["dl_node.cpp"],
    deps = [
        "//third_party:openvino",
        "//src:executingstreamidguard",
        "//src:modelinstance_h",
        "//src:modelinstanceunloadguard",
        "//src:libovms_model_instance_provider",
        "//src:libovms_ov_utils",
        "//src:libovmslogging",
        "//src/metrics:libovmsmetrics",
        "//src:libovmsmodelversion",
        "//src:libovmstimer",
        "node",
        "nodesession",
        "dlnodesession",
        "nodestreamidguard",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "nodeinfo",
    hdrs = ["nodeinfo.hpp"],
    deps = [
        "@fmtlib",
        "//src:libovms_threadsafequeue",
        "//src:libovmsmodelversion",
        "//src:libovms_tensorinfo",
        "node_library",
        "aliases",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "node_library",
    hdrs = ["node_library.hpp"],
    srcs = ["node_library.cpp"],
    deps = [
        "//src:custom_node_interface",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "node_library_utils",
    hdrs = ["node_library_utils.hpp"],
    srcs = ["node_library_utils.cpp"],
    deps = [
        "//src:custom_node_interface",
        "node_library",
        "tensormap",
        "//src:libovmsprecision",
        "//src:libovmsshape",
        "//src:libovmslayout",
        "//src:libovms_ov_utils",
        "//src:libovms_tensorinfo",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "custom_node_output_allocator",
    hdrs = ["custom_node_output_allocator.hpp"],
    srcs = ["custom_node_output_allocator.cpp"],
    deps = [
        "//src:custom_node_interface",
        "node_library",
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "custom_node_library_manager",
    hdrs = ["custom_node_library_manager.hpp"],
    srcs = ["custom_node_library_manager.cpp"],
    deps = [
        "node_library",
        "//src/filesystem:libovmsfilesystem",
        "//src:libovmslogging",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "custom_node_library_internal_manager_wrapper",
    hdrs = ["custom_node_library_internal_manager_wrapper.hpp"],
    deps = [
        "node_library",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "customnodesession",
    hdrs = ["customnodesession.hpp"],
    srcs = ["customnodesession.cpp"],
    deps = [
        "//src:custom_node_interface",
        "pipelineeventqueue",
        "node",
        "node_library",
        "node_library_utils",
        "nodeinputhandler",
        "nodesession",
        "tensormap",
        "custom_node_output_allocator",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovmstimer",
        "//src:libovmsprofiler",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "custom_node",
    hdrs = ["custom_node.hpp"],
    srcs = ["custom_node.cpp"],
    deps = [
        "//src:custom_node_interface",
        "pipelineeventqueue",
        "nodeinfo",
        "node_library",
        "node",
        "node_library_utils",
        "customnodesession",
        "custom_node_output_allocator",
        "custom_node_library_internal_manager_wrapper",
        "//src:libovmslogging",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "exitnodesession",
    hdrs = ["exitnodesession.hpp",
            "exitnodesession_impl.hpp",
            "gatherexitnodeinputhandler.hpp",],
    deps = [
            "nodesession",
            "nodeinputhandler",
            "nodesessionmetadata",
            "gathernodeinputhandler",
            "tensormap",
            "//src:libovmslogging",
            "//src:libovmsprofiler",
            "//src:libovmsstatus",
            "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "exit_node",
    hdrs = ["exit_node.hpp",
            "exit_node_impl.hpp",],
    deps = [
            "node",
            "exitnodesession",
            "//src:libovmslogging",
            "//src:libovms_ov_utils",
            "//src:libovms_tensorinfo",
            "//src:libovmsprofiler",
            "//src:serialization_common",
            "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "entry_node",
    hdrs = ["entry_node.hpp",
            "entry_node_impl.hpp",],
    deps = [
            "node",
            "nodesession",
            "//src:deserialization_main",
            "//src:predict_request_validation_utils",
            "//src:libovmstensorfactory",
            "//src:tensor_conversion",
            "//src:libovms_tensorinfo",
            "//src:libovmslogging",
            "//src:libovms_ov_utils",
            "//src:libovmsprofiler",
            "//src:ovms_header",
            "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
)


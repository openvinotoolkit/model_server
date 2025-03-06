#
# Copyright (c) 2020-2023 Intel Corporation
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
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@mediapipe//mediapipe/framework:more_selects.bzl", "more_selects")
load("//:common_settings.bzl",
     "COMMON_STATIC_TEST_COPTS", "COMMON_STATIC_LIBS_COPTS_VISIBLE", "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS", "COMMON_LOCAL_DEFINES", "COPTS_PYTHON", "COPTS_MEDIAPIPE", "COPTS_DROGON",
     "create_config_settings")

COPTS_OV_TRACE = select({
    "//conditions:default": ["-DOV_TRACE=0"],
    "//:not_disable_ov_trace" : ["-DOV_TRACE=1"],
})
COPTS_ADJUSTED = COMMON_STATIC_LIBS_COPTS + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_COPTS,
})
LINKOPTS_ADJUSTED = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})
COPTS_CLOUD = select({
    "//conditions:default": ["-DCLOUD_DISABLE=1"],
    "//:not_disable_cloud" : ["-DCLOUD_DISABLE=0"],
})

COPTS_TESTS = COMMON_STATIC_TEST_COPTS + COPTS_PYTHON + COPTS_MEDIAPIPE + COPTS_DROGON + COPTS_CLOUD

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "custom_nodes_common_buffersqueue",
    linkstatic = 1,
    hdrs = ["custom_nodes/common/buffersqueue.hpp"],
    srcs = [
        "custom_nodes/common/buffersqueue.hpp",
        "custom_nodes/common/buffersqueue.cpp",
    ],
    deps = ["libovms_queue"],
    copts = COMMON_STATIC_LIBS_COPTS_VISIBLE,
)

cc_shared_library(
    name = "ovms_shared",
    dynamic_deps = [],
    static_deps = [
        #"//src:libovmsfilesystemfactory", // TODO missing symbols
        "@tensorflow_serving//:__subpackages__",
        "@minitrace//:__subpackages__",
        "@com_github_gabime_spdlog//:__subpackages__",
        "@fmtlib//:__subpackages__",
        "@com_github_jarro2783_cxxopts//:__subpackages__",
        "@aws-sdk-cpp//:__subpackages__",
        "@boost//:__subpackages__",
        "@com_github_googleapis_google_cloud_cpp//:__subpackages__",
        "@curl//:__subpackages__",
        "@com_github_google_crc32c//:__subpackages__",
        "@com_github_googleapis_google_cloud_cpp_common//:__subpackages__",
        "@com_github_libevent_libevent//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@com_github_tencent_rapidjson//:__subpackages__",
        "@org_tensorflow//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@gif//:__subpackages__",
        "@libjpeg_turbo//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@fft2d//:__subpackages__",
        "@highwayhash//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@snappy//:__subpackages__",
        "@double_conversion//:__subpackages__",
        "@nsync//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "//third_party:openvino//:__subpackages__",
        "@com_github_jupp0r_prometheus_cpp//:__subpackages__",
        "@//:__subpackages__",
        "@com_github_grpc_grpc//:__subpackages__",
        "@upb//:__subpackages__",
        "@zlib//:__subpackages__",
        "@com_github_glog_glog//:__subpackages__",
        "@com_github_gflags_gflags//:__subpackages__",
        "@XNNPACK//:__subpackages__",
        "@pthreadpool//:__subpackages__",
        "@FXdiv//:__subpackages__",
        "@FP16//:__subpackages__",
        "@flatbuffers//:__subpackages__",
        "@ruy//:__subpackages__",
        "@azure//:__subpackages__",
        "@cpprest//:__subpackages__",
        "//third_party:opencv//:__subpackages__",
    ] + select({
        "//:not_disable_mediapipe": [
            "@mediapipe//:__subpackages__",
            "@mediapipe_calculators//:__subpackages__",
            "@model_api//:__subpackages__",
            "@arm_neon_2_x86_sse//:__subpackages__",
            "@ruy//:__subpackages__",
            "@cpuinfo//:__subpackages__",
            "@clog//:__subpackages__",
            "@gemmlowp//:__subpackages__",
            "@flatbuffers//:__subpackages__",],
        "//:disable_mediapipe": []
        })
     + select({
        "//:not_disable_python": [
            "@pybind11//:__subpackages__",
            "//third_party:=python3//:__subpackages__",
            "@local_config_python//:__subpackages__",
            ],
        "//:disable_python": []
        })
     + select({
            "//conditions:default": [
                "@boringssl//:__subpackages__",
            ],
            "//src:windows" : [],
        }),
    features = [],
    roots = [
        "//src:ovms_lib",
    ],
)

load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

mediapipe_proto_library(
    name = "ovmscalculatoroptions_proto", # ovmscalculatoroptions_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["mediapipe_calculators/ovmscalculator.proto",],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "libovms_queue",
    hdrs = ["queue.hpp"],
    copts = [],
    visibility = ["//visibility:public",],
    linkopts = [],
)
cc_library(
    name = "libovms_ovinferrequestsqueue",
    hdrs = ["ovinferrequestsqueue.hpp"],
    srcs = ["ovinferrequestsqueue.cpp"],
    deps = [
        "libovms_queue",
        "libovmslogging",
        "//third_party:openvino",
    ],
    copts = [],
    visibility = ["//visibility:public",],
    linkopts = [],
)
cc_library(
    name = "ovms_header",
    hdrs = ["ovms.h"],
    copts = [],
    visibility = ["//visibility:public",],
    linkopts = [],
    alwayslink = 1,
)
cc_library(
    name = "libovms_server_settings",
    hdrs = ["capi_frontend/server_settings.hpp"],
    srcs = [],
    deps = [],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_version",
    hdrs = ["version.hpp"],
    srcs = [],
    deps = [],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_systeminfo",
    hdrs = ["systeminfo.hpp"],
    srcs = ["systeminfo.cpp"],
    deps = [
        "libovmslogging",
        "libovmsstatus",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_config",
    hdrs = ["config.hpp"],
    srcs = ["config.cpp"],
    deps = [
        "libovms_server_settings",
        "libovmslogging",
        "libovmsmodelconfig",
        "libovms_cliparser",
        "libovms_systeminfo",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED + COPTS_DROGON,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_cliparser",
    hdrs = ["cli_parser.hpp"],
    srcs = ["cli_parser.cpp"],
    deps = [
        "@com_github_jarro2783_cxxopts//:cxxopts",
        "libovms_server_settings",
        "libovms_version",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_capi_servable_metadata",
    hdrs = ["capi_frontend/servablemetadata.hpp"],
    srcs = ["capi_frontend/servablemetadata.cpp"],
    deps = [
        "libovms_tensorinfo",
        "libovmsmodelversion",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_sequence",
    hdrs = ["sequence.hpp"],
    srcs = ["sequence.cpp"],
    deps = [
        "libovmsstatus",
        "libovms_ov_utils",
        "//third_party:openvino",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_sequence_manager",
    hdrs = ["sequence_manager.hpp"],
    srcs = ["sequence_manager.cpp"],
    deps = [
        "libovms_sequence",
        "libovmsmodelversion",
        "libovmsstatus",
        "libovmslogging",
        "libovms_sequence_processing_spec",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_sequence_processing_spec",
    hdrs = ["sequence_processing_spec.hpp"],
    deps = [],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_kfs_utils",
    hdrs = ["kfs_frontend/kfs_utils.hpp"],
    srcs = ["kfs_frontend/kfs_utils.cpp"],
    deps = [
        "//src/kfserving_api:kfserving_api_cpp",
        "libovmslogging",
        "libovmsstatus",
        "libovmsprofiler",
        "libovms_tensorinfo",
        "libovmsprecision",
        "libovmsmodelversion",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_kfs_grpc_inference_service_h",
    hdrs = ["kfs_frontend/kfs_grpc_inference_service.hpp"],
    deps = [
        "//src/kfserving_api:kfserving_api_cpp",
        "//third_party:openvino",
        "@com_github_grpc_grpc//:grpc++",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "ovms_lib",
    hdrs = ["ovms.h",
            "ovms_internal.h",
    ],
    srcs = [
        "capi_frontend/capi.cpp",
        "cleaner_utils.cpp",
        "cleaner_utils.hpp",
        "custom_node_interface.h",
        "customloaderconfig.hpp",
        "customloaders.hpp",
        "customloaders.cpp",
        "customloaderinterface.hpp",
        "deserialization.cpp",
        "deserialization.hpp",
        "dags/custom_node.cpp",
        "dags/custom_node.hpp",
        "dags/custom_node_library_manager.cpp",
        "dags/custom_node_library_manager.hpp",
        "dags/custom_node_output_allocator.cpp",
        "dags/custom_node_output_allocator.hpp",
        "dags/custom_node_library_internal_manager_wrapper.hpp",
        "dags/custom_node_library_internal_manager_wrapper.cpp",
        "dags/customnodesession.cpp",
        "dags/customnodesession.hpp",
        "dags/dl_node.cpp",
        "dags/dl_node.hpp",
        "dags/dlnodesession.cpp",
        "dags/dlnodesession.hpp",
        "dags/entry_node.cpp",
        "dags/entry_node.hpp",
        "dags/exit_node.cpp",
        "dags/exit_node.hpp",
        "dags/exitnodesession.cpp",
        "dags/exitnodesession.hpp",
        "dags/gatherexitnodeinputhandler.hpp",
        "dags/nodeinfo.hpp",
        "dags/node_library.cpp",
        "dags/node_library.hpp",
        "dags/node_library_utils.cpp",
        "dags/node_library_utils.hpp",
        "dags/nodestreamidguard.cpp",
        "dags/nodestreamidguard.hpp",
        "dags/pipeline.cpp",
        "dags/pipeline.hpp",
        "dags/pipelinedefinition.cpp",
        "dags/pipelinedefinition.hpp",
        "dags/pipelinedefinitionunloadguard.cpp",
        "dags/pipelinedefinitionunloadguard.hpp",
        "dags/pipeline_factory.cpp",
        "dags/pipeline_factory.hpp",
        "executingstreamidguard.cpp",
        "executingstreamidguard.hpp",
        "get_model_metadata_impl.cpp",
        "get_model_metadata_impl.hpp",
        "global_sequences_viewer.hpp",
        "global_sequences_viewer.cpp",
        "http_server.hpp",
        "http_server.cpp",
        "http_rest_api_handler.hpp",
        "httpservermodule.hpp",
        "http_rest_api_handler.cpp",
        "httpservermodule.cpp",
        "grpc_utils.cpp",
        "grpc_utils.hpp",
        "grpcservermodule.cpp",
        "grpcservermodule.hpp",
        "kfs_frontend/kfs_grpc_inference_service.cpp",
        "model.cpp",
        "model.hpp",
        "modelchangesubscription.cpp",
        "modelchangesubscription.hpp",
        "modelmanager.cpp",
        "modelmanager.hpp",
        "modelinstance.cpp",
        "modelinstance.hpp",
        "modelinstanceunloadguard.cpp",
        "modelinstanceunloadguard.hpp",
        "model_service.hpp",
        "model_service.cpp",
        "model_metric_reporter.cpp",
        "model_metric_reporter.hpp",
        "ovms.h",
        "ovms_exit_codes.hpp",
        "prediction_service.cpp",
        "prediction_service.hpp",
        "prediction_service_utils.hpp",
        "prediction_service_utils.cpp",
        "predict_request_validation_utils.hpp",
        "predict_request_validation_utils.cpp",
        "rest_parser.cpp",
        "rest_parser.hpp",
        "rest_utils.cpp",
        "rest_utils.hpp",
        "serialization.cpp",
        "serialization.hpp",
        "servablemanagermodule.cpp",
        "servablemanagermodule.hpp",
        "server.cpp",
        "server.hpp",
        "statefulmodelinstance.cpp",
        "statefulmodelinstance.hpp",
        "tensor_conversion.hpp",
        "tensor_conversion.cpp",
         ] + select({
            "//:not_disable_mediapipe": [
                "mediapipe_internal/mediapipefactory.cpp",
                "mediapipe_internal/mediapipefactory.hpp",
                "mediapipe_internal/mediapipegraphconfig.hpp",
                "mediapipe_internal/mediapipegraphconfig.cpp",
                "mediapipe_internal/mediapipegraphdefinition.cpp",
                "mediapipe_internal/mediapipegraphdefinition.hpp",
                "mediapipe_internal/mediapipegraphexecutor.cpp",
                "mediapipe_internal/mediapipegraphexecutor.hpp",
                "mediapipe_internal/packettypes.hpp",
                "kfs_frontend/kfs_graph_executor_impl.cpp",
                "kfs_frontend/kfs_graph_executor_impl.hpp",
                "http_frontend/http_graph_executor_impl.cpp",
                "http_frontend/http_graph_executor_impl.hpp",
            ],
            "//:disable_mediapipe" : [],
        }),
        deps = select({
            "//:not_disable_python": [
                "//src/python:libovmspythonmodule",
                # Jinja template processing is done in Python
                "//src/llm:llmcalculator",
                "//src/llm:genai_servables",
                "//src/llm:text_processor",
            ],
            "//:disable_python": []
        }) + select({
            "//conditions:default": [],
            "//:not_disable_mediapipe" : [
                "//src/llm:openai_completions_api_handler",
                "//src/embeddings:embeddingscalculator",
                "//src/rerank:rerankcalculator",],
        }) + select({
            "//:enable_drogon": [
                "libdrogon_http_server",
                "libovmsmulti_part_parser_drogon_impl",
            ],
            "//conditions:default" : [
                "libnet_http_server",
                # TODO
            ],
        }) + [
            "cpp_headers",
            "libovms_capi_servable_metadata",
            "libmodelconfigjsonparser",
            "libovmsmodelconfig",
            "libovmscapibuffer",
            "libovmscapi_utils",
            "libovmscapiinferenceparameter",
            "libovmscapiinferencerequest",
            "libovmscapiinferenceresponse",
            "libovmscapiinferencetensor",
            "libovms_cliparser",
            "libovms_config",
            "libovmsdagspipelinedefinitionstatus",
            "libovms_dags_aliases",
            "libovms_dags_node",
            "libovms_dags_session_id",
            "libovms_dags_pipelineeventqueue",
            "libovms_dags_nodesessionmetadata",
            "libovms_dags_nodesessionresult",
            "libovms_dags_nodeinputhandler",
            "libovms_execution_context",
            "libovmsfilesystem",
            "libovmsfilesystemfactory_h",
            "libovmsfilesystemfactory", # TODO missing symbols when not in ovms_lib
            "libovms_kfs_utils",
            "libovms_kfs_grpc_inference_service_h",
            "libovmslocalfilesystem", # indirectly & directly through factory
            "libovmslogging",
            "libovmsmetrics",
            "libovms_ov_utils",
            "libovmsprecision",
            "libovmsshape",
            "libovmsschema",
            "libovmslayout",
            "libovmslayout_configuration",
            "libovms_tensorinfo",
            "libovmstimer",
            "libovmstfs_grpc",
            "libovms_version",
            "libovmsprofiler",
            "libovmsstatus",
            "libovms_sequence",
            "libovms_sequence_manager",
            "libovms_sequence_processing_spec",
            "libovms_queue",
            "libovms_ovinferrequestsqueue",
            "libovmstensor_utils",
            "libovms_dags_tensormap",
            "libovmsstring_utils",
            "libovmsmodelversioning",
            "httppayload",
            "libhttpclientconnection",
	    #"libovmsoclutils", # TODO decide if will need
            "libovmstensorfactory",
            "libovms_threadsafequeue",
            "//:ovms_dependencies", # TODO dispose or replace
            "//src/kfserving_api:kfserving_api_cpp",
            "capimodule",
         ] + select({
                "//:not_disable_cloud": [
                    "libovmsazurefilesystem",
                    ],
                "//:disable_cloud": []
                })
            + select({
                "//conditions:default": [],
                "//src:windows" : [
                    "@windows_opencl//:opencl",
                    "@windows_opencl2//:opencl2",
                    ],
                })
            + select({
            "//conditions:default": [
                "@mediapipe_calculators//:mediapipe_calculators", # Need this dependencies here because we use ovms/src - cannot add in ovms_dependencies because we copy src directory later in Dockerfile
                "@mediapipe//mediapipe/graphs/holistic_tracking:holistic_tracking_to_render_data",
                "@mediapipe//mediapipe/graphs/iris_tracking:iris_tracking_cpu_deps",
                "@mediapipe//mediapipe/calculators/tensor:image_to_tensor_calculator",
                "@mediapipe//mediapipe/modules/holistic_landmark:holistic_landmark_cpu",
                "libovmsmediapipe_utils",
                "@mediapipe//mediapipe/calculators/geti/inference:inference_calculators",
                "@mediapipe//mediapipe/calculators/geti/utils:utils",
                "@mediapipe//mediapipe/calculators/geti/utils:emptylabel_calculators",
                "@mediapipe//mediapipe/calculators/geti/serialization:calculators",
            ],
            "//:disable_mediapipe" : [],
        }),
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED + COPTS_PYTHON + COPTS_MEDIAPIPE + COPTS_DROGON + COPTS_OV_TRACE,
    linkopts = LINKOPTS_ADJUSTED + select({
                "//conditions:default": ["-lOpenCL"], # TODO make as direct dependency
                "//src:windows" : ["/DEFAULTLIB:Rpcrt4.lib"],}),
    data = select({
        "//:not_disable_python": [
            "//src/python/binding:pyovms.so",
        ],
        "//:disable_python": []
        }) + select({
            "//:is_windows_and_python_is_enabled": [
                "//src/python/binding:copy_pyovms",
            ],
            "//conditions:default": [],
        }),
    alwayslink = 1,
)

cc_library(
    name = "libovmsprofiler",
    hdrs = ["profilermodule.hpp",
            "profiler.hpp"],
    srcs = [
        "profiler.cpp",
        "profilermodule.cpp",],
    deps = [
        "@minitrace//:trace",
        "libovms_module",
        "//src:cpp_headers",
        "//src:libovmslogging",
        "//src:libovmsstatus",
        "//src:libovmsprecision",
        "//src:libovmsmodelversioning",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libovmslogging",
    hdrs = ["logging.hpp"],
    srcs = ["logging.cpp",],
    deps = [
        "@com_github_gabime_spdlog//:spdlog",
        "@com_github_glog_glog//:glog", # used to manage Mediapipe logging
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED + COPTS_OV_TRACE,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "httppayload",
    hdrs = ["http_payload.hpp"],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
        "libovmsclient_connection",
        "libovmsmulti_part_parser",
    ],
    visibility = ["//visibility:public"],
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libovmsclient_connection",
    hdrs = ["client_connection.hpp"],
    deps = [
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libhttpclientconnection",
    hdrs = ["http_frontend/http_client_connection.hpp"],
    deps = [
        "libovmsclient_connection",
        "libhttp_async_writer_interface",  # interface for http output streaming
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libovmsmulti_part_parser",
    hdrs = ["multi_part_parser.hpp"],
    deps = [
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libovmsmulti_part_parser_drogon_impl",
    hdrs = ["http_frontend/multi_part_parser_drogon_impl.hpp"],
    srcs = ["http_frontend/multi_part_parser_drogon_impl.cpp"],
    deps = [
        "libovmsmulti_part_parser",
        "@drogon//:drogon_cmake",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libovms_module",
    hdrs = ["module.hpp"],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsprecision",
    hdrs = ["precision.hpp"],
    srcs = ["precision.cpp",],
    deps = ["//third_party:openvino",], # TODO dispose
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

load("@pip_deps//:requirements.bzl", "all_requirements")

cc_binary(
    name = "lib_node_mock.so",
    srcs = [
	    "test/custom_nodes/node_mock.c",
	    "custom_node_interface.h",
    ],
    linkshared = 1,
)

cc_binary(
    name = "lib_node_missing_implementation.so",
    srcs = [
	    "test/custom_nodes/node_missing_implementation.c",
	    "custom_node_interface.h",
    ],
    linkshared = 1,
)

cc_binary(
    name = "lib_node_add_sub.so",
    srcs = [
	    "test/custom_nodes/node_add_sub.c",
	    "custom_node_interface.h",
    ],
    linkshared = 1,
)

cc_binary(
    name = "lib_node_dynamic_image.so",
    srcs = [
	    "test/custom_nodes/node_dynamic_image.c",
	    "custom_node_interface.h",
    ],
    linkshared = 1,
)

cc_binary(
    name = "lib_node_choose_maximum.so",
    srcs = [
        "test/custom_nodes/node_choose_maximum.cpp",
        "custom_node_interface.h",
    ],
    linkshared = 1,
)
cc_binary(
    name = "lib_node_perform_different_operations.so",
    srcs = [
        "test/custom_nodes/node_perform_different_operations.cpp",
        "custom_node_interface.h",
    ],
    linkshared = 1,
)

cc_binary(
    name = "lib_node_dynamic_demultiplex.so",
    srcs = [
        "test/custom_nodes/node_dynamic_demultiplex.cpp",
        "custom_node_interface.h",
    ],
    linkshared = 1,
)

cc_binary(
    name = "libcustom_node_east_ocr.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/opencv_utils.hpp",
        "custom_nodes/east_ocr/east_ocr.cpp",
        "custom_nodes/east_ocr/nms.hpp",
        "custom_node_interface.h",
    ],
    deps = [
        "//third_party:opencv",
    ],
    linkshared = 1,
    copts = COMMON_STATIC_LIBS_COPTS_VISIBLE,
)

cc_binary(
    name = "libcustom_node_face_blur.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/opencv_utils.hpp",
        "custom_nodes/face_blur/face_blur.cpp",
        "custom_node_interface.h",
    ],
    deps = [
        "//third_party:opencv",
    ],
    linkshared = 1,
    copts =COMMON_STATIC_LIBS_COPTS_VISIBLE,
)

cc_binary(
    name = "libcustom_node_add_one.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/custom_node_library_internal_manager.hpp",
        "custom_nodes/common/custom_node_library_internal_manager.cpp",
        "custom_nodes/add_one/add_one.cpp",
        "custom_node_interface.h",
        "custom_nodes/add_one/add_one_internal_manager.hpp"
    ],
    deps = [
        "custom_nodes_common_buffersqueue",
        "libovms_queue",
    ],
    linkshared = 1,
    copts = COMMON_STATIC_LIBS_COPTS_VISIBLE,
)

cc_binary(
    name = "libcustom_node_model_zoo_intel_object_detection.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/opencv_utils.hpp",
        "custom_nodes/common/custom_node_library_internal_manager.hpp",
        "custom_nodes/common/custom_node_library_internal_manager.cpp",
        "custom_nodes/model_zoo_intel_object_detection/model_zoo_intel_object_detection.cpp",
        "custom_node_interface.h",
    ],
    deps = [
        "custom_nodes_common_buffersqueue",
        "//third_party:opencv",
        "libovms_queue",
    ],
    linkshared = 1,
    copts = COMMON_STATIC_LIBS_COPTS_VISIBLE,
)

cc_binary(
    name = "libcustom_node_image_transformation.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/opencv_utils.hpp",
        "custom_nodes/image_transformation/image_transformation.cpp",
        "custom_node_interface.h",
    ],
    deps = [
        "//third_party:opencv",
    ],
    linkshared = 1,
    copts = COMMON_STATIC_LIBS_COPTS_VISIBLE,
)

cc_binary(
    name = "libcustom_node_horizontal_ocr.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/opencv_utils.hpp",
        "custom_nodes/horizontal_ocr/horizontal_ocr.cpp",
        "custom_node_interface.h",
    ],
    deps = [
        "//third_party:opencv",
    ],
    linkshared = 1,
    copts = COMMON_STATIC_LIBS_COPTS_VISIBLE,
)

# POC which can start OVMS with C-API built by bazel
# Standalone example for building outside of bazel (with bare g++ is inside MakefileCapi)
cc_binary(
    name = "capi_cpp_example",
    srcs = [
        "main_capi.cpp",
    ],
    linkopts = LINKOPTS_ADJUSTED,
    copts = [
    ],
    deps = [
        "//src:ovms_lib",
        "//src:libovmsfilesystemfactory",
    ],
    linkstatic = True,
)

# Use for dynamic linking when necessary
cc_import(
  name = "shared_lib",
  hdrs = ["ovms.h"],
  shared_library = "ovms_shared",
)

cc_library(
    name = "libovmstfs_grpc",
    hdrs = ["tfs_frontend/tfs_utils.hpp",],
    srcs = [
        "tfs_frontend/tfs_utils.cpp",],
    deps = [
        "@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto",
        "@tensorflow_serving//tensorflow_serving/apis:model_service_cc_proto",
        "@org_tensorflow//tensorflow/core:framework", # Eigen Tensor
        "libovmslogging",
        "libovmsprofiler",
        "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

# Unused
cc_library( # TODO split dependencies
    name = "libovmshttpservermodule",
    hdrs = [
        "http_server.hpp",
        "http_rest_api_handler.hpp",
        "httpservermodule.hpp",
    ],
    srcs = [
        "http_rest_api_handler.cpp",
        "http_server.cpp",
        "httpservermodule.cpp",
    ],
    deps = [
        "@com_github_jupp0r_prometheus_cpp//core",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto",
        "@tensorflow_serving//tensorflow_serving/util:threadpool_executor",
        "@tensorflow_serving//tensorflow_serving/util:json_tensor",
        "libovms_module",
        "libovmslogging",
        "libovmsprofiler",
        "libovmsstatus",
        "libdrogon_http_server",
        "//src/kfserving_api:kfserving_api_cpp",
    ] + select({
            "//:not_disable_python": [
                "//src/python:libovmspythonmodule",
            ],
            "//:disable_python": []
        }),
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmsschema",
    hdrs = ["schema.hpp",],
    srcs = ["schema.cpp",],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
        "libovmsstatus",
        "libovmslogging",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED + COPTS_MEDIAPIPE + COPTS_DROGON,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmsmetrics",
    hdrs = [
        "metric.hpp",
        "metric_config.hpp",
        "metric_family.hpp",
        "metric_module.hpp",
        "metric_registry.hpp",
    ],
    srcs = [
        "metric.cpp",
        "metric_config.cpp",
        "metric_family.cpp",
        "metric_registry.cpp",
        "metric_module.cpp",
    ],
    deps = [
        "@com_github_jupp0r_prometheus_cpp//core",
        "@com_github_tencent_rapidjson//:rapidjson",
        "cpp_headers",
        "libovms_module",
        "libovmslogging",
        "libovmsschema",
        "libovmsstatus",
        "libovmsstring_utils",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmsstatus",
    hdrs = ["status.hpp",],
    srcs = ["status.cpp",],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmsmediapipe_utils",
    hdrs = ["mediapipe_internal/mediapipe_utils.hpp",
            "mediapipe_internal/packettypes.hpp",],
    srcs = ["mediapipe_internal/mediapipe_utils.cpp",],
    deps = ["libovmsstring_utils",
            "libovmsstatus",
            "libovmslogging",],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library( # make ovms_lib dependent, use share doptions
    name = "libovmsstring_utils",
    hdrs = ["stringutils.hpp",],
    srcs = ["stringutils.cpp",],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmsmodelversioning",
    hdrs = [
            "model_version_policy.hpp",],
    srcs = ["model_version_policy.cpp",],
    deps = [
            "libovmsmodelversion",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library( # make ovms_lib dependent, use share doptions
    name = "libovmsazurefilesystem",
    hdrs = ["azurefilesystem.hpp",],
    srcs = [
            "azurefilesystem.cpp",
            "azurestorage.cpp",
            "azurestorage.hpp",
    ],
    deps = [
        "@azure//:storage",
        "@cpprest//:sdk",
        "libovmsfilesystem",
        "libovmslogging",
        "libovmsstring_utils",
        "libovmsmodelversioning",
        "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsgcsfilesystem",
    hdrs = ["gcsfilesystem.hpp",],
    srcs = [
            "gcsfilesystem.cpp",
    ],
    deps = [
        "@com_github_googleapis_google_cloud_cpp//google/cloud/storage:storage_client",
        "libovmsfilesystem",
        "libovmslogging",
        "libovmsstring_utils",
        "libovmsmodelversioning",
        "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
# TODO split
cc_library(
    name = "libovmstensorfactory",
    hdrs = ["itensorfactory.hpp",
            "regularovtensorfactory.hpp",] 
            + select({
                "//conditions:default": [
                    "vaapitensorfactory.hpp",
                    "opencltensorfactory.hpp",
                ],
                "//src:windows" : [],
            }),
    srcs = ["regularovtensorfactory.cpp"]
            + select({
                "//conditions:default": [
                    "vaapitensorfactory.cpp",
                    "opencltensorfactory.cpp",
                ],
                "//src:windows" : [],
            }),
    deps = ["//third_party:openvino",
            "libovmscapibuffer",
            "libovmslogging",
            "ovms_header",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmscapibuffer",
    hdrs = ["capi_frontend/buffer.hpp",
    ],
    srcs = [
        "capi_frontend/buffer.cpp",
    ],
    deps = [
            "ovms_header",
            "libovmslogging",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmscapi_utils_h",
    hdrs = ["capi_frontend/capi_utils.hpp",],
    deps = [
            "ovms_header",
            "libovmsshape",
            "libovmsprecision",
            "libovmsdagspipelinedefinitionstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmscapi_utils",
    hdrs = [],
    srcs = ["capi_frontend/capi_utils.cpp",],
    deps = [
            "libovmscapi_utils_h",
            "libovmslogging",
            "libovmsstatus",
            "libovmscapibuffer",
            "libovmscapiinferencerequest",
            "libovmscapiinferenceresponse",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmscapiinferenceparameter",
    hdrs = ["capi_frontend/inferenceparameter.hpp",],
    srcs = ["capi_frontend/inferenceparameter.cpp",],
    deps = [
            "ovms_header",
            "libovmscapi_utils_h",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmscapiinferencetensor",
    hdrs = ["capi_frontend/inferencetensor.hpp",
    ],
    srcs = [
        "capi_frontend/inferencetensor.cpp",
    ],
    deps = [
            "ovms_header",
            "libovmscapibuffer",
            "libovmslogging",
            "libovmsshape",
            "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmscapiinferencerequest",
    hdrs = ["capi_frontend/inferencerequest.hpp",],
    srcs = ["capi_frontend/inferencerequest.cpp",],
    deps = [
            "ovms_header",
            "libovmscapiinferencetensor",
            "libovmscapiinferenceparameter",
            "libovmscapibuffer",
            "libovmsmodelversion",
            "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmscapiinferenceresponse",
    hdrs = ["capi_frontend/inferenceresponse.hpp",],
    srcs = ["capi_frontend/inferenceresponse.cpp",],
    deps = [
            "ovms_header",
            "libovmscapiinferencetensor",
            "libovmscapiinferenceparameter",
            "libovmsmodelversion",
            "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsmodelconfig",
    hdrs = ["modelconfig.hpp",],
    srcs = ["modelconfig.cpp",],
    deps = [
            "@com_github_tencent_rapidjson//:rapidjson", # TODO split into parser
            "libovmsfilesystem",
            "libovmslayout_configuration",
            "libovmsmodelversioning",
            "libovmsschema",
            "libovmsshape",
            "libovmsstatus",
            "libovmsstring_utils",
            "libmodelconfigjsonparser",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsshape",
    hdrs = ["shape.hpp",],
    srcs = ["shape.cpp",],
    deps = [
            "//third_party:openvino",
            "ovms_header",
            "libovmslogging",
            "libovmsstatus",
            "libovmsstring_utils",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmstimer",
    hdrs = ["timer.hpp",],
    deps = [
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmslayout",
    hdrs = ["layout.hpp",],
    srcs = ["layout.cpp",],
    deps = [
            "libovmsshape",
            "//third_party:openvino",
            "ovms_header",
            "libovmslogging",
            "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmslayout_configuration",
    hdrs = ["layout_configuration.hpp",],
    srcs = ["layout_configuration.cpp",],
    deps = [
            "//third_party:openvino",
            "libovmslayout",
            "ovms_header",
            "libovmsstatus",
            "libovmsstring_utils",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsmodelversion",
    hdrs = [
        "modelversion.hpp",
    ],
    srcs = [],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsmodelversionstatus",
    hdrs = ["modelversionstatus.hpp",],
    srcs = ["modelversionstatus.cpp",],
    deps = [
            "libovmslogging",
            "libovmsmodelversion",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmstensor_utils",
    hdrs = ["tensor_utils.hpp",],
    deps = [
            "//third_party:openvino",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_nodesessionresult",
    hdrs = [
        "dags/nodesessionresult.hpp",
    ],
    srcs = [
    ],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_nodesessionmetadata",
    hdrs = [
        "dags/nodesessionmetadata.hpp",
    ],
    srcs = [
        "dags/nodesessionmetadata.cpp",
    ],
    deps = [
        "libovmslogging",
        "libovms_execution_context",
        "libovms_dags_session_id",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_aliases",
    hdrs = ["dags/aliases.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_pipelineeventqueue",
    hdrs = ["dags/pipelineeventqueue.hpp",],
    deps = [
        "libovms_threadsafequeue",
        "libovms_dags_session_id",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_threadsafequeue",
    hdrs = ["threadsafequeue.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_tensorinfo",
    hdrs = ["tensorinfo.hpp",],
    srcs = ["tensorinfo.cpp",],
    deps = [
            "//third_party:openvino",
            "libovmslogging",
            "libovmsstring_utils",
            "libovmsprecision",
            "libovmslayout",
            "libovmsshape",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_ov_utils",
    hdrs = ["ov_utils.hpp",],
    srcs = ["ov_utils.cpp",],
    deps = [
            "//third_party:openvino",
            "libovmslogging",
            "libovmsstatus",
            "libovmsshape",
            "libovmsprofiler",
            "libovms_tensorinfo",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_execution_context",
    hdrs = ["execution_context.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_session_id",
    hdrs = ["dags/session_id.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_tensormap",
    hdrs = ["dags/tensormap.hpp",],
    deps = [
            "//third_party:openvino",
            "libovmstensor_utils",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_gathernodeinputhandler",
    hdrs = ["dags/gathernodeinputhandler.hpp",],
    srcs = ["dags/gathernodeinputhandler.cpp",],
    deps = [
            "//third_party:openvino",
            "libovmslogging",
            "libovms_dags_tensormap",
            "libovms_dags_nodeinputhandler",
            "libovms_dags_nodesessionmetadata",
            "libovms_dags_session_id",
            "libovmsstatus",
            "libovms_tensorinfo",
            "libovmsprofiler",
            "libovms_ov_utils",
            "libovmsshape",
            "libovmsprecision",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_nodeinputhandler",
    hdrs = ["dags/nodeinputhandler.hpp",],
    srcs = ["dags/nodeinputhandler.cpp",],
    deps = [
            "libovmslogging",
            "libovms_dags_tensormap",
            "libovms_dags_session_id",
            "libovmstensor_utils",
            "libovmstimer",
            "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_nodesession",
    hdrs = ["dags/nodesession.hpp",],
    srcs = ["dags/nodesession.cpp",],
    deps = [
            "libovmslogging",
            "libovms_dags_tensormap",
            "libovms_dags_nodesessionmetadata",
            "libovms_dags_gathernodeinputhandler",
            "libovms_dags_session_id",
            "libovms_dags_nodeinputhandler",
            "libovmstensor_utils",
            "libovmsstatus",
            "libovmsshape",
            "libovmsprecision",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovms_dags_node",
    hdrs = ["dags/node.hpp",],
    srcs = ["dags/node.cpp",],
    deps = [
            "libovmslogging",
            "libovmsmodelversionstatus",
            "libovms_execution_context",
            "libovms_dags_aliases",
            "libovms_dags_pipelineeventqueue",
            "libovms_dags_session_id",
            "libovms_dags_tensormap",
            "libovms_dags_nodesession",
            "libovms_dags_nodesessionresult",
            "libovms_dags_nodesessionmetadata",
            "libovms_ov_utils",
            "libovmsprofiler",
            "libovmsshape",
            "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsdagspipelinedefinitionstatus",
    hdrs = ["dags/pipelinedefinitionstatus.hpp",],
    srcs = ["dags/pipelinedefinitionstatus.cpp",],
    deps = [
            "libovmslogging",
            "libovmsmodelversionstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsoclutils",
    hdrs = ["ocl_utils.hpp",
    ],
    srcs = [
        "ocl_utils.cpp",
    ],
    deps = [
            "//third_party:openvino",
            "libovmslogging",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmss3filesystem",
    hdrs = ["s3filesystem.hpp",],
    srcs = [
            "s3filesystem.cpp",
    ],
    deps = [
        "@aws-sdk-cpp//:aws-sdk-cpp",
        "libovmsfilesystem",
        "libovmslogging",
        "libovmsstring_utils",
        "libovmsmodelversioning",
        "libovmsstatus",
    ],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "libovmsfilesystem",
    hdrs = ["filesystem.hpp",],
    srcs = ["filesystem.cpp",],
    deps = [
            "libovmslogging",
            "libovmsmodelversioning",
            "libovmsstatus",
            "libovmsstring_utils",
            "@boringssl//:ssl"],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmslocalfilesystem",
    hdrs = ["localfilesystem.hpp",],
    srcs = ["localfilesystem.cpp",],
    deps = ["libovmsfilesystem",
            "libovmsstatus",
            "libovmslogging"],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmsfilesystemfactory_h",
    hdrs = ["filesystemfactory.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libovmsfilesystemfactory",
    hdrs = [],
    srcs = ["filesystemfactory.cpp",],
    deps = ["libovmsfilesystem",
            "libovmsfilesystemfactory_h",
            "libovmsstatus",
            "libovmslocalfilesystem",
            "libovmslogging"
            ] + select({
            "//:not_disable_cloud": [
                "libovmsazurefilesystem",
                "libovmsgcsfilesystem",
                "libovmss3filesystem",
                ],
            "//:disable_cloud": []
            }),
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED + COPTS_CLOUD,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libmodelconfigjsonparser",
    hdrs = [
        "json_parser.hpp",
    ],
    srcs = ["json_parser.cpp"],
    deps = ["cpp_headers",
            "@com_github_tencent_rapidjson//:rapidjson",
            "//third_party:openvino",
            "libovmslogging"],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

  # interface for http output streaming
cc_library(
    name = "libhttp_async_writer_interface",
    hdrs = [
        "http_async_writer_interface.hpp",
    ],
    deps = [
        "libhttp_status_code"
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libhttp_status_code",
    srcs = [
        "http_status_code.hpp",
    ],
    deps = [
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

# HTTP Server implementation using Drogon
# To use other library simply create new target and implementation of libhttp_async_writer_interface
cc_library(
    name = "libdrogon_http_server",
    srcs = [
        "drogon_http_server.hpp",
        "drogon_http_server.cpp",
        "drogon_http_async_writer_impl.hpp",
        "drogon_http_async_writer_impl.cpp",
        "timer.hpp"
    ],
    deps = [
        "libhttp_async_writer_interface",
        "@drogon//:drogon_cmake",
        "@mediapipe//mediapipe/framework/port:threadpool",
        "libovmslogging",
        "libovmsstatus",
        "libhttp_status_code"
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

# HTTP Server implementation using net_http of tensorflow
# To use other library simply create new target and implementation of libhttp_async_writer_interface
cc_library(
    name = "libnet_http_server",
    srcs = [
        "net_http_async_writer_impl.hpp",
        "net_http_async_writer_impl.cpp",
    ],
    deps = [
        "libhttp_async_writer_interface",
        "@tensorflow_serving//tensorflow_serving/util/net_http/server/public:http_server",
        "@tensorflow_serving//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@tensorflow_serving//tensorflow_serving/util:threadpool_executor",
        "libovmslogging",
        "libovmsstatus",
        "libhttp_status_code"
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library( # TODO split further
    name = "cpp_headers",
    hdrs = [
        "capi_frontend/server_settings.hpp",
    ],
    srcs = ["server.hpp",
        "config.hpp",
        "status.hpp",
        "module_names.hpp",
        "module_names.cpp",
    ],
    deps = [],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_binary(
    name = "capi_benchmark",
    srcs = [
        "main_benchmark.cpp",
    ],
    linkopts = [
        "-lpthread",
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
    ],
    copts = [
    ],
    deps = [
        "//src:ovms_lib",
        "//src:libovmsfilesystemfactory",
    ],
    linkstatic = True,
)

cc_binary(
    name = "ovms",
    srcs = [
        "main.cpp",
    ],
    linkopts = LINKOPTS_ADJUSTED,
    copts = COMMON_STATIC_LIBS_COPTS,
    deps = [
        "//src:ovms_lib",
        "//src:libovmsfilesystemfactory",]
    + select({
            "//conditions:default": [
                "//src:libcustom_node_east_ocr.so",
                "//src:libcustom_node_face_blur.so",
                "//src:libcustom_node_horizontal_ocr.so",
                "//src:libcustom_node_model_zoo_intel_object_detection.so",
                "//src:libcustom_node_image_transformation.so",
            ],
            "//src:windows" : [],
        })
     + select({
            "//conditions:default": [
                "@mediapipe//mediapipe/calculators/ovms:ovms_calculator",
            ],
            "//:disable_mediapipe" : [],
        }),
    data = select({
        "//:not_disable_python": [
            "//src/python/binding:pyovms.so",
        ],
        "//:disable_python": []
    }),
    # linkstatic = False,  # Use for dynamic linking when necessary
)

cc_test(
    name = "ovms_test",
    linkstatic = 1,
    srcs = [
        "test/c_api_stress_tests.cpp",
        "test/c_api_tests.cpp",
        "test/capi_predict_validation_test.cpp",
        "test/custom_loader_test.cpp",
        "test/custom_node_buffersqueue_test.cpp",
        "test/custom_node_output_allocator_test.cpp",
        "test/demultiplexer_node_test.cpp",
        "test/deserialization_tests.cpp",
        "test/ensemble_config_change_stress.cpp",
        "test/ensemble_flow_custom_node_tests.cpp",
        "test/ensemble_mapping_config_tests.cpp",
        "test/ensemble_metadata_test.cpp",
        "test/ensemble_tests.cpp",
        "test/gather_node_test.cpp",
        "test/get_model_metadata_response_test.cpp",
        "test/standalone_http_server_test.cpp",
        "test/get_model_metadata_signature_test.cpp",
        "test/get_model_metadata_validation_test.cpp",
        "test/get_pipeline_metadata_response_test.cpp",
        "test/http_rest_api_handler_test.cpp",
        "test/kfs_metadata_test.cpp",
        "test/kfs_rest_test.cpp",
        "test/kfs_rest_parser_test.cpp",
        "test/layout_test.cpp",
        "test/metric_config_test.cpp",
        "test/metrics_test.cpp",
        "test/metrics_flow_test.cpp",
        "test/mockmodelinstancechangingstates.hpp",
        "test/model_cache_test.cpp",
        "test/model_service_test.cpp",
        "test/model_test.cpp",
        "test/model_version_policy_test.cpp",
        "test/modelconfig_test.cpp",
        "test/modelinstance_test.cpp",
        "test/modelmanager_test.cpp",
        "test/modelversionstatus_test.cpp",
        "test/node_library_manager_test.cpp",
        "test/nodesessionmetadata_test.cpp",
        "test/ov_utils_test.cpp",
        "test/ovinferrequestqueue_test.cpp",
        "test/ovmsconfig_test.cpp",
        "test/pipelinedefinitionstatus_test.cpp",
        "test/predict_validation_test.cpp",
        "test/prediction_service_test.cpp",
        "test/rest_utils_test.cpp",
        "test/schema_test.cpp",
        "test/sequence_manager_test.cpp",
        "test/sequence_test.cpp",
        "test/serialization_tests.cpp",
        "test/server_test.cpp",
        "test/shape_test.cpp",
        "test/stateful_config_test.cpp",
        "test/stateful_modelinstance_test.cpp",
        "test/stateful_test_utils.hpp",
        "test/status_test.cpp",
        "test/stringutils_test.cpp",
        "test/systeminfo_test.cpp",
        "test/tensor_conversion_test.cpp",
        "test/tensorinfo_test.cpp",
        "test/tensorutils_test.cpp",
        "test/test_http_utils.hpp",
        "test/tfs_rest_parser_binary_inputs_test.cpp",
        "test/tfs_rest_parser_column_test.cpp",
        "test/tfs_rest_parser_nonamed_test.cpp",
        "test/tfs_rest_parser_row_test.cpp",
        "test/threadsafequeue_test.cpp",
        "test/unit_tests.cpp",
        "test/localfilesystem_test.cpp",
        ] + select({
            "//:not_disable_cloud": [
                "test/gcsfilesystem_test.cpp",
                "test/azurefilesystem_test.cpp",
            ],
            "//:disable_cloud": []
            })
        + select({
            "//:not_disable_mediapipe": [
                "test/embeddingsnode_test.cpp",
                "test/mediapipeflow_test.cpp",
                "test/mediapipe/inputsidepacketusertestcalc.cc",
                "test/reranknode_test.cpp",
                "test/rerank_handler_test.cpp",
                "test/rerank_chunking_test.cpp",
                "test/streaming_test.cpp", # Mediapipe enabled
                "test/mediapipe_validation_test.cpp", # Mediapipe enabled
                "test/get_mediapipe_graph_metadata_response_test.cpp",
                "test/mediapipe_framework_test.cpp",
                "test/http_openai_handler_test.cpp",
            ],
            "//:disable_mediapipe" : [
                "test/disabled_mediapipe_test.cpp",
            ],
        }) + select({
            "//:not_disable_python": [
                # OvmsPyTensor is currently not used in OVMS core and is just a base for the binding.
                # "test/python/ovms_py_tensor_test.cpp",
                "test/pythonnode_test.cpp",
                # LLM logic uses Python for processing Jinja templates
                "test/llmnode_test.cpp",
                "test/llmtemplate_test.cpp",
                "test/text_streamer_test.cpp",
            ],
            "//:disable_python" : [],
        }),
    data = [
        "test/add_two_inputs_model/1/add.xml",
        "test/add_two_inputs_model/1/add.bin",
        "test/binaryutils/grayscale.jpg",
        "test/binaryutils/rgb.jpg",
        "test/binaryutils/rgb2x2.jpg",
        "test/binaryutils/rgb4x4.jpg",
        "test/configs/config.json",
        "test/configs/config_benchmark.json",
        "test/configs/config_dummy_dag.json",
        "test/configs/config_dummy_dynamic_entry_dag.json",
        "test/configs/config_metadata_all.json",
        "test/configs/config_standard_dummy.json",
        "test/configs/config_standard_scalar.json",
        "test/configs/config_dummy_dynamic_shape.json",
        "test/configs/emptyConfigWithMetrics.json",
        "test/llm/graphkfspass.pbtxt",
        "test/dummy/1/dummy.xml",
        "test/dummy/1/dummy.bin",
        "test/dummy_fp64/1/saved_model.xml",
        "test/dummy_fp64/1/saved_model.bin",
        "test/dummy_saved_model/1/saved_model.pb",
        "test/dummy_tflite/1/dummy.tflite",
        "test/dummyUppercase/1/dummy.xml",
        "test/dummyUppercase/1/dummy.bin",
        "test/no_name_output/1/model.xml",
        "test/no_name_output/1/model.bin",    
        "test/increment_1x3x4x5/1/increment_1x3x4x5.xml",
        "test/increment_1x3x4x5/1/increment_1x3x4x5.bin",
        "test/mediapipe/config_mediapipe_openai_chat_completions_mock.json",
        "test/mediapipe/config_mediapipe_add_adapter_full.json",
        "test/mediapipe/config_mediapipe_all_graphs_adapter_full.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full_dag.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full_dummy_in_both_config_and_subconfig.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full_subconfig.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_scalar.json",
        "test/mediapipe/config_mediapipe_dummy_nonexistent_calculator.json",
        "test/mediapipe/config_mediapipe_dummy_two_outputs.json",
        "test/mediapipe/config_mediapipe_graph_with_side_packets.json",
        "test/mediapipe/config_mediapipe_two_inputs.json",
        "test/mediapipe/config_mediapipe_two_outputs_dag.json",
        "test/mediapipe/config_mp_tf_passthrough.json",
        "test/mediapipe/config_standard_add.json",
        "test/mediapipe/config_standard_dummy.json",
        "test/mediapipe/graph_two_inputs_model.pbtxt",
        "test/mediapipe/config_mediapipe_pytensor_ovtensor_converter.json",
        "test/mediapipe/graph_gpt.pbtxt",
        "test/mediapipe/graphadd.pbtxt",
        "test/mediapipe/graphaddadapterfull.pbtxt",
        "test/mediapipe/graphdummy_tf.pbtxt",
        "test/mediapipe/graphdummy.pbtxt",
        "test/mediapipe/graphdummyadapterfull.pbtxt",
        "test/mediapipe/graphdummynonexistentcalculator.pbtxt",
        "test/mediapipe/graphtfpassthrough.pbtxt",
        "test/mediapipe/graphscalar.pbtxt",
        "test/mediapipe/graphWithParams.pbtxt",
        "test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt",
        "test/mediapipe/graphadapterfull_two_outputs_dag.pbtxt",
        "test/mediapipe/graphdummyadapterfull_two_outputs.pbtxt",
        "test/mediapipe/negative/config_exception_during_process.json",
        "test/mediapipe/negative/config_no_calc_output_stream.json",
        "test/mediapipe/negative/graph_exception_during_process.pbtxt",
        "test/mediapipe/negative/graph_no_calc_output_stream/graph_no_calc_output_stream.pbtxt",
        "test/mediapipe/relative_paths/config_mp_passthrough.json",
        "test/mediapipe/relative_paths/config_relative_dummy_negative.json",
        "test/mediapipe/relative_paths/config_relative_add_subconfig_negative.json",
        "test/mediapipe/relative_paths/config_relative_add_subconfig.json",
        "test/mediapipe/relative_paths/config_relative_dummy.json",
        "test/mediapipe/relative_paths/config_tflite_passthrough.json",
        "test/mediapipe/relative_paths/config_relative_dummy_subconfig_base_path.json",
        "test/mediapipe/relative_paths/graph1/dummy1/1/dummy.xml",
        "test/mediapipe/relative_paths/graph1/graph.pbtxt",
        "test/mediapipe/relative_paths/graph1/dummy1/1/dummy.bin",
        "test/mediapipe/relative_paths/graph2/graph.pbtxt",
        "test/mediapipe/relative_paths/graph2/dummy2/1/dummy.xml",
        "test/mediapipe/relative_paths/graph2/dummy2/1/dummy.bin",
        "test/mediapipe/relative_paths/graph1/graphaddadapterfull.pbtxt",
        "test/mediapipe/relative_paths/graph1/subconfig.json",
        "test/mediapipe/relative_paths/graph2/subconfig.json",
        "test/mediapipe/relative_paths/graph2/graphadd.pbtxt",
        "test/mediapipe/python/scripts/bad_execute_read_more_than_one_input.py",
        "test/mediapipe/python/scripts/bad_execute_no_method.py",
        "test/mediapipe/python/scripts/bad_missing_class.py",
        "test/mediapipe/python/scripts/concatenate_input_arrays.py",
        "test/mediapipe/python/scripts/symmetric_increment.py",
        "test/mediapipe/python/scripts/symmetric_increment_by_2.py",
        "test/mediapipe/python/scripts/single_io_increment.py",
        "test/mediapipe/python/scripts/symmetric_scalar_increment.py",
        "test/mediapipe/python/scripts/symmetric_scalar_increment_generator.py",
        "test/mediapipe/python/scripts/good_finalize_remove_file.py",
        "test/mediapipe/python/scripts/return_custom_datatype.py",
        "test/mediapipe/python/scripts/return_non_tensor_object.py",
        "test/mediapipe/python/scripts/return_none_object.py",
        "test/passthrough/1/passthrough.xml",
        "test/passthrough/1/passthrough.bin",
        "test/passthrough_string/1/passthrough.xml",
        "test/passthrough_string/1/passthrough.bin",
        "test/summator/1/summator.xml",
        "test/summator/1/summator.bin",
        "test/scalar/1/saved_model.xml",
        "test/scalar/1/saved_model.bin",
        "//src:lib_node_mock.so",
        "//src:lib_node_missing_implementation.so",
        "//src:lib_node_add_sub.so",
        "//src:lib_node_dynamic_image.so",
        "//src:lib_node_dynamic_demultiplex.so",
        "//src:lib_node_perform_different_operations.so",
        "//src:lib_node_choose_maximum.so",
        "//src:libcustom_node_east_ocr.so",
        "//src:libcustom_node_face_blur.so",
        "//src:libcustom_node_model_zoo_intel_object_detection.so",
        "//src:libcustom_node_image_transformation.so",
        "//src:libcustom_node_add_one.so",
        "//src:libcustom_node_horizontal_ocr.so",
    ],
    linkopts = LINKOPTS_ADJUSTED,
    deps = [
            "@cpp_httplib//:cpp_httplib", # server_test.cpp
            ":test_utils",
            "//src/rerank:rerank_api_handler",
            ":embeddings_handler_tests",
            ":inferencerequest_test",
            ":libtest_environment",
            ":libtest_gpuenvironment",
            "//src:ovms_lib",
            "//src:libovmsfilesystemfactory",
            "//src:custom_nodes_common_buffersqueue",
            "@com_google_googletest//:gtest",
            ] + select({
                "//conditions:default": [
                    ":openvino_remote_tensors_tests",
                    ":openvino_tests",
                ],
                "//src:windows" : [
                    "@mediapipe//mediapipe/framework/port:gtest_main",
                    ],
            })  + select({
            "//conditions:default": [
                "//src/llm:genai_servables",
                "//src/test/mediapipe/calculators:mediapipe_test_calculators",
                "@mediapipe//mediapipe/calculators/ovms:ovms_calculator",
                "@mediapipe//mediapipe/framework:calculator_runner",
            ],
            "//:disable_mediapipe" : [],
        }),
    copts = COPTS_TESTS,
)

cc_library(
    name = "test_utils",
    linkstatic = 1,
    hdrs = [
        "test/c_api_test_utils.hpp",
        "test/test_utils.hpp",
        "test/stress_test_utils.hpp",
        ],
    srcs = [
        "test/c_api_test_utils.cpp",
        "test/test_utils.cpp",
        ],
    data = [
    ],
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    deps = [
        "//src:ovms_lib",
        "@com_google_googletest//:gtest",
    ],
    copts = COPTS_TESTS,
)

cc_library(
    name = "libtest_environment",
    linkstatic = 1,
    alwayslink = True,
    hdrs = [
        "test/environment.hpp",
        ],
    srcs = [
        "test/environment.cpp",
        ],
    linkopts = [],
    deps = [
	"libovmslogging",
        "@com_google_googletest//:gtest",
    ],
    copts = COPTS_TESTS,
)
cc_library(
    name = "libtest_gpuenvironment",
    linkstatic = 1,
    alwayslink = True,
    hdrs = ["test/gpuenvironment.hpp",],
    srcs = ["test/gpuenvironment.cpp",],
    linkopts = [],
    deps = [
            "libovmslogging",
            "@com_google_googletest//:gtest",
    ],
    copts = COPTS_TESTS,
)
cc_library(
    name = "inferencerequest_test",
    linkstatic = 1,
    alwayslink = True,
    srcs = ["test/inferencerequest_test.cpp",],
    data = [],
    linkopts = [],
    deps = [
            ":libovmscapibuffer",
            ":libovmslogging",
            ":libovmsstatus",
            ":libovmsshape",
            ":libovmscapiinferenceparameter",
            ":libovmscapiinferencerequest",
            ":libovmscapiinferenceresponse",
            "@com_google_googletest//:gtest",
    ],
    copts = COPTS_TESTS,
)
cc_library(
    name = "openvino_remote_tensors_tests",
    linkstatic = 1,
    alwayslink = True,
    srcs = ["test/openvino_remote_tensors_tests.cpp"],
    data = [
        "test/configs/config_standard_dummy.json",
        "test/configs/config_gpu_dummy.json",
        "test/configs/config_gpu_face_detection_adas.json",
        "test/dummy/1/dummy.xml",
        "test/dummy/1/dummy.bin",
    ],
    linkopts = [
        "-lstdc++fs",
	# lib va
    ] + select({
        "//conditions:default": [
            "-lva",
            "-lva-drm",
	],
	"//:redhat_build": [],
    }) + [
	"-lOpenCL",
    ],
    deps = [
	":libtest_gpuenvironment",
        ":test_utils",
        "//src:ovms_lib",
	":libovmsoclutils",
        "@com_google_googletest//:gtest",
    ],
    copts = COPTS_TESTS + ["-lOpenCL",
        ] + select({
            "//:redhat_build": [],
            "//conditions:default": ["-DBUILD_VAAPITESTS=1"],
	}), # + ["-isystem /usr/include"],
)

filegroup(
  name = "release_custom_nodes",
  srcs = [
    "//src:libcustom_node_add_one.so",
    "//src:libcustom_node_east_ocr.so",
    "//src:libcustom_node_face_blur.so",
    "//src:libcustom_node_horizontal_ocr.so",
    "//src:libcustom_node_model_zoo_intel_object_detection.so",
    "//src:libcustom_node_image_transformation.so",
  ]
)
cc_library(
    name = "openvino_tests",
    linkstatic = 1,
    alwayslink = True,
    srcs = ["test/openvino_tests.cpp"],
    data = [
        "test/configs/config_standard_dummy.json",
        "test/configs/config_gpu_dummy.json",
        "test/dummy/1/dummy.xml",
        "test/dummy/1/dummy.bin",
    ],
    deps = [
        ":test_utils",
        "//src:ovms_lib",
        "@com_google_googletest//:gtest",
    ],
    copts = COPTS_TESTS,
    linkopts = [],
)
cc_library(
    name = "embeddings_handler_tests",
    linkstatic = 1,
    alwayslink = True,
    srcs = ["test/embeddings_handler_test.cpp"],
    data = [],
    deps = [
        "//src/embeddings:embeddings_api",
        "@com_google_googletest//:gtest",
    ],
    copts = COPTS_TESTS,
    linkopts = [],
)
cc_library(
    name = "capimodule",
    hdrs = ["capi_frontend/capimodule.hpp"],
    srcs = ["capi_frontend/capimodule.cpp"],
    deps = [
        "libovms_module",
        "//src:cpp_headers",
        "//src:libovmslogging",
        "//src:libovmsstatus",
    ],
    copts = COPTS_TESTS,
    linkopts = [],
)
filegroup(
  name = "static_analysis",
  srcs = [
    "//src:ovms",
    "//src:release_custom_nodes",
    "//src:capi_benchmark",
    "//src:capi_cpp_example"
  ]
)

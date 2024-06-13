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
     "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS", "COMMON_LOCAL_DEFINES",
     "create_config_settings")
COPTS_ADJUSTED = COMMON_STATIC_LIBS_COPTS + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_COPTS,
}) + select({
        "//conditions:default": ["-DPYTHON_DISABLE=1"],
        "//:not_disable_python" : ["-DPYTHON_DISABLE=0"],
}) + select({
        "//conditions:default": ["-DMEDIAPIPE_DISABLE=1"],
        "//:not_disable_mediapipe" : ["-DMEDIAPIPE_DISABLE=0"],
})

LINKOPTS_ADJUSTED = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})

COPTS_TESTS = [
    "-Wall",
    "-Wno-unknown-pragmas",
    "-Werror",
    "-Isrc",
    "-fconcepts", # for gmock related utils
    "-fvisibility=hidden",# Needed for pybind targets
] + select({
        "//conditions:default": ["-DPYTHON_DISABLE=1"],
        "//:not_disable_python" : ["-DPYTHON_DISABLE=0"],
}) + select({
        "//conditions:default": ["-DMEDIAPIPE_DISABLE=1"],
        "//:not_disable_mediapipe" : ["-DMEDIAPIPE_DISABLE=0"],
})

cc_library(
    name = "custom_nodes_common_lib",
    linkstatic = 1,
    hdrs = ["custom_nodes/common/buffersqueue.hpp"],
    srcs = [
        "queue.hpp",
        "custom_nodes/common/buffersqueue.hpp",
        "custom_nodes/common/buffersqueue.cpp",
    ],
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror",
        "-Wno-sign-compare",
    ],
)

cc_shared_library(
    name = "ovms_shared",
    dynamic_deps = [],
    static_deps = ["@tensorflow_serving//:__subpackages__",
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
        "@linux_openvino//:__subpackages__",
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
        "@boringssl//:__subpackages__",
        "@linux_opencv//:__subpackages__",
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
            "@python3_linux//:__subpackages__",
            "@local_config_python//:__subpackages__",
            ],
        "//:disable_python": []
        })
    ,
    features = [],
    roots = ["//src:ovms_lib"],
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
    name = "ovms_header",
    hdrs = ["ovms.h"],
    copts = [],
    visibility = ["//visibility:public",],
    linkopts = [],
    alwayslink = 1,
)

cc_library(
    name = "ovms_lib",
    hdrs = ["ovms.h"],
    srcs = [
        "capi_frontend/buffer.cpp",
        "capi_frontend/buffer.hpp",
        "capi_frontend/capi.cpp",
        "capi_frontend/capi_utils.cpp",
        "capi_frontend/capi_utils.hpp",
        "capi_frontend/inferenceparameter.cpp",
        "capi_frontend/inferenceparameter.hpp",
        "capi_frontend/inferenceresponse.cpp",
        "capi_frontend/inferenceresponse.hpp",
        "capi_frontend/inferencerequest.cpp",
        "capi_frontend/inferencerequest.hpp",
        "capi_frontend/inferencetensor.cpp",
        "capi_frontend/inferencetensor.hpp",
        "capi_frontend/servablemetadata.cpp",
        "capi_frontend/servablemetadata.hpp",
        "capi_frontend/server_settings.hpp",
        "cleaner_utils.cpp",
        "cleaner_utils.hpp",
        "cli_parser.cpp",
        "cli_parser.hpp",
        "config.cpp",
        "config.hpp",
        "custom_node_interface.h",
        "customloaderconfig.hpp",
        "customloaders.hpp",
        "customloaders.cpp",
        "customloaderinterface.hpp",
        "deserialization.cpp",
        "deserialization.hpp",
        "dags/aliases.hpp",
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
        "dags/gathernodeinputhandler.cpp",
        "dags/gathernodeinputhandler.hpp",
        "dags/gatherexitnodeinputhandler.hpp",
        "dags/node.cpp",
        "dags/node.hpp",
        "dags/nodeinfo.hpp",
        "dags/node_library.cpp",
        "dags/node_library.hpp",
        "dags/node_library_utils.cpp",
        "dags/node_library_utils.hpp",
        "dags/nodesession.cpp",
        "dags/nodesession.hpp",
        "dags/nodesessionresult.hpp",
        "dags/nodeinputhandler.cpp",
        "dags/nodeinputhandler.hpp",
        "dags/nodeoutputhandler.cpp",
        "dags/nodeoutputhandler.hpp",
        "dags/nodesessionmetadata.hpp",
        "dags/nodesessionmetadata.cpp",
        "dags/nodestreamidguard.cpp",
        "dags/nodestreamidguard.hpp",
        "dags/pipeline.cpp",
        "dags/pipeline.hpp",
        "dags/pipelinedefinition.cpp",
        "dags/pipelinedefinition.hpp",
        "dags/pipelinedefinitionstatus.cpp",
        "dags/pipelinedefinitionstatus.hpp",
        "dags/pipelinedefinitionunloadguard.cpp",
        "dags/pipelinedefinitionunloadguard.hpp",
        "dags/pipelineeventqueue.hpp",
        "dags/pipeline_factory.cpp",
        "dags/pipeline_factory.hpp",
        "dags/session_id.hpp",
        "dags/tensormap.hpp",
        "execution_context.hpp",
        "executingstreamidguard.cpp",
        "executingstreamidguard.hpp",
        "get_model_metadata_impl.cpp",
        "get_model_metadata_impl.hpp",
        "global_sequences_viewer.hpp",
        "global_sequences_viewer.cpp",
        "layout.cpp",
        "layout.hpp",
        "layout_configuration.cpp",
        "layout_configuration.hpp",
        "grpc_utils.cpp",
        "grpc_utils.hpp",
        "grpcservermodule.cpp",
        "grpcservermodule.hpp",
        "http_rest_api_handler.cpp",
        "http_server.cpp",
        "httpservermodule.cpp",
        "http_rest_api_handler.hpp",
        "http_server.hpp",
        "httpservermodule.hpp",
        "kfs_frontend/kfs_grpc_inference_service.cpp",
        "kfs_frontend/kfs_grpc_inference_service.hpp",
        "kfs_frontend/kfs_utils.cpp",
        "kfs_frontend/kfs_utils.hpp",
        "model.cpp",
        "model.hpp",
        "modelchangesubscription.cpp",
        "modelchangesubscription.hpp",
        "modelconfig.cpp",
        "modelconfig.hpp",
        "modelmanager.cpp",
        "modelmanager.hpp",
        "modelinstance.cpp",
        "modelinstance.hpp",
        "modelinstanceunloadguard.cpp",
        "modelinstanceunloadguard.hpp",
        "modelversionstatus.cpp",
        "modelversionstatus.hpp",
        "model_service.hpp",
        "model_service.cpp",
        "model_metric_reporter.cpp",
        "model_metric_reporter.hpp",
        "module.hpp",
        "ovinferrequestsqueue.hpp",
        "ov_utils.cpp",
        "ov_utils.hpp",
        "ovms.h",
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
        "schema.hpp",
        "schema.cpp",
        "serialization.cpp",
        "serialization.hpp",
        "servablemanagermodule.cpp",
        "servablemanagermodule.hpp",
        "server.cpp",
        "server.hpp",
        "sequence.cpp",
        "sequence.hpp",
        "sequence_manager.cpp",
        "sequence_manager.hpp",
        "sequence_processing_spec.hpp",
        "shape.cpp",
        "shape.hpp",
        "statefulmodelinstance.cpp",
        "statefulmodelinstance.hpp",
        "systeminfo.cpp",
        "systeminfo.hpp",
        "queue.hpp",
        "tensorinfo.cpp",
        "tensorinfo.hpp",
        "tensor_utils.hpp",
        "threadsafequeue.hpp",
        "timer.hpp",
        "version.hpp",
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
                "//src/llm:llmnoderesources",
            ],
            "//:disable_python": []
        }) + [
            "cpp_headers",
            "libmodelconfigjsonparser",
            "libovmsfilesystem",
            "libovmsfilesystemfactory",
            "libovmslocalfilesystem", # indirectly & directly through factory
            "libovmslogging",
            "libovmsmetrics",
            "libovmsprecision",
            "libovmstfs_grpc",
            "//src:libovmsprofiler",
            "libovmsstatus",
            "libovmsstring_utils",
            "libovmsazurefilesystem",
            "libovmsmodelversioning",
            "//:ovms_dependencies", # TODO dispose or replace
            "//src/kfserving_api:kfserving_api_cpp",
        ] + select({
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
                "//src/llm:httppayload",
            ],
            "//:disable_mediapipe" : [],
        }),
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    data = select({
        "//:not_disable_python": [
            "//src/python/binding:pyovms.so",
        ],
        "//:disable_python": []
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
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

cc_library(
    name = "libovmsprecision",
    hdrs = ["precision.hpp"],
    srcs = ["precision.cpp",],
    deps = ["@linux_openvino//:openvino",], # TODO dispose
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    alwayslink = 1,
)

load("@pip_deps//:requirements.bzl", "all_requirements")

cc_binary(
    name = "libsampleloader.so",
    srcs = [
	    "example/SampleCustomLoader/sampleCustLoader.cpp",
	    "customloaderinterface.hpp",
    ],
    linkshared = 1,
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
    ],
)

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
        "@linux_opencv//:opencv"
    ],
    linkshared = 1,
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror"
    ]
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
        "@linux_opencv//:opencv"
    ],
    linkshared = 1,
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror"
    ]
)

cc_binary(
    name = "libcustom_node_add_one.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/buffersqueue.hpp",
        "custom_nodes/common/buffersqueue.cpp",
        "custom_nodes/common/custom_node_library_internal_manager.hpp",
        "custom_nodes/common/custom_node_library_internal_manager.cpp",
        "queue.hpp",
        "custom_nodes/add_one/add_one.cpp",
        "custom_node_interface.h",
        "custom_nodes/add_one/add_one_internal_manager.hpp"
    ],
    linkshared = 1,
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror"
    ]
)

cc_binary(
    name = "libcustom_node_model_zoo_intel_object_detection.so",
    srcs = [
        "custom_nodes/common/utils.hpp",
        "custom_nodes/common/opencv_utils.hpp",
        "custom_nodes/common/buffersqueue.hpp",
        "custom_nodes/common/buffersqueue.cpp",
        "custom_nodes/common/custom_node_library_internal_manager.hpp",
        "custom_nodes/common/custom_node_library_internal_manager.cpp",
        "queue.hpp",
        "custom_nodes/model_zoo_intel_object_detection/model_zoo_intel_object_detection.cpp",
        "custom_node_interface.h",
    ],
    deps = [
        "@linux_opencv//:opencv"
    ],
    linkshared = 1,
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror"
    ]
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
        "@linux_opencv//:opencv"
    ],
    linkshared = 1,
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror"
    ]
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
        "@linux_opencv//:opencv"
    ],
    linkshared = 1,
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror"
    ]
)

# POC which can start OVMS with C-API built by bazel
# Standalone example for building outside of bazel (with bare g++ is inside MakefileCapi)
cc_binary(
    name = "capi_cpp_example",
    srcs = [
        "main_capi.cpp",
    ],
    linkopts = [
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
    ],
    copts = [
    ],
    deps = [
        "//src:ovms_lib",
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

cc_library( # TODO split dependencies
    name = "libovmshttpservermodule",
    hdrs = [
        "http_rest_api_handler.hpp",
        "http_server.hpp",
        "httpservermodule.hpp",
    ],
    srcs = [
        "http_rest_api_handler.cpp",
        "http_server.cpp",
        "httpservermodule.cpp",
    ],
    deps = [
        "@tensorflow_serving//tensorflow_serving/util/net_http/server/public:http_server",
        "@tensorflow_serving//tensorflow_serving/util/net_http/server/public:http_server_api",
        "@tensorflow_serving//tensorflow_serving/util:threadpool_executor",
        "@tensorflow_serving//tensorflow_serving/util:json_tensor",
        "libovmslogging",
        "libovmsprofiler",
        "libovmsstatus",
    ],
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
    copts = COPTS_ADJUSTED,
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
    hdrs = ["modelversion.hpp",
            "model_version_policy.hpp",],
    srcs = ["model_version_policy.cpp",],
    deps = [],
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

cc_library( # make ovms_lib dependent, use share doptions
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

cc_library( # make ovms_lib dependent, use share doptions
    name = "libovmss3filesystem",
    hdrs = ["s3filesystem.hpp",],
    srcs = [
            "s3filesystem.cpp",
    ],
    deps = select({
            "//:ubuntu_build": [
                "@aws-sdk-cpp//:aws-sdk-cpp_ubuntu",
            ],
            "//:redhat_build": [
                "@aws-sdk-cpp//:aws-sdk-cpp_redhat",
            ],
        }) + [
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
            "@boringssl//:ssl",
            "libovmslogging",
            "libovmsmodelversioning",
            "libovmsstatus",],
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
    name = "libovmsfilesystemfactory",
    hdrs = ["filesystemfactory.hpp",],
    srcs = ["filesystemfactory.cpp",],
    deps = ["libovmsfilesystem",
            "libovmsstatus",
            "libovmsazurefilesystem",
            "libovmsgcsfilesystem",
            "libovmss3filesystem",
            "libovmslocalfilesystem",
            "libovmslogging"],
    visibility = ["//visibility:public"],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)

cc_library(
    name = "libmodelconfigjsonparser",
    hdrs = [
        "json_parser.hpp",
    ],
    srcs = ["json_parser.cpp"],
    deps = ["cpp_headers",
            "@com_github_tencent_rapidjson//:rapidjson",
            "@linux_openvino//:openvino",
            "@com_github_gabime_spdlog//:spdlog"],
    visibility = ["//visibility:public"],
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
        "module.hpp",],
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
    ],
    linkstatic = True,
)

cc_binary(
    name = "ovms",
    srcs = [
        "main.cpp",
    ],
    linkopts = [
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
        # "-lovms_shared",  # Use for dynamic linking when necessary
    ],
    copts = [
        "-Wall",
        "-Wconversion",
        "-Werror",
    ],
    deps = [
        "//src:ovms_lib",
        "//src:libcustom_node_east_ocr.so",
        "//src:libcustom_node_face_blur.so",
        "//src:libcustom_node_horizontal_ocr.so",
        "//src:libcustom_node_model_zoo_intel_object_detection.so",
        "//src:libcustom_node_image_transformation.so",
    ] + select({
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
        "test/azurefilesystem_test.cpp",
        "test/tensor_conversion_test.cpp",
        "test/c_api_test_utils.hpp",
        "test/c_api_tests.cpp",
        "test/c_api_stress_tests.cpp",
        "test/custom_loader_test.cpp",
        "test/custom_node_output_allocator_test.cpp",
        "test/custom_node_buffersqueue_test.cpp",
        "test/demultiplexer_node_test.cpp",
        "test/deserialization_tests.cpp",
        "test/ensemble_tests.cpp",
        "test/ensemble_flow_custom_node_tests.cpp",
        "test/ensemble_mapping_config_tests.cpp",
        "test/ensemble_metadata_test.cpp",
        "test/ensemble_config_change_stress.cpp",
        "test/environment.hpp",
        "test/gather_node_test.cpp",
        "test/gcsfilesystem_test.cpp",
        "test/get_model_metadata_response_test.cpp",
        "test/get_pipeline_metadata_response_test.cpp",
        "test/get_model_metadata_signature_test.cpp",
        "test/get_model_metadata_validation_test.cpp",
        "test/http_rest_api_handler_test.cpp",
        "test/inferencerequest_test.cpp",
        "test/kfs_metadata_test.cpp",
        "test/kfs_rest_test.cpp",
        "test/layout_test.cpp",
        "test/localfilesystem_test.cpp",
        "test/metrics_flow_test.cpp",
        "test/metrics_test.cpp",
        "test/metric_config_test.cpp",
        "test/mockmodelinstancechangingstates.hpp",
        "test/model_cache_test.cpp",
        "test/model_service_test.cpp",
        "test/model_version_policy_test.cpp",
        "test/model_test.cpp",
        "test/modelinstance_test.cpp",
        "test/modelconfig_test.cpp",
        "test/node_library_manager_test.cpp",
        "test/modelmanager_test.cpp",
        "test/modelversionstatus_test.cpp",
        "test/nodesessionmetadata_test.cpp",
        "test/ovmsconfig_test.cpp",
        "test/ovinferrequestqueue_test.cpp",
        "test/ov_utils_test.cpp",
        "test/pipelinedefinitionstatus_test.cpp",
        "test/capi_predict_validation_test.cpp",
        "test/predict_validation_test.cpp",
        "test/prediction_service_test.cpp",
        "test/tfs_rest_parser_row_test.cpp",
        "test/tfs_rest_parser_column_test.cpp",
        "test/tfs_rest_parser_binary_inputs_test.cpp",
        "test/tfs_rest_parser_nonamed_test.cpp",
        "test/kfs_rest_parser_test.cpp",
        "test/rest_utils_test.cpp",
        "test/schema_test.cpp",
        "test/sequence_test.cpp",
        "test/serialization_tests.cpp",
        "test/server_test.cpp",
        "test/sequence_manager_test.cpp",
        "test/shape_test.cpp",
        "test/stateful_config_test.cpp",
        "test/stateful_modelinstance_test.cpp",
        "test/stateful_test_utils.hpp",
        "test/status_test.cpp",
        "test/stringutils_test.cpp",
        "test/systeminfo_test.cpp",
        "test/tensorinfo_test.cpp",
        "test/tensorutils_test.cpp",
        "test/test_utils.cpp",
        "test/test_utils.hpp",
        "test/stress_test_utils.hpp",
        "test/threadsafequeue_test.cpp",
        "test/unit_tests.cpp",
        ] + select({
            "//:not_disable_mediapipe": [
                "test/http_openai_handler_test.cpp",
                "test/get_mediapipe_graph_metadata_response_test.cpp",
                "test/mediapipe/inputsidepacketusertestcalc.cc",
                "test/mediapipeflow_test.cpp",
                "test/mediapipe_framework_test.cpp",
                "test/mediapipe_validation_test.cpp",
                "test/streaming_test.cpp",
            ],
            "//:disable_mediapipe" : [],
        }) + select({
            "//:not_disable_python": [
                "test/pythonnode_test.cpp",
                # OvmsPyTensor is currently not used in OVMS core and is just a base for the binding.
                # "test/python/ovms_py_tensor_test.cpp",
                # LLM logic uses Python for processing Jinja templates
                "test/llmnode_test.cpp",
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
        "test/c_api/config.json",
        "test/c_api/config_benchmark.json",
        "test/c_api/config_dummy_dag.json",
        "test/c_api/config_dummy_dynamic_entry_dag.json",
        "test/c_api/config_metadata_all.json",
        "test/c_api/config_standard_dummy.json",
        "test/c_api/config_standard_scalar.json",
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
    ],
    linkopts = [
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
        "-lssl",
    ],
    deps = [
        "//src:ovms_lib",
        "//src:custom_nodes_common_lib",
        "//src:libsampleloader.so",
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
        "@com_google_googletest//:gtest",
        ] + select({
            "//conditions:default": [
                "//src/test/mediapipe/calculators:mediapipe_test_calculators",
                "@mediapipe//mediapipe/calculators/ovms:ovms_calculator",
                "@mediapipe//mediapipe/framework:calculator_runner",
            ],
            "//:disable_mediapipe" : [],
        }) +select({
            "//:not_disable_python": [
                "//src/llm:llmnoderesources",
            ],
            "//:disable_python": []
        }),
    copts = COPTS_TESTS,
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

filegroup(
  name = "static_analysis",
  srcs = [
    "//src:ovms",
    "//src:release_custom_nodes",
    "//src:capi_benchmark",
    "//src:capi_cpp_example"
  ]
)

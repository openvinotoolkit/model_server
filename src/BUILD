#
# Copyright (c) 2020-2022 Intel Corporation
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
#config_setting(
#    name = "linux_distribution_family",
#    constraint_values = [
#        ":debian", # like Ubuntu
#        ":fedora", # like RHEL/CentOS
#    ],
#)

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@mediapipe//mediapipe/framework:more_selects.bzl", "more_selects")

#To build without mediapipe use flags - bazel build --define MEDIAPIPE_DISABLE=1 --cxxopt=-DMEDIAPIPE_DISABLE=1 //src:ovms
config_setting(
    name = "disable_mediapipe",
    define_values = {
        "MEDIAPIPE_DISABLE": "1",
    },
    visibility = ["//visibility:public"],
)

more_selects.config_setting_negation(
    name = "not_disable_mediapipe",
    negate = ":disable_mediapipe",
)

constraint_setting(name = "linux_distribution_family")
constraint_value(constraint_setting = "linux_distribution_family", name = "fedora") # like RHEL/CentOS
constraint_value(constraint_setting = "linux_distribution_family", name = "debian") # like Ubuntu

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

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@tensorflow_serving//tensorflow_serving:serving.bzl", "serving_tensorflow_proto_dep")

cc_shared_library(
    name = "ovms_shared",
    dynamic_deps = [],
    static_deps = ["@tensorflow_serving//:__subpackages__",
        "@minitrace//:__subpackages__",
        "@com_github_gabime_spdlog//:__subpackages__",
        "@fmtlib//:__subpackages__",
        "@com_github_jarro2783_cxxopts//:__subpackages__",
        "@awssdk//:__subpackages__",
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
        "@oneTBB//:__subpackages__",
        "@com_github_glog_glog//:__subpackages__",
        "@com_github_gflags_gflags//:__subpackages__",
    ] + select({
        "//conditions:default": [
            "@mediapipe//:__subpackages__",
            "@mediapipe_calculators//:__subpackages__",
            "@model_api//:__subpackages__",
            "@arm_neon_2_x86_sse//:__subpackages__",
            "@ruy//:__subpackages__",
            "@cpuinfo//:__subpackages__",
            "@clog//:__subpackages__",
            "@gemmlowp//:__subpackages__",
            "@flatbuffers//:__subpackages__",
            ],
        "//src:disable_mediapipe" : [],
        }),
    features = [],
    roots = ["//src:ovms_lib"],
)

load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_cc_proto_library", "mediapipe_proto_library")

mediapipe_proto_library(
    name = "ovmscalculatoroptions_proto", # ovmscalculatoroptions_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["mediapipe_calculators/ovmscalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)
mediapipe_proto_library(
    name = "modelapiovmsinferencecalculator_proto", # ovmscalculatoroptions_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["mediapipe_calculators/modelapiovmsinferencecalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)
mediapipe_proto_library(
    name = "modelapiovmssessioncalculator_proto", # ovmscalculatoroptions_cc_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
    srcs = ["mediapipe_calculators/modelapiovmssessioncalculator.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_options_proto",
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "ovms_lib",
    hdrs = ["ovms.h"],
    srcs = [
        "azurestorage.hpp",
        "azurestorage.cpp",
        "azurefilesystem.cpp",
        "azurefilesystem.hpp",
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
        "gcsfilesystem.cpp",
        "execution_context.hpp",
        "executingstreamidguard.cpp",
        "executingstreamidguard.hpp",
        "filesystem.cpp",
        "filesystem.hpp",
        "get_model_metadata_impl.cpp",
        "get_model_metadata_impl.hpp",
        "global_sequences_viewer.hpp",
        "global_sequences_viewer.cpp",
        "http_rest_api_handler.cpp",
        "http_rest_api_handler.hpp",
        "http_server.cpp",
        "http_server.hpp",
        "httpservermodule.cpp",
        "httpservermodule.hpp",
        "layout.cpp",
        "layout.hpp",
        "layout_configuration.cpp",
        "layout_configuration.hpp",
        "localfilesystem.cpp",
        "localfilesystem.hpp",
        "gcsfilesystem.hpp",
        "grpc_utils.cpp",
        "grpc_utils.hpp",
        "grpcservermodule.cpp",
        "grpcservermodule.hpp",
        "kfs_frontend/kfs_grpc_inference_service.cpp",
        "kfs_frontend/kfs_grpc_inference_service.hpp",
        "kfs_frontend/kfs_utils.cpp",
        "kfs_frontend/kfs_utils.hpp",
        "metric.cpp",
        "metric.hpp",
        "metric_config.cpp",
        "metric_config.hpp",
        "metric_family.cpp",
        "metric_family.hpp",
        "metric_registry.cpp",
        "metric_registry.hpp",
        "metric_module.cpp",
        "metric_module.hpp",
        "model.cpp",
        "model.hpp",
        "model_version_policy.cpp",
        "model_version_policy.hpp",
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
        "modelversion.hpp",
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
        "precision.cpp",
        "precision.hpp",
        "prediction_service.cpp",
        "prediction_service.hpp",
        "prediction_service_utils.hpp",
        "prediction_service_utils.cpp",
        "predict_request_validation_utils.hpp",
        "predict_request_validation_utils.cpp",
        "profiler.cpp",
        "profiler.hpp",
        "profilermodule.cpp",
        "profilermodule.hpp",
        "rest_parser.cpp",
        "rest_parser.hpp",
        "rest_utils.cpp",
        "rest_utils.hpp",
        "s3filesystem.cpp",
        "s3filesystem.hpp",
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
        "status.cpp",
        "status.hpp",
        "stringutils.cpp",
        "stringutils.hpp",
        "systeminfo.cpp",
        "systeminfo.hpp",
        "queue.hpp",
        "tensorinfo.cpp",
        "tensorinfo.hpp",
        "tfs_frontend/tfs_utils.cpp",
        "tfs_frontend/tfs_utils.hpp",
        "tensor_utils.hpp",
        "threadsafequeue.hpp",
        "timer.hpp",
        "version.hpp",
        "logging.hpp",
        "logging.cpp",
        "tensor_conversion.hpp",
        "tensor_conversion.cpp",
         ] + select({
            "//conditions:default": [
                "mediapipe_internal/mediapipefactory.cpp",
                "mediapipe_internal/mediapipefactory.hpp",
                "mediapipe_internal/mediapipegraphconfig.hpp",
                "mediapipe_internal/mediapipegraphconfig.cpp",
                "mediapipe_internal/mediapipegraphdefinition.cpp",
                "mediapipe_internal/mediapipegraphdefinition.hpp",
                "mediapipe_internal/mediapipegraphexecutor.cpp",
                "mediapipe_internal/mediapipegraphexecutor.hpp",
                "mediapipe_calculators/modelapiovmsadapter.cc",
                "mediapipe_calculators/modelapiovmsadapter.hpp",
                "mediapipe_calculators/ovms_calculator.cc",
                "mediapipe_calculators/modelapiovmsinferencecalculator.cc",
                "mediapipe_calculators/modelapiovmssessioncalculator.cc",
            ],
            "//src:disable_mediapipe" : [],
        }),
    deps = [
        "//:ovms_dependencies",
        "//src/kfserving_api:kfserving_api_cpp",
        ] + select({
            "//conditions:default": [
                "//src:ovmscalculatoroptions_cc_proto", # ovmscalculatoroptions_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
                "//src:modelapiovmsinferencecalculator_cc_proto", # modelapiovmsinferencecalculator_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
                "//src:modelapiovmssessioncalculator_cc_proto", # modelapiovmssessioncalculator_proto - just mediapipe stuff with mediapipe_proto_library adding nonvisible target
            ],
            "//src:disable_mediapipe" : [],
        }),
    local_defines = [
        "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE",
    ],
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror",
    ],
    visibility = [
        "//visibility:public",
    ],
    linkopts = [
        "-lxml2",
        "-luuid",
        "-lstdc++fs",
        "-lcrypto",
    ],
    alwayslink = 1,
)

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
    name = "cpp_headers",
    srcs = ["server.hpp",
        "module.hpp",],
    deps = ["//src:shared_lib"]
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
        # "//src:cpp_headers",  # Use for dynamic linking when necessary
    ],
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
        "test/threadsafequeue_test.cpp",
        "test/unit_tests.cpp",
        ] + select({
        "//conditions:default": [
            "test/mediapipe/inputsidepacketusertestcalc.cc",
            "test/mediapipeflow_test.cpp",
            "test/mediapipe_validation_test.cpp",
        ],
        "//src:disable_mediapipe" : [],
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
        "test/configs/emptyConfigWithMetrics.json",
        "test/dummy/1/dummy.xml",
        "test/dummy/1/dummy.bin",
        "test/dummy_fp64/1/saved_model.xml",
        "test/dummy_fp64/1/saved_model.bin",
        "test/dummy_saved_model/1/saved_model.pb",
        "test/dummyUppercase/1/dummy.xml",
        "test/dummyUppercase/1/dummy.bin",
        "test/increment_1x3x4x5/1/increment_1x3x4x5.xml",
        "test/increment_1x3x4x5/1/increment_1x3x4x5.bin",
        "test/mediapipe/subconfig.json",
        "test/mediapipe/config_mediapipe_add_adapter_full.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full_subconfig.json",
        "test/mediapipe/config_mediapipe_all_graphs_adapter_full.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full_dag.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full_dummy_in_both_config_and_subconfig.json",
        "test/mediapipe/config_mediapipe_dummy_adapter_full.json",
        "test/mediapipe/config_mediapipe_graph_with_side_packets.json",
        "test/mediapipe/config_standard_add.json",
        "test/mediapipe/config_standard_dummy.json",
        "test/mediapipe/graphadd.pbtxt",
        "test/mediapipe/graphaddadapterfull.pbtxt",
        "test/mediapipe/graphdummy.pbtxt",
        "test/mediapipe/graphdummyadapterfull.pbtxt",
        "test/mediapipe/graphWithParams.pbtxt",
        "test/mediapipe/relative_paths/config_relative_dummy_negative.json",
        "test/mediapipe/relative_paths/config_relative_add_subconfig_negative.json",
        "test/mediapipe/relative_paths/config_relative_add_subconfig.json",
        "test/mediapipe/relative_paths/config_relative_dummy.json",
        "test/mediapipe/relative_paths/graph1/dummy1/1/dummy.xml",
        "test/mediapipe/relative_paths/graph2/dummy2/1/dummy.xml",
        "test/mediapipe/relative_paths/graph1/graphaddadapterfull.pbtxt",
        "test/mediapipe/relative_paths/graph1/subconfig.json",
        "test/mediapipe/relative_paths/graph2/graphadd.pbtxt",
        "test/mediapipe/graphdummyadapterfull_dummyinputnames.pbtxt",
        "test/mediapipe/relative_paths/graph2/subconfig.json",
        "test/passthrough/1/passthrough.xml",
        "test/passthrough/1/passthrough.bin",
        "test/summator/1/summator.xml",
        "test/summator/1/summator.bin",
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
    ],
    copts = [
        "-Wall",
        "-Wno-unknown-pragmas",
        "-Werror",
    ],
)

filegroup(
  name = "static_analysis",
  srcs = [
    "//src:ovms",
    "//src:libcustom_node_add_one.so",
    "//src:libcustom_node_east_ocr.so",
    "//src:libcustom_node_face_blur.so",
    "//src:libcustom_node_horizontal_ocr.so",
    "//src:libcustom_node_model_zoo_intel_object_detection.so",
    "//src:libcustom_node_image_transformation.so",
  ]
)

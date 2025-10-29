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

# TensorFlow Serving API protobuf definitions
# This package contains only the proto files needed for TensorFlow Serving API compatibility
# without requiring the full TensorFlow or TensorFlow Serving build dependencies.

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# TensorFlow Core Framework protos
proto_library(
    name = "tensorflow_core_framework_protos",
    srcs = glob(["tensorflow/core/framework/*.proto"]),
    deps = ["@com_google_protobuf//:cc_wkt_protos"],
)

cc_proto_library(
    name = "tensorflow_core_framework_cc_proto",
    srcs = glob(["tensorflow/core/framework/*.proto"]),
    deps = ["@com_google_protobuf//:cc_wkt_protos"],
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)

# TensorFlow Core Example protos
proto_library(
    name = "tensorflow_core_example_protos",
    srcs = glob(["tensorflow/core/example/*.proto"]),
)

cc_proto_library(
    name = "tensorflow_core_example_cc_proto",
    srcs = glob(["tensorflow/core/example/*.proto"]),
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)

# TensorFlow Core Protobuf protos
proto_library(
    name = "tensorflow_core_protobuf_protos",
    srcs = glob(["tensorflow/core/protobuf/*.proto"]),
    deps = [
        ":tensorflow_core_framework_protos",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_proto_library(
    name = "tensorflow_core_protobuf_cc_proto",
    srcs = glob(["tensorflow/core/protobuf/*.proto"]),
    deps = [
        ":tensorflow_core_framework_cc_proto",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)

# TensorFlow Serving Config protos
proto_library(
    name = "tensorflow_serving_config_protos",
    srcs = glob(["tensorflow_serving/config/*.proto"]),
    deps = ["@com_google_protobuf//:cc_wkt_protos"],
)

cc_proto_library(
    name = "tensorflow_serving_config_cc_proto",
    srcs = glob(["tensorflow_serving/config/*.proto"]),
    deps = ["@com_google_protobuf//:cc_wkt_protos"],
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)

# TensorFlow Serving API protos (non-service)
proto_library(
    name = "tensorflow_serving_apis_protos",
    srcs = [
        "tensorflow_serving/apis/classification.proto",
        "tensorflow_serving/apis/get_model_metadata.proto",
        "tensorflow_serving/apis/get_model_status.proto",
        "tensorflow_serving/apis/inference.proto",
        "tensorflow_serving/apis/input.proto",
        "tensorflow_serving/apis/logging.proto",
        "tensorflow_serving/apis/model.proto",
        "tensorflow_serving/apis/model_management.proto",
        "tensorflow_serving/apis/predict.proto",
        "tensorflow_serving/apis/prediction_log.proto",
        "tensorflow_serving/apis/regression.proto",
        "tensorflow_serving/apis/session_service.proto",
        "tensorflow_serving/apis/status.proto",
    ],
    deps = [
        ":tensorflow_core_example_protos",
        ":tensorflow_core_framework_protos",
        ":tensorflow_core_protobuf_protos",
        ":tensorflow_serving_config_protos",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
)

cc_proto_library(
    name = "tensorflow_serving_apis_cc_proto",
    srcs = [
        "tensorflow_serving/apis/classification.proto",
        "tensorflow_serving/apis/get_model_metadata.proto",
        "tensorflow_serving/apis/get_model_status.proto",
        "tensorflow_serving/apis/inference.proto",
        "tensorflow_serving/apis/input.proto",
        "tensorflow_serving/apis/logging.proto",
        "tensorflow_serving/apis/model.proto",
        "tensorflow_serving/apis/model_management.proto",
        "tensorflow_serving/apis/predict.proto",
        "tensorflow_serving/apis/prediction_log.proto",
        "tensorflow_serving/apis/regression.proto",
        "tensorflow_serving/apis/session_service.proto",
        "tensorflow_serving/apis/status.proto",
    ],
    deps = [
        ":tensorflow_core_example_cc_proto",
        ":tensorflow_core_framework_cc_proto",
        ":tensorflow_core_protobuf_cc_proto",
        ":tensorflow_serving_config_cc_proto",
        "@com_google_protobuf//:cc_wkt_protos",
    ],
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)

# Prediction Service proto (with gRPC service)
proto_library(
    name = "prediction_service_proto",
    srcs = ["tensorflow_serving/apis/prediction_service.proto"],
    deps = [":tensorflow_serving_apis_protos"],
)

cc_proto_library(
    name = "prediction_service_cc_proto",
    srcs = ["tensorflow_serving/apis/prediction_service.proto"],
    deps = [":tensorflow_serving_apis_cc_proto"],
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)

cc_grpc_library(
    name = "prediction_service_cc_grpc",
    srcs = [":prediction_service_proto"],
    grpc_only = True,
    deps = [":prediction_service_cc_proto"],
)

# Model Service proto (with gRPC service)
proto_library(
    name = "model_service_proto",
    srcs = ["tensorflow_serving/apis/model_service.proto"],
    deps = [":tensorflow_serving_apis_protos"],
)

cc_proto_library(
    name = "model_service_cc_proto",
    srcs = ["tensorflow_serving/apis/model_service.proto"],
    deps = [":tensorflow_serving_apis_cc_proto"],
    default_runtime = "@com_google_protobuf//:protobuf",
    protoc = "@com_google_protobuf//:protoc",
)

cc_grpc_library(
    name = "model_service_cc_grpc",
    srcs = [":model_service_proto"],
    grpc_only = True,
    deps = [":model_service_cc_proto"],
)

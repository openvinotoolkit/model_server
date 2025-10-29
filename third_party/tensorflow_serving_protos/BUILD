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

load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# TensorFlow Core Framework protos
cc_proto_library(
    name = "tensorflow_core_framework_cc_proto",
    srcs = glob(["tensorflow/core/framework/*.proto"]),
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
)

# TensorFlow Core Example protos
cc_proto_library(
    name = "tensorflow_core_example_cc_proto",
    srcs = glob(["tensorflow/core/example/*.proto"]),
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
)

# TensorFlow Core Protobuf protos
cc_proto_library(
    name = "tensorflow_core_protobuf_cc_proto",
    srcs = glob(["tensorflow/core/protobuf/*.proto"]),
    deps = [
        ":tensorflow_core_framework_cc_proto",
    ],
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
)

# TensorFlow Serving Config protos
cc_proto_library(
    name = "tensorflow_serving_config_cc_proto",
    srcs = glob(["tensorflow_serving/config/*.proto"]),
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
)

# TensorFlow Serving API protos (non-service)
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
    ],
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
)

# Prediction Service proto (with gRPC service)
cc_proto_library(
    name = "prediction_service_cc_proto",
    srcs = ["tensorflow_serving/apis/prediction_service.proto"],
    deps = [":tensorflow_serving_apis_cc_proto"],
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
    use_grpc_plugin = True,
)

# Model Service proto (with gRPC service)
cc_proto_library(
    name = "model_service_cc_proto",
    srcs = ["tensorflow_serving/apis/model_service.proto"],
    deps = [":tensorflow_serving_apis_cc_proto"],
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
    use_grpc_plugin = True,
)



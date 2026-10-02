#
# Copyright (c) 2022 Intel Corporation
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
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

proto_library(
    name = "kfserving_api_proto",
    srcs = ["grpc_predict_v2.proto"],
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "kfserving_api_cpp_proto",
    deps = [":kfserving_api_proto"],
    visibility = ["//visibility:public"],
)

cc_grpc_library(
    name = "kfserving_api_cpp_grpc",
    srcs = [":kfserving_api_proto"],
    grpc_only = True,
    deps = [":kfserving_api_cpp_proto"],
    visibility = ["//visibility:public"],
)

# Aggregate target that both the messages and the grpc service, keeping the
# historical `kfserving_api_cpp` label used across the codebase.
cc_library(
    name = "kfserving_api_cpp",
    deps = [
        ":kfserving_api_cpp_grpc",
        ":kfserving_api_cpp_proto",
    ],
    visibility = ["//visibility:public"],
)

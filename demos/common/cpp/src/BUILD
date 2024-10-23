#
# Copyright (c) 2021 Intel Corporation
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

filegroup(
  name = "ovms_cpp_image_classification",
  srcs = [
    "//src:classification_client_sync",
    "//src:classification_client_async_benchmark",
  ]
)

filegroup(
  name = "ovms_cpp_benchmark",
  srcs = [
    "//src:synthetic_client_async_benchmark",
  ]
)

cc_binary(
    name = "classification_client_sync",
    srcs = [
        "classification_client_sync.cpp",
        "common.hpp",
    ],
    deps = [
        "@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@linux_opencv//:opencv",
    ]
)

cc_binary(
    name = "classification_client_async_benchmark",
    srcs = [
        "classification_client_async_benchmark.cpp",
        "common.hpp",
    ],
    deps = [
        "@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@linux_opencv//:opencv",
    ]
)

cc_binary(
    name = "synthetic_client_async_benchmark",
    srcs = [
        "synthetic_client_async_benchmark.cpp",
        "common.hpp",
    ],
    deps = [
        "@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ]
)

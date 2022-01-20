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
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")

cc_proto_library(
    name = "kfserving_api_cpp",
    srcs = ["grpc_predict_v2.proto",],
    deps = [],
    cc_libs = ["@com_google_protobuf//:protobuf"],
    protoc = "@com_google_protobuf//:protoc",
    default_runtime = "@com_google_protobuf//:protobuf",
    use_grpc_plugin = True, # ??
    testonly = False, # ??
    visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "kfserving_api_lib",
    hdrs = ["kfserving_api/grpc_predict_v2.pb.h",
            "kfserving_api/grpc_predict_v2.grpc.pb.h",
    ],
    deps = [":kfserving_api_cpp"],
)

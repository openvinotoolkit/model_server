#
# Copyright (c) 2020 Intel Corporation
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

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "core",
    srcs = ["build/aws-cpp-sdk-core/libaws-cpp-sdk-core.a"],
    hdrs = glob([
        "aws-cpp-sdk-core/include/**/*.h",
    ]),
    strip_include_prefix = "aws-cpp-sdk-core/include",
)

cc_library(
    name = "s3",
    srcs = ["build/aws-cpp-sdk-s3/libaws-cpp-sdk-s3.a"],
    hdrs = glob([
        "aws-cpp-sdk-s3/include/**/*.h",
    ]),
    strip_include_prefix = "aws-cpp-sdk-s3/include"
)

cc_library(
    name = "deps",
    srcs = [
        "build/.deps/install/lib/libaws-c-event-stream.a",
        "build/.deps/install/lib/libaws-checksums.a",
        "build/.deps/install/lib/libaws-c-common.a"
    ],
    hdrs = glob([
        "build/.deps/install/include/**/*.*",
    ]),
    strip_include_prefix = "build/.deps/install/include"
)

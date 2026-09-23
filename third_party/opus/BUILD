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

load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"], exclude = ["tests/**", "src/**/tests/**"]),
)

cmake(
    name = "opus_cmake",
    lib_source = ":all_srcs",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "BUILD_SHARED_LIBS": "OFF",
        "OPUS_BUILD_SHARED_LIBRARY": "OFF",
        "OPUS_BUILD_TESTING": "OFF",
        "OPUS_BUILD_PROGRAMS": "OFF",
        "OPUS_INSTALL_PKG_CONFIG_MODULE": "OFF",
        "OPUS_INSTALL_CMAKE_CONFIG_MODULE": "OFF",
        "OPUS_ENABLE_FLOAT_API": "ON",
    },
    out_static_libs = ["libopus.a"],
    out_include_dir = "include",
    tags = ["requires-network"],
)

cc_library(
    name = "libopus",
    deps = [":opus_cmake"],
    linkopts = ["-lm"],
)

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
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
visibility = ["//visibility:public"]
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

visibility = ["//visibility:public"]

config_setting(
    name = "dbg",
    values = {"compilation_mode": "dbg"},
)

config_setting(
    name = "opt",
    values = {"compilation_mode": "opt"},
)

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

build_release = {"CMAKE_BUILD_TYPE": "Release"}
build_debug = {"CMAKE_BUILD_TYPE": "Debug"}
cmake(
    name = "cpp-httplib_cmake",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant parallel compilation support
        "VERBOSE=1",
        "-j 8",
    ],
    cache_entries = {},
    lib_source = ":all_srcs",
    out_lib_dir = "{lib_path}",
    out_static_libs = [],
    tags = ["requires-network"],
    alwayslink = False,
    visibility = ["//visibility:public"],
)
cc_library(
    name = "cpp_httplib",
    srcs = [],
    hdrs = glob([
        "*.h",
    ]),
    includes = [
        "",
    ],
    copts = [],
    defines = [],
    visibility = ["//visibility:public"],
)

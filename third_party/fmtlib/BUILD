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
    name = "fmtlib2",
    srcs = glob([
        "src/*.cc",
    ]),
    hdrs = glob([
        "include/fmt/*.h",
    ]),
    includes = [
        "include/",
    ],
    visibility = ["//visibility:public"],
    defines = ["FMT_HEADER_ONLY"],

    alwayslink = 1,
        )


filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
cmake(
    name = "fmtlib_cmake_ubuntu",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant paralell compilation support
        "VERBOSE=1",
        "-j 4",
    ],
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_ONLY": "s3", # core builds always
        "ENABLE_TESTING": "OFF",
        "AUTORUN_UNIT_TESTS": "OFF",
        "BUILD_SHARED_LIBS": "OFF",
        "MINIMIZE_SIZE": "ON",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "FORCE_SHARED_CRT": "OFF",
        "SIMPLE_INSTALL": "OFF",
        "CMAKE_CXX_FLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized",
    },
    lib_source = ":all_srcs",
    out_lib_dir = "lib",
    # linking order
    out_static_libs = ["libfmt.a"],
    tags = ["requires-network"],
    alwayslink = False,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fmtlib",
    deps = [
        ":fmtlib_cmake_ubuntu",
    ],
    visibility = ["//visibility:public"],
    alwayslink = False,
)

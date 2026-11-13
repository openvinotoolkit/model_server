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

load("@bazel_skylib//rules:select_file.bzl", "select_file")
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"], exclude = ["test/**", "examples/**"]),
)

# Static build avoids SONAME/symlink handling for the shared .so and keeps
# linking into ovms straightforward; media (SRTP/RTP) stays enabled for
# audio/video tracks, built-in WebSocket signaling is disabled since OVMS
# exposes its own HTTP interface for signaling.
cmake(
    name = "libdatachannel_cmake",
    lib_source = ":all_srcs",
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "BUILD_SHARED_LIBS": "OFF",
        "USE_GNUTLS": "OFF",
        "USE_MBEDTLS": "OFF",
        "NO_MEDIA": "OFF",
        "NO_WEBSOCKET": "ON",
        "NO_EXAMPLES": "ON",
        "NO_TESTS": "ON",
    },
    out_static_libs = [
        "libdatachannel.pic.a",
        "libusrsctp.pic.a",
        "libjuice.pic.a",
        "libsrtp2.pic.a",
    ],
    # rules_foreign_cc identifies PIC archives by their .pic.a suffix.
    postfix_script = """
        for lib in datachannel usrsctp juice srtp2; do
            mv "$$INSTALLDIR$$/lib/lib$lib.a" "$$INSTALLDIR$$/lib/lib$lib.pic.a"
        done
    """,
    out_include_dir = "include",
    tags = ["requires-network"],
)

[
    select_file(
        name = lib + "_archive",
        srcs = ":libdatachannel_cmake",
        subpath = "lib/lib" + lib + ".pic.a",
    )
    for lib in ["datachannel", "usrsctp", "juice", "srtp2"]
]

cc_library(
    name = "libdatachannel",
    # Native ownership keeps these archives in cc_shared_library's link graph.
    srcs = [
        ":datachannel_archive",
        ":usrsctp_archive",
        ":juice_archive",
        ":srtp2_archive",
    ],
    deps = [":libdatachannel_cmake"],
    linkopts = [
        "-lssl",
        "-lcrypto",
        "-lpthread",
    ],
)

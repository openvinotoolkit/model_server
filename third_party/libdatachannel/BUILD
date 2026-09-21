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
        "libdatachannel.a",
        "libusrsctp.a",
        "libjuice.a",
        "libsrtp2.a",
    ],
    out_include_dir = "include",
    tags = ["requires-network"],
)

cc_library(
    name = "libdatachannel",
    deps = [":libdatachannel_cmake"],
    linkopts = [
        "-lssl",
        "-lcrypto",
        "-lpthread",
    ],
)

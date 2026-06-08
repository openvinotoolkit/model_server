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

load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "ovms_cc_library")

package(default_visibility = ["//visibility:public"])

# Intel Media Processing Interface (Intel MPI) library
# Windows-only for now — wraps GStreamer for HW-accelerated media decode/encode
# with zero-copy OpenVINO tensor integration.
#
# On Linux this target is a no-op (empty lib) so dependents can
# unconditionally list it without breaking the build.

cc_library(
    name = "intel_mpi",
    srcs = select({
        "//src:windows": ["intel_mpi.cpp"],
        "//conditions:default": [],
    }),
    hdrs = select({
        "//src:windows": [
            "intel_mpi.h",
            "imp_mpi_impl.h",
        ],
        "//conditions:default": ["intel_mpi.h"],
    }),
    copts = select({
        "//src:windows": [
            "/std:c++17",
            "/EHsc",
            "/MD",
            "/DGST_USE_UNSTABLE_API",
            "/external:anglebrackets",
            "/external:W0",
            "/wd4005",
            "/wd4100",
            "/wd4267",
            "/wd4244",
            "/wd4996",
        ],
        "//conditions:default": [],
    }),
    local_defines = COMMON_LOCAL_DEFINES + select({
        "//src:windows": ["INTEL_MPI_AVAILABLE=1"],
        "//conditions:default": [],
    }),
    deps = select({
        "//src:windows": [
            "//third_party:openvino",
            "//third_party:gstreamer",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# Thin audio-decode/encode utilities that wrap Intel MPI for OVMS calculators.
# On Linux this is a stub (all functions return false → callers fall back to dr_libs).
ovms_cc_library(
    name = "imp_audio_utils",
    srcs = ["imp_audio_utils.cpp"],
    hdrs = ["imp_audio_utils.hpp"],
    local_defines = COMMON_LOCAL_DEFINES + select({
        "//src:windows": ["INTEL_MPI_AVAILABLE=1"],
        "//conditions:default": [],
    }),
    deps = select({
        "//src:windows": [
            ":intel_mpi",
            "//third_party:openvino",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

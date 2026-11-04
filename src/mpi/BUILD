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

load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "ovms_cc_library")

package(default_visibility = ["//visibility:public"])

# Intel Media Processing Interface (Intel MPI) library
# Windows: delay-loaded GStreamer imports with D3D11 HW pipeline.
# Linux:   GStreamer loaded at runtime via dlopen (gst_loader.cpp).
#          CPU decode only; VA-API GPU path deferred.

cc_library(
    name = "intel_mpi",
    srcs = select({
        "//src:windows": ["intel_mpi.cpp"],
        "//conditions:default": ["intel_mpi.cpp", "gst_loader.cpp"],
    }),
    hdrs = select({
        "//src:windows": [
            "intel_mpi.h",
            "imp_mpi_impl.h",
        ],
        "//conditions:default": [
            "intel_mpi.h",
            "imp_mpi_impl.h",
            "gst_loader.h",
        ],
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
        "//conditions:default": [
            "-std=c++17",
        ],  # TODO @atobisze comply with OVMS copts
    }),
    local_defines = COMMON_LOCAL_DEFINES + select({
        "//src:windows": ["INTEL_MPI_AVAILABLE=1"],
        "//conditions:default": ["INTEL_MPI_AVAILABLE=1"],  # TODO @atobisze not needed anymore?
    }),
    linkopts = select({
        "//src:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = select({
        "//src:windows": [
            "//third_party:openvino",
            "//third_party:gstreamer",
        ],
        "//conditions:default": [
            "//third_party:openvino",
            "//third_party:gstreamer",
        ],
    }),
    visibility = ["//visibility:public"],
)

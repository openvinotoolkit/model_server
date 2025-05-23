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

load("//:common_settings.bzl",
     "COMMON_STATIC_LIBS_COPTS", "COMMON_STATIC_LIBS_LINKOPTS", "COMMON_FUZZER_COPTS", "COMMON_FUZZER_LINKOPTS", "COMMON_LOCAL_DEFINES", "PYBIND_DEPS")

COPTS_ADJUSTED = COMMON_STATIC_LIBS_COPTS + select({
        "//conditions:default": [],
        "//:fuzzer_build" : COMMON_FUZZER_COPTS,
})

LINKOPTS_ADJUSTED = COMMON_STATIC_LIBS_LINKOPTS + select({
    "//conditions:default": [],
    "//:fuzzer_build" : COMMON_FUZZER_LINKOPTS,
})

cc_library(
    name = "config_export",
    hdrs = ["config_export.hpp",],
    srcs = ["config_export.cpp",],
    deps = [
        "@com_github_tencent_rapidjson//:rapidjson",
        "config_export_types",
        "//src/graph_export:graph_export_types",
        "//src:libovms_server_settings",
        "//src:libovmsfilesystem",
        "//src:libovmslogging",
        "//src:libovmsschema",
    ],
    visibility = ["//visibility:public",],
    local_defines = COMMON_LOCAL_DEFINES,
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
)
cc_library(
    name = "config_export_module",
    srcs = ["config_export_module.cpp"],
    hdrs = ["config_export_module.hpp"],
    deps =  [
             "config_export",
             "config_export_types",
             "//src/graph_export:graph_export_types",
             "//src:cpp_headers",
             "//src:libovmsstring_utils",
             "//src:libovmslogging",
             "//src:libovms_server_settings",
             "//src:libovms_module",
    ],
    visibility = ["//visibility:public"],
    copts = COPTS_ADJUSTED,
    linkopts = LINKOPTS_ADJUSTED,
    local_defines = COMMON_LOCAL_DEFINES,
)

cc_library(
    name = "config_export_types",
    srcs = ["config_export_types.cpp"],
    hdrs = ["config_export_types.hpp"],
    deps =  [],
    visibility = ["//visibility:public"],
)
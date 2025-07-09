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

load("//:common_settings.bzl", "ovms_cc_library")

ovms_cc_library(
    name = "listmodels",
    hdrs = ["listmodels.hpp",],
    srcs = ["listmodels.cpp",],
    deps = [
        "//src:libovmsfilesystem",
        "//src:libovmslogging",
        "//src:modelextensions",
    ],
    visibility = ["//visibility:public",],
)
ovms_cc_library(
    name = "servablesconfigmanagermodule",
    srcs = ["servablesconfigmanagermodule.cpp"],
    hdrs = ["servablesconfigmanagermodule.hpp"],
    deps =  [
             "listmodels",
             "//src:cpp_headers",
             "//src:libovmsstring_utils",
             "//src:libovmslogging",
             "//src:libovms_server_settings",
             "//src:libovms_module",
             "//src/config_export_module:config_export",
             "//src/graph_export:graph_export_types",
             "//src/config_export_module:config_export_types",
    ],
    visibility = ["//visibility:public"],
)

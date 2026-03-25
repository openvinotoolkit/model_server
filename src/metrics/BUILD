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
load("//:common_settings.bzl", "ovms_cc_library")

ovms_cc_library(
    name = "libovmsmetrics",
    hdrs = [
        "metric.hpp",
        "metric_config.hpp",
        "metric_family.hpp",
        "metric_module.hpp",
        "metric_registry.hpp",
    ],
    srcs = [
        "metric.cpp",
        "metric_config.cpp",
        "metric_family.cpp",
        "metric_registry.cpp",
        "metric_module.cpp",
    ],
    deps = [
        "@com_github_jupp0r_prometheus_cpp//core",
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src/port:rapidjson_stringbuffer",
        "//src/port:rapidjson_writer",
        "//src:cpp_headers",
        "//src:libovms_module",
        "//src:libovmslogging",
        "//src:libovmsschema",
        "//src:libovmsstatus",
        "//src:libovmsstring_utils",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovms_metric_provider",
    hdrs = ["metric_provider.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
)

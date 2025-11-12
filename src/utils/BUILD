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
    name = "rapidjson_utils",
    hdrs = ["rapidjson_utils.hpp"],
    srcs = ["rapidjson_utils.cpp"],
    deps = [
            "//src/port:rapidjson_document",
            "//src/port:rapidjson_writer",
            "//src/port:rapidjson_stringbuffer",
    ],
    visibility = ["//visibility:public",],
)

ovms_cc_library(
    name = "env_guard",
    hdrs = ["env_guard.hpp",],
    srcs = ["env_guard.cpp",],
    deps = [
        "@ovms//src:libovmslogging",
        "@com_google_googletest//:gtest",
    ],
    visibility = ["//visibility:public"],
)
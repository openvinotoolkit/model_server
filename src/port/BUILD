#
# Copyright (c) 2020-2023 Intel Corporation
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
load("//:common_settings.bzl","ovms_cc_library")

ovms_cc_library(
    name = "rapidjson_document",
    hdrs = ["rapidjson_document.hpp"],
    deps = ["@com_github_tencent_rapidjson//:rapidjson"],
    visibility = ["//visibility:public",],
)

ovms_cc_library(
    name = "rapidjson_stringbuffer",
    hdrs = ["rapidjson_stringbuffer.hpp"],
    deps = ["@com_github_tencent_rapidjson//:rapidjson"],
    visibility = ["//visibility:public",],
)

ovms_cc_library(
    name = "rapidjson_writer",
    hdrs = ["rapidjson_writer.hpp"],
    deps = ["@com_github_tencent_rapidjson//:rapidjson"],
    visibility = ["//visibility:public",],
)

ovms_cc_library(
    name = "dr_audio",
    hdrs = ["dr_audio.hpp"],
    deps = ["@dr_libs//:dr"],
    visibility = ["//visibility:public",],
)

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
    name = "image_conversion",
    hdrs = [
        "image_conversion.hpp",
    ],
    srcs = [
        "image_conversion.cpp",
    ],
    deps = [
        "@stb//:image",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status",
        "//third_party:openvino",
        "//src:libovmslogging",
        "//src:libovmsprofiler",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "decoded_image_size",
    hdrs = ["decoded_image_size.hpp"],
    srcs = ["decoded_image_size.cpp"],
    deps = [
        ":image_conversion",
    ],
    visibility = ["//visibility:public"],
)

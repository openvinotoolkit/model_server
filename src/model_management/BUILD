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
    name = "servable_loading_task",
    hdrs = ["servable_loading_task.hpp"],
    deps = select({
        "//conditions:default": [],
        "//:not_disable_mediapipe": [
            "//src/mediapipe_internal:mediapipegraphconfig",
        ],
    }) + [
        "//src:modelconfig",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "servable_loading_queue",
    hdrs = ["servable_loading_queue.hpp"],
    srcs = ["servable_loading_queue.cpp"],
    deps = [
        "servable_loading_task",
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

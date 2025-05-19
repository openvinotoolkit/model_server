#
# Copyright (c) 2024 Intel Corporation
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
package(default_visibility = ["//visibility:public"])

alias(
    name = "opencv",
    actual = select({
        "//src:windows": "@windows_opencv//:opencv",
        "//conditions:default": "@linux_opencv//:opencv",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "openvino",
    actual = select({
        "//src:windows": "@windows_openvino//:openvino",
        "//conditions:default": "@linux_openvino//:openvino",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "genai",
    actual = select({
        "//src:windows": "@windows_genai//:genai",
        "//conditions:default": "@linux_genai//:genai",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "python3",
    actual = select({
        "//src:windows": "@python3_windows//:python3-lib",
        "//conditions:default": "@python3_linux//:python3-lib",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "curl",
    actual = select({
        "//src:windows": "@windows_curl//:curl",
        "//conditions:default": "@linux_curl//:curl",
    }),
    visibility = ["//visibility:public"],
)
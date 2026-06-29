#
# Copyright (c) 2020 Intel Corporation
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

package(
    default_visibility = ["//visibility:public"],
)
#/openvino/src/plugins/intel_gpu/include/va/va.h
cc_library(
    name = "openvino_old_headers",
    hdrs = glob([
        "include/ie/**/*.*"
    ]),
    strip_include_prefix = "include/ie",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "openvino_new_headers",
    hdrs = glob([
        "include/openvino/**/*.*"
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [
        "@linux_openvino//:openvino_old_headers",
    ]
)

cc_library(
    name = "ngraph",
    hdrs = glob([
        "include/ngraph/**/*.*"
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "openvino",
    # OpenVINO places its runtime libs under lib/intel64 on x86_64 and lib/aarch64
    # on ARM. glob ignores patterns that match nothing, so exactly one resolves per
    # architecture and the link finds libopenvino.so on both.
    srcs = glob([
        "lib/intel64/libopenvino.so",
        "lib/aarch64/libopenvino.so",
    ]),
    strip_include_prefix = "include/ie",
    visibility = ["//visibility:public"],
    deps = [
        "@linux_openvino//:ngraph",
        "@linux_openvino//:openvino_new_headers",
    ],
)

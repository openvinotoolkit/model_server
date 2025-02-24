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

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "genai_headers",
    hdrs = glob([
        "include/openvino/genai/*.*"
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)


cc_library(
    name = "genai",
    srcs = glob([
        "lib/intel64/libopenvino_genai.so",
        "lib/intel64/libopenvino_tokenizers.so",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":genai_headers",
    ],
)
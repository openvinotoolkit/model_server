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

cc_library(
    name = "openvino_genai_shared",
    srcs = glob([
        "runtime/lib/intel64/libopenvino_genai.so.2430",
        "runtime/lib/intel64/libopenvino_genai.so"
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@llm_engine//:llm_engine",
    ],
)

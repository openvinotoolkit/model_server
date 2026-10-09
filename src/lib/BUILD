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


# This target ensures placing of those libs in the same structure like in release package
# In ensures build environment matches the release.
# In is useful for libs loaded dynamically 

load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "COMMON_STATIC_LIBS_LINKOPTS")

cc_binary(
    name = "ovms_llm_calculators",
    srcs = ["//src/llm:llm_calculators_plugin_src"],
    deps = [
        "//src/llm:llm_calculators_plugin_api",
    ],
    linkopts = COMMON_STATIC_LIBS_LINKOPTS + select({
        "//src:windows": ["/FORCE:UNRESOLVED"],
        "//conditions:default": [
            "-Wl,--allow-shlib-undefined",
            "-Wl,-rpath,/ovms/lib:/opt/intel/openvino/runtime/lib/intel64",
        ],
    }),
    linkshared = True,
    local_defines = COMMON_LOCAL_DEFINES,
    visibility = ["//visibility:public"],
)

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
load("//src/mediapipe_internal:runtime_plugin.bzl", "runtime_plugin_deps")
load("@aspect_bazel_lib//:e2e/copy_action/copy.bzl", "simple_copy_file")

# Resolve common framework, protobuf and registry references against the host.
# Do not statically embed a second copy of any of their registration objects.
runtime_plugin_deps(
    name = "opencv_calculators_plugin_deps",
    deps = ["@mediapipe_calculators//:opencv_calculators"],
    provided_by = [
        "//src:ovms_lib",
        "//src/mediapipe_internal:libovms_mediapipe_runtime_owner_no_python_calculators",
        "//src/llm:llmcalculator_proto_owner",
    ],
)

cc_binary(
    name = "libovms_opencv_calculators.so",
    srcs = ["//src/mediapipe_internal:opencv_calculators_plugin_src"],
    deps = select({
        "//src:windows": ["@mediapipe_calculators//:opencv_calculators"],
        "//conditions:default": [":opencv_calculators_plugin_deps"],
    }),
    linkopts = select({
        "//src:windows": [],
        "//conditions:default": [
            "-Wl,--allow-shlib-undefined",
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
    linkshared = True,
    linkstatic = True,
    copts = select({
        "//src:windows": [],
        "//conditions:default": ["-Wno-deprecated-declarations"],
    }),
    visibility = ["//visibility:public"],
)

simple_copy_file(
    name = "copy_ovms_opencv_calculators",
    src = "libovms_opencv_calculators.so",
    out = "ovms_opencv_calculators.dll",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ovms_opencv_calculators_plugin",
    srcs = select({
        "//src:windows": [":copy_ovms_opencv_calculators"],
        "//conditions:default": [":libovms_opencv_calculators.so"],
    }),
    visibility = ["//visibility:public"],
)

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

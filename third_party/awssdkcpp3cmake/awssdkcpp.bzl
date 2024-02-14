#
# Copyright (c) 2023 Intel Corporation
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

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")
    # Note we need to escape '{/}' by doubling them due to call to format
    build_file_content = """
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
#load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

visibility = ["//visibility:public"]
# TODO
filegroup(
    name = "all_srcs",
    srcs = glob(["model_api/cpp/**"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "core",
    cache_entries = {{
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_ONLY": "s3",
        "ENABLE_TESTING": "OFF",
        "BUILD_SHARED_LIBS": "OFF",
        "MINIMIZE_SIZE": "ON",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "FORCE_SHARED_CRT": "OFF",
        "SIMPLE_INSTALL": "OFF",
    }},
    env = {{
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_static_libs = ["libmodel_api.a"],
    tags = ["requires-network"]
)

cmake(
    name = "model_api_cmake",
    cache_entries = {{
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "OpenVINO_DIR": "/opt/intel/openvino/runtime/cmake",
    }},
    env = {{
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_static_libs = ["libmodel_api.a"],
    tags = ["requires-network"]
)

cc_library(
    name = "model_api",
    deps = [
        "@mediapipe//mediapipe/framework/port:opencv_core",
        "@linux_openvino//:openvino",
        ":model_api_cmake",
    ],
    visibility = ["//visibility:public"],
)

"""
    repository_ctx.file("BUILD", build_file_content.format(http_proxy=http_proxy, https_proxy=https_proxy))

awssdkcpp3cmake_repository = repository_rule(
    implementation = _impl,
    local=False,
)

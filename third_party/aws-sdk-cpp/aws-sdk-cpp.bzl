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
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
def aws_sdk_cpp():
    aws_sdk_cpp_repository(name="_aws_sdk_cpp")
    new_git_repository(
        name = "aws-sdk-cpp",
        remote = "https://github.com/aws/aws-sdk-cpp.git",
        build_file = "@_aws_sdk_cpp//:BUILD",
        tag = "1.11.268",
        init_submodules = True,
        recursive_init_submodules = True,
        # https://github.com/bazelbuild/bazel/issues/374
        # issues with ASCI handling of file_test.c *xample file.txt in bazel
        patch_cmds = ["find . -name '*xample.txt' -delete"],
    )

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")

    result = repository_ctx.execute(["cat","/etc/os-release"],quiet=False)
    ubuntu20_count = result.stdout.count("PRETTY_NAME=\"Ubuntu 20")
    ubuntu22_count = result.stdout.count("PRETTY_NAME=\"Ubuntu 22")

    if ubuntu20_count == 1 or ubuntu22_count == 1:
        lib_path = "lib"
    else: # for redhat
        lib_path = "lib64"

    # Note we need to escape '{/}' by doubling them due to call to format
    build_file_content = """
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

visibility = ["//visibility:public"]

config_setting(
    name = "dbg",
    values = {{"compilation_mode": "dbg"}},
)

config_setting(
    name = "opt",
    values = {{"compilation_mode": "opt"}},
)

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

build_release = {{"CMAKE_BUILD_TYPE": "Release"}}
build_debug = {{"CMAKE_BUILD_TYPE": "Debug"}}
cmake(
    name = "aws-sdk-cpp_cmake",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant parallel compilation support
        "VERBOSE=1",
        "-j 32",
    ],
    cache_entries = {{
        "BUILD_ONLY": "s3", # core builds always
        "ENABLE_TESTING": "OFF",
        "AUTORUN_UNIT_TESTS": "OFF",
        "BUILD_SHARED_LIBS": "OFF",
        "MINIMIZE_SIZE": "ON",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "FORCE_SHARED_CRT": "OFF",
        "SIMPLE_INSTALL": "OFF",
        "CMAKE_CXX_FLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized\",
    }} | select({{
           "//conditions:default": dict(
               build_release
            ),
            ":dbg":  dict(
               build_debug
            ),
        }}),
    env = {{
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_lib_dir = "{lib_path}",
    # linking order
    out_static_libs = select({{
           "//conditions:default": [
                "linux/intel64/Release/libaws-cpp-sdk-s3.a",
                "linux/intel64/Release/libaws-cpp-sdk-core.a",
            ],
            ":dbg": [
                "linux/intel64/Debug/libaws-cpp-sdk-s3.a",
                "linux/intel64/Debug/libaws-cpp-sdk-core.a",
            ],
        }}) + [
            "libaws-crt-cpp.a",
            "libaws-c-s3.a",
            "libaws-c-auth.a",
            "libaws-c-cal.a",
            "libaws-c-http.a",
            "libaws-c-io.a",
            "libs2n.a",
            "libaws-c-compression.a",
            "libaws-c-sdkutils.a",
            "libaws-c-mqtt.a",
            "libaws-c-event-stream.a",
            "libaws-checksums.a",
            "libaws-c-common.a",
        ],
    tags = ["requires-network"],
    alwayslink = False,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "aws-sdk-cpp",
    deps = [
        ":aws-sdk-cpp_cmake",
    ],
    visibility = ["//visibility:public"],
    alwayslink = False,
)

"""
    repository_ctx.file("BUILD", build_file_content.format(http_proxy=http_proxy, https_proxy=https_proxy, lib_path=lib_path))

aws_sdk_cpp_repository = repository_rule(
    implementation = _impl,
    local=False,
)

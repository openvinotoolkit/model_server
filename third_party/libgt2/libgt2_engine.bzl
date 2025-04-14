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

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def libgt2_engine():
    libgt2_repository(name="_libgt2_engine")
    new_git_repository(
        name = "libgt2_engine",
        remote = "https://github.com/libgit2/libgit2.git",
        commit = "338e6fb681369ff0537719095e22ce9dc602dbf0", # Dec 28, 2024 - v1.9.0
        build_file = "@_libgt2_engine//:BUILD",
        patch_args = ["-p1"],
        patches = ["lfs.patch"],
    )

    #native.new_local_repository(
    #    name = "libgt2_engine",
    #    path = "/libgt2_engine",
    #    build_file = "@_libgt2_engine//:BUILD",
    #)

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    if http_proxy == "":
        http_proxy = repository_ctx.os.environ.get("HTTP_PROXY", "")
    
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")
    if https_proxy == "":
        https_proxy = repository_ctx.os.environ.get("HTTPS_PROXY", "")

    if _is_windows(repository_ctx):
        lib_name = "git2"
        out_static = "out_interface_libs = [\"{lib_name}.lib\"],".format(lib_name=lib_name)
        out_libs = "out_shared_libs = [\"{lib_name}.dll\"],".format(lib_name=lib_name)
        cache_entries = """
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": " -s -D_GLIBCXX_USE_CXX11_ABI=1",
        "CMAKE_LIBRARY_OUTPUT_DIRECTORY": "Debug",
        "WIN32": "True",
        "X86_64": "True",
        "BUILD_EXAMPLES": "OFF",
        "BUILD_TESTS": "OFF",
        "BUILD_CLI": "OFF"
        """
    else:
        lib_name = "libgit2"
        out_static = ""
        out_libs = "out_shared_libs = [\"{lib_name}.so\"],".format(lib_name=lib_name)
        cache_entries = """
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": " -s -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized",
        "CMAKE_ARCHIVE_OUTPUT_DIRECTORY": "lib",
        "CMAKE_INSTALL_LIBDIR": "lib",
        "BUILD_EXAMPLES": "OFF",
        "BUILD_TESTS": "OFF",
        "BUILD_CLI": "OFF",
        """

    # Note that we need to escape '{/}' by doubling them due to call to format
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
    srcs = glob(["**"], exclude = ["tests/**"]),
    visibility = ["//visibility:public"],
)

build_release = {{"CMAKE_BUILD_TYPE": "Release"}}
build_debug = {{"CMAKE_BUILD_TYPE": "Debug"}}

cmake(
    name = "libgt2_cmake",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant parallel compilation support
        #"-j 32",
    ],
    cache_entries = {{ 
        {cache_entries}
    }} | select({{
           "//conditions:default": dict(
               build_release
            ),
            ":dbg":  dict(
               build_debug
            ),
        }}),
    env = {{
        "http_proxy": "{http_proxy}",
        "https_proxy": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_include_dir = "include",
    {out_libs}
    {out_static}
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
    lib_name = "{lib_name}",
    deps = []
)

cc_library(
    name = "libgt2_engine",
    deps = [
        ":libgt2_cmake",
    ],
    visibility = ["//visibility:public"],
)
"""
    repository_ctx.file("BUILD", build_file_content.format(http_proxy=http_proxy, https_proxy=https_proxy,
                                                            lib_name=lib_name, out_libs=out_libs, cache_entries=cache_entries, out_static=out_static))

libgt2_repository = repository_rule(
    implementation = _impl,
    local=True,
)

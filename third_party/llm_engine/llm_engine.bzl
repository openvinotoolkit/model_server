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

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def llm_engine():
    llm_engine_repository(name="_llm_engine")
    new_git_repository(
        name = "llm_engine",
        remote = "https://github.com/mzegla/openvino.genai",
        commit = "4fe7c52ee5ded67837852f1eeb986dbe1efaf6d8", # temporary commit with fix for prompt lookup
        build_file = "@_llm_engine//:BUILD",
        init_submodules = True,
        recursive_init_submodules = True,
        patch_args = ["-p1"],
        patches = ["cb.patch"],
    )
    # when using local repository manually run: git submodule update --recursive 
    #native.new_local_repository(
    #    name = "llm_engine",
    #    path = "/openvino.genai",
    #    build_file = "@_llm_engine//:BUILD",
    #)

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    if http_proxy == "":
        http_proxy = repository_ctx.os.environ.get("HTTP_PROXY", "")
    
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")
    if https_proxy == "":
        https_proxy = repository_ctx.os.environ.get("HTTPS_PROXY", "")
    
    OpenVINO_DIR = repository_ctx.os.environ.get("OpenVINO_DIR", "")

    if _is_windows(repository_ctx):
        icudt = "icudt70"
        icuuc = "icuuc70"
        tokenizers = "openvino_tokenizers"
        lib_name = "openvino_genai"
        OpenVINO_DIR = OpenVINO_DIR.replace("\\", "\\\\").replace("/", "\\\\")
        out_dll_dir_win = "out_dll_dir = \"runtime/bin/Release\","
        out_lib_dir = "out_lib_dir = \"runtime/lib/Release\""
        out_static = "out_interface_libs = [\"{lib_name}.lib\"],".format(lib_name=lib_name)
        out_libs = "out_shared_libs = [\"{lib_name}.dll\"],".format(lib_name=lib_name)
        cache_entries = """
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": " -s -D_GLIBCXX_USE_CXX11_ABI=1",
        "CMAKE_LIBRARY_OUTPUT_DIRECTORY": "runtime/bin/Release",
        "WIN32": "True",
        "X86_64": "True",
        "BUILD_TOKENIZERS": "OFF",
        """
        jobs_param = "\"-j 8\"" # on Windows we do not need to specify number of jobs, it's set to all available cores number
    else:
        lib_name = "libopenvino_genai"
        out_dll_dir_win = ""
        out_lib_dir = "out_lib_dir = \"runtime/lib/intel64\""
        out_static = ""
        out_libs = "out_shared_libs = [\"{lib_name}.so.2520\"],".format(lib_name=lib_name)
        cache_entries = """
        "BUILD_SHARED_LIBS": "OFF",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": " -s -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized",
        "CMAKE_ARCHIVE_OUTPUT_DIRECTORY": "lib",
        "ENABLE_SYSTEM_ICU": "True",
        "BUILD_TOKENIZERS": "OFF",
        """
        jobs_param = "\"-j 8\"" # on Linux we need to specify jobs number, by default it's set to 1

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
    name = "llm_engine_cmake",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant parallel compilation support - lets go with default - CORES + 2 for ninja
        {jobs_param}
    ],
    cache_entries = {{ 
        {cache_entries}
    }} | select({{
           "//conditions:default": dict(
               build_release
            ),
            # Debug does not build
            #":dbg":  dict(
            #   build_debug
            #),
        }}),
    env = {{
        "OpenVINO_DIR": "{OpenVINO_DIR}",
        "http_proxy": "{http_proxy}",
        "https_proxy": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_include_dir = "runtime/include",
    {out_lib_dir},
    {out_libs}
    {out_static}
    {out_dll_dir_win}
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
    lib_name = "{lib_name}",
    deps = [
        "@ovms//third_party:openvino",
    ]
)

cc_library(
    name = "llm_engine",
    deps = [
        "@ovms//third_party:openvino",
        ":llm_engine_cmake",
    ],
    visibility = ["//visibility:public"],
)
"""
    repository_ctx.file("BUILD", build_file_content.format(OpenVINO_DIR=OpenVINO_DIR, http_proxy=http_proxy, https_proxy=https_proxy,
                                                            out_dll_dir_win=out_dll_dir_win, out_lib_dir=out_lib_dir, lib_name=lib_name, out_libs=out_libs, cache_entries=cache_entries, out_static=out_static,
                                                            jobs_param=jobs_param))

llm_engine_repository = repository_rule(
    implementation = _impl,
    local=True,
)

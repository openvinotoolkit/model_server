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
        remote = "https://github.com/openvinotoolkit/openvino.genai",
        commit = "e5cf8ce1cb68c56d833a4959d4afae5a0ee2996e", # / Jan 24
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

    core = "core_tokenizers"
    icudt = "icudt70"
    icuuc = "icuuc70"
    tokenizers = "openvino_tokenizers"
    lib_name = "openvino_genai"

    if _is_windows(repository_ctx):
        OpenVINO_DIR = OpenVINO_DIR.replace("\\", "\\\\").replace("/", "\\\\")
        out_dll_dir_win = "out_dll_dir = \"runtime/bin/Release\","
        out_lib_dir = "out_lib_dir = \"runtime/lib/Release\""
        out_static = "out_static_libs = [\"{lib_name}.lib\"],".format(lib_name=lib_name)
        out_libs = "out_shared_libs = [\"{lib_name}.dll\", \"{core}.dll\", \"{icudt}.dll\", \"{icuuc}.dll\", \"{tokenizers}.dll\"]".format(lib_name=lib_name, core=core, icuuc=icuuc, icudt=icudt, tokenizers=tokenizers)
        cache_entries = """
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": " -s -D_GLIBCXX_USE_CXX11_ABI=1",
        "CMAKE_LIBRARY_OUTPUT_DIRECTORY": "runtime/bin/Release",
        "WIN32": "True",
        "X86_64": "True"
        """
    else:
        out_dll_dir_win = ""
        out_lib_dir = "out_lib_dir = \"runtime/lib/intel64\""
        out_static = ""
        out_libs = "out_shared_libs = [\"lib{lib_name}.so.2500\", \"lib{core}.so\", \"lib{icudt}.so\", \"lib{icuuc}.so\", \"lib{tokenizers}.so\"]".format(lib_name=lib_name, core=core, icuuc=icuuc, icudt=icudt, tokenizers=tokenizers)
        cache_entries = """
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": " -s -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized",
        "CMAKE_ARCHIVE_OUTPUT_DIRECTORY": "lib"
        """

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
        #"-j 32",
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
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
        "http_proxy": "{http_proxy}",
        "https_proxy": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_include_dir = "runtime/include",
    {out_lib_dir},
    {out_libs},
    {out_static}
    {out_dll_dir_win}
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
    lib_name = "{lib_name}",
    deps = [
        "@ovms//third_party:openvino",
    ],
    alwayslink = 1,
)

cc_library(
    name = "llm_engine",
    deps = [
        "@ovms//third_party:openvino",
        ":llm_engine_cmake",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
"""
    repository_ctx.file("BUILD", build_file_content.format(OpenVINO_DIR=OpenVINO_DIR, http_proxy=http_proxy, https_proxy=https_proxy,
                                                            out_dll_dir_win=out_dll_dir_win, out_lib_dir=out_lib_dir, lib_name=lib_name, out_libs=out_libs, cache_entries=cache_entries, out_static=out_static))

llm_engine_repository = repository_rule(
    implementation = _impl,
    local=False,
)

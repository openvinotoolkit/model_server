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
def jinja2cpp():
    llm_engine_repository(name="_jinja2cpp")
    new_git_repository(
        name = "jinja2cpp",
        remote = "https://github.com/atobiszei/Jinja2Cpp",
        commit = "98a69ee68502d7c186a93870512cf112fdd3fd17",
#        commit = "20ab985f8798c72e0443d9bc8d335bf772c45dd6", # does not work
#        commit = "9ebc98b46ee8c9c6a2ad2475d1c7c31b9d959ac9", # works
        #commit = "73ef8f2699611c7058135c1a7a316c4c748bd92e", works with Ilya jinja
        build_file = "@_jinja2cpp//:BUILD",
        init_submodules = True,
        recursive_init_submodules = True,
    )

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")
    OpenVINO_DIR = repository_ctx.os.environ.get("OpenVINO_DIR", "")
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

cmake(
    name = "jinja2cpp_cmake_ubuntu",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant paralell compilation support
        "VERBOSE=1",
        "-j 12",
    ],
    cache_entries = {{
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_SHARED_LIBS": "OFF",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized\",
        #"CMAKE_ARCHIVE_OUTPUT_DIRECTORY": "lib",
        "JINJA2CPP_BUILD_TESTS": "OFF",
 #       "JINJA2CPP_INSTALL": "ON",
 #       "JINJA2CPP_BUILD_SHARED": "OFF",
 #       "JINJA2CPP_DEPS_MODE": "internal",
        "JINJA2CPP_DEPS_MODE": "external-boost",
 #       "JINJA2CPP_DEPS_MODE": "conan-build",
         "JINJA2CPP_STRICT_WARNINGS": "OFF",
 #       "JINJA2CPP_WITH_JSON_BINDINGS": "none",
    }},
    env = {{
 #       "OpenVINO_DIR": "{OpenVINO_DIR}",
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    #out_lib_dir = "lib",
    out_lib_dir = "",
    out_bin_dir = "",
    out_binaries = ["noizatkao"],
    out_data_dirs = ["_deps"],
    # linking order
    out_static_libs = [
            #"libopenvino_continuous_batching.a",
            "lib/libopenvino_continuous_batching.a",
            "_deps/jinja2cpp-build/lib/libjinja2cpp.a",
        ],
    install = True,
    tags = ["requires-network"],
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "jinja2cpp_ubuntu",
    deps = [
        ":jinja2cpp_cmake_ubuntu",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
    linkstatic = True,
    features = ["fully_static_link"],
)

cmake(
    name = "llm_engine_cmake_redhat",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant paralell compilation support
        "VERBOSE=1",
        "-j 4",
    ],
    cache_entries = {{
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_SHARED_LIBS": "OFF",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "CMAKE_CXX_FLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized\",
        "CMAKE_ARCHIVE_OUTPUT_DIRECTORY": "lib"
    }},
    env = {{
        "OpenVINO_DIR": "{OpenVINO_DIR}",
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_lib_dir = "lib64",
   
    # linking order
    out_static_libs = [
            "libopenvino_continuous_batching.a",
        ],
    tags = ["requires-network"],
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "llm_engine_redhat",
    deps = [
        ":llm_engine_cmake_redhat",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
    linkstatic = True,
)
"""
    repository_ctx.file("BUILD", build_file_content.format(OpenVINO_DIR=OpenVINO_DIR, http_proxy=http_proxy, https_proxy=https_proxy))

llm_engine_repository = repository_rule(
    implementation = _impl,
    local=False,
)

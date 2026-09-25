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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

DROGON_COMMIT = "686f68a12ff82c6c3114a4c48502334261f258f6"
TRANTOR_COMMIT = "8bf280ba043eb77c3eb02121065e0891a9a4f431"

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def drogon_cpp():
    http_archive(
        name = "jsoncpp",
        sha256 = "f93b6dd7ce796b13d02c108bc9f79812245a82e577581c4c9aabe57075c90ea2",
        strip_prefix = "jsoncpp-1.9.6",
        urls = [
            "https://github.com/open-source-parsers/jsoncpp/archive/1.9.6.tar.gz",
        ],
        build_file_content = """
cc_library(
    name = "jsoncpp",
    srcs = glob(["src/lib_json/*.cpp"]),
    hdrs = glob(["include/json/*.h", "src/lib_json/*.h", "src/lib_json/*.inl"]),
    includes = [
        "include",  # Add the top-level include directory
        "src/lib_json",  # Add the source-level include directory
    ],
    visibility = ["//visibility:public"],
    linkstatic = True,
)
        """,
    )
    drogon_cpp_repository(
        name = "drogon",
        patches = [
            "@ovms//third_party/drogon:ovms_drogon_trantor.patch",
        ],
    )

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    if http_proxy == "":
        http_proxy = repository_ctx.os.environ.get("HTTP_PROXY", "")

    https_proxy = repository_ctx.os.environ.get("https_proxy", "")
    if https_proxy == "":
        https_proxy = repository_ctx.os.environ.get("HTTPS_PROXY", "")

    repository_ctx.download_and_extract(
        url = "https://github.com/drogonframework/drogon/archive/{}.tar.gz".format(DROGON_COMMIT),
        sha256 = "47baf8d02acb21ef3ccaca407aeef9bc3c692e9b9a12d5f8f8431ec486eeea28",
        stripPrefix = "drogon-{}".format(DROGON_COMMIT),
    )
    repository_ctx.download_and_extract(
        url = "https://github.com/an-tao/trantor/archive/{}.tar.gz".format(TRANTOR_COMMIT),
        output = "trantor",
        sha256 = "f32272fe923bdbc20143fe5de4b66908747b25b7e32e068c0e83f58171197990",
        stripPrefix = "trantor-{}".format(TRANTOR_COMMIT),
    )
    for patch in repository_ctx.attr.patches:
        repository_ctx.patch(patch, strip = 1)

    # Create the Python script dynamically
    repository_ctx.file("remove_japanese_txt.py", """
import os
import fnmatch
import sys

def remove_japanese_txt(directory):
    for root, dirs, files in os.walk(directory):
        for file in fnmatch.filter(files, '*.txt'):
            file_path = os.path.join(root, file)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    directory = f"{os.getcwd()}/lib/tests/integration_test/server/"
    print(f"Working in {directory}")
    remove_japanese_txt(directory)
""")

    # Locate the Python binary
    python_binary = repository_ctx.which("python3") or repository_ctx.which("python")
    if not python_binary:
        fail("Python interpreter not found in PATH")

    # Execute the Python script
    # This patches drogon repo to remove txt files which break building, such as 中文.txt
    result = repository_ctx.execute([python_binary, "remove_japanese_txt.py"], environment=repository_ctx.os.environ)

    # Log the script's stdout and stderr
    if result.return_code == 0:
        print("Script executed successfully!")
        print("Output:", result.stdout)  # Log the standard output
    else:
        fail("Script execution failed: " + result.stderr)  # Log the error output

    if _is_windows(repository_ctx):
        cache_entries = """
            "ZLIB_INCLUDE_DIR": "../drogon_cmake.ext_build_deps/include",  # This is a hack because drogon does not allow absolute path
            "ZLIB_LIBRARY": "@zlib//:zlib",
            "JSONCPP_INCLUDE_DIR": "@jsoncpp//:jsoncpp",
            "BUILD_CTL": "OFF",
            "BUILD_EXAMPLES": "OFF",
            "BUILD_ORM": "OFF",
            "BUILD_BROTLI": "OFF",
            "BUILD_YAML_CONFIG": "OFF",
            "CMAKE_INSTALL_LIBDIR": "lib",
            "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
            "CMAKE_CXX_STANDARD": "17",
            "CXX_FILESYSTEM_HAVE_FS": "1",
            "CMAKE_CXX_FLAGS": " /guard:cf /GS -s -D_GLIBCXX_USE_CXX11_ABI=1"
        """
        out_static_libs = [
            "drogon.lib",
            "trantor.lib"
        ]
    else:
        cache_entries  = """
            "CMAKE_CXX_FLAGS": " -s -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-error=deprecated-declarations -Wuninitialized ",
            "ZLIB_INCLUDE_DIR": "../drogon_cmake.ext_build_deps/include",  # This is a hack because drogon does not allow absolute path
            "ZLIB_LIBRARY": "@zlib//:zlib",
            "JSONCPP_INCLUDE_DIR": "@jsoncpp//:jsoncpp",
            "BUILD_CTL": "OFF",
            "BUILD_EXAMPLES": "OFF",
            "BUILD_ORM": "OFF",
            "BUILD_BROTLI": "OFF",
            "BUILD_YAML_CONFIG": "OFF",
            "CMAKE_INSTALL_LIBDIR": "lib",
            "CMAKE_POSITION_INDEPENDENT_CODE": "ON"
        """
        out_static_libs = [
            "libdrogon.a",
            "libtrantor.a"
        ]

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

build_release = {{"CMAKE_BUILD_TYPE": "Release"}}
build_debug = {{"CMAKE_BUILD_TYPE": "Debug"}}

filegroup(
    name = "all_srcs",
    srcs = glob(["**"], exclude = ["lib/tests/**/*.txt"]),
    visibility = ["//visibility:public"],
)

cmake(
    name = "drogon_cmake",
    build_args = [
        "--verbose",
        "--",  # <- Pass remaining options to the native tool.
        # https://github.com/bazelbuild/rules_foreign_cc/issues/329
        # there is no elegant parallel compilation support
        "-j 6",
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
        "HTTP_PROXY": "{http_proxy}",
        "HTTPS_PROXY": "{https_proxy}",
    }},
    deps = ["@jsoncpp//:jsoncpp", "@zlib//:zlib"],
    lib_source = ":all_srcs",
    out_lib_dir = "lib",
    # linking order
    out_static_libs = {out_static_libs},
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
)
"""
    repository_ctx.file("BUILD", build_file_content.format(
        http_proxy=http_proxy, https_proxy=https_proxy, is_windows=_is_windows(repository_ctx),
        cache_entries=cache_entries, out_static_libs=out_static_libs))

drogon_cpp_repository = repository_rule(
    implementation = _impl,
    attrs = {
        "patches": attr.label_list(),
    },
    local=False,
)

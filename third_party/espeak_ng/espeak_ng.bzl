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
# Fetches and builds espeak-ng from source using rules_foreign_cc (CMake)
# so OVMS no longer relies on a system-installed espeak-ng package.
#
# Works on both Linux (libespeak-ng.so + share/espeak-ng-data) and Windows
# (espeak-ng.dll + espeak-ng-data). The build is gated by the //:espeak
# build flag; when set to "off", the rule is still defined but no targets
# in OVMS depend on it.

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

# Pinned to espeak-ng release tag 1.52.0 (commit hash).
_ESPEAK_NG_COMMIT = "212928b394a96e8fd2096616bfd54e17845c48f6"  # 1.52.0
_ESPEAK_NG_REMOTE = "https://github.com/espeak-ng/espeak-ng.git"

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def espeak_ng():
    _espeak_ng_repository(name = "_espeak_ng")
    new_git_repository(
        name = "espeak_ng",
        remote = _ESPEAK_NG_REMOTE,
        commit = _ESPEAK_NG_COMMIT,
        build_file = "@_espeak_ng//:BUILD",
        init_submodules = False,
        shallow_since = "1709251200 +0000",  # roughly 2024-03-01, around 1.52.0
        patches = ["@ovms//third_party/espeak_ng:out_of_source_phsource.patch"],
        patch_args = ["-p1"],
    )

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    if http_proxy == "":
        http_proxy = repository_ctx.os.environ.get("HTTP_PROXY", "")
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")
    if https_proxy == "":
        https_proxy = repository_ctx.os.environ.get("HTTPS_PROXY", "")

    if _is_windows(repository_ctx):
        # On Windows espeak-ng builds espeak-ng.dll.
        out_shared = "out_shared_libs = [\"espeak-ng.dll\"],"
        out_interface = "out_interface_libs = [\"espeak-ng.lib\"],"
        out_dll_dir = "out_dll_dir = \"bin\","
        out_lib_dir = "out_lib_dir = \"lib\""
        platform_cache = """
            "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
            "CMAKE_CXX_FLAGS": " /guard:cf /GS /DN_PATH_HOME=1024",
            "CMAKE_C_FLAGS": " /DN_PATH_HOME=1024",
            "WIN32": "True",
        """
        jobs_param = "\"-j 8\""
    else:
        # On Linux espeak-ng produces libespeak-ng.so.<ABI>.
        out_shared = "out_shared_libs = [\"libespeak-ng.so.1\"],"
        out_interface = ""
        out_dll_dir = ""
        out_lib_dir = "out_lib_dir = \"lib\""
        platform_cache = """
            "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
            "CMAKE_C_FLAGS": " -fPIC -Wno-error -DN_PATH_HOME=1024",
        """
        jobs_param = "\"-j 8\""

    # NOTE: braces inside the format string are doubled to escape them.
    build_file_content = """
load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")

visibility = ["//visibility:public"]

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

# Build espeak-ng as a minimal text-to-phoneme library; we do not need
# audio output, MBROLA, klatt, libsonic or the CLI executable at runtime.
cmake(
    name = "espeak_ng_cmake",
    build_args = [
        "--verbose",
        "--",
        {jobs_param}
    ],
    cache_entries = {{
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_SHARED_LIBS": "ON",
        "USE_ASYNC": "OFF",
        "USE_MBROLA": "OFF",
        "USE_LIBSONIC": "OFF",
        "USE_LIBPCAUDIO": "OFF",
        "USE_KLATT": "OFF",
        "USE_SPEECHPLAYER": "OFF",
        "BUILD_ESPEAK_NG_EXE": "OFF",
        "BUILD_SPEAK_EXE": "OFF",
        "EXTRA_cmn": "OFF",
        "EXTRA_ru": "OFF",
        {platform_cache}
    }},
    env = {{
        "http_proxy": "{http_proxy}",
        "https_proxy": "{https_proxy}",
    }},
    lib_source = ":all_srcs",
    out_include_dir = "include",
    {out_lib_dir},
    {out_shared}
    {out_interface}
    {out_dll_dir}
    out_data_dirs = ["share/espeak-ng-data"],
    tags = ["requires-network"],
    visibility = ["//visibility:public"],
    lib_name = "espeak-ng",
)

# Headers + shared library suitable for cc_library deps.
cc_library(
    name = "espeak_ng",
    deps = [":espeak_ng_cmake"],
    visibility = ["//visibility:public"],
)

# Exposes the espeak-ng-data directory as a filegroup so create_package.sh
# (or similar packaging) can locate it under bazel-out.
filegroup(
    name = "espeak_ng_data",
    srcs = [":espeak_ng_cmake"],
    output_group = "gen_dir",
    visibility = ["//visibility:public"],
)
"""
    repository_ctx.file(
        "BUILD",
        build_file_content.format(
            http_proxy = http_proxy,
            https_proxy = https_proxy,
            out_shared = out_shared,
            out_interface = out_interface,
            out_dll_dir = out_dll_dir,
            out_lib_dir = out_lib_dir,
            platform_cache = platform_cache,
            jobs_param = jobs_param,
        ),
    )

_espeak_ng_repository = repository_rule(
    implementation = _impl,
    local = True,
)

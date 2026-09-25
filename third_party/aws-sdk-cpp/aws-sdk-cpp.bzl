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
AWS_SDK_CPP_ARCHIVES = [
    ("aws", "aws-sdk-cpp", "f07aeab3d044f09d32c3b9d92b7e6dba8573fc45", "552f906867f0bd31c99d6bd6f02aa01c1ac9b246377b6897a8ac97b58824ff75", ""),
    ("awslabs", "aws-crt-cpp", "e4514b7fb8b1fe67429aa7b0e00f628999722174", "4d3559f53752f0aaa9dc5744043d96413d0aac97b429fe5ec5a1a62ce3a15bbd", "crt/aws-crt-cpp"),
    ("awslabs", "aws-c-auth", "6ba7a0f8688c713dfe137716dbd5be324c2315b0", "d9cf3931f352f82cb608e26dbdbfaab0d861a7ea1939f1ff09fc346af368e5bd", "crt/aws-crt-cpp/crt/aws-c-auth"),
    ("awslabs", "aws-c-cal", "56f0a79ceb10f2efcf92f525ace717f84d8c8a11", "cf3cbf0ef28f088f705f2dec63a863eb8c25e52e42c951e8257af5da6a362e10", "crt/aws-crt-cpp/crt/aws-c-cal"),
    ("awslabs", "aws-c-common", "8eaa0986ad3cfd46c87432a2e4c8ab81a786085f", "57d9d62f66877a55085f9d02b2feb72d400c37e356e44af7014468c4f50924d6", "crt/aws-crt-cpp/crt/aws-c-common"),
    ("awslabs", "aws-c-compression", "99ec79ee2970f1a045d4ced1501b97ee521f2f85", "8da75f0578157fd3d225672c66088f7a9426235fbba31545a2b4e894cb38af16", "crt/aws-crt-cpp/crt/aws-c-compression"),
    ("awslabs", "aws-c-event-stream", "63d1e1021b04ce3c3b1fc1895078ac85e0430b24", "d4eea3f912aa9b15bb8734bd0ecf0010b5dda9c750d40bd39a10dc77251f3e02", "crt/aws-crt-cpp/crt/aws-c-event-stream"),
    ("awslabs", "aws-c-http", "6a1c157c20640a607102738909e89561a41e91e9", "6b98b52acbf15f37138da950f3dba33640bd1aa30b6164a30bae439113d098ba", "crt/aws-crt-cpp/crt/aws-c-http"),
    ("awslabs", "aws-c-io", "6225ebb9da28f1023ad5e21694de9d165cd65f3b", "4acf6ec5b20ff3ea151d7527fe54b08b391d8a574490403f590debb4f9f1d13f", "crt/aws-crt-cpp/crt/aws-c-io"),
    ("awslabs", "aws-c-mqtt", "17ee24a2177fc64cf9773d430a24e6fa06a89dd0", "f251a8df0da82c00d88fcc07bc4c856315d09166ac864822e2a9b58e098a7216", "crt/aws-crt-cpp/crt/aws-c-mqtt"),
    ("awslabs", "aws-c-s3", "1dd55be83b19a55cd9c155e2da977cdc76112a91", "2e9e96a51bbcdcbff645272a37ba2dd2e25a32ea2e69841fe3c7608e83ab5b35", "crt/aws-crt-cpp/crt/aws-c-s3"),
    ("awslabs", "aws-c-sdkutils", "fd8c0ba2e233997eaaefe82fb818b8b444b956d3", "03f4aedf78f119c7962d7862400182ac9ec668453aa50481fb810a311f8dbd46", "crt/aws-crt-cpp/crt/aws-c-sdkutils"),
    ("awslabs", "aws-checksums", "321b805559c8e911be5bddba13fcbd222a3e2d3a", "ebf31bc916d2319bc135353ba22ba31e24b18d8ff402b959614fb9b5cda8decd", "crt/aws-crt-cpp/crt/aws-checksums"),
    ("awslabs", "aws-lc", "dc4e28145ceb6d46b5475e833f2da8def6d583fe", "182c9cca691a6550bb1823eec6fa90fd9af5ac11be02d75b904da20ed1c6983d", "crt/aws-crt-cpp/crt/aws-lc"),
    ("awslabs", "s2n", "0998358a6ef7c4f22295deba088796fe354c5f4c", "6c5b0a60a4e2a495bde40d056dc1ce9237db27ebf93db0a68281c1414a707f78", "crt/aws-crt-cpp/crt/s2n"),
    ("awslabs", "aws-verification-model-for-libcrypto", "440a07ca02d60ff8158cb99fb89833de35d76ae6", "ab99f0451b568652558489fa05735a5a6039b49af3157c474fc552e80389181d", "crt/aws-crt-cpp/crt/s2n/tests/cbmc/aws-verification-model-for-libcrypto"),
]

def aws_sdk_cpp():
    aws_sdk_cpp_repository(name = "aws-sdk-cpp")

def _impl(repository_ctx):
    http_proxy = repository_ctx.os.environ.get("http_proxy", "")
    https_proxy = repository_ctx.os.environ.get("https_proxy", "")

    for owner, repository, commit, sha256, output in AWS_SDK_CPP_ARCHIVES:
        archive_root = "s2n-tls" if repository == "s2n" else repository
        repository_ctx.download_and_extract(
            url = "https://github.com/{}/{}/archive/{}.tar.gz".format(owner, repository, commit),
            output = output,
            sha256 = sha256,
            stripPrefix = "{}-{}".format(archive_root, commit),
        )

    # https://github.com/bazelbuild/bazel/issues/374
    repository_ctx.execute(["find", ".", "-name", "*xample.txt", "-delete"])

    result = repository_ctx.execute(["cat","/etc/os-release"],quiet=False)
    ubuntu20_count = result.stdout.count("PRETTY_NAME=\"Ubuntu 20")
    ubuntu22_count = result.stdout.count("PRETTY_NAME=\"Ubuntu 22")
    ubuntu24_count = result.stdout.count("PRETTY_NAME=\"Ubuntu 24")

    if ubuntu24_count == 1 or ubuntu22_count == 1:
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

#
# Copyright (c) 2020 Intel Corporation
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

workspace(name = "ovms")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# 2023-06-05
# This version of Glog is required for Windows support, but currently causes
# crashes on some Android devices.
# OVMS - must be before MP
http_archive(
    name = "com_github_glog_glog_windows",
    strip_prefix = "glog-3a0d4d22c5ae0b9a2216988411cfa6bf860cc372",
    sha256 = "170d08f80210b82d95563f4723a15095eff1aad1863000e8eeb569c96a98fefb",
    urls = [
      "https://github.com/google/glog/archive/3a0d4d22c5ae0b9a2216988411cfa6bf860cc372.zip",
    ],
    patches = [
        "@mediapipe//third_party:com_github_glog_glog.diff",
        "@mediapipe//third_party:com_github_glog_glog_windows_patch.diff",
    ],
    patch_args = [
        "-p1",
    ],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "37cdfbc6faefea94f7b37760a305c98c08981116c2bc9e821e3b423221fad8c8",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.9.2/bazel-skylib-1.9.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.9.2/bazel-skylib-1.9.2.tar.gz",
    ],
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check(minimum_bazel_version = "7.0.0")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# platforms bumped to include ppc64le (needed by rules_go under bazel 7 / grpc_extra_deps).
http_archive(
    name = "platforms",
    sha256 = "3384eb1c30762704fbe38e440204e114154086c8fc8a8c2e3e28441028c019a8",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
        "https://github.com/bazelbuild/platforms/releases/download/1.0.0/platforms-1.0.0.tar.gz",
    ],
)

# bazel_features - required by newer rules_cc / mediapipe upstream.
http_archive(
    name = "bazel_features",
    sha256 = "5ac743bf5f05d88e84962e978811f2524df09602b789c92cf7ae2111ecdeda94",
    strip_prefix = "bazel_features-1.14.0",
    url = "https://github.com/bazel-contrib/bazel_features/releases/download/v1.14.0/bazel_features-v1.14.0.tar.gz",
)
load("@bazel_features//:deps.bzl", "bazel_features_deps")
bazel_features_deps()

# Newer rules_cc required by upstream mediapipe (@rules_cc//cc:cc_library.bzl, cc_binary.bzl).
http_archive(
    name = "rules_cc",
    sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6",
    strip_prefix = "rules_cc-0.1.5",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz",
)

http_archive(
    name = "rules_python",
    sha256 = "0cc05ddb27614baecace068986931e2a6e9f69114e6115fc5dc58250faf56e0f",
    strip_prefix = "rules_python-0.37.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.37.0/rules_python-0.37.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# ABSL pinned to a release with `absl/status/status_macros.h` (required by
# mediapipe upstream) and compatible with protobuf 6.31 / gRPC 1.74.
http_archive(
    name = "com_google_absl",
    sha256 = "6e1aee535473414164bf83e4ebc40240dec71a4701f8a642d906e95bea1aea0c",
    strip_prefix = "abseil-cpp-20260526.0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20260526.0.tar.gz"],
    patches = ["@ovms//third_party/absl:absl_constexpr_fix.patch"],
    patch_args = ["-p1"],
    patch_tool = "patch",
)

# Pre-declare `com_google_googletest` (satisfies protobuf_deps's existing-rule
# check) so it does not pull in a second abseil under the name `@abseil-cpp`.
http_archive(
    name = "com_google_googletest",
    sha256 = "65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c",
    strip_prefix = "googletest-1.17.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz"],
)

http_archive(
    name = "zlib",
    build_file = "@mediapipe//third_party:zlib.BUILD",
    sha256 = "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16",
    strip_prefix = "zlib-1.3.2",
    url = "https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz",
)

# RapidJSON
# Must be defined earlier than tensorflow_serving because TFS is using older rapidjson
# Version must match openvino.genai -> jinja2cpp -> rapidjson
# git/Jinja2Cpp/third_party/internal_deps.cmake
# Date:   Tue May 9 21:31:22 2023 +0000 Avoid ptrdiff between pointers to different allocations
http_archive(
    name = "com_github_tencent_rapidjson",
    url = "https://github.com/Tencent/rapidjson/archive/973dc9c06dcd3d035ebd039cfb9ea457721ec213.tar.gz",
    sha256 = "d0c9e52823d493206eb721d38cb3a669ca0212360862bd15a3c2f7d35ea7c6f7",
    strip_prefix = "rapidjson-973dc9c06dcd3d035ebd039cfb9ea457721ec213",
    build_file = "@//third_party/rapidjson:BUILD"
)

http_archive(
    name = "com_github_libevent_libevent",
    url = "https://github.com/libevent/libevent/archive/release-2.1.12-stable.zip",
    sha256 = "8836ad722ab211de41cb82fe098911986604f6286f67d10dfb2b6787bf418f49",
    strip_prefix = "libevent-release-2.1.12-stable",
    build_file = "@//third_party/libevent:BUILD",
)

# overriding tensorflow serving bazel dependency
# alternative would be to use cmake build of grpc and flag
# to use system ssl instead
new_local_repository(
    name = "boringssl",
    path = "/usr/",
    build_file = "@//third_party/boringssl:BUILD",
)

# `@curl` is used by google_cloud_cpp (via repo_mapping) and OVMS itself.
new_local_repository(
    name = "curl",
    path = "/usr/",
    build_file_content = """
cc_library(
    name = "curl",
    hdrs = glob(["include/x86_64/curl/*"]),
    srcs = glob(["lib/x86_64-linux-gnu/libcurl.so"]),
    copts = ["-lcrypto", "-lssl"],
    visibility = ["//visibility:public"],
)
""",
)

new_local_repository(
    name = "windows_curl",
    path = "C:\\opt\\curl-8.21.0_7-win64-mingw",
    build_file_content = """
cc_import(
    name = "curl_lib",
    hdrs = [],
    interface_library = "bin/libcurl-x64.lib",
    shared_library = "bin/libcurl-x64.dll",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "curl",
    hdrs = glob(["include/curl/curl.h"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
    deps = [":curl_lib"],
)
""",
)


########################################################### Mediapipe
http_archive(
    name = "com_google_protobuf",
    sha256 = "597071a340acc5346494c119ba3a541825c3f81071fc783521b24e29a485d60f",
    strip_prefix = "protobuf-6.31.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.tar.gz"],
    patches = [
        "@ovms//third_party/protobuf:windows_zlib_dependency.patch",
        "@ovms//third_party/protobuf:windows_msvc_untyped_message_hash_fix.patch",
    ],
    patch_args = ["-p1"],
    repo_mapping = {"@abseil-cpp": "@com_google_absl"},
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

load("@rules_java//java:rules_java_deps.bzl", "rules_java_dependencies")
rules_java_dependencies()
load("@rules_java//java:repositories.bzl", "rules_java_toolchains")
rules_java_toolchains()

################################### Upstream mediapipe repository ###############
#### Will be used on feature release
git_repository(
    name = "mediapipe",
    remote = "https://github.com/google-ai-edge/mediapipe",
    commit = "2bce9dd15fa45f267c9e5f77086997c984a9f107", # top of mediapipe master branch as of 17.09.2026
    patches = [
        "@ovms//third_party/mediapipe:ovms_strip.diff",
        "@ovms//third_party/mediapipe:ovms_no_litert.diff",
        "@ovms//third_party/mediapipe:ovms_calculator_graph_visibility.diff",
        "@ovms//third_party/mediapipe:ovms_constinit_definition.diff",
        "@ovms//third_party/mediapipe:ovms_msvc_api3_calculator_context.diff",
    ],
    patch_args = ["-p1"],
    patch_tool = "patch",
)

# DEV mediapipe 1 source - adjust local repository path for build
#local_repository(
#    name = "mediapipe",
#    path = "C:\\git\\mediapipe",
#)

# Protobuf for Node dependencies
http_archive(
    name = "rules_proto_grpc",
    sha256 = "bbe4db93499f5c9414926e46f9e35016999a4e9f6e3522482d3760dc61011070",
    strip_prefix = "rules_proto_grpc-4.2.0",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.2.0.tar.gz"],
)

# Node dependencies
http_archive(
    name = "build_bazel_rules_nodejs",
    sha256 = "d3476f6dc146766d3f35aa8aa39e65df6467ed7dc8dfdcabf9b7cad464516cd5",
    strip_prefix = "rules_nodejs-5.7.1",
    type = "tar.gz",
    urls = ["https://codeload.github.com/bazelbuild/rules_nodejs/tar.gz/refs/tags/5.7.1"],
)

load("@build_bazel_rules_nodejs//:repositories.bzl", "build_bazel_rules_nodejs_dependencies")
build_bazel_rules_nodejs_dependencies()

# fetches nodejs, npm, and yarn
load("@build_bazel_rules_nodejs//:index.bzl", "node_repositories", "yarn_install")
node_repositories()
yarn_install(
    name = "npm",
    package_json = "//:package.json",
    yarn_lock = "//:yarn.lock",
)

http_archive(
    name = "com_google_protobuf_javascript",
    sha256 = "35bca1729532b0a77280bf28ab5937438e3dcccd6b31a282d9ae84c896b6f6e3",
    strip_prefix = "protobuf-javascript-3.21.2",
    urls = ["https://github.com/protocolbuffers/protobuf-javascript/archive/refs/tags/v3.21.2.tar.gz"],
)

http_archive( # Using commit past 0.9.0 that adds cmake 3.26.2 for model api. Be sure to update to 0.10.0 when available.
    name = "rules_foreign_cc",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/1fb8a1ed9100236590c2a9441d79811aefaabcf8.tar.gz",
    sha256 = "9d55e5b295dc9de77dfac8c1c2f365894363f2acc54cccdde41f7555a9b28dc6",
    strip_prefix = "rules_foreign_cc-1fb8a1ed9100236590c2a9441d79811aefaabcf8",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies(cmake_version="3.26.2")

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.zip",
)

# 2020-08-21 SHA 3a0d4d22c5ae0b9a2216988411cfa6bf860cc372
load("@ovms//third_party/glog:glog.bzl", "glog_dependencies")
glog_dependencies()

http_archive(
    name = "com_github_glog_glog_no_gflags",
    strip_prefix = "glog-3a0d4d22c5ae0b9a2216988411cfa6bf860cc372",
    sha256 = "170d08f80210b82d95563f4723a15095eff1aad1863000e8eeb569c96a98fefb",
    urls = [
        "https://github.com/google/glog/archive/3a0d4d22c5ae0b9a2216988411cfa6bf860cc372.zip",
    ],
    build_file = "@mediapipe//third_party:glog_no_gflags.BUILD",
)

load("@mediapipe//third_party:external_files.bzl", "external_files")
external_files()

new_local_repository(
    name = "linux_openvino",
    build_file = "@//third_party/openvino:BUILD",
    path = "/opt/intel/openvino/runtime",
)

new_local_repository(
    name = "windows_openvino",
    build_file = "@//third_party/openvino:openvino_windows.BUILD",
    path = "C:\\opt\\openvino\\runtime",
)

new_local_repository(
    name = "windows_genai",
    build_file = "@//third_party/genai:genai_windows.BUILD",
    path = "C:\\opt\\openvino\\runtime",
)

new_local_repository(
    name = "linux_genai",
    build_file = "@//third_party/genai:BUILD",
    path = "/opt/intel/openvino/runtime",
)

new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party/opencv:BUILD",
    path = "/opt/opencv/",
)

new_local_repository(
    name = "windows_opencv",
    build_file = "@//third_party/opencv:opencv_windows.BUILD",
    path = "C:\\opt\\opencv_4.14.0",
)

new_local_repository(
    name = "windows_opencl",
    build_file = "@//third_party/opencl:opencl_windows.BUILD",
    path = "C:\\opt\\opencl\\external\\OpenCL-CLHPP",
)

new_local_repository(
    name = "windows_opencl2",
    build_file = "@//third_party/opencl:opencl_windows2.BUILD",
    path = "C:\\opt\\opencl\\external\\OpenCL-Headers",
)

########################################################### Mediapipe end

########################################################### Python support start

http_archive(
    name = "aspect_bazel_lib",
    sha256 = "7b39d9f38b82260a8151b18dd4a6219d2d7fc4a0ac313d4f5a630ae6907d205d",
    strip_prefix = "bazel-lib-2.10.0",
    url = "https://github.com/bazel-contrib/bazel-lib/releases/download/v2.10.0/bazel-lib-v2.10.0.tar.gz",
)

load("@aspect_bazel_lib//lib:repositories.bzl", "register_coreutils_toolchains")
register_coreutils_toolchains()

load("@ovms//third_party/python:python_repo.bzl", "python_repository")
python_repository(name = "_python3-linux")

load("@ovms//third_party/python:python_repo_win.bzl", "python_repository")
python_repository(name = "_python3-windows")

new_local_repository(
    name = "python3_linux",
    path = "/usr",
    build_file = "@_python3-linux//:BUILD"
)

new_local_repository(
    name = "python3_windows",
    path = "C:\\opt\\",
    build_file = "@_python3-windows//:BUILD"
)

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-b162c7c88a253e3f6b673df0c621aca27596ce6b",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/b162c7c88a253e3f6b673df0c621aca27596ce6b.zip"],
)
# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.11.1",
  urls = ["https://github.com/pybind/pybind11/archive/v2.11.1.tar.gz"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
bind(
    name = "python_headers",
    actual = "@local_config_python//:python_headers",
)

load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pip_deps",
    requirements_lock = "//src/python/binding:tests/requirements.txt",
)

load("@pip_deps//:requirements.bzl", "install_deps")

install_deps()

########################################################### Python support end

# minitrace
http_archive(
    name = "minitrace",
    url = "https://github.com/hrydgard/minitrace/archive/020f42b189e8d6ad50e4d8f45d69edee0a6b3f23.tar.gz",
    sha256 = "314bcab4dc069c61f2eb813c9bd649efb2bd1fd75f91205a269a4c5abcd3d66b",
    strip_prefix = "minitrace-020f42b189e8d6ad50e4d8f45d69edee0a6b3f23",
    build_file_content = """
cc_library(
    name = "trace",
    hdrs = ["minitrace.h"],
    srcs = ["minitrace.c"],
    visibility = ["//visibility:public"],
    local_defines = [
    ],
)
""",
)

# grpc must be defined before grpc_deps() below so this version is picked up.
http_archive( # 1.74.1
    name = "com_github_grpc_grpc",
    urls = [
        "https://github.com/grpc/grpc/archive/893bdadd56dbb75fb156175afdaa2b0d47e1c15b.tar.gz",
    ],
    strip_prefix = "grpc-893bdadd56dbb75fb156175afdaa2b0d47e1c15b",
    patches = ["@ovms//third_party/grpc:grpc_missing_algorithm_include.patch"],
    patch_args = ["-p1"],
    repo_mapping = {"@abseil-cpp": "@com_google_absl"},
)

# rules_pkg was previously provided transitively via TensorFlow's workspace macros.
http_archive(
    name = "rules_pkg",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
    ],
    sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
)

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

load("@ovms//third_party/aws-sdk-cpp:aws-sdk-cpp.bzl", "aws_sdk_cpp")
aws_sdk_cpp()

### Libgit2
load("@ovms//third_party/libgit2:libgit2_engine.bzl", "libgit2_engine")
libgit2_engine()

load("@ovms//third_party/drogon:drogon.bzl", "drogon_cpp")
drogon_cpp()

### espeak-ng (built from source via Bazel; gated by --//:espeak flag)
load("@ovms//third_party/espeak_ng:espeak_ng.bzl", "espeak_ng")
espeak_ng()

# Azure SDK for C++ (azure-core, azure-storage-blobs, azure-storage-files-shares)
new_local_repository(
    name = "azure",
    build_file = "@//third_party/azure:BUILD",
    path = "/azure-sdk-install",
)

# Google Cloud SDK
http_archive(
    name = "google_cloud_cpp",
    sha256 = "629cbfcc5bd581d38277ba8fa94a5b6591af1e0f6af0dab6d1d9ed796bf48b61",
    strip_prefix = "google-cloud-cpp-2.39.0",
    url = "https://github.com/googleapis/google-cloud-cpp/archive/v2.39.0.tar.gz",
    repo_mapping = {"@com_github_curl_curl" : "@curl"}
)

load("@google_cloud_cpp//bazel:workspace0.bzl", "gl_cpp_workspace0")

gl_cpp_workspace0()

load("@google_cloud_cpp//bazel:workspace1.bzl", "gl_cpp_workspace1")

gl_cpp_workspace1()

load("@google_cloud_cpp//bazel:workspace2.bzl", "gl_cpp_workspace2")

gl_cpp_workspace2()

load("@google_cloud_cpp//bazel:workspace4.bzl", "gl_cpp_workspace4")

gl_cpp_workspace4()

load("@google_cloud_cpp//bazel:workspace5.bzl", "gl_cpp_workspace5")

gl_cpp_workspace5()

# grpc

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# cxxopts
http_archive(
    name = "com_github_jarro2783_cxxopts",
    url = "https://github.com/jarro2783/cxxopts/archive/v3.1.1.zip",
    sha256 = "25b644a2bfa9c6704d723be51b026bc02420dfdee1277a49bfe5df3f19b0eaa4",
    strip_prefix = "cxxopts-3.1.1",
    build_file = "@//third_party/cxxopts:BUILD",
)

# spdlog
http_archive(
    name = "com_github_gabime_spdlog",
    urls = [ 
        "https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz",
        "https://mirror.bazel.build/github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz",
    ],
    sha256 = "15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67",
    strip_prefix = "spdlog-1.15.3",
    build_file = "@//third_party/spdlog:BUILD"
)

# fmtlib
http_archive(
    name = "fmtlib",
    url = "https://github.com/fmtlib/fmt/archive/refs/tags/11.2.0.tar.gz",
    sha256 = "bc23066d87ab3168f27cef3e97d545fa63314f5c79df5ea444d41d56f962c6af",
    strip_prefix = "fmt-11.2.0",
    build_file = "@//third_party/fmtlib:BUILD"
)

# prometheus-cpp
http_archive(
    name = "com_github_jupp0r_prometheus_cpp",
    strip_prefix = "prometheus-cpp-1.0.1",
    urls = ["https://github.com/jupp0r/prometheus-cpp/archive/refs/tags/v1.0.1.zip"],
)
load("@com_github_jupp0r_prometheus_cpp//bazel:repositories.bzl", "prometheus_cpp_repositories")
prometheus_cpp_repositories()

load("@rules_foreign_cc//foreign_cc:cmake.bzl", "cmake")
load("@ovms//third_party/model_api:model_api.bzl", "workspace_model_api")
workspace_model_api()

new_local_repository(
    name = "mediapipe_calculators",
    build_file = "@//third_party/mediapipe_calculators:BUILD",
    path = "third_party/mediapipe_calculators",
)

# Eigen — referenced by mediapipe upstream calculators (e.g. matrix_to_vector).
http_archive(
    name = "eigen",
    build_file = "@mediapipe//third_party:eigen.BUILD",
    sha256 = "35c6126e246585d9cf6600b65471582c2701aae64b784a6fd19168a90cfc841e",
    strip_prefix = "eigen-ea13a98decd497a8c5588fb5de71b57bcf10d864",
    urls = ["https://gitlab.com/libeigen/eigen/-/archive/ea13a98decd497a8c5588fb5de71b57bcf10d864/eigen-ea13a98decd497a8c5588fb5de71b57bcf10d864.tar.gz"],
)

http_archive(
    name = "nlohmann_json",
    sha256 = "0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406",
    strip_prefix = "json-3.11.3",
    type = "tar.gz",
    url = "https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3",
)
# for rest client in unit tests (server_test.cpp)
http_archive(
    name = "cpp_httplib",
    url = "https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.7.tar.gz",
    sha256 = "b7b1e9e4e77565a5a9bc95e761d5df3e7c0e8ca37c90fd78b1b031bc6cb90fc1",
    strip_prefix = "cpp-httplib-0.18.7",
    build_file = "@//third_party/cpp-httplib:BUILD"
)

http_archive(
    name = "stb",
    url = "https://github.com/nothings/stb/archive/5c205738c191bcb0abc65c4febfa9bd25ff35234.tar.gz",
    sha256 = "cfeab9f800961882d6d22ddf36e965523b33002f4f937de08321304c9ba72af3",
    strip_prefix = "stb-5c205738c191bcb0abc65c4febfa9bd25ff35234",
    build_file_content = """
cc_library(
    name = "image",
    hdrs = ["stb_image.h", "stb_image_write.h"],
    visibility = ["//visibility:public"],
    local_defines = [
    ],
)
""",
)

http_archive(
    name = "dr_libs",
    url = "https://github.com/mackron/dr_libs/archive/47a4f08e777faddf59a8955c4ea84f69f41020d5.tar.gz",
    sha256 = "9f451ba81a65ffa06c14113b13cf257ac70a9193a37782f9a934e5240c371772",
    strip_prefix = "dr_libs-47a4f08e777faddf59a8955c4ea84f69f41020d5",
    build_file_content = """
cc_library(
    name = "dr",
    hdrs = ["dr_mp3.h", "dr_wav.h"],
    visibility = ["//visibility:public"],
    local_defines = [
    ],
)
""",
)

http_archive(
    name = "winreg",
    url = "https://github.com/GiovanniDicanio/WinReg/archive/4e1fab61959ca7a43c2627251ba306ebbbec7f7a.tar.gz",
    sha256 = "9066cc3eaf2319c23ec49f412b9625f50121608d709ef214ec86a2fd6c0c570e",
    strip_prefix = "WinReg-4e1fab61959ca7a43c2627251ba306ebbbec7f7a", # master Aug 22 2025
    build_file_content = """
cc_library(
    name = "winreg",
    hdrs = glob(["WinReg/WinReg.hpp"]),
    visibility = ["//visibility:public"],
    local_defines = [],
)
""",
)


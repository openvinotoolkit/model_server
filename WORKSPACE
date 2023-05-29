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
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# overriding tensorflow serving bazel dependency
# alternative would be to use cmake build of grpc and flag
# to use system ssl instead
new_local_repository(
    name = "boringssl",
    path = "/usr/",
    build_file_content = """
cc_library(
    name = "ssl",
    hdrs = glob(["include/openssl/*"]),
    srcs = glob(["lib/x86_64-linux-gnu/libssl.so"]),
    copts = ["-lcrypto", "-lssl"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "crypto",
    hdrs = glob(["include/openssl/*"]),
    srcs = glob(["lib/x86_64-linux-gnu/libssl.so"]),
    copts = ["-lcrypto", "-lssl"],
    visibility = ["//visibility:public"],
)
""",
)
# overriding GCS curl dependency to force using system provided openssl
new_local_repository(
    name = "libcurl",
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
# Tensorflow serving
git_repository(
    name = "tensorflow_serving",
    remote = "https://github.com/tensorflow/serving.git",
    tag = "2.6.5",
    patch_args = ["-p1"],
    patches = ["net_http.patch", "listen.patch"]
    #                             ^^^^^^^^^^^^
    #                       make bind address configurable
    #          ^^^^^^^^^^^^
    #        allow all http methods
)

########################################################### Mediapipe
http_archive(
    name = "com_google_protobuf",
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.19.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.1.tar.gz"],
    #patches = [
    #    "@//third_party:com_google_protobuf_fixes.diff"
    #],
    #patch_args = [
    #    "-p1",
    #],
)

################################### Official mediapipe repository #########
#### Will be used on feature release
git_repository(
    name = "mediapipe",
    remote = "https://github.com/google/mediapipe",
    tag = "v0.9.1",
)

# DEV mediapipe 1 source - adjust local repository path for build
#local_repository(
#    name = "mediapipe",
#    path = "/mediapipe/",
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
    sha256 = "5aae76dced38f784b58d9776e4ab12278bc156a9ed2b1d9fcd3e39921dc88fda",
    urls = ["https://github.com/bazelbuild/rules_nodejs/releases/download/5.7.1/rules_nodejs-5.7.1.tar.gz"],
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

http_archive(
   name = "rules_foreign_cc",
   strip_prefix = "rules_foreign_cc-0.1.0",
   url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.1.0.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.zip",
)

git_repository(
    name = "com_github_glog_glog",
    remote = "https://github.com/google/glog",
    tag = "v0.5.0",
)

load("@mediapipe//third_party:external_files.bzl", "external_files")
external_files()

new_local_repository(
    name = "linux_openvino",
    build_file = "@//third_party/openvino:BUILD",
    path = "/opt/intel/openvino/runtime",
)

new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party/opencv:BUILD",
    path = "/opt/opencv/",
)

########################################################### Mediapipe end

# minitrace
new_git_repository(
    name = "minitrace",
    remote = "https://github.com/hrydgard/minitrace.git",
    commit = "020f42b189e8d6ad50e4d8f45d69edee0a6b3f23",
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

load("@tensorflow_serving//tensorflow_serving:repo.bzl", "tensorflow_http_archive")
tensorflow_http_archive(
    name = "org_tensorflow",
    sha256 = "fd687f8e26833cb917ae0bd8e434c9bd30c92042361c8ae69679983d3c66a440",
    git_commit = "15198b1818bd2bf1b5b55bf5b02bf42398d222fc",
    patch = "tf.patch",
    repo_mapping = {"@curl" : "@curl"}
)

load("@tensorflow_serving//tensorflow_serving:workspace.bzl", "tf_serving_workspace")
tf_serving_workspace()

# Check bazel version requirement, which is stricter than TensorFlow's.
load(
    "@org_tensorflow//tensorflow:version_check.bzl",
    "check_bazel_version_at_least"
)
check_bazel_version_at_least("5.3.1")

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")
workspace()

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

# AWS S3 SDK
new_local_repository(
    name = "awssdk",
    build_file = "@//third_party/aws:BUILD",
    path = "/awssdk",
)

# Azure Storage SDK
new_local_repository(
    name = "azure",
    build_file = "@//third_party/azure:BUILD",
    path = "/azure/azure-storage-cpp",
)

# Azure Storage SDK dependency - cpprest
new_local_repository(
    name = "cpprest",
    build_file = "@//third_party/cpprest:BUILD",
    path = "/azure/cpprestsdk",
)

# Boost (needed for Azure Storage SDK)

new_local_repository(
    name = "boost",
    path = "/usr/local/lib/",
    build_file = "@//third_party/boost:BUILD"
)

# Google Cloud SDK
http_archive(
    name = "com_github_googleapis_google_cloud_cpp",
    sha256 = "a370bcf2913717c674a7250c4a310250448ffeb751b930be559a6f1887155f3b",
    strip_prefix = "google-cloud-cpp-0.21.0",
    url = "https://github.com/googleapis/google-cloud-cpp/archive/v0.21.0.tar.gz",
    repo_mapping = {"@com_github_curl_curl" : "@curl"}
)

load("@com_github_googleapis_google_cloud_cpp//bazel:google_cloud_cpp_deps.bzl", "google_cloud_cpp_deps")
google_cloud_cpp_deps()

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")
switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,  # C++ support is only "Partially implemented", roll our own.
    grpc = True,
)

load("@com_github_googleapis_google_cloud_cpp_common//bazel:google_cloud_cpp_common_deps.bzl", "google_cloud_cpp_common_deps")
google_cloud_cpp_common_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

# cxxopts
http_archive(
    name = "com_github_jarro2783_cxxopts",
    url = "https://github.com/jarro2783/cxxopts/archive/v2.2.0.zip",
    sha256 = "f9640c00d9938bedb291a21f9287902a3a8cee38db6910b905f8eba4a6416204",
    strip_prefix = "cxxopts-2.2.0",
    build_file = "@//third_party/cxxopts:BUILD",
)

# RapidJSON
http_archive(
    name = "com_github_tencent_rapidjson",
    url = "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
    sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
    strip_prefix = "rapidjson-1.1.0",
    build_file = "@//third_party/rapidjson:BUILD"
)

# spdlog
http_archive(
    name = "com_github_gabime_spdlog",
    url = "https://github.com/gabime/spdlog/archive/v1.4.0.tar.gz",
    sha256 = "afd18f62d1bc466c60bef088e6b637b0284be88c515cedc59ad4554150af6043",
    strip_prefix = "spdlog-1.4.0",
    build_file = "@//third_party/spdlog:BUILD"
)

# fmtlib
http_archive(
    name = "fmtlib",
    url = "https://github.com/fmtlib/fmt/archive/6.0.0.tar.gz",
    sha256 = "f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78",
    strip_prefix = "fmt-6.0.0",
    build_file = "@//third_party/fmtlib:BUILD"
)

# libevent
http_archive(
    name = "com_github_libevent_libevent",
    url = "https://github.com/libevent/libevent/archive/release-2.1.8-stable.zip",
    sha256 = "70158101eab7ed44fd9cc34e7f247b3cae91a8e4490745d9d6eb7edc184e4d96",
    strip_prefix = "libevent-release-2.1.8-stable",
    build_file = "@//third_party/libevent:BUILD",
)

# prometheus-cpp
http_archive(
    name = "com_github_jupp0r_prometheus_cpp",
    strip_prefix = "prometheus-cpp-1.0.1",
    urls = ["https://github.com/jupp0r/prometheus-cpp/archive/refs/tags/v1.0.1.zip"],
)
load("@com_github_jupp0r_prometheus_cpp//bazel:repositories.bzl", "prometheus_cpp_repositories")
prometheus_cpp_repositories()

##################### OPEN VINO ######################
# OPENVINO DEFINITION FOR BUILDING FROM BINARY RELEASE: ##########################
new_local_repository(
    name = "openvino",
    build_file = "@//third_party/openvino:BUILD",
    path = "/opt/intel/openvino/runtime",
)
################## END OF OPENVINO DEPENDENCY ##########

##################### OPEN CV ######################
# OPENCV DEFINITION FOR ARTIFACTS BUILT FROM SOURCE: ##########################
new_local_repository(
    name = "opencv",
    build_file = "@//third_party/opencv:BUILD",
    path = "/opt/opencv",
)
################## END OF OPENCV DEPENDENCY ##########

new_git_repository(
    name = "model_api",
    remote = "https:///github.com/openvinotoolkit/model_api/",
    build_file_content = """
cc_library(
    name = "adapter_api",
    hdrs = ["model_api/cpp/adapters/include/adapters/inference_adapter.h",],
    includes = ["model_api/cpp/adapters/include"],
    deps = ["@linux_openvino//:openvino"],
    visibility = ["//visibility:public"],
)
    """,
    commit = "7e163416c60ba9ccdf440c6c049d6c7e7137e144"
)

git_repository(
    name = "oneTBB",
    branch = "v2021.8.0",
    remote = "https://github.com/oneapi-src/oneTBB/",
    patch_args = ["-p1"],
    patches = ["mwaitpkg.patch",]
)

new_local_repository(
    name = "mediapipe_calculators",
    build_file = "@//third_party/mediapipe_calculators:BUILD",
    path = "/ovms/third_party/mediapipe_calculators",
)

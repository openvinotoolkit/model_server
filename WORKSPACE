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

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.zip",
)

http_archive(
    name = "com_github_glog_glog",
    strip_prefix = "glog-0a2e5931bd5ff22fd3bf8999eb8ce776f159cda6",
    sha256 = "58c9b3b6aaa4dd8b836c0fd8f65d0f941441fb95e27212c5eeb9979cfd3592ab",
    urls = [
        "https://github.com/google/glog/archive/0a2e5931bd5ff22fd3bf8999eb8ce776f159cda6.zip",
    ],
)

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

# Tensorflow serving branch master / Fri Feb 10 09:56:11 2023
git_repository(
    name = "tensorflow_serving",
    remote = "https://github.com/tensorflow/serving.git",
    branch = "master",
    #patch_args = ["-p1"],
    #patches = ["net_http.patch", "listen.patch"]
    #                             ^^^^^^^^^^^^
    #                       make bind address configurable
    #          ^^^^^^^^^^^^
    #        allow all http methods
)

load("@tensorflow_serving//tensorflow_serving:repo.bzl", "tensorflow_http_archive")
git_repository(
    name = "org_tensorflow",
    remote = "https://github.com/tensorflow/tensorflow.git",
    commit = "84f40925e929d05e72ab9234e53c729224e3af38",
    #patch_args = ["-p1"],
    #patches = ["tf.patch"],
    repo_mapping = {"@curl" : "@curl"}
)


load("@tensorflow_serving//tensorflow_serving:workspace.bzl", "tf_serving_workspace")
tf_serving_workspace()

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")
workspace()

########################################################### Mediapipe

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

http_archive(
    name = "com_github_jupp0r_prometheus_cpp",
    strip_prefix = "prometheus-cpp-master",
    urls = ["https://github.com/jupp0r/prometheus-cpp/archive/master.zip"],
)

load("@com_github_jupp0r_prometheus_cpp//bazel:repositories.bzl", "prometheus_cpp_repositories")
prometheus_cpp_repositories()

git_repository(
    name = "google_mediapipe",
    remote = "https://github.com/google/mediapipe",
    tag = "v0.9.1",
)

# Protobuf for Node dependencies
http_archive(
    name = "rules_proto_grpc",
    sha256 = "bbe4db93499f5c9414926e46f9e35016999a4e9f6e3522482d3760dc61011070",
    strip_prefix = "rules_proto_grpc-4.2.0",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.2.0.tar.gz"],
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")
rules_proto_grpc_toolchains()
rules_proto_grpc_repos()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

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
    name = "com_google_protobuf",
    sha256 = "87407cd28e7a9c95d9f61a098a53cf031109d451a7763e7dd1253abf8b4df422",
    strip_prefix = "protobuf-3.21.9",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.21.9.tar.gz"],
    patches = [
        "@//third_party:com_google_protobuf_fixes.diff"
    ],
    patch_args = [
        "-p1",
    ],
)



http_archive(
   name = "rules_foreign_cc",
   strip_prefix = "rules_foreign_cc-0.1.0",
   url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.1.0.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# ABSL cpp library lts_2021_03_24, patch 2.
http_archive(
    name = "com_google_absl",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz",
    ],
    # Remove after https://github.com/abseil/abseil-cpp/issues/326 is solved.
    #patches = [
    #    "@//third_party:com_google_absl_f863b622fe13612433fdf43f76547d5edda0c93001.diff"
    #],
    #patch_args = [
    #    "-p1",
    #],
    strip_prefix = "abseil-cpp-20220623.1",
    sha256 = "91ac87d30cc6d79f9ab974c51874a704de9c2647c40f6932597329a282217ba8"
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

http_archive(
	name = "bazel_toolchains",
	urls = ["https://github.com/bazelbuild/bazel-toolchains/archive/dac71231098d891e5c4b74a2078fe9343feef510.tar.gz"],
	strip_prefix = "bazel-toolchains-dac71231098d891e5c4b74a2078fe9343feef510",
	sha256 = "56d5370eb99559b4c74f334f81bc8a298f728bd16d5a4333c865c2ad10fae3bc",
)

load("@bazel_toolchains//repositories:repositories.bzl", bazel_toolchains_repositories = "repositories")
bazel_toolchains_repositories()

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "56d8c5a5c91e1af73eca71a6fab2ced959b67c86d12ba37feedb0a2dfea441a6",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.37.0/rules_go-v0.37.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.37.0/rules_go-v0.37.0.zip",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "ecba0f04f96b4960a5b250c8e8eeec42281035970aa8852dda73098274d14a1d",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.29.0/bazel-gazelle-v0.29.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.29.0/bazel-gazelle-v0.29.0.tar.gz",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies", "go_repository")

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

#load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
#grpc_extra_deps()

# cxxopts
http_archive(
    name = "cxxopts",
    url = "https://github.com/jarro2783/cxxopts/archive/v2.2.0.zip",
    sha256 = "f9640c00d9938bedb291a21f9287902a3a8cee38db6910b905f8eba4a6416204",
    strip_prefix = "cxxopts-2.2.0",
    build_file = "@//third_party/cxxopts:BUILD",
)

# RapidJSON
http_archive(
    name = "rapidjson",
    url = "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
    sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
    strip_prefix = "rapidjson-1.1.0",
    build_file = "@//third_party/rapidjson:BUILD"
)

# spdlog
http_archive(
    name = "spdlog",
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

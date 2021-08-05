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

# overriding tensorflow serving bazel dependency
# alternative would be to use cmake build of grpc and flag
# to use system ssl instead
new_local_repository(
    name = "boringssl",
    path = "/usr/",
    build_file_content = """
#constraint_setting(name = "linux_distribution_family")
#constraint_value(constraint_setting = "linux_distribution_family", name = "fedora") # like RHEL/CentOS
#constraint_value(constraint_setting = "linux_distribution_family", name = "debian") # like Ubuntu
cc_library(
    name = "ssl",
    hdrs = glob(["include/openssl/*"]),
    srcs = select({
        ":debian": glob(["lib64/libssl.so"]),
        ":fedora": glob(["lib/x86_64-linux-gnu/libssl.so"])
        }),
    copts = ["-lcrypto", "-lssl"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "crypto",
    hdrs = glob(["include/openssl/*"]),
    srcs = select({
        ":debian": glob(["lib64/libssl.so"]),
        ":fedora": glob(["lib/x86_64-linux-gnu/libssl.so"])
        }),
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
    srcs = select(
        ":debian": glob(["lib64/libcurl.so"]),
        ":fedora": glob(["lib/x86_64-linux-gnu/libcurl.so"])
        }),
    copts = ["-lcrypto", "-lssl"],
    visibility = ["//visibility:public"],
)
""",
)
# Tensorflow serving
git_repository(
    name = "tensorflow_serving",
    remote = "https://github.com/tensorflow/serving.git",
    tag = "2.5.1",
    patch_args = ["-p1"],
    patches = ["net_http.patch", "listen.patch"]
    #                             ^^^^^^^^^^^^
    #                       make bind address configurable
)

load("@tensorflow_serving//tensorflow_serving:repo.bzl", "tensorflow_http_archive")
tensorflow_http_archive(
    name = "org_tensorflow",
    sha256 = "cb99f136dc5c89143669888a44bfdd134c086e1e2d9e36278c1eb0f03fe62d76",
    git_commit = "a4dfb8d1a71385bd6d122e4f27f86dcebb96712d",
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
check_bazel_version_at_least("3.7.2")

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
    path = "/opt/intel/openvino/deployment_tools",
)
################## END OF OPENVINO DEPENDENCY ##########

##################### OPEN CV ######################
# OPENCV DEFINITION FOR BUILDING FROM BINARY RELEASE: ##########################
new_local_repository(
    name = "opencv",
    build_file = "@//third_party/opencv:BUILD",
    path = "/opt/intel/openvino/opencv",
)
################## END OF OPENCV DEPENDENCY ##########

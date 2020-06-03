workspace(name = "ovms")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Tensorflow serving
git_repository(
    name = "tensorflow_serving",
    remote = "https://github.com/tensorflow/serving.git",
    tag = "2.2.0-rc2"
)

# Tensorflow core
git_repository(
    name = "org_tensorflow",
    remote = "https://github.com/tensorflow/tensorflow.git",
    tag = "v2.2.0-rc2"
)

http_archive(
    name = "rules_pkg",
    sha256 = "5bdc04987af79bd27bc5b00fe30f59a858f77ffa0bd2d8143d5b31ad8b1bd71c",
    url = "https://github.com/bazelbuild/rules_pkg/releases/download/0.2.0/rules_pkg-0.2.0.tar.gz",
)

load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

load(
    "@org_tensorflow//third_party/toolchains/preconfig/generate:archives.bzl",
    "bazel_toolchains_archive",
)
bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)
bazel_toolchains_repositories()


# START: Upstream TensorFlow dependencies
# TensorFlow build depends on these dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)
http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)  # https://github.com/bazelbuild/bazel-skylib/releases


# END: Upstream TensorFlow dependencies

load("@tensorflow_serving//tensorflow_serving:workspace.bzl", "tf_serving_workspace")
tf_serving_workspace()

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("2.0.0")

# GPRC deps, required to match TF's.  Only after calling tf_serving_workspace()
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")

bazel_version_repository(name = "bazel_version")

# OpenVINO
new_local_repository(
    name = "openvino",
    build_file_content = """
cc_library(
    name = "openvino",
    srcs = ["bin/intel64/Release/lib/libinference_engine_legacy.so", "bin/intel64/Release/lib/libinference_engine.so", "bin/intel64/Release/lib/libinference_engine_c_api.so"],
    hdrs = glob([
        "inference-engine/include/**/*.h",
        "inference-engine/include/**/*.hpp"
    ]),
    strip_include_prefix = "inference-engine/include",
    visibility = ["//visibility:public"],
)
""",
    path = "/openvino",
)

# AWS S3 SDK
new_local_repository(
    name = "awssdk",
    build_file_content = """
cc_library(
    name = "core",
    srcs = ["build/aws-cpp-sdk-core/libaws-cpp-sdk-core.a"],
    hdrs = glob([
        "aws-cpp-sdk-core/include/**/*.h",
    ]),
    strip_include_prefix = "aws-cpp-sdk-core/include",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "s3",
    srcs = ["build/aws-cpp-sdk-s3/libaws-cpp-sdk-s3.a"],
    hdrs = glob([
        "aws-cpp-sdk-s3/include/**/*.h",
    ]),
    strip_include_prefix = "aws-cpp-sdk-s3/include",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "deps",
    srcs = [
        "build/.deps/install/lib/libaws-c-event-stream.a",
        "build/.deps/install/lib/libaws-checksums.a",
        "build/.deps/install/lib/libaws-c-common.a"
    ],
    hdrs = glob([
        "build/.deps/install/include/**/*.*",
    ]),
    strip_include_prefix = "build/.deps/install/include",
    visibility = ["//visibility:public"],
)
""",
    path = "/awssdk",
)

# Google Cloud SDK
http_archive(
    name = "com_github_googleapis_google_cloud_cpp",
    sha256 = "a370bcf2913717c674a7250c4a310250448ffeb751b930be559a6f1887155f3b",
    strip_prefix = "google-cloud-cpp-0.21.0",
    url = "https://github.com/googleapis/google-cloud-cpp/archive/v0.21.0.tar.gz",
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
    name="cxxopts",
    url="https://github.com/jarro2783/cxxopts/archive/v2.2.0.zip",
    sha256="f9640c00d9938bedb291a21f9287902a3a8cee38db6910b905f8eba4a6416204",
    strip_prefix = "cxxopts-2.2.0",
    build_file_content= """
cc_library(
    name = "cxxopts",
    srcs = [],
    hdrs = [
        "include/cxxopts.hpp",
    ],
    copts = [
        "-Iexternal/cxxopts/",
        "--std=c++14"
    ],
    includes = [
        "include/",
    ],
    visibility = ["//visibility:public"],
)    """
)

# RapidJSON
http_archive(
    name = "rapidjson",
    urls = [
        "https://github.com/Tencent/rapidjson/archive/v1.1.0.zip",
    ],
    sha256 = "8e00c38829d6785a2dfb951bb87c6974fa07dfe488aa5b25deec4b8bc0f6a3ab",
    strip_prefix = "rapidjson-1.1.0",
    build_file_content="""
package(
    default_visibility = ["//visibility:public"],
)
cc_library(
    name = "rapidjson",
    hdrs = glob(["include/rapidjson/**/*.h"]),
    includes = ["include"],
) """
)

http_archive(
    name="spdlog",
    url="https://github.com/gabime/spdlog/archive/v1.4.0.tar.gz",
    sha256="afd18f62d1bc466c60bef088e6b637b0284be88c515cedc59ad4554150af6043",
    strip_prefix = "spdlog-1.4.0",
    build_file_content= """

cc_library(
    name = "spdlog",
    srcs = [],
    hdrs = glob([
        "include/**/*.cc",
        "include/**/*.h",
    ]),
    includes = [
        "include/",
    ],
    copts = ["-Iexternal/spdlog/"],
    visibility = ["//visibility:public"],
    deps = ["@fmtlib"],
    defines = ["SPDLOG_FMT_EXTERNAL"],
)    """
)

http_archive(
    name="fmtlib",
    url="https://github.com/fmtlib/fmt/archive/6.0.0.tar.gz",
    sha256="f1907a58d5e86e6c382e51441d92ad9e23aea63827ba47fd647eacc0d3a16c78",
    strip_prefix = "fmt-6.0.0",
    build_file_content= """

cc_library(
    name = "fmtlib",
    srcs = glob([
        "fmt/*.cc",
    ]),
    hdrs = glob([
        "include/fmt/*.h",
    ]),
    includes = [
        "include/",
    ],
    visibility = ["//visibility:public"],
    defines = ["FMT_HEADER_ONLY"],
)    """
)

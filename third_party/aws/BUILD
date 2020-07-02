package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "core",
    srcs = ["build/aws-cpp-sdk-core/libaws-cpp-sdk-core.a"],
    hdrs = glob([
        "aws-cpp-sdk-core/include/**/*.h",
    ]),
    strip_include_prefix = "aws-cpp-sdk-core/include"
)

cc_library(
    name = "s3",
    srcs = ["build/aws-cpp-sdk-s3/libaws-cpp-sdk-s3.a"],
    hdrs = glob([
        "aws-cpp-sdk-s3/include/**/*.h",
    ]),
    strip_include_prefix = "aws-cpp-sdk-s3/include"
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
    strip_include_prefix = "build/.deps/install/include"
)

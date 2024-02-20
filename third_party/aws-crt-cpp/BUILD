# Description:
#   AWS C Common

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws-crt-cpp",
    srcs = glob([
        "include/aws/common/*.h",
        "include/aws/common/private/*.h",
        "source/*.c",
    ]),
    hdrs = glob([
        "include/aws/common/*.h",
        "include/aws/common/private/*.h",
    ]),
    defines = [],
    includes = [
        "include",
    ],
    textual_hdrs = glob([
        "include/**/*.inl",
    ]),
    deps = [
        "@aws-c-common//:aws-c-common", # TODO check genrule
    ],
)


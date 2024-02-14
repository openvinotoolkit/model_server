# Description:
#   AWS C Event Stream

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

cc_library(
    name = "aws-c-event-stream",
    srcs = glob([
        "include/**/*.h",
        "source/**/*.c",
    ]),
    hdrs = [
    ],
    defines = [],
    includes = [
        "include",
    ],
    deps = [
        "@aws-c-common",
        "@aws-checksums",
    ],
)

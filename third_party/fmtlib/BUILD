package(
    default_visibility = ["//visibility:public"],
)

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
)
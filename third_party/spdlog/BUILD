package(
    default_visibility = ["//visibility:public"],
)

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
    deps = ["@fmtlib"],
    defines = ["SPDLOG_FMT_EXTERNAL"],
)
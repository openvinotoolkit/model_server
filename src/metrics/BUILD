load("//:common_settings.bzl", "ovms_cc_library")

ovms_cc_library(
    name = "libovmsmetrics",
    hdrs = [
        "metric.hpp",
        "metric_config.hpp",
        "metric_family.hpp",
        "metric_module.hpp",
        "metric_registry.hpp",
    ],
    srcs = [
        "metric.cpp",
        "metric_config.cpp",
        "metric_family.cpp",
        "metric_registry.cpp",
        "metric_module.cpp",
    ],
    deps = [
        "@com_github_jupp0r_prometheus_cpp//core",
        "@com_github_tencent_rapidjson//:rapidjson",
        "//src/port:rapidjson_stringbuffer",
        "//src/port:rapidjson_writer",
        "//src:cpp_headers",
        "//src:libovms_module",
        "//src:libovmslogging",
        "//src:libovmsschema",
        "//src:libovmsstatus",
        "//src:libovmsstring_utils",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovms_metric_provider",
    hdrs = ["metric_provider.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
)


# third_party/espeak_ng/BUILD

config_setting(
    name = "is_macos",
    values = {"cpu": "darwin"},
)

cc_library(
    name = "espeak_ng",
    linkopts = [
        "-L/usr/local/lib",
        "-lespeak-ng",
    ],
    includes = [
        "/usr/local/include",
        "/usr/local/include/espeak-ng",
    ],
    visibility = ["//visibility:public"],
)

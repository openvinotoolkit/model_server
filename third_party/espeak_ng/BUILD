
# third_party/espeak_ng/BUILD

config_setting(
    name = "is_macos",
    values = {"cpu": "darwin"},
)

cc_library(
    name = "espeak_ng",
    hdrs = glob(["include/**/*.h"]),
    includes = [
        "include",
        "include/espeak-ng",
    ],
    deps = [":espeak_ng_lib"],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "espeak_ng_lib",
    static_library = "lib/libespeak-ng.a",
    visibility = ["//visibility:private"],
)

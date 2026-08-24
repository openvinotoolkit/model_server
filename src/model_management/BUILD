load("//:common_settings.bzl", "ovms_cc_library")

ovms_cc_library(
    name = "servable_loading_task",
    hdrs = ["servable_loading_task.hpp"],
    deps = select({
        "//conditions:default": [],
        "//:not_disable_mediapipe": [
            "//src/mediapipe_internal:mediapipegraphconfig",
        ],
    }) + [
        "//src:modelconfig",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "servable_loading_queue",
    hdrs = ["servable_loading_queue.hpp"],
    srcs = ["servable_loading_queue.cpp"],
    deps = [
        "servable_loading_task",
        "//src:libovmslogging",
    ],
    visibility = ["//visibility:public"],
)

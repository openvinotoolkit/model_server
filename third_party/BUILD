alias(
    name = "opencv",
    actual = select({
        "//src:windows": "@windows_opencv//:opencv",
        "//conditions:default": "@linux_opencv//:opencv",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "boringssl",
    actual = select({
        "//src:windows": "@windows_boringssl//:ssl",
        "//conditions:default": "@linux_boringssl//:ssl",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "openvino",
    actual = select({
        "//src:windows": "@windows_openvino//:openvino",
        "//conditions:default": "@linux_openvino//:openvino",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "boost",
    actual = select({
        "//src:windows": "@windows_boost//:boost",
        "//conditions:default": "@linux_boost//:boost",
    }),
    visibility = ["//visibility:public"],
)
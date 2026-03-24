load("//:common_settings.bzl", "ovms_cc_library")

COPTS_CLOUD = select({
    "//conditions:default": ["-DCLOUD_DISABLE=1"],
    "//:not_disable_cloud" : ["-DCLOUD_DISABLE=0"],
})

ovms_cc_library(
    name = "libovmsfilesystem",
    hdrs = ["filesystem.hpp",],
    srcs = ["filesystem.cpp",],
    deps = [
            "//src:libovmslogging",
            "//src:libovmsmodelversioning",
            "//src:libovmsstatus",
            "//src:libovmsstring_utils",
            "@boringssl//:ssl"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovmslocalfilesystem",
    hdrs = ["localfilesystem.hpp",],
    srcs = ["localfilesystem.cpp",],
    deps = ["libovmsfilesystem",
            "//src:libovmsstatus",
            "//src:libovmslogging"],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovmsfilesystemfactory_h",
    hdrs = ["filesystemfactory.hpp",],
    deps = [],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovmsfilesystemfactory",
    hdrs = [],
    srcs = ["filesystemfactory.cpp",],
    deps = ["libovmsfilesystem",
            "libovmsfilesystemfactory_h",
            "//src:libovmsstatus",
            "libovmslocalfilesystem",
            "//src:libovmslogging"
            ] + select({
            "//:not_disable_cloud": [
                "libovmsazurefilesystem",
                "libovmsgcsfilesystem",
                "libovmss3filesystem",
                ],
            "//:disable_cloud": []
            }),
    additional_copts = COPTS_CLOUD,
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovmsazurefilesystem",
    hdrs = ["azurefilesystem.hpp",],
    srcs = [
            "azurefilesystem.cpp",
            "azurestorage.cpp",
            "azurestorage.hpp",
    ],
    deps = [
        "@azure//:storage",
        "@cpprest//:sdk",
        "libovmsfilesystem",
        "//src:libovmslogging",
        "//src:libovmsstring_utils",
        "//src:libovmsmodelversioning",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovmsgcsfilesystem",
    hdrs = ["gcsfilesystem.hpp",],
    srcs = [
            "gcsfilesystem.cpp",
    ],
    deps = [
        "@google_cloud_cpp//:storage",
        "libovmsfilesystem",
        "//src:libovmslogging",
        "//src:libovmsstring_utils",
        "//src:libovmsmodelversioning",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

ovms_cc_library(
    name = "libovmss3filesystem",
    hdrs = ["s3filesystem.hpp",],
    srcs = [
            "s3filesystem.cpp",
    ],
    deps = [
        "@aws-sdk-cpp//:aws-sdk-cpp",
        "libovmsfilesystem",
        "//src:libovmslogging",
        "//src:libovmsstring_utils",
        "//src:libovmsmodelversioning",
        "//src:libovmsstatus",
    ],
    visibility = ["//visibility:public"],
)

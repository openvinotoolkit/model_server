
package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_binary(
    name = "server_cc",
    srcs = [
        "server.cc",
        "config.cpp",
        "config.h",
        "model.cpp",
        "model.h",
        "modelconfig.h",
        "modelmanager.cpp",
        "modelmanager.h",
        "modelinstance.cpp",
        "modelinstance.h",
        "ovstreams.cpp",
        "ovstreams.h",
        "prediction_service.cpp",
        "prediction_service.hpp",
        "status.h",
        "tensorinfo.h",
        "timer.h",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@openvino//:headers",
        "@openvino//:library",
        "@cxxopts//:cxxopts"
    ],
)

cc_test(
    name = "server_cc_test",
    srcs = [
        "test/unit_tests.cpp",
        "test/modelmanager_test.cpp",
        "config.cpp",
        "config.h",
        "model.cpp",
        "model.h",
        "modelconfig.h",
        "modelmanager.cpp",
        "modelmanager.h",
        "modelinstance.cpp",
        "modelinstance.h",
        "ovstreams.cpp",
        "ovstreams.h",
        "prediction_service.cpp",
        "prediction_service.hpp",
        "status.h",
        "tensorinfo.h",
        "timer.h",
    ],
    deps = [
        "//tensorflow_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@com_github_tencent_rapidjson//:rapidjson",
        "@openvino//:headers",
        "@openvino//:library",
        "@com_google_googletest//:gtest",
        "@cxxopts//:cxxopts"
    ],
)

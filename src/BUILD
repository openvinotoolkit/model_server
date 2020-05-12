cc_library(
    name = "ovms_lib",
    linkstatic = 1,
    srcs = [
        "config.cpp",
        "config.hpp",
        "deserialization.hpp",
        "directoryversionreader.hpp",
        "get_model_metadata_impl.cpp",
        "get_model_metadata_impl.hpp",
        "model.cpp",
        "model.hpp",
        "model_version_policy.cpp",
        "model_version_policy.hpp",
        "modelconfig.hpp",
        "modelmanager.cpp",
        "modelmanager.hpp",
        "modelinstance.cpp",
        "modelinstance.hpp",
        "model_service.hpp",
        "model_service.cpp",
        "ovinferrequestsqueue.cpp",
        "ovinferrequestsqueue.hpp",
        "prediction_service.cpp",
        "prediction_service.hpp",
        "serialization.hpp",
        "server.cpp",
        "status.hpp",
        "stringutils.hpp",
        "tensorinfo.hpp",
        "timer.hpp",
        "modelversionstatus.hpp",
    ],
    copts = [
         "--std=c++17"
    ],
    deps = [
        "@tensorflow_serving//tensorflow_serving/apis:prediction_service_cc_proto",
        "@tensorflow_serving//tensorflow_serving/apis:model_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:framework",
        "@rapidjson//:rapidjson",
        "@openvino//:openvino",
        "@spdlog//:spdlog",
        "@cxxopts//:cxxopts",
    ],
    local_defines = [
        "SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG"
    ]
)

cc_binary(
    name = "ovms",
    srcs = [
        "main.cpp",
    ],
    copts = [
        "--std=c++17"
    ],
    linkopts = [
        "-lstdc++fs"
    ],
    deps = [
        "//src:ovms_lib",
    ]
)

cc_test(
    name = "ovms_test",
    linkstatic = 1,
    srcs = [
        "test/deserialization_tests.cpp",
        "test/get_model_metadata_response_test.cpp",
        "test/get_model_metadata_signature_test.cpp",
        "test/get_model_metadata_validation_test.cpp",
        "test/unit_tests.cpp",
        "test/model_version_policy_test.cpp",
        "test/modelinstance_test.cpp",
        "test/modelmanager_test.cpp",
        "test/modelconfig_test.cpp",
        "test/ovtestutils.hpp",
        "test/ovinferrequestqueue_test.cpp",
        "test/predict_validation_test.cpp",
        "test/serialization_tests.cpp",
        "test/modelversionstatus_test.cpp",
        "test/model_service_test.cpp",
        "test/stringutils_test.cpp",
    ],
    data = [
        "test/dummy/0/dummy.xml",
        "test/dummy/0/dummy.bin",
    ],
    copts = [
        "--std=c++17"
    ],
    linkopts = [
        "-lstdc++fs"
    ],
    deps = [
        "//src:ovms_lib",
        "@com_google_googletest//:gtest",
    ],
)

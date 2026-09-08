load("//:common_settings.bzl", "COMMON_LOCAL_DEFINES", "COMMON_STATIC_LIBS_LINKOPTS", "COPTS_TESTS")

cc_test(
    name = "gemma4_generation_contract_test",
    srcs = ["gemma4_generation_contract_test.cpp"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "//src/llm:generation_config_builders",
        "//third_party:genai",
    ],
    copts = COPTS_TESTS,
    local_defines = COMMON_LOCAL_DEFINES,
    linkopts = COMMON_STATIC_LIBS_LINKOPTS,
    linkstatic = 1,
)
